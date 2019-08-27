"""Train CIFAR10 with PyTorch."""
import argparse
import os
import random
import time
from pprint import pprint

import numpy as np
from skimage.color import rgb2gray
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

from lib.datasets import PseudoCIFAR10
from lib.utils import AverageMeter, accuracy, CosineAnnealingLRWithRestart
from lib.models import WideResNet, resnet18_cifar
from test import validate


def get_dataloader(args):
    if not args.input_gray:
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616))
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        to_gray = transforms.Lambda(lambda img: torch.from_numpy(
            rgb2gray(np.array(img))).unsqueeze(0).float())
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            to_gray,
        ])
        transform_test = to_gray

    testset = CIFAR10(root=args.data_dir, train=False,
                      download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    trainset = CIFAR10(root=args.data_dir, train=True,
                       download=True, transform=transform_test)

    args.ndata = len(trainset)
    num_labeled_data = args.num_labeled
    num_unlabeled_data = args.ndata - num_labeled_data

    if args.pseudo_file is not None:
        pseudo_dict = torch.load(args.pseudo_file)
        labeled_indexes = pseudo_dict['labeled_indexes']
    else:
        torch.manual_seed(args.rng_seed)
        perm = torch.randperm(args.ndata)
        labeled_indexes = perm[:num_labeled_data]

    pseudo_trainset = PseudoCIFAR10(
        labeled_indexes=labeled_indexes, root=args.data_dir,
        train=True, transform=transform_train)

    # load pseudo labels
    if args.pseudo_file is not None:
        pseudo_num = int(num_unlabeled_data * args.pseudo_ratio)
        pseudo_indexes = pseudo_dict['pseudo_indexes'][:pseudo_num]
        pseudo_labels = pseudo_dict['pseudo_labels'][:pseudo_num]
        pseudo_trainset.set_pseudo(pseudo_indexes, pseudo_labels)

    pseudo_trainloder = torch.utils.data.DataLoader(
        pseudo_trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    print('-' * 80)
    print('selected labeled indexes: ', labeled_indexes)

    return testloader, pseudo_trainloder


def build_model(args):
    if args.architecture == 'resnet18':
        net = resnet18_cifar(low_dim=args.num_class, norm=False)
    elif args.architecture.startswith('wrn'):
        split = args.architecture.split('-')
        net = WideResNet(depth=int(split[1]), widen_factor=int(split[2]),
                         num_classes=args.num_class, norm=False)
    else:
        raise ValueError('architecture should be resnet18 or wrn')
    if args.input_gray:
        net.conv1 = nn.Conv2d(1, net.conv1.out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
    net = net.to(args.device)

    print('#param: {}'.format(sum([p.nelement() for p in net.parameters()])))

    if args.device == 'cuda':
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # resume from unsupervised pretrain
    if len(args.resume) > 0:
        # Load checkpoint.
        print('==> Resuming from unsupervised pretrained checkpoint..')
        checkpoint = torch.load(args.resume)
        # only load shared conv layers, don't load fc
        model_dict = net.state_dict()
        if not args.input_gray:
            pretrained_dict = checkpoint['net']
        else:
            lst = ['conv1', 'block1', 'block2', 'block3']
            pretrained_dict = {
                'module.' + lst[int(k[0])] + k[1:]: v for k, v in checkpoint.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict
                           and v.size() == model_dict[k].size()}
        assert len(pretrained_dict) > 0
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    return net


def get_lr_scheduler(optimizer, lr_scheduler, max_iters):
    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, max_iters, eta_min=0.00001)
    elif args.lr_scheduler == 'cosine-with-restart':
        scheduler = CosineAnnealingLRWithRestart(optimizer, eta_min=0.00001)
    else:
        raise ValueError("not supported")

    return scheduler


# Training
def train(net, optimizer, scheduler, trainloader, testloader, criterion, summary_writer, args):
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    best_acc = 0
    end = time.time()

    def inf_generator(trainloader):
        while True:
            for data in trainloader:
                yield data

    for step, (inputs, targets) in enumerate(inf_generator(trainloader)):
        if step >= args.max_iters:
            break

        data_time.update(time.time() - end)

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # switch to train mode
        net.train()
        scheduler.step()
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets).mean()
        prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
        summary_writer.add_scalar('top1', top1.val, step)
        summary_writer.add_scalar('top2', top2.val, step)
        summary_writer.add_scalar('batch_time', batch_time.val, step)
        summary_writer.add_scalar('data_time', data_time.val, step)
        summary_writer.add_scalar('train_loss', train_loss.val, step)

        if step % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f'Train: [{step}/{args.max_iters}] '
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Lr: {lr:.5f} '
                  f'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                  f'prec2: {top2.val:.3f} ({top2.avg:.3f}) '
                  f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})')

        if (step + 1) % args.eval_freq == 0 or step == args.max_iters - 1:
            acc = validate(testloader, net, criterion,
                           device=args.device, print_freq=args.print_freq)

            summary_writer.add_scalar('val_top1', acc, step)

            if acc > best_acc:
                best_acc = acc
                state = {
                    'step': step,
                    'best_acc': best_acc,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                os.makedirs(args.model_dir, exist_ok=True)
                torch.save(state, os.path.join(args.model_dir, 'ckpt.pth.tar'))

            print('best accuracy: {:.2f}\n'.format(best_acc))


def main(args):
    # Data
    print('==> Preparing data..')
    testloader, pseudo_trainloder = get_dataloader(args)

    print('==> Building model..')
    net = build_model(args)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)

    criterion = nn.__dict__[args.criterion]().to(args.device)
    scheduler = get_lr_scheduler(optimizer, args.lr_scheduler, args.max_iters)

    if args.eval:
        return validate(testloader, net, criterion,
                        device=args.device, print_freq=args.print_freq)
    # summary writer
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(args.log_dir)

    train(net, optimizer, scheduler, pseudo_trainloder,
          testloader, criterion, summary_writer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--data_dir', '--dataDir', default='./data',
                        type=str, metavar='DIR')
    parser.add_argument('--model-root', default='./checkpoint/cifar10-semi',
                        type=str, metavar='DIR',
                        help='root directory to save checkpoint')
    parser.add_argument('--log-root', default='./tensorboard/cifar10-semi',
                        type=str, metavar='DIR',
                        help='root directory to save tensorboard logs')
    parser.add_argument('--exp-name', default='exp', type=str,
                        help='experiment name, used to determine log_dir and model_dir')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='learning rate')
    parser.add_argument('--lr-scheduler', default='cosine', type=str,
                        choices=['multi-step', 'cosine',
                                 'cosine-with-restart'],
                        help='which lr scheduler to use')
    parser.add_argument('--resume', '-r', default='', type=str,
                        metavar='FILE', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='test only')
    parser.add_argument('--finetune', action='store_true',
                        help='only training last fc layer')
    parser.add_argument('-j', '--num-workers', default=2, type=int,
                        metavar='N', help='number of workers to load data')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='batch size')
    parser.add_argument('--max-iters', default=500000, type=int,
                        metavar='N', help='number of iterations')
    parser.add_argument('--num-labeled', default=500, type=int,
                        metavar='N', help='number of labeled data')
    parser.add_argument('--rng-seed', default=0, type=int,
                        metavar='N', help='random number generator seed')
    parser.add_argument('--gpus', default='0', type=str, metavar='GPUS')
    parser.add_argument('--eval-freq', default=500, type=int,
                        metavar='N', help='eval frequence')
    parser.add_argument('--print-freq', default=100, type=int,
                        metavar='N', help='print frequence')
    parser.add_argument('--criterion', default='CrossEntropyLoss', type=str,
                        choices=['CrossEntropyLoss', 'MultiMarginLoss'])
    parser.add_argument('--pseudo-file', type=str,
                        metavar='FILE', help='pseudo file to load', required=True)
    parser.add_argument('--input-gray', action='store_true',
                        help='set for load colorization pretrained model, '
                             '(colorization model use gray image as input)')
    parser.add_argument('--pseudo-ratio', default=1, type=float, metavar='0-1',
                        help='ratio of unlabeled data to use for pseudo labels')
    parser.add_argument('--architecture', '--arch', default='wrn-28-2', type=str,
                        help='which backbone to use')
    args, rest = parser.parse_known_args()
    print(rest)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.num_class = 10
    args.log_dir = os.path.join(args.log_root, args.exp_name)
    args.model_dir = os.path.join(args.model_root, args.exp_name)

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed(args.rng_seed)
    random.seed(args.rng_seed)
    torch.set_printoptions(threshold=50, precision=4)

    print('-' * 80)
    pprint(vars(args))

    main(args)

    print('-' * 80)
    pprint(vars(args))
