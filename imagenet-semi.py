"""Train ImageNet with PyTorch."""
import argparse
import os
import random
import time
import glob
from pprint import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler

from lib.datasets import PseudoDatasetFolder
from lib.utils import AverageMeter, accuracy, CosineAnnealingLRWithRestart
from test import validate

best_acc = 0
global_step = 0


def get_dataloader(args):
    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')

    testset = datasets.ImageFolder(valdir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    trainset = datasets.ImageFolder(traindir, transform=transform_train)

    # split labeled and unlabeled
    args.ndata = len(trainset)
    num_labeled = args.num_labeled
    num_unlabeled = args.ndata - num_labeled

    torch.manual_seed(args.rng_seed)
    perm = torch.randperm(args.ndata)

    index_labeled = []
    index_unlabeled = []
    data_per_class = num_labeled // args.num_class
    train_labels = torch.Tensor([x[1] for x in trainset.samples])
    for c in range(args.num_class):
        indexes_c = perm[train_labels[perm] == c]
        index_labeled.append(indexes_c[:data_per_class])
        index_unlabeled.append(indexes_c[data_per_class:])

    args.index_labeled = torch.cat(index_labeled)
    args.index_unlabeled = torch.cat(index_unlabeled)

    print('-' * 80)
    print('selected labeled indexes: ', args.index_labeled)

    pseudo_trainset = PseudoDatasetFolder(trainset, labeled_indexes=args.index_labeled)
    # load pseudo labels
    if args.pseudo_dir is not None:
        pseudo_files = glob.glob(args.pseudo_dir + '/*')
        pseudo_num_per_chunk = int(
            num_unlabeled * args.pseudo_ratio / len(pseudo_files))

        pseudo_indexes = []
        pseudo_labels = []
        for pseudo_file in pseudo_files:
            pseudo_dict = torch.load(pseudo_file)
            pseudo_indexes.append(pseudo_dict['pseudo_indexes'][:pseudo_num_per_chunk])
            pseudo_labels.append(pseudo_dict['pseudo_labels'][:pseudo_num_per_chunk])
            assert (args.index_labeled == pseudo_dict['labeled_indexes']).all()
        pseudo_indexes = torch.cat(pseudo_indexes)
        pseudo_labels = torch.cat(pseudo_labels)

        assert num_labeled == args.index_labeled.shape[0]

        pseudo_trainset.set_pseudo(pseudo_indexes, pseudo_labels)

        print('num_pseudo = {}'.format(pseudo_indexes.shape[0]))

    pseudo_trainloder = torch.utils.data.DataLoader(
        pseudo_trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    return testloader, pseudo_trainloder


def build_model(args):
    print("=> creating model '{}'".format(args.architecture))
    net = models.__dict__[args.architecture]()
    net = net.to(args.device)

    print('#param: {}'.format(sum([p.nelement() for p in net.parameters()])))

    if args.device == 'cuda':
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=0, nesterov=True)

    # resume from unsupervised pretrain
    if len(args.resume) > 0:
        print('==> Resuming from {}'.format(args.resume))
        global best_acc, global_step
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        global_step = checkpoint['step'] + 1
    elif len(args.pretrained) > 0:
        # Load checkpoint.
        print('==> Load pretrained model: {}'.format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model_dict = net.state_dict()
        # only load shared conv layers, don't load fc
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items()
                           if k in model_dict
                           and v.size() == model_dict[k].size()}
        assert len(pretrained_dict) > 0
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    return net, optimizer


def get_lr_scheduler(optimizer, lr_scheduler, max_iters):
    if lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, max_iters, eta_min=0.00001)
    elif lr_scheduler == 'cosine-with-restart':
        scheduler = CosineAnnealingLRWithRestart(optimizer, eta_min=0.00001)
    elif lr_scheduler == 'multi-step':
        scheduler = MultiStepLR(optimizer, [max_iters * 3 // 7, max_iters * 6 // 7], gamma=0.1)
    else:
        raise ValueError("not supported")

    return scheduler


def inf_generator(trainloader):
    while True:
        for data in trainloader:
            yield data


# Training
def train(net, optimizer, scheduler, trainloader, testloader, criterion, summary_writer, args):
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    best_acc = 0
    end = time.time()

    global global_step
    for inputs, targets in inf_generator(trainloader):
        if global_step >= args.max_iters:
            break

        data_time.update(time.time() - end)

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # switch to train mode
        net.train()
        scheduler.step(global_step)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        summary_writer.add_scalar('top1', top1.val, global_step)
        summary_writer.add_scalar('top5', top5.val, global_step)
        summary_writer.add_scalar('batch_time', batch_time.val, global_step)
        summary_writer.add_scalar('data_time', data_time.val, global_step)
        summary_writer.add_scalar('train_loss', train_loss.val, global_step)

        if global_step % args.print_freq == 0:
            print('Train: [{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Lr: {lr:.5f} '
                  'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                  'prec5: {top5.val:.3f} ({top5.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                global_step, args.max_iters, lr=optimizer.param_groups[0]['lr'],
                batch_time=batch_time, data_time=data_time,
                top1=top1, top5=top5, train_loss=train_loss))

        if (global_step + 1) % args.eval_freq == 0 or global_step == args.max_iters - 1:
            acc = validate(testloader, net, criterion,
                           device=args.device, print_freq=args.print_freq)

            summary_writer.add_scalar('val_top1', acc, global_step)

            if acc > best_acc:
                best_acc = acc
                state = {
                    'step': global_step,
                    'best_acc': best_acc,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                os.makedirs(args.model_dir, exist_ok=True)
                torch.save(state, os.path.join(args.model_dir, 'ckpt.pth.tar'))

            print('best accuracy: {:.2f}\n'.format(best_acc))
        global_step += 1


def main(args):
    # Data
    print('==> Preparing data..')
    testloader, pseudo_trainloder = get_dataloader(args)

    print('==> Building model..')
    net, optimizer = build_model(args)

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
    parser = argparse.ArgumentParser(description='PyTorch Imagenet Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', '--dataDir', required=True,
                        type=str, metavar='DIR', help='data dir')
    parser.add_argument('--model-root', default='./checkpoint/imagenet',
                        type=str, metavar='DIR',
                        help='root directory to save checkpoint')
    parser.add_argument('--log-root', default='./tensorboard/imagenet',
                        type=str, metavar='DIR',
                        help='root directory to save tensorboard logs')
    parser.add_argument('--exp-name', default='exp', type=str,
                        help='experiment name, used to determine log_dir and model_dir')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='learning rate')
    parser.add_argument('--lr-scheduler', default='multi-step', type=str,
                        choices=['multi-step', 'cosine',
                                 'cosine-with-restart'],
                        help='which lr scheduler to use')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='FILE', help='The pretrained checkpoint to load. Only load model parametric')
    parser.add_argument('--resume', '-r', default='', type=str,
                        metavar='FILE', help='resume from checkpoint. Optimizer state will be resumed too')
    parser.add_argument('--eval', action='store_true', help='test only')
    parser.add_argument('--finetune', action='store_true',
                        help='only training last fc layer')
    parser.add_argument('-j', '--num-workers', default=32, type=int,
                        metavar='N', help='number of workers to load data')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='batch size')
    parser.add_argument('--max-iters', default=50000, type=int,
                        metavar='N', help='number of iterations')
    parser.add_argument('--num-labeled', default=13000, type=int,
                        metavar='N', help='number of labeled data')
    parser.add_argument('--rng-seed', default=0, type=int,
                        metavar='N', help='random number generator seed')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, metavar='GPUS',
                        help='ids of GPU to use')
    parser.add_argument('--eval-freq', default=500, type=int,
                        metavar='N', help='eval frequence')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequence')
    parser.add_argument('--criterion', default='CrossEntropyLoss', type=str,
                        choices=['CrossEntropyLoss', 'MultiMarginLoss'], help='Criterion to use')
    parser.add_argument('--pseudo-dir', type=str,
                        metavar='PATH', help='pseudo folder to load')
    parser.add_argument('--pseudo-ratio', default=0.1, type=float, metavar='0-1',
                        help='ratio of unlabeled data to use for pseudo labels')
    parser.add_argument('--architecture', '--arch', default='resnet18', type=str,
                        help='which backbone to use')
    args_, rest = parser.parse_known_args()
    print(rest)

    os.environ["CUDA_VISIBLE_DEVICES"] = args_.gpus
    args_.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_.num_class = 1000
    args_.log_dir = os.path.join(args_.log_root, args_.exp_name)
    args_.model_dir = os.path.join(args_.model_root, args_.exp_name)

    torch.manual_seed(args_.rng_seed)
    torch.cuda.manual_seed(args_.rng_seed)
    random.seed(args_.rng_seed)
    torch.set_printoptions(threshold=50, precision=4)

    print('-' * 80)
    pprint(vars(args_))

    main(args_)

    print('-' * 80)
    pprint(vars(args_))
