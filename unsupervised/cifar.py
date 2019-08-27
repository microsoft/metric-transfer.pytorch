"""Train CIFAR10 with PyTorch."""
import argparse
import os
import sys
import time
from pprint import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from lib import datasets, models
from lib.LinearAverage import LinearAverage
from lib.NCEAverage import NCEAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter, CosineAnnealingLRWithRestart
from test import kNN


# Training
def train(net, optimizer, trainloader, criterion, lemniscate, epoch):
    print('\nEpoch: {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(args.device), targets.to(
            args.device), indexes.to(args.device)
        optimizer.zero_grad()

        features = net(inputs)
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs, indexes)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 100 == 0:
            print(f'Epoch: [{epoch}/{args.epoch}][{batch_idx}/{len(trainloader)}] '
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})')


def get_data_loader():
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if args.transform_crop == 'RandomResizedCrop':
        crop = transforms.RandomResizedCrop(
            size=32, scale=(args.transform_scale, 1.))
    else:
        crop = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32)
        ])
    transform_train = transforms.Compose([
        crop,
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    print('-' * 80)
    print('transform_train = ', transform_train)
    print('-' * 80)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.CIFAR10Instance(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10Instance(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    ndata = trainset.__len__()

    return trainloader, testloader, ndata


def build_model():
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.architecture == 'resnet18':
        net = models.__dict__['resnet18_cifar'](low_dim=args.low_dim)
    elif args.architecture == 'wrn-28-2':
        net = models.WideResNet(
            depth=28, num_classes=args.low_dim, widen_factor=2, dropRate=0).to(args.device)
    elif args.architecture == 'wrn-28-10':
        net = models.WideResNet(
            depth=28, num_classes=args.low_dim, widen_factor=10, dropRate=0).to(args.device)

    # define leminiscate
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, args.ndata,
                                args.nce_k, args.nce_t, args.nce_m)
    else:
        lemniscate = LinearAverage(
            args.low_dim, args.ndata, args.nce_t, args.nce_m)

    if args.device == 'cuda':
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True)
    # Model
    if args.test_only or len(args.resume) > 0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lemniscate = checkpoint['lemniscate']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    if args.lr_scheduler == 'multi-step':
        if args.epochs == 200:
            steps = [60, 120, 160]
        elif args.epochs == 600:
            steps = [180, 360, 480, 560]
        else:
            raise RuntimeError(
                f"need to config steps for epoch = {args.epochs} first.")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, steps, gamma=0.2, last_epoch=start_epoch - 1)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0.00001, last_epoch=start_epoch - 1)
    elif args.lr_scheduler == 'cosine-with-restart':
        scheduler = CosineAnnealingLRWithRestart(
            optimizer, eta_min=0.00001, last_epoch=start_epoch - 1)
    else:
        raise ValueError("not supported")

    # define loss function
    if hasattr(lemniscate, 'K'):
        criterion = NCECriterion(args.ndata)
    else:
        criterion = nn.CrossEntropyLoss()

    net.to(args.device)
    lemniscate.to(args.device)
    criterion.to(args.device)

    return net, lemniscate, optimizer, criterion, scheduler, best_acc, start_epoch


def main():
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    trainloader, testloader, args.ndata = get_data_loader()

    print('==> Building model..')
    net, lemniscate, optimizer, criterion, scheduler, best_acc, start_epoch = build_model()

    if args.test_only:
        kNN(net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
        sys.exit(0)

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        train(net, optimizer, trainloader, criterion, lemniscate, epoch)
        acc = kNN(net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'lemniscate': lemniscate,
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(state, os.path.join(
                args.model_dir, 'ckpt.cifar.pth.tar'))
            best_acc = acc

        print('best accuracy: {:.2f}'.format(best_acc * 100))

    acc = kNN(net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    print('last accuracy: {:.2f}'.format(acc * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--data-dir', '--dataDir',
                        default='./data', type=str, metavar='DIR')
    parser.add_argument('--model-dir', '--modelDir', default='./checkpoint/instance_cifar10', type=str,
                        metavar='DIR', help='directory to save checkpoint')
    parser.add_argument('--log-dir', '--logDir', default='./tensorboard/instance_cifar10', type=str,
                        metavar='DIR', help='directory to save tensorboard logs')
    parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
    parser.add_argument('--lr-scheduler', default='cosine', type=str,
                        choices=['multi-step', 'cosine',
                                 'cosine-with-restart'],
                        help='which lr scheduler to use')
    parser.add_argument('--resume', '-r', default='',
                        type=str, help='resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--low-dim', default=128, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--nce-k', default=0, type=int,
                        metavar='K', help='number of negative samples for NCE')
    parser.add_argument('--nce-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.5, type=float,
                        metavar='M', help='momentum for non-parametric updates')
    parser.add_argument('--epochs', default=600, type=int,
                        metavar='N', help='number of epochs')
    parser.add_argument('--architecture', '--arch', default='wrn-28-2', type=str,
                        choices=['resnet18', 'wrn-28-2', 'wrn-28-10'],
                        help='which backbone to use')
    parser.add_argument('--transform-scale', default=0.2, type=float)
    parser.add_argument('--transform-crop', type=str, default='RandomResizedCrop',
                        choices=['RandomResizedCrop', 'PadCrop'])
    parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4)
    args = parser.parse_args()

    pprint(vars(args))

    main()

    pprint(vars(args))
