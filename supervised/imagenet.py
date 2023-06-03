import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from lib import models, datasets

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from lib.SupervisedMetricPretraining import SupervisedSoftmax
from test import NN, kNN

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset', required=True)
parser.add_argument('--model-dir', metavar='DIR',
                    default='./checkpoint/instance_imagenet', help='path to save model')
parser.add_argument('--log-dir', metavar='DIR',
                    default='./tensorboard/instance_imagenet', help='path to save log')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-vb', '--val-batch-size', default=128, type=int,
                    metavar='N', help='validation mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--auto-resume', action='store_true', help='auto resume')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-t', default=0.07, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--iter-size', default=1, type=int,
                    help='caffe style iter size')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](low_dim=args.low_dim)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    print("=> loading dataset")
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolderInstance(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolderInstance(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define lemniscate and loss function (criterion)
    print("=> building optimizer")
    ndata = train_dataset.__len__()
    
    lemniscate = LinearAverage(
        args.low_dim, ndata, args.nce_t, args.nce_m).cuda()
    criterion = SupervisedSoftmax(train_loader,device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    model_filename_to_resume = None
    if args.resume:
        if os.path.isfile(args.resume):
            model_filename_to_resume = args.resume
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.auto_resume:
        for epoch in range(args.epochs, args.start_epoch + 1, -1):
            model_filename = get_model_name(epoch)
            if os.path.exists(model_filename):
                model_filename_to_resume = model_filename
                break
        else:
            print("=> no checkpoint found at '{}'".format(args.model_dir))

    if model_filename_to_resume is not None:
        print("=> loading checkpoint '{}'".format(model_filename_to_resume))
        checkpoint = torch.load(model_filename_to_resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        lemniscate = checkpoint['lemniscate']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_filename_to_resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, lemniscate, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = NN(model, lemniscate, train_loader, val_loader)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename=get_model_name(epoch))
        
    # evaluate KNN after last epoch
    kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)


def train(train_loader, model, lemniscate, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (inputs, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        index = index.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # compute output
        feature = model(inputs)
        output = lemniscate(feature, index)
        loss = criterion(output, target)

        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, inputs.size(0))

        if (i + 1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}/{args.epochs}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')


def get_model_name(epoch):
    return os.path.join(args.model_dir, 'ckpt-{}.pth.tar'.format(epoch))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            args.model_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    if epoch < 120:
        lr = args.lr
    elif 120 <= epoch < 160:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    # lr = args_.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
