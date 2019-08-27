import time

import torch

from lib.utils import AverageMeter, get_train_labels, accuracy


def NN(net, lemniscate, trainloader, testloader, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    train_features = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        train_labels = torch.LongTensor(
            [y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        train_labels = get_train_labels(trainloader)
    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(
            trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            batch_size = inputs.size(0)
            features = net(inputs)
            train_features[:, batch_idx * batch_size:batch_idx *
                           batch_size + batch_size] = features.data.t()
        train_labels = get_train_labels(trainloader)
        trainloader.dataset.transform = transform_bak

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda(non_blocking=True)
            batch_size = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, train_features)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()

            cls_time.update(time.time() - end)
            end = time.time()

            print(f'Test [{total}/{testsize}]\t'
                  f'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  f'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  f'Top1: {correct * 100. / total:.2f}')

    return correct / total


def kNN(net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    train_features = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        train_labels = torch.LongTensor(
            [y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        train_labels = get_train_labels(trainloader)
    C = train_labels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(
            trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            bs = inputs.size(0)
            features = net(inputs)
            train_features[:, batch_idx * bs:batch_idx *
                           bs + bs] = features.data.t()
        train_labels = get_train_labels(trainloader)
        trainloader.dataset.transform = transform_bak

    top1 = 0.
    top5 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            bs = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, train_features)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(bs, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(bs * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(
                bs, -1, C), yd_transform.view(bs, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 2).sum().item()

            total += targets.size(0)

            if batch_idx % 100 == 0:
                print(f'Test [{total}/{testsize}]\t'
                      f'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                      f'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                      f'Top1: {top1 * 100. / total:.2f}  top5: {top5 * 100. / total:.2f}')

    print(top1 * 100. / total)

    return top1 / total


def validate(val_loader, model, criterion, device='cpu', print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}] '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                      f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

        print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')

    return top1.avg
