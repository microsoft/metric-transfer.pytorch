from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import numpy as np


class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class PseudoCIFAR10(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    def __init__(self, labeled_indexes, **kwargs):
        super(PseudoCIFAR10, self).__init__(**kwargs)
        assert self.train
        self.labeled_indexes = labeled_indexes.cpu().numpy().copy()
        self.C = 10
        self.labels = np.array(self.targets)[self.labeled_indexes]
        self.indexes = self.labeled_indexes

    def __len__(self):
        return self.indexes.shape[0]

    def set_pseudo(self, pseudo_indexes, pseudo_labels):
        assert pseudo_indexes.shape == pseudo_labels.shape

        self.labels = np.concatenate(
            [np.array(self.targets)[self.labeled_indexes], pseudo_labels.cpu().numpy().copy()], axis=0)
        self.indexes = np.concatenate([self.labeled_indexes, pseudo_indexes.cpu().numpy().copy()], axis=0)

    def __getitem__(self, index):
        real_index = self.indexes[index]
        img = self.data[real_index]
        target = self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == '__main__':
    import torchvision.transforms as transforms

    _labeled_indexes = torch.arange(10)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = PseudoCIFAR10(
        labeled_indexes=_labeled_indexes,
        root='./data',
        transform=transform_train,
        download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True, num_workers=0)
    assert len(loader) == 2
    for i, (_img, _target) in enumerate(loader):
        print(_img.shape, _target)
        break

    # test pseudo
    _pseudo_indexes = torch.arange(100, 200)
    _pseudo_labels = torch.zeros([100])
    loader.dataset.set_pseudo(_pseudo_indexes, _pseudo_labels)
    assert len(loader) == 22  # (100 + 10) / 5
    for i, (_img, _target) in enumerate(loader):
        print(_img.shape, _target)
        break
