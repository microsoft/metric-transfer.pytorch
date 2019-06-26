import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class PseudoDatasetFolder(Dataset):

    def __init__(self, ds, labeled_indexes):
        self.ds = ds
        self.labeled_indexes = labeled_indexes
        self.num_labeled = len(self.labeled_indexes)
        # self.labeled_indexes_set = set(labeled_indexes.cpu().numpy())
        self.pseudo_indexes = []
        self.pseudo_labels = None

    def __len__(self):
        return self.num_labeled + len(self.pseudo_indexes)

    def __getitem__(self, index):

        if index < self.num_labeled:
            # labeled
            real_index = self.labeled_indexes[index]
            sample, target = self.ds[real_index]
        else:
            # pseudo
            real_index = self.pseudo_indexes[index - self.num_labeled]
            sample, _ = self.ds[real_index]
            target = self.pseudo_labels[index - self.num_labeled]
        return sample, target

    def set_pseudo(self, pseudo_indexes, pseudo_labels):
        assert len(pseudo_indexes) == len(pseudo_labels)
        self.pseudo_indexes = pseudo_indexes
        if isinstance(pseudo_labels, torch.Tensor):
            pseudo_labels = pseudo_labels.cpu().numpy()
        self.pseudo_labels = pseudo_labels


if __name__ == '__main__':
    from torchvision import datasets
    import torchvision.transforms as transforms
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    trainset = datasets.ImageFolder('/home/liubin/data/imagenet/train/', transform=transform_test)
    # test list
    labeled_indexes_ = [1]
    pseudo_indexes_, pseudo_labels_ = [2], [10]
    pseudo_trainset = PseudoDatasetFolder(trainset, labeled_indexes=labeled_indexes_)
    pseudo_trainset.set_pseudo(pseudo_indexes_, pseudo_labels_)
    for i, (_, target_) in enumerate(pseudo_trainset):
        if i == 0:
            assert target_ == 0
        else:
            assert target_ == 10

    # test np array
    import numpy as np
    labeled_indexes_ = np.array([1])
    pseudo_indexes_, pseudo_labels_ = np.array([2]), np.array([10])
    pseudo_trainset = PseudoDatasetFolder(trainset, labeled_indexes=labeled_indexes_)
    pseudo_trainset.set_pseudo(pseudo_indexes_, pseudo_labels_)
    for i, (_, target_) in enumerate(pseudo_trainset):
        if i == 0:
            assert target_ == 0
        else:
            assert target_ == 10

    # test torch tensor
    n = len(trainset)
    num_labeled = n // 2
    labeled_indexes_ = torch.arange(num_labeled)
    pseudo_indexes_ = torch.arange(num_labeled, n)
    pseudo_labels_ = torch.zeros([n - num_labeled], dtype=torch.int64)
    pseudo_trainset = PseudoDatasetFolder(trainset, labeled_indexes=labeled_indexes_)
    pseudo_trainset.set_pseudo(pseudo_indexes_, pseudo_labels_)
    assert pseudo_trainset[0][1] == trainset.samples[0][1]
    assert pseudo_trainset[num_labeled][1] == 0

    # test loader
    pseudo_trainloder = torch.utils.data.DataLoader(
        pseudo_trainset, batch_size=256,
        shuffle=True, num_workers=8)

    for data_, target_ in pseudo_trainloder:
        print(data_, target_)
        break
