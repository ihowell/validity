from collections import namedtuple

import torch
from torchvision import datasets, transforms

DatasetInfo = namedtuple('DatasetInfo', ['num_labels', 'num_output_channels'])


class Binarize(object):
    """ This class introduces a binarization transformation
    """

    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def _bin_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        Binarize(),
    ])


def get_dataset_info(dataset):
    if dataset == 'mnist':
        return DatasetInfo(10, 1)
    elif dataset == 'fmnist':
        return DatasetInfo(10, 1)
    elif dataset == 'cifar10':
        return DatasetInfo(10, 3)
    else:
        raise Exception(f'get_dataset_info recieved unknown dataset: {dataset}')


def load_detector_datasets(dataset, detector_train_prop=0.1, data_root='./datasets'):
    """
    Args:
        - `dataset`: The name of the dataset to load
        - `detector_train_prop`: The proportion of elements to include in each detector's training set
    Returns:
        List of:
            - `train_cls`: Dataset used to train a classifier
            - `train_adv`: Dataset used to train a adversarial detector
            - `train_ood`: Dataset used to train a ood detector
            - `test`: Dataset used to test the detectors
    """
    assert detector_train_prop <= 0.5
    train_ds, test_ds = load_datasets(dataset, data_root=data_root)
    train_size = int(len(test_ds) * detector_train_prop)
    return [train_ds] + torch.utils.data.random_split(
        test_ds,
        [train_size, train_size, len(test_ds) - 2 * train_size],
        generator=torch.Generator().manual_seed(42))


def load_datasets(dataset, data_root='./datasets'):
    if dataset == 'mnist':
        train_ds = datasets.MNIST(root=data_root,
                                  train=True,
                                  download=True,
                                  transform=transforms.ToTensor())
        test_ds = datasets.MNIST(root=data_root,
                                 train=False,
                                 download=True,
                                 transform=transforms.ToTensor())
    elif dataset == 'fmnist':
        train_ds = datasets.FashionMNIST(root=data_root,
                                         train=True,
                                         download=True,
                                         transform=transforms.ToTensor())
        test_ds = datasets.FashionMNIST(root=data_root,
                                        train=False,
                                        download=True,
                                        transform=transforms.ToTensor())
    elif dataset == 'mnist_bin':
        train_ds = datasets.MNIST(root=data_root,
                                  train=True,
                                  download=True,
                                  transform=_bin_transforms())
        test_ds = datasets.MNIST(root=data_root,
                                 train=False,
                                 download=True,
                                 transform=_bin_transforms())
    elif dataset == 'fmnist_bin':
        train_ds = datasets.FashionMNIST(root=data_root,
                                         train=True,
                                         download=True,
                                         transform=_bin_transforms())
        test_ds = datasets.FashionMNIST(root=data_root,
                                        train=False,
                                        download=True,
                                        transform=_bin_transforms())
    elif dataset == 'cifar10':
        train_ds = datasets.CIFAR10(root=data_root,
                                    train=True,
                                    download=True,
                                    transform=transforms.ToTensor())
        test_ds = datasets.CIFAR10(root=data_root,
                                   train=False,
                                   download=True,
                                   transform=transforms.ToTensor())
    elif dataset == 'svhn':
        train_ds = datasets.SVHN(root=data_root,
                                 split='train',
                                 download=True,
                                 transform=transforms.ToTensor())
        test_ds = datasets.SVHN(root=data_root,
                                split='test',
                                download=True,
                                transform=transforms.ToTensor())
    else:
        raise Exception(f'Attempted to load unknown dataset: {dataset}')

    return train_ds, test_ds
