import torch
from torchvision import datasets, transforms


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
