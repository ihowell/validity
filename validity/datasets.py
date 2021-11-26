from torchvision import datasets, transforms


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
    elif dataset == 'cifar10':
        train_ds = datasets.CIFAR10(root=data_root,
                                    train=True,
                                    download=True,
                                    transform=transforms.ToTensor())
        test_ds = datasets.CIFAR10(root=data_root,
                                   train=False,
                                   download=True,
                                   transform=transforms.ToTensor())
    elif datasets == 'svhn':
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
