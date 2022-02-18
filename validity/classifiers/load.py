from pathlib import Path

import torch

from .resnet import ResNet18, ResNet34, ResNet50, \
    train_network as resnet_train
#from .densenet import *
from .mnist import MnistClassifier, \
    train_network as mnist_train


def get_cls_path(cls_type, dataset, id=None):
    name = f'cls_{cls_type}_{dataset}'
    if id:
        name = f'{name}_{id}'
    return Path(f'models/{name}.pt')


def construct_cls(cls_type, dataset):
    if dataset == 'mnist':
        num_labels = 10
        in_channels = 1
    elif dataset == 'cifar10':
        num_labels = 10
        in_channels = 3

    if cls_type == 'mnist':
        net = MnistClassifier()
    elif cls_type == 'resnet18':
        net = ResNet18(num_labels, in_channels)
    elif cls_type == 'resnet34':
        net = ResNet34(num_labels, in_channels)
    elif cls_type == 'resnet50':
        net = ResNet50(num_labels, in_channels)

    else:
        raise Exception(f'Unknown classifier type {classifier_type}')

    return net


def load_cls(cls_type, weights_path, dataset):
    net = construct_cls(cls_type, dataset)
    net.load_state_dict(torch.load(weights_path))
    return net
