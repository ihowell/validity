from pathlib import Path

import torch

from .resnet import ResNet, ResNet18, ResNet34, ResNet50
#from .densenet import *
from .mnist import MnistClassifier


def get_cls_path(cls_type, dataset, id=None):
    name = f'cls_{cls_type}_{dataset}'
    if id:
        name = f'{name}_{id}'
    return Path(f'models/{name}.pt')


def construct_cls(cls_type, dataset, cls_kwargs=None):
    if cls_kwargs is None:
        cls_kwargs = {}
    if dataset == 'mnist':
        num_labels = 10
        in_channels = 1
    elif dataset == 'cifar10':
        num_labels = 10
        in_channels = 3

    if cls_type == 'mnist':
        net = MnistClassifier(**cls_kwargs)
    elif cls_type == 'resnet18':
        net = ResNet18(num_labels, in_channels, **cls_kwargs)
    elif cls_type == 'resnet34':
        net = ResNet34(num_labels, in_channels, **cls_kwargs)
    elif cls_type == 'resnet50':
        net = ResNet50(num_labels, in_channels, **cls_kwargs)

    else:
        raise Exception(f'Unknown classifier type {cls_type}')

    return net


def load_cls(weights_path):
    saved_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    cls_type = saved_dict['type']

    if cls_type == 'mnist':
        net = MnistClassifier.load(saved_dict)
    elif cls_type == 'resnet':
        net = ResNet.load(saved_dict)
    else:
        raise Exception(f'Unknown classifier type: {cls_type}')
    return net
