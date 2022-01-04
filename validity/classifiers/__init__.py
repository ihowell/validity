import torch

from .resnet import ResNet34, ResNet18
#from .densenet import *
from .mnist import MnistClassifier


def load_cls(classifier_type, weights_path, dataset):
    if dataset == 'mnist':
        num_ch = 1
    elif dataset == 'cifar10':
        num_ch = 3

    if classifier_type == 'mnist':
        classifier = MnistClassifier()
        classifier.load_state_dict(torch.load(weights_path, map_location=f'cuda:0'))
    elif classifier_type in ['resnet', 'resnet34']:
        # Normalization is for cifar10
        classifier = ResNet34(
            10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        classifier.load_state_dict(torch.load(weights_path, map_location=f'cuda:{cuda_idx}'))
    elif classifier_type in ['resnet18']:
        classifier = ResNet18(10, num_ch)
        classifier.load_state_dict(torch.load(weights_path, map_location=f'cuda:0'))
    else:
        raise Exception(f'Unknown classifier type {classifier_type}')

    classifier = classifier.cuda()
    return classifier
