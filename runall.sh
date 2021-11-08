#!/bin/bash

python -m validity.detectors.odin train_multiple_odin resnet ./classifiers/cifar10/resnet_cifar10.pth
python -m validity.detectors.mahalanobis train_multiple_mahalanobis_ood resnet ./classifiers/cifar10/resnet_cifar10.pth
python -m validity.detectors.lid train_multiple_lid_adv resnet ./classifiers/cifar10/resnet_cifar10.pth cwl2
python -m validity.detectors.mahalanobis train_multiple_mahalanobis_adv resnet ./classifiers/cifar10/resnet_cifar10.pth cwl2

python -m validity.joint_ood_adv resnet cifar10 svhn cwl2
