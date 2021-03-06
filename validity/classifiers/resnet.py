'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import os
import math
from pathlib import Path

import fire
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm

from validity.classifiers.train import train_ds
from validity.datasets import get_dataset_info


def conv3x3(in_planes, out_planes, stride=1, spectral_normalization=False):
    conv = nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
    if spectral_normalization:
        conv = spectral_norm(conv)
    return conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, spectral_normalization=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes,
                             planes,
                             stride,
                             spectral_normalization=spectral_normalization)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, spectral_normalization=spectral_normalization)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            conv = nn.Conv2d(in_planes,
                             self.expansion * planes,
                             kernel_size=1,
                             stride=stride,
                             bias=False), nn.BatchNorm2d(self.expansion * planes)
            if spectral_normalization:
                conv = spectral_norm(conv)
            self.shortcut = nn.Sequential(conv)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, spectral_normalization=False):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes,
                             planes,
                             stride,
                             spectral_normalization=spectral_normalization)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, spectral_normalization=spectral_normalization)

        if stride != 1 or in_planes != self.expansion * planes:
            conv = nn.Conv2d(in_planes,
                             self.expansion * planes,
                             kernel_size=1,
                             stride=stride,
                             bias=False)
            if spectral_normalization:
                conv = spectral_norm(conv)
            self.shortcut = nn.Sequential(conv)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, spectral_normalization=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=1,
                               bias=False,
                               spectral_normalization=spectral_normalization)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False,
                               spectral_normalization=spectral_normalization)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False,
                               spectral_normalization=spectral_normalization)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            conv = nn.Conv2d(in_planes,
                             self.expansion * planes,
                             kernel_size=1,
                             stride=stride,
                             bias=False), nn.BatchNorm2d(self.expansion * planes)
            if spectral_normalization:
                conv = spectral_norm(conv)
            self.shortcut = nn.Sequential(conv)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, spectral_normalization=False):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=1,
                               bias=False,
                               spectral_normalization=spectral_normalization)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False,
                               spectral_normalization=spectral_normalization)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False,
                               spectral_normalization=spectral_normalization)

        if stride != 1 or in_planes != self.expansion * planes:
            conv = nn.Conv2d(in_planes,
                             self.expansion * planes,
                             kernel_size=1,
                             stride=stride,
                             bias=False)
            if spectral_normalization:
                conv = spectral_norm(conv)
            self.shortcut = nn.Sequential(conv)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):

    @classmethod
    def load(cls, saved_dict):
        (args, kwargs) = saved_dict['args']
        model = cls(*args, **kwargs)
        model.load_state_dict(saved_dict['state_dict'])
        return model

    def __init__(self,
                 block_name,
                 num_blocks,
                 num_classes=10,
                 in_channels=3,
                 spectral_normalization=False):
        super(ResNet, self).__init__()
        self.block_name = block_name
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.spectral_normalization = spectral_normalization
        block = get_block_cls(block_name)
        self.in_planes = 64

        self.conv1 = conv3x3(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def get_type(self):
        return 'resnet'

    def get_args(self):
        return ((self.block_name, self.num_blocks), {
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'spectral_normalization': self.spectral_normalization,
        })

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes,
                      planes,
                      stride,
                      spectral_normalization=self.spectral_normalization))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate


def get_block_cls(block_name):
    if block_name == 'pre_act':
        return PreActBlock
    elif block_name == 'basic':
        return BasicBlock
    elif block_name == 'bottleneck':
        return Bottleneck
    else:
        raise Exception(f'Could not find block with name "{block_name}"')


def ResNet18(num_c, in_ch, spectral_normalization=False):
    return ResNet('pre_act', [2, 2, 2, 2],
                  num_classes=num_c,
                  in_channels=in_ch,
                  spectral_normalization=spectral_normalization)


def ResNet34(num_c, in_ch, spectral_normalization=False):
    return ResNet('basic', [3, 4, 6, 3],
                  num_classes=num_c,
                  in_channels=in_ch,
                  spectral_normalization=spectral_normalization)


def ResNet50(num_c, in_ch, spectral_normalization=False):
    return ResNet('bottleneck', [3, 4, 6, 3],
                  num_classes=num_c,
                  in_channels=in_ch,
                  spectral_normalization=spectral_normalization)


def ResNet101():
    return ResNet('bottleneck', [3, 4, 23, 3])


def ResNet152():
    return ResNet('bottleneck', [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())


# test()

# def eval_network(weights_path, data_root='./', cuda_idx=0):
#     from validity.classifiers.resnet import ResNet34
#     cuda_device = None
#     if torch.cuda.is_available():
#         cuda_device = cuda_idx

#     network = ResNet34(10)
#     weights = torch.load(weights_path, map_location=torch.device('cpu'))
#     network.load_state_dict(weights)
#     if cuda_device is not None:
#         network = network.cuda(cuda_device)

#     in_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     in_train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=data_root,
#                                                                    train=False,
#                                                                    download=True,
#                                                                    transform=in_transform),
#                                                   batch_size=64,
#                                                   shuffle=True)

#     acc = []
#     for data, labels in tqdm(in_train_loader):
#         if cuda_device is not None:
#             data = data.cuda(cuda_device)
#         outputs = network(data).data.cpu()
#         preds = torch.argmax(outputs, 1)
#         acc.append(torch.mean(torch.where(preds == labels, 1.0, 0.0)))
#     print('accuracy', np.mean(acc))


def train_network(dataset, net_type, id=None, spectral_normalization=False, **kwargs):
    ds_info = get_dataset_info(dataset)

    if net_type == 'resnet18':
        net = ResNet18
    elif net_type == 'resnet34':
        net = ResNet34
    elif net_type == 'resnet50':
        net = ResNet50
    net = net(ds_info.num_labels,
              ds_info.num_channels,
              spectral_normalization=spectral_normalization)
    net = net.cuda()
    net.train()

    name = f'cls_{net_type}_{dataset}'
    if id:
        name = f'{name}_{id}'
    net_path = Path(f'models/{name}.pt')
    tensorboard_path = f'tensorboard/resnet/{name}'

    train_ds(net, net_path, dataset, tensorboard_path=tensorboard_path, **kwargs)


if __name__ == '__main__':
    fire.Fire()
