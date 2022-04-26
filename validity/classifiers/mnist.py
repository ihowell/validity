import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from tqdm import tqdm
from tensorboardX import SummaryWriter
import fire

from torchvision import transforms, datasets
import numpy as np

from validity.classifiers.train import standard_train, adversarial_train_for_free
from validity.datasets import load_datasets
from validity.util import EarlyStopping


class MnistClassifier(nn.Module):

    def __init__(self, spectral_normalization=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 20, 5)
        self.dense = nn.Linear(3380, 10)

        if spectral_normalization:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.dense = spectral_norm(self.dense)

    def forward(self, x):
        out = x
        out = F.interpolate(x, size=(64, 64))
        out = self.conv1(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out

    def penultimate_forward(self, x):
        out = x
        out = F.interpolate(x, size=(64, 64))
        out = self.conv1(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        penultimate = out
        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out, penultimate

    def feature_list(self, x):
        features = []
        out = x
        out = F.interpolate(x, size=(64, 64))
        out = self.conv1(out)
        features.append(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        out = self.conv2(out)
        features.append(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out, features

    def post_relu_features(self, x):
        features = []
        out = x
        out = F.interpolate(x, size=(64, 64))
        out = self.conv1(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        features.append(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        features.append(out)

        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out, features

    def intermediate_forward(self, x, layer_idx):
        out = x
        out = F.interpolate(x, size=(64, 64))
        out = self.conv1(out)
        if layer_idx == 0:
            return out
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        out = self.conv2(out)
        if layer_idx == 1:
            return out
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out