import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class MnistClassifier(nn.Module):

    @classmethod
    def load(cls, saved_dict):
        args = saved_dict['args']
        self = cls(**args)
        self.load_state_dict(saved_dict['state_dict'])
        return self

    def __init__(self, spectral_normalization=False):
        super().__init__()
        self.spectral_normalization = spectral_normalization
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 20, 5)
        self.dense = nn.Linear(3380, 10)

        if spectral_normalization:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.dense = spectral_norm(self.dense)

    def get_type(self):
        return 'mnist'

    def get_args(self):
        return {'spectral_normalization': self.spectral_normalization}

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