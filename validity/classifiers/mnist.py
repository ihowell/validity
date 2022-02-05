import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import fire

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import transforms, datasets

from validity.classifiers.train import standard_train, adversarial_train_for_free
from validity.datasets import load_datasets
from validity.util import EarlyStopping


class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Resize((64, 64))
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 20, 5)
        self.dense = nn.Linear(3380, 10)

    def forward(self, x):
        out = x
        out = self.transform(out)
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
        out = self.transform(out)
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
        out = self.transform(out)
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

    def intermediate_forward(self, x, layer_idx):
        out = x
        out = self.transform(out)
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


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())


def train_network(max_epochs=1000, data_root='./datasets/', train_method=None, **kwargs):
    net = MnistClassifier()
    net = net.cuda()
    train_set, test_set = load_datasets('mnist')
    train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

    if train_method is None:
        standard_train(net, 'models/cls_mnist_mnist.pt', train_set, val_set, test_set,
                       **kwargs)
    elif train_method == 'adv_free':
        writer = SummaryWriter('tensorboard/cls_mnist_mnist_adv_free')
        adversarial_train_for_free(net,
                                   'models/cls_mnist_mnist_adv_free.pt',
                                   train_set,
                                   val_set,
                                   test_set,
                                   writer=writer,
                                   **kwargs)


def eval_network(weights_path, data_root='./', cuda_idx=0):
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = cuda_idx

    network = ResNet34(10)
    weights = torch.load(weights_path, map_location=torch.device('cpu'))
    network.load_state_dict(weights)
    if cuda_device is not None:
        network = network.cuda(cuda_device)

    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    in_train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=data_root,
                                                                   train=False,
                                                                   download=True,
                                                                   transform=in_transform),
                                                  batch_size=64,
                                                  shuffle=True)

    acc = []
    for data, labels in tqdm(in_train_loader):
        if cuda_device is not None:
            data = data.cuda(cuda_device)
        outputs = network(data).data.cpu()
        preds = torch.argmax(outputs, 1)
        acc.append(torch.mean(torch.where(preds == labels, 1.0, 0.0)))
    print('accuracy', np.mean(acc))


if __name__ == '__main__':
    fire.Fire()
