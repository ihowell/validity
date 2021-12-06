import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import fire

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import transforms, datasets


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


def train_network(num_epochs=10, data_root='./datasets/', cuda_idx=0):
    net = MnistClassifier()
    net = net.cuda()
    in_train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        root=data_root, train=True, download=True, transform=transforms.ToTensor()),
                                                  batch_size=64,
                                                  shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(num_epochs):
        for data, labels in tqdm(in_train_loader):
            optimizer.zero_grad()
            outputs = net(data.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

    in_test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        root=data_root, train=False, download=True, transform=transforms.ToTensor()),
                                                 batch_size=64,
                                                 shuffle=False)

    losses = []
    correct = 0.
    total = 0.
    with torch.no_grad():
        for data, labels in in_test_loader:
            outputs = net(data.cuda())
            loss = criterion(outputs, labels.cuda())
            losses.append(loss)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()

    print(f'Accuracy {correct / total}')

    torch.save(net.state_dict(), 'classifiers/mnist.pth')


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
