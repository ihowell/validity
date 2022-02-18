import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

from validity.generators.nvae.model import load_nvae

if __name__ == '__main__':
    model = ResNet34(10,
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    model.load_state_dict(
        torch.load('/mnt/d/research/classifiers/cifar10/resnet_cifar10.pth',
                   map_location='cuda:0'))
    model.cuda()

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='/mnt/d/research/datasets/',
        train=False,
        download=True,
        transform=transforms.ToTensor()),
                                              batch_size=64,
                                              shuffle=True)

    accuracies = []
    for data, label in tqdm(test_loader):
        data = data.cuda()
        label = label.cuda()

        logits = model(data)
        preds = torch.argmax(logits, 1)
        accuracies.append(
            torch.mean(torch.where(preds == label, 1., 0.)).cpu().detach().numpy())

    print('Accuracy', np.mean(accuracies))
