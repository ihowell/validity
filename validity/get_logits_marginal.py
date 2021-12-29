import fire
import numpy as np
import torch

from tqdm import tqdm
from torchvision import datasets, transforms

from validity.classifiers.mnist import MnistClassifier
from validity.classifiers.resnet import ResNet34
from validity.datasets import load_datasets


def main(ds_name, net_type, weights_location, data_root='./datasets/', cuda_idx=0):
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    if net_type == 'mnist':
        network = MnistClassifier()
        network.load_state_dict(torch.load(weights_location, map_location=f'cuda:0'))
    elif net_type == 'resnet':
        network = ResNet34(
            10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        network.load_state_dict(torch.load(weights_location, map_location=f'cuda:0'))

    network = network.cuda()

    # Load datasets
    train_ds, _ = load_datasets(ds_name, data_root=data_root)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    logit_labels = []
    marginals = []
    for data, labels in tqdm(loader):
        labels = labels.cuda()
        logits = network(data.cuda())
        for logit, label in zip(logits, labels):
            logit_labels.append(logit[label].unsqueeze(0).cpu().detach().numpy())
        logits = logits.sort()[0]
        marginal = logits[:, -1] - logits[:, -2]
        marginals.append(marginal.cpu().detach().numpy())

    logit_labels = np.concatenate(logit_labels)
    marginals = np.concatenate(marginals)

    print('Logit[label]:')
    print(f'{np.mean(logit_labels)=}')
    print(f'{np.std(logit_labels)=}')
    print(f'{np.min(logit_labels)=}')
    print(f'{np.max(logit_labels)=}')

    print('Marginals:')
    print(f'{np.mean(marginals)=}')
    print(f'{np.std(marginals)=}')
    print(f'{np.min(marginals)=}')
    print(f'{np.max(marginals)=}')


if __name__ == '__main__':
    fire.Fire(main)
