import fire
import numpy as np
import torch

from tqdm import tqdm
from torchvision import datasets, transforms

from validity.classifiers.resnet import ResNet34


def main(net_type, weights_path, in_ds_name, data_root='./datasets/', cuda_idx=0):
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    network = ResNet34(10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    weights = torch.load(weights_path, map_location=f'cuda:{cuda_idx}')
    network.load_state_dict(weights)
    network.cuda()

    # Load datasets
    if in_ds_name == 'cifar10':
        loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=data_root,
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.ToTensor()),
                                             batch_size=64,
                                             shuffle=True)
    if in_ds_name == 'svhn':
        loader = torch.utils.data.DataLoader(datasets.SVHN(root=data_root,
                                                           split='test',
                                                           download=True,
                                                           transform=transforms.ToTensor()),
                                             batch_size=64,
                                             shuffle=True)

    marginals = []
    for data, labels in tqdm(loader):
        logits = network(data.cuda())
        logits = logits.sort()[0]
        marginal = logits[:, -1] - logits[:, -2]
        marginals.append(marginal.cpu().detach().numpy())
    marginals = np.concatenate(marginals)
    print('Marginals:')
    print(f'{np.mean(marginals)=}')
    print(f'{np.std(marginals)=}')
    print(f'{np.min(marginals)=}')
    print(f'{np.max(marginals)=}')


if __name__ == '__main__':
    fire.Fire(main)
