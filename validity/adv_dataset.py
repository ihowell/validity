import pathlib

import fire
import torch
import numpy as np
import matplotlib.pyplot as plt

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from torchvision import transforms, datasets
from tqdm import tqdm

from validity.classifiers.resnet import ResNet34


def construct_dataset(dataset,
                      attack,
                      net_type,
                      weights_location,
                      data_root='./'):
    """Construct an adversarial dataset.

    Args:
        net_type (str): resnet
        attack (str): fgsm | cwl2
    """
    if attack == 'fgsm':
        adv_noise = 0.05

    if net_type == 'resnet':
        network = ResNet34(10)
        network.load_state_dict(
            torch.load(weights_location, map_location=f'cuda:0'))
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        min_pixel = -2.42906570435
        max_pixel = 2.75373125076

        if dataset == 'cifar10':
            if attack == 'fgsm':
                random_noise_size = 0.25 / 4
            elif attack == 'cwl2':
                random_noise_size = 0.05 / 2

    if dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=data_root,
                                        train=False,
                                        download=True,
                                        transform=in_transform)
        n_classes = 10

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False)
    network.cuda()
    network.eval()

    adv_data = []
    noisy_data = []
    for data, labels in tqdm(test_loader):
        data = data.cuda()

        noisy_d = torch.add(data,
                            torch.randn(data.shape).cuda(),
                            alpha=random_noise_size)
        noisy_d = torch.clamp(noisy_d, min_pixel, max_pixel)
        noisy_data.append(noisy_d.cpu().numpy())

        if attack == 'fgsm':
            adv_batch = fast_gradient_method(network, data, adv_noise, np.inf)
        if attack == 'cwl2':
            adv_batch = carlini_wagner_l2(network, data, n_classes)
        adv_data.append(adv_batch.cpu().detach().numpy())
    adv_data = np.concatenate(adv_data)
    noisy_data = np.concatenate(noisy_data)

    adv_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_{net_type}.npy'
    noisy_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_noise_{net_type}.npy'

    adv_ds_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(adv_ds_path, adv_data)
    np.save(noisy_ds_path, noisy_data)


def load_adv_dataset(dataset, attack, net_type):
    adv_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_{net_type}.npy'
    noisy_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_noise_{net_type}.npy'

    adv_data = np.load(adv_ds_path)
    noisy_data = np.save(noisy_ds_path)

    return adv_data, noisy_data


if __name__ == '__main__':
    fire.Fire(construct_dataset)
