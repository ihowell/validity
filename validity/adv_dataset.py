import pathlib

import fire
import torch
import numpy as np
import matplotlib.pyplot as plt

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from torchvision import transforms, datasets
from tqdm import tqdm

from validity.classifiers.resnet import ResNet34


def construct_dataset(dataset,
                      attack,
                      net_type,
                      weights_location,
                      data_root='./datasets'):
    """Construct an adversarial dataset.

    Args:
        net_type (str): resnet
        attack (str): fgsm | bim | cwl2
    """
    if attack == 'fgsm':
        adv_noise = 0.10
    elif attack == 'bim':
        adv_noise = 0.02

    if net_type == 'resnet':
        network = ResNet34(
            10,
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)))
        network.load_state_dict(
            torch.load(weights_location, map_location=f'cuda:0'))

        min_pixel = -2.42906570435
        max_pixel = 2.75373125076

        if dataset == 'cifar10':
            if attack == 'fgsm':
                random_noise_size = 0.25 / 4 * 0.2
            elif attack == 'bim':
                random_noise_size = 0.21 / 4 * 0.2
            elif attack == 'cwl2':
                random_noise_size = 0.05 / 2 * 0.2

    if dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=data_root,
                                        train=False,
                                        download=True,
                                        transform=transforms.ToTensor())
        n_classes = 10

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False)
    network.cuda()
    network.eval()

    clean_data = []
    adv_data = []
    noisy_data = []

    accuracy = []
    noise_accuracy = []
    adv_accuracy = []

    for data, labels in tqdm(test_loader):
        data = data.cuda()
        labels = labels.cuda()

        noisy_d = torch.add(data,
                            torch.randn(data.shape).cuda(),
                            alpha=random_noise_size)
        noisy_d = torch.clamp(noisy_d, min_pixel, max_pixel)

        if attack == 'fgsm':
            adv_batch = fast_gradient_method(network,
                                             data,
                                             adv_noise,
                                             np.inf,
                                             clip_min=0.,
                                             clip_max=1.)
        elif attack == 'bim':
            adv_batch = projected_gradient_descent(network,
                                                   data,
                                                   np.inf,
                                                   adv_noise,
                                                   5,
                                                   np.inf,
                                                   sanity_checks=False,
                                                   rand_init=False,
                                                   clip_min=0.,
                                                   clip_max=1.)
        elif attack == 'cwl2':
            adv_batch = carlini_wagner_l2(network, data, n_classes)
        adv_batch = adv_batch

        logits = network(data)
        preds = torch.argmax(logits, 1)
        noise_logits = network(noisy_d)
        noise_preds = torch.argmax(noise_logits, 1)
        adv_logits = network(adv_batch)
        adv_preds = torch.argmax(adv_logits, 1)

        accuracy.append(
            torch.mean(torch.where(preds == labels, 1.,
                                   0.)).cpu().detach().numpy())
        noise_accuracy.append(
            torch.mean(torch.where(noise_preds == labels, 1.,
                                   0.)).cpu().detach().numpy())
        adv_accuracy.append(
            torch.mean(torch.where(adv_preds == labels, 1.,
                                   0.)).cpu().detach().numpy())

        indices = torch.where(
            torch.logical_and(noise_preds == labels, adv_preds != labels))[0]
        data = torch.index_select(data, 0, indices)
        noisy_d = torch.index_select(noisy_d, 0, indices)
        adv_batch = torch.index_select(adv_batch, 0, indices)

        clean_data.append(data.cpu().numpy())
        noisy_data.append(noisy_d.cpu().numpy())
        adv_data.append(adv_batch.cpu().detach().numpy())
    clean_data = np.concatenate(clean_data)
    adv_data = np.concatenate(adv_data)
    noisy_data = np.concatenate(noisy_data)

    print(f'Accuracy: {np.mean(accuracy)}')
    print(f'Noise Accuracy: {np.mean(noise_accuracy)}')
    print(f'Adversarial Accuracy: {np.mean(adv_accuracy)}')
    print(f'Data after filtering: {clean_data.shape[0]}')

    clean_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_clean_{net_type}.npy'
    adv_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_adv_{net_type}.npy'
    noisy_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_noise_{net_type}.npy'

    adv_ds_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(clean_ds_path, clean_data)
    np.save(adv_ds_path, adv_data)
    np.save(noisy_ds_path, noisy_data)


def load_adv_dataset(dataset, attack, net_type):
    clean_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_clean_{net_type}.npy'
    adv_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_adv_{net_type}.npy'
    noisy_ds_path = pathlib.Path(
        'adv_datasets') / f'{dataset}_{attack}_noise_{net_type}.npy'

    assert clean_ds_path.exists()
    assert adv_ds_path.exists()
    assert noisy_ds_path.exists()

    clean_data = np.load(clean_ds_path)
    adv_data = np.load(adv_ds_path)
    noisy_data = np.load(noisy_ds_path)

    return clean_data, adv_data, noisy_data


if __name__ == '__main__':
    fire.Fire(construct_dataset)
