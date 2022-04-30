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

from validity.classifiers.load import load_cls
from validity.datasets import load_datasets


def construct_example(dataset, attack, net_type, weights_location, data_root='./datasets'):
    network = ResNet34(
        10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    network.load_state_dict(torch.load(weights_location, map_location=f'cuda:0'))
    network = network.cuda()

    confidence = 7.439
    test_dataset = datasets.CIFAR10(root=data_root,
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    for data, label in test_loader:
        data = data.cuda()
        adv_data = carlini_wagner_l2(network,
                                     data,
                                     10,
                                     confidence=confidence,
                                     clip_min=0.,
                                     clip_max=1.)

        logits = network(data)
        adv_logits = network(adv_data)

        # print(f'{label=}')
        # print(f'{logits.argmax(axis=1)=}')
        # print(f'{adv_logits.argmax(axis=1)=}')
        logits = logits.sort()[0]
        adv_logits = adv_logits.sort()[0]
        marginal = logits[:, -1] - logits[:, -2]
        adv_marginal = adv_logits[:, -1] - adv_logits[:, -2]
        # print(f'{marginal=}')
        # print(f'{adv_marginal=}')

        data = data.cpu().detach().numpy()
        adv_data = adv_data.cpu().detach().numpy()
        break
    img = np.transpose(np.concatenate([data, adv_data], 2)[0], (1, 2, 0))
    plt.imshow(img)
    plt.show()


def construct_dataset(dataset,
                      attack,
                      net_type,
                      weights_location,
                      data_root='./datasets',
                      id=None):
    """Construct an adversarial dataset.

    Args:
        net_type (str): resnet
        attack (str): fgsm | bim | cwl2
    """
    if attack == 'fgsm':
        adv_noise = 0.10
    elif attack == 'bim':
        adv_noise = 0.02
    elif attack == 'cwl2':
        if dataset == 'mnist':
            confidence = 0.
            # confidence = 17.198
        elif dataset == 'cifar10':
            confidence = 7.439  # Average logit marginal in training set

    network = load_cls(weights_location)
    network.eval()

    if dataset == 'mnist':
        ds_mean = (0.1307, )
        ds_std = (0.3081, )

        ds_mean = torch.tensor(ds_mean).reshape((1, 1, 1))
        ds_std = torch.tensor(ds_std).reshape((1, 1, 1))
        if attack == 'fgsm':
            random_noise_size = 0.25 / 4 * 0.2
        elif attack == 'bim':
            random_noise_size = 0.21 / 4 * 0.2
        elif attack == 'cwl2':
            random_noise_size = 0.05 / 2 * 0.2

    elif dataset == 'cifar10':
        ds_mean = (0.4914, 0.4822, 0.4465)
        ds_std = (0.2023, 0.1994, 0.2010)

        ds_mean = torch.tensor(ds_mean).reshape((3, 1, 1))
        ds_std = torch.tensor(ds_std).reshape((3, 1, 1))
        if attack == 'fgsm':
            random_noise_size = 0.25 / 4 * 0.2
        elif attack == 'bim':
            random_noise_size = 0.21 / 4 * 0.2
        elif attack == 'cwl2':
            random_noise_size = 0.05 / 2 * 0.2

    if attack == 'cwl2':
        if dataset == 'mnist':
            n_classes = 10
        elif dataset == 'cifar10':
            n_classes = 10

    network = network.cuda()
    network.eval()

    _, test_dataset = load_datasets(dataset, data_root=data_root)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    clean_data = []
    adv_data = []
    noisy_data = []

    accuracy = []
    noise_accuracy = []
    adv_accuracy = []

    for data, labels in tqdm(test_loader):
        data = data.cuda()
        labels = labels.cuda()

        mean = torch.ones(data.shape) * ds_mean
        std = torch.ones(data.shape) * ds_std
        noise = torch.normal(mean, std).cuda()
        noisy_d = torch.add(data, noise, alpha=random_noise_size)
        noisy_d = torch.clamp(noisy_d, 0., 1.)

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
            adv_batch = carlini_wagner_l2(network, data, n_classes, confidence=confidence)
        adv_batch = adv_batch

        logits = network(data)
        preds = torch.argmax(logits, 1)
        noise_logits = network(noisy_d)
        noise_preds = torch.argmax(noise_logits, 1)
        adv_logits = network(adv_batch)
        adv_preds = torch.argmax(adv_logits, 1)

        accuracy.append(
            torch.mean(torch.where(preds == labels, 1., 0.)).cpu().detach().numpy())
        noise_accuracy.append(
            torch.mean(torch.where(noise_preds == labels, 1., 0.)).cpu().detach().numpy())
        adv_accuracy.append(
            torch.mean(torch.where(adv_preds == labels, 1., 0.)).cpu().detach().numpy())

        indices = torch.where(torch.logical_and(noise_preds == labels, adv_preds != labels))[0]
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

    files = [(f'{dataset}_{attack}_clean_{net_type}', clean_data),
             (f'{dataset}_{attack}_adv_{net_type}', adv_data),
             (f'{dataset}_{attack}_noise_{net_type}', noisy_data)]
    if id:
        files = [(f'{name}_{id}', data) for (name, data) in files]

    data_root = pathlib.Path('adv_datasets')
    data_root.mkdir(exist_ok=True, parents=True)
    for name, data in files:
        path = data_root / f'{name}.npy'
        np.save(path, data)


def load_adv_datasets(dataset, attack, net_type, classifier_id=None):
    data_root = pathlib.Path('adv_datasets')
    partitions = ['clean', 'adv', 'noise']
    paths = [f'{dataset}_{attack}_{partition}_{net_type}' for partition in partitions]
    if classifier_id:
        paths = [f'{path}_{classifier_id}' for path in paths]

    paths = [data_root / f'{path}.npy' for path in paths]
    for path in paths:
        if not path.exists():
            return False

    return [np.load(path) for path in paths]


def adv_dataset_exists(dataset, attack, net_type, id=None):
    data_root = pathlib.Path('adv_datasets')
    partitions = ['clean', 'adv', 'noise']
    paths = [f'{dataset}_{attack}_{partition}_{net_type}' for partition in partitions]
    if id:
        paths = [f'{path}_{id}' for path in paths]

    paths = [data_root / f'{path}.npy' for path in paths]
    for path in paths:
        if not path.exists():
            return False
    return True


def calculate_channel_means(dataset):
    train_ds, _ = load_datasets(dataset)
    data = np.stack([data for data, _ in train_ds])
    print(f'Mean: {np.mean(data, (0, 2, 3))}')
    print(f'Std: {np.std(data, (0, 2, 3))}')


if __name__ == '__main__':
    fire.Fire()
