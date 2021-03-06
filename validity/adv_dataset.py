import pathlib

import fire
import torch
import numpy as np
import matplotlib.pyplot as plt

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from tqdm import tqdm

from validity.classifiers.load import load_cls
from validity.datasets import load_datasets, load_detector_datasets


def construct_example(dataset, weights_location, data_root='./datasets'):
    network = load_cls(weights_location)
    network = network.cuda()

    confidence = 7.439
    _, test_dataset = load_datasets(dataset, data_root=data_root)

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
                      classifier_id=None):
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
    network = network.cuda()
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

    clean_arr = []
    adv_arr = []
    noise_arr = []

    _, adv_train, _, adv_test = load_detector_datasets(dataset, data_root=data_root)
    for ds in [adv_train, adv_test]:
        test_loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

        clean_data = []
        adv_data = []
        noise_data = []

        clean_accuracy = []
        adv_accuracy = []
        noise_accuracy = []

        for data, labels in tqdm(test_loader):
            data = data.cuda()
            labels = labels.cuda()

            mean = torch.ones(data.shape) * ds_mean
            std = torch.ones(data.shape) * ds_std
            noise = torch.normal(mean, std).cuda()
            noise_d = torch.add(data, noise, alpha=random_noise_size)
            noise_d = torch.clamp(noise_d, 0., 1.)

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

            logits = network(data)
            preds = torch.argmax(logits, 1)
            noise_logits = network(noise_d)
            noise_preds = torch.argmax(noise_logits, 1)
            adv_logits = network(adv_batch)
            adv_preds = torch.argmax(adv_logits, 1)

            clean_accuracy.append(
                torch.mean(torch.where(preds == labels, 1., 0.)).cpu().detach().numpy())
            noise_accuracy.append(
                torch.mean(torch.where(noise_preds == labels, 1., 0.)).cpu().detach().numpy())
            adv_accuracy.append(
                torch.mean(torch.where(adv_preds == labels, 1., 0.)).cpu().detach().numpy())

            indices = torch.where(torch.logical_and(noise_preds == labels,
                                                    adv_preds != labels))[0]
            data = torch.index_select(data, 0, indices)
            noise_d = torch.index_select(noise_d, 0, indices)
            adv_batch = torch.index_select(adv_batch, 0, indices)

            clean_data.append(data.cpu().numpy())
            noise_data.append(noise_d.cpu().numpy())
            adv_data.append(adv_batch.cpu().detach().numpy())
        clean_data = np.concatenate(clean_data)
        adv_data = np.concatenate(adv_data)
        noise_data = np.concatenate(noise_data)

        clean_arr.append(clean_data)
        adv_arr.append(adv_data)
        noise_arr.append(noise_data)

        print(f'Clean Accuracy: {np.mean(clean_accuracy)}')
        print(f'Noise Accuracy: {np.mean(noise_accuracy)}')
        print(f'Adversarial Accuracy: {np.mean(adv_accuracy)}')
        print(f'Data after filtering: {clean_data.shape[0]}')

    save_path = get_adv_dataset_path(dataset, attack, net_type, classifier_id=classifier_id)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(save_path,
             clean_train=clean_arr[0],
             clean_test=clean_arr[1],
             adv_train=adv_arr[0],
             adv_test=adv_arr[1],
             noise_train=noise_arr[0],
             noise_test=noise_arr[1])


def get_adv_dataset_path(dataset, attack, net_type, classifier_id=None):
    data_root = pathlib.Path('adv_datasets')
    file_name = f'{dataset}_{attack}_{net_type}'
    if classifier_id:
        file_name = f'{file_name}_{classifier_id}'
    return data_root / f'{file_name}.npz'


def load_adv_datasets(dataset, attack, net_type, classifier_id=None):
    save_path = get_adv_dataset_path(dataset, attack, net_type, classifier_id=classifier_id)
    data_dict = np.load(save_path)
    return {
        'clean': (data_dict['clean_train'], data_dict['clean_test']),
        'adv': (data_dict['adv_train'], data_dict['adv_test']),
        'noise': (data_dict['noise_train'], data_dict['noise_test']),
    }


def adv_dataset_exists(dataset, attack, net_type, classifier_id=None):
    save_path = get_adv_dataset_path(dataset, attack, net_type, classifier_id=classifier_id)
    return save_path.exists()


def calculate_channel_means(dataset):
    train_ds, _ = load_datasets(dataset)
    data = np.stack([data for data, _ in train_ds])
    print(f'Mean: {np.mean(data, (0, 2, 3))}')
    print(f'Std: {np.std(data, (0, 2, 3))}')


if __name__ == '__main__':
    fire.Fire()
