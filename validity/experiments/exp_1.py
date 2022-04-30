from pathlib import Path

import fire

from validity.adv_dataset import construct_dataset as construct_adv_dataset, \
    adv_dataset_exists
from validity.classifiers.load import load_cls, get_cls_path, construct_cls
from validity.classifiers.train import train_ds
from validity.contrastive.dataset import get_contrastive_dataset_path
from validity.detectors.density import train_density_adv, get_density_path, DensityDetector
from validity.detectors.lid import train_multiple_lid_adv, get_best_lid_path, LIDDetector
from validity.detectors.llr import train_llr_ood, get_llr_path, LikelihoodRatioDetector
from validity.detectors.odin import train_multiple_odin, get_best_odin_path, \
    load_best_odin, ODINDetector
from validity.detectors.mahalanobis import MahalanobisDetector, \
    get_best_mahalanobis_ood_path, get_best_mahalanobis_adv_path, \
    train_multiple_mahalanobis_adv, train_multiple_mahalanobis_ood
from validity.generators.mnist_vae import train as train_mnist_vae, \
    get_save_path as get_mnist_vae_path
from validity.generators.wgan_gp import train as train_wgan_gp, get_save_path as get_wgan_gp_path

from .joint_ood_adv import joint_ood_adv


def c_func(path, func, *params, **kwargs):
    if Path(path).exists():
        print(f'Found cached {func.__name__}, {params}, {kwargs}')
    else:
        print(f'Running {func.__name__}, {params}, {kwargs}')
        func(*params, **kwargs)


def c_adv_dataset(dataset, adv_attack, cls_type, cls_path):
    if adv_dataset_exists(dataset, adv_attack, cls_type):
        print(f'Found cached adv dataset {dataset} {adv_attack} {cls_type}')
    else:
        print(f'Constructing adv dataset {dataset} {adv_attack} {cls_type}')
        construct_adv_dataset(dataset, adv_attack, cls_type, cls_path)


def train_func(cls_type, dataset, batch_size):
    cls = construct_cls(cls_type, dataset)
    cls_path = get_cls_path(cls_type, dataset)
    cls = cls.cuda()
    cls.train()
    train_ds(cls, cls_path, dataset, batch_size)


def run_experiment(cls_type, in_dataset, out_dataset):
    adv_attacks = ['fgsm', 'bim', 'cwl2']

    cls_path = get_cls_path(cls_type, in_dataset)
    eval_vae_path = get_mnist_vae_path(beta=20., id='eval')
    eval_bg_vae_path = get_mnist_vae_path(beta=20., mutation_rate=0.3, id='eval')

    # Train classifier
    c_func(cls_path, train_func, cls_type, in_dataset, 64)

    # Train generative functions
    # c_func(wgan_gp_path, train_wgan_gp, in_dataset, lambda_term=10., critic_iter=5)
    c_func(eval_vae_path, train_mnist_vae, beta=20., id='eval')
    c_func(eval_bg_vae_path, train_mnist_vae, beta=20., mutation_rate=0.3, id='eval')

    # Create adversarial datasets
    for adv_attack in adv_attacks:
        c_adv_dataset(in_dataset, adv_attack, cls_type, cls_path)

    # Train OOD detectors
    odin_path = get_best_odin_path(cls_type, in_dataset, out_dataset)
    llr_path = get_llr_path(in_dataset, out_dataset, 0.3)
    mahalanobis_ood_path = get_best_mahalanobis_ood_path(cls_type, in_dataset, out_dataset)
    c_func(odin_path, train_multiple_odin, in_dataset, out_dataset, cls_type, cls_path)
    c_func(llr_path, train_llr_ood, in_dataset, out_dataset, 'mnist_vae', eval_vae_path,
           eval_bg_vae_path, 0.3)
    c_func(mahalanobis_ood_path, train_multiple_mahalanobis_ood, in_dataset, out_dataset,
           cls_type, cls_path)

    # Train ADV detectors
    for adv_attack in adv_attacks:
        density_path = get_density_path(cls_type, in_dataset, adv_attack)
        lid_path = get_best_lid_path(cls_type, in_dataset, adv_attack)
        mahalanobis_adv_path = get_best_mahalanobis_adv_path(cls_type, in_dataset, adv_attack)
        c_func(density_path, train_density_adv, in_dataset, cls_type, cls_path, adv_attack)
        c_func(lid_path, train_multiple_lid_adv, in_dataset, cls_type, cls_path, adv_attack)
        c_func(mahalanobis_adv_path, train_multiple_mahalanobis_adv, in_dataset, cls_type,
               cls_path, adv_attack)

    # Evaluate detector joint distribution and error
    joint_ood_adv(cls_type, in_dataset, out_dataset, adv_attacks)


if __name__ == '__main__':
    fire.Fire(run_experiment)
