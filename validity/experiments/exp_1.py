from pathlib import Path

from validity.adv_dataset import construct_dataset as construct_adv_dataset, \
    adv_dataset_exists
from validity.classifiers import load_cls
from validity.classifiers.mnist import train_network as train_mnist_cls
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
from validity.generators.wgan_gp import train as train_wgan_gp

from .joint_ood_adv import joint_ood_adv


def c_func(path, func, *params, **kwargs):
    if Path(path).exists():
        print(f'Found cached {func.__name__}, {params}, {kwargs}')
    else:
        print(f'Running {func.__name__}, {params}, {kwargs}')
        func(*params, **kwargs)


def c_adv_dataset(dataset, adv_attack, net_type, cls_path):
    if adv_dataset_exists('mnist', adv_attack, dataset):
        print(f'Found cached adv dataset {dataset} {adv_attack} {net_type}')
    else:
        print(f'Constructing adv dataset {dataset} {adv_attack} {net_type}')
        construct_adv_dataset(dataset, adv_attack, net_type, cls_path)


def run_experiment():
    mnist_cls_path = Path('models/cls_mnist_mnist.pt')
    adv_attacks = ['fgsm', 'bim', 'cwl2']
    vae_path = get_mnist_vae_path(beta=20.)
    bg_vae_path = get_mnist_vae_path(beta=20., mutation_rate=0.3)

    # Train classifiers
    c_func(mnist_cls_path, train_mnist_cls)

    # Train generative functions
    c_func(vae_path, train_mnist_vae, beta=20.)
    c_func(bg_vae_path, train_mnist_vae, beta=20., mutation_rate=0.3)
    c_func('models/wgan_gp_mnist_lam_10_iter_5.pt',
           train_wgan_gp,
           'mnist',
           lambda_term=10.,
           critic_iter=5)

    # Create adversarial datasets
    for adv_attack in adv_attacks:
        c_adv_dataset('mnist', adv_attack, 'mnist', mnist_cls_path)

    # Train OOD detectors
    c_func(get_best_odin_path('mnist', 'mnist', 'fmnist'), train_multiple_odin, 'mnist',
           'fmnist', 'mnist', mnist_cls_path)
    c_func(get_llr_path('mnist', 'fmnist', 0.3), train_llr_ood, 'mnist', 'fmnist', 'mnist_vae',
           vae_path, bg_vae_path, 0.3)
    c_func(get_best_mahalanobis_ood_path('mnist', 'mnist', 'fmnist'),
           train_multiple_mahalanobis_ood, 'mnist', 'fmnist', 'mnist', mnist_cls_path)

    # Train ADV detectors
    for adv_attack in adv_attacks:
        c_func(get_density_path('mnist', 'mnist', adv_attack), train_density_adv, 'mnist',
               'mnist', mnist_cls_path, adv_attack)
        c_func(get_best_lid_path('mnist', 'mnist', adv_attack), train_multiple_lid_adv,
               'mnist', 'mnist', mnist_cls_path, adv_attack)
        c_func(get_best_mahalanobis_adv_path('mnist', 'mnist', adv_attack),
               train_multiple_mahalanobis_adv, 'mnist', 'mnist', mnist_cls_path, adv_attack)

    # Evaluate detector joint distribution and error
    joint_ood_adv('mnist', 'mnist', 'fmnist', adv_attacks)


if __name__ == '__main__':
    run_experiment()
