from pathlib import Path
import json

import fire
from tabulate import tabulate

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

from .eval_contrastive import eval_contrastive_ds, get_eval_res_path


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


def run_experiment(high_performance=False):
    mnist_cls_path = 'models/cls_mnist_mnist.pt'
    adv_attacks = ['fgsm', 'bim', 'cwl2']
    contrastive_methods = ['am', 'xgems', 'cdeepex']
    vae_path = get_mnist_vae_path(beta=10.)
    wgan_gp_path = 'models/wgan_gp_mnist_lam_10_iter_5.pt'
    eval_vae_path = get_mnist_vae_path(beta=20., id='eval')
    eval_bg_vae_path = get_mnist_vae_path(beta=20., mutation_rate=0.3, id='eval')

    # Train classifiers
    c_func(mnist_cls_path, train_mnist_cls)

    # Train generative functions
    c_func(wgan_gp_path, train_wgan_gp, 'mnist', lambda_term=10., critic_iter=5)
    c_func(vae_path, train_mnist_vae, beta=10.)
    c_func(eval_vae_path, train_mnist_vae, beta=20., id='eval')
    c_func(eval_bg_vae_path, train_mnist_vae, beta=20., mutation_rate=0.3, id='eval')

    # Create adversarial datasets
    for adv_attack in adv_attacks:
        c_adv_dataset('mnist', adv_attack, 'mnist', mnist_cls_path)

    # Train OOD detectors
    c_func(get_best_odin_path('mnist', 'mnist', 'fmnist'), train_multiple_odin, 'mnist',
           'fmnist', 'mnist', mnist_cls_path)
    c_func(get_llr_path('mnist', 'fmnist', 0.3), train_llr_ood, 'mnist', 'fmnist', 'mnist_vae',
           eval_vae_path, eval_bg_vae_path, 0.3)
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

    # Create contrastive examples
    for contrastive_method in contrastive_methods:
        if not get_contrastive_dataset_path(contrastive_method, 'mnist', 'mnist',
                                            'wgan_gp').exists():
            if not high_performance:
                raise Exception(
                    f'Could not find contrastive dataset for method {contrastive_method}. Please run this expeirment in a high-performance setting and set high_performance=True.'
                )

            make_contrastive_dataset(contrastive_method, 'mnist', 'mnist', mnist_cls_path,
                                     'wgan_gp', wgan_gp_path)

    # Evaluate contrastive examples
    results = {}
    for contrastive_method in contrastive_methods:
        results[contrastive_method] = {}
        for adv_attack in adv_attacks:
            contrastive_ds_path = get_contrastive_dataset_path(contrastive_method, 'mnist',
                                                               'mnist', 'wgan_gp')
            contrastive_res_path = get_eval_res_path(contrastive_method, 'mnist', 'mnist',
                                                     'fmnist', adv_attack)
            if not contrastive_res_path.exists():
                print(f'Evaluating:')
                print(f'Contrastive method: {contrastive_method}')
                print(f'Adversarial attack: {adv_attack}')
                eval_contrastive_ds(contrastive_method,
                                    contrastive_ds_path,
                                    'mnist',
                                    mnist_cls_path,
                                    'mnist',
                                    'fmnist',
                                    adv_attack,
                                    verbose=True)

            with open(contrastive_res_path) as in_file:
                results[contrastive_method][adv_attack] = json.load(in_file)

    headers = [([''] * 2) + sum([[c] + [''] * 3 for c in contrastive_methods], []),
               ['OOD Method', 'ADV Method'] +
               ['TCV', 'ID', 'NAdv', 'CValid'] * len(contrastive_methods)]

    table = headers
    for adv_attack in adv_attacks:
        rows = []
        for contrastive_method in contrastive_methods:
            for ood_name in results[contrastive_method][adv_attack]['validity']:
                res = results[contrastive_method][adv_attack]
                print(res.keys())
                for adv_name, valid in results[contrastive_method][adv_attack]['validity'][
                        ood_name].items():
                    found = False
                    for i, entry in enumerate(rows):
                        if entry[0] == ood_name and entry[1] == adv_name:
                            entry += [
                                f'{res["target_class_validity"]:.4f}',
                                f'{res["ood_validity"][ood_name]:.4f}',
                                f'{res["adv_validity"][adv_name]:.4f}',
                                f'{valid:.4f}',
                            ]
                            found = True
                            break
                    if not found:
                        rows.append([
                            ood_name,
                            adv_name,
                            f'{res["target_class_validity"]:.4f}',
                            f'{res["ood_validity"][ood_name]:.4f}',
                            f'{res["adv_validity"][adv_name]:.4f}',
                            f'{valid:.4f}',
                        ])
        table = table + rows

    print(tabulate(table, tablefmt='tsv', floatfmt='.4f'))


if __name__ == '__main__':
    fire.Fire(run_experiment)
