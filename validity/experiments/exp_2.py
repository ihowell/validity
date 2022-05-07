from collections import defaultdict
from itertools import product
from pathlib import Path
import json

import fire
from tabulate import tabulate

from validity.adv_dataset import construct_dataset as construct_adv_dataset, \
    adv_dataset_exists
from validity.classifiers.load import load_cls, get_cls_path, construct_cls
from validity.classifiers.train import train_ds
from validity.contrastive.dataset import get_contrastive_dataset_path
from validity.detectors.lid import train_multiple_lid_adv, get_best_lid_path, LIDDetector
from validity.detectors.llr import train_llr_ood, get_llr_path, LikelihoodRatioDetector
from validity.detectors.odin import train_multiple_odin, get_best_odin_path, \
    load_best_odin, ODINDetector
from validity.detectors.mahalanobis import MahalanobisDetector, \
    get_best_mahalanobis_ood_path, get_best_mahalanobis_adv_path, \
    train_multiple_mahalanobis_adv, train_multiple_mahalanobis_ood
from validity.generators.mnist_vae import train as train_mnist_vae, \
    get_save_path as get_mnist_vae_path, encode_dataset as mnist_vae_encode_dataset
from validity.generators.wgan_gp import train as train_wgan_gp, get_save_path as get_wgan_gp_path, \
    encode_dataset as wgan_gp_encode_dataset
from validity.contrastive.dataset import make_contrastive_dataset

from .eval_contrastive import eval_contrastive_ds, get_eval_res_path


def c_func(path, func, *params, **kwargs):
    if Path(path).exists():
        print(f'Found cached {func.__name__}, {params}, {kwargs}')
    else:
        print(f'Running {func.__name__}, {params}, {kwargs}')
        func(*params, **kwargs)


def c_adv_dataset(dataset, adv_attack, net_type, cls_path):
    if adv_dataset_exists(dataset, adv_attack, dataset):
        print(f'Found cached adv dataset {dataset} {adv_attack} {net_type}')
    else:
        print(f'Constructing adv dataset {dataset} {adv_attack} {net_type}')
        construct_adv_dataset(dataset, adv_attack, net_type, cls_path)


def train_func(cls_type, dataset, batch_size):
    cls = construct_cls(cls_type, dataset)
    cls_path = get_cls_path(cls_type, dataset)
    cls = cls.cuda()
    cls.train()
    train_ds(cls, cls_path, dataset, batch_size)


def run_experiment(cls_type, in_dataset, out_dataset, high_performance=False, subset=None):
    adv_attacks = ['fgsm', 'bim', 'cwl2']
    contrastive_methods = ['am', 'xgems', 'cdeepex']

    cls_path = get_cls_path(cls_type, in_dataset)
    vae_path = get_mnist_vae_path(beta=10.)
    mnist_encode_wgan_gp_path = Path('data/wgan_gp_encode_mnist_test.npz')
    wgan_gp_path = get_wgan_gp_path(in_dataset, 10, 5)
    eval_vae_path = get_mnist_vae_path(beta=20., id='eval')
    eval_bg_vae_path = get_mnist_vae_path(beta=20., mutation_rate=0.3, id='eval')

    generators = [{
        'type': 'mnist_vae',
        'path': vae_path
    }, {
        'type': 'wgan_gp',
        'path': wgan_gp_path
    }]

    # Train classifiers
    c_func(cls_path, train_func, cls_type, in_dataset, 64)

    # Train generative functions
    c_func(wgan_gp_path, train_wgan_gp, in_dataset, lambda_term=10., critic_iter=5)
    c_func(vae_path, train_mnist_vae, beta=10.)
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
    c_func(llr_path, train_llr_ood, in_dataset, out_dataset, eval_vae_path, eval_bg_vae_path,
           0.3)
    c_func(mahalanobis_ood_path, train_multiple_mahalanobis_ood, in_dataset, out_dataset,
           cls_type, cls_path)

    # Train ADV detectors
    for adv_attack in adv_attacks:
        lid_path = get_best_lid_path(cls_type, in_dataset, adv_attack)
        mahalanobis_adv_path = get_best_mahalanobis_adv_path(cls_type, in_dataset, adv_attack)
        c_func(lid_path, train_multiple_lid_adv, in_dataset, cls_type, cls_path, adv_attack)
        c_func(mahalanobis_adv_path, train_multiple_mahalanobis_adv, in_dataset, cls_type,
               cls_path, adv_attack)

    # Encode datasets
    if not mnist_encode_wgan_gp_path.exists():
        print(f'{mnist_encode_wgan_gp_path=}')
        if not high_performance:
            raise Exception(f'High-performance required to run to encode dataset.')
        wgan_gp_encode_dataset(wgan_gp_path)

    # Create contrastive examples
    for contrastive_method in contrastive_methods:
        for gen in generators:
            contrastive_path = get_contrastive_dataset_path(contrastive_method, in_dataset,
                                                            cls_type, gen['type'], subset)
            if not contrastive_path.exists():
                if not high_performance:
                    raise Exception(
                        f'Could not find contrastive dataset for method {contrastive_method} with generator {gen["type"]} at {contrastive_path}. Please run this expeirment in a high-performance setting and set high_performance=True.'
                    )

                make_contrastive_dataset(contrastive_method,
                                         in_dataset,
                                         cls_type,
                                         cls_path,
                                         gen['type'],
                                         gen['path'],
                                         subset=subset)

    # Evaluate contrastive examples
    results = {}
    for contrastive_method in contrastive_methods:
        results[contrastive_method] = {}
        for gen in generators:
            results[contrastive_method][gen['type']] = {}
            for adv_attack in adv_attacks:
                contrastive_ds_path = get_contrastive_dataset_path(
                    contrastive_method, in_dataset, cls_type, gen['type'], subset)
                contrastive_res_path = get_eval_res_path(contrastive_method, cls_type,
                                                         in_dataset, out_dataset, gen['type'],
                                                         adv_attack)
                if not contrastive_res_path.exists():
                    print(f'Evaluating:')
                    print(f'Contrastive method: {contrastive_method} with {gen["type"]}')
                    print(f'Adversarial attack: {adv_attack}')
                    eval_contrastive_ds(contrastive_method,
                                        contrastive_ds_path,
                                        cls_type,
                                        cls_path,
                                        in_dataset,
                                        out_dataset,
                                        gen['type'],
                                        adv_attack,
                                        verbose=True)

                with open(contrastive_res_path) as in_file:
                    results[contrastive_method][gen['type']][adv_attack] = json.load(in_file)

    _grid_output(adv_attacks, contrastive_methods, generators, results)


def _grid_output(adv_attacks, contrastive_methods, generators, results):
    headers = [['', 'Detectors', ''] +
               sum([[c + ' ' + g['type']] + [''] * 3
                    for (c, g) in product(contrastive_methods, generators)], []),
               ['Attack', 'OOD Method', 'ADV Method'] +
               ['TCV', 'ID', 'NAdv', 'CValid'] * len(contrastive_methods) * len(generators)]

    table = headers

    for adv_attack in adv_attacks:
        rows = defaultdict(list)
        for contrastive_method in contrastive_methods:
            for gen in generators:
                for ood_name in results[contrastive_method][
                        gen['type']][adv_attack]['validity']:
                    res = results[contrastive_method][gen['type']][adv_attack]
                    for adv_name, valid in results[contrastive_method][
                            gen['type']][adv_attack]['validity'][ood_name].items():
                        rows[(ood_name, adv_name)] += [
                            f'{res["target_class_validity"]:.4f}',
                            f'{res["ood_validity"][ood_name]:.4f}',
                            f'{res["adv_validity"][adv_name]:.4f}', f'{valid:.4f}'
                        ]
        rows = [['', ood, adv] + vals for ((ood, adv), vals) in rows.items()]
        rows[0][0] = adv_attack

        table = table + rows

    print('')
    print(tabulate(table, tablefmt='tsv', floatfmt='.4f'))


def _long_table(adv_attacks, contrastive_methods, generators, results):
    #headers = [([''] * 2) + sum([[c] + [''] * 3 for c in contrastive_methods], []),
    #           ['OOD Method', 'ADV Method'] +
    #           ['TCV', 'ID', 'NAdv', 'CValid'] * len(contrastive_methods)]
    headers = ['Contrastive Method', 'OOD Method', 'ADV Method', 'TCV', 'ID', 'NAdv', 'CValid']

    table = [headers]

    for adv_attack in adv_attacks:
        rows = []
        for contrastive_method in contrastive_methods:
            for gen in generators:
                for ood_name in results[contrastive_method][
                        gen['type']][adv_attack]['validity']:
                    res = results[contrastive_method][gen['type']][adv_attack]
                    for adv_name, valid in results[contrastive_method][
                            gen['type']][adv_attack]['validity'][ood_name].items():
                        rows.append([
                            contrastive_method + ' ' + gen['type'],
                            ood_name,
                            adv_name,
                            f'{res["target_class_validity"]:.4f}',
                            f'{res["ood_validity"][ood_name]:.4f}',
                            f'{res["adv_validity"][adv_name]:.4f}',
                            f'{valid:.4f}',
                        ])
        table = table + rows

    print('')
    print(tabulate(table, tablefmt='csv', floatfmt='.4f'))


if __name__ == '__main__':
    fire.Fire(run_experiment)
