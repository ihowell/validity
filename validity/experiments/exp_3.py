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
from validity.detectors.density import train_density_adv, get_density_path, DensityDetector
from validity.detectors.lid import train_multiple_lid_adv, get_best_lid_path, LIDDetector
from validity.detectors.llr import train_llr_ood, get_llr_path, LikelihoodRatioDetector
from validity.detectors.odin import train_multiple_odin, get_best_odin_path, \
    load_best_odin
from validity.detectors.mahalanobis import get_best_mahalanobis_ood_path, get_best_mahalanobis_adv_path, \
    train_multiple_mahalanobis_adv, train_multiple_mahalanobis_ood
from validity.generators.mnist_vae import train as train_mnist_vae, \
    get_save_path as get_mnist_vae_path, encode_dataset as mnist_vae_encode_dataset
from validity.generators.wgan_gp import train as train_wgan_gp, get_save_path as get_wgan_gp_path, \
    encode_dataset as wgan_gp_encode_dataset
from validity.contrastive.dataset import make_contrastive_dataset
from validity.util import get_executor

from .eval_contrastive import eval_contrastive_ds, get_eval_res_path


def c_func(executor):

    def thunk(path, func, *params, **kwargs):
        if Path(path).exists():
            print(f'Found cached {func.__name__}, {params}, {kwargs}')
        else:
            print(f'Running {func.__name__}, {params}, {kwargs}')
            return executor.submit(func, *params, **kwargs)

    return thunk


def c_adv_dataset(executor):

    def thunk(dataset, adv_attack, net_type, cls_path, id=None):
        if adv_dataset_exists(dataset, adv_attack, dataset, id=id):
            print(f'Found cached adv dataset {dataset} {adv_attack} {net_type} {id}')
        else:
            print(f'Constructing adv dataset {dataset} {adv_attack} {net_type} {id}')
            return executor.submit(construct_adv_dataset,
                                   dataset,
                                   adv_attack,
                                   net_type,
                                   cls_path,
                                   id=id)

    return thunk


def train_cls_func(cls_type,
                   cls_path,
                   dataset,
                   batch_size,
                   cls_kwargs=None,
                   train_kwargs=None):
    train_kwargs = train_kwargs or {}
    cls = construct_cls(cls_type, dataset, cls_kwargs=cls_kwargs)
    cls = cls.cuda()
    cls.train()
    train_ds(cls, cls_path, dataset, batch_size=batch_size, **train_kwargs)


def get_gen_path(gen_type, dataset, **kwargs):
    if gen_type == 'mnist_vae':
        return get_mnist_vae_path(**kwargs)
    elif gen_type == 'wgan_gp':
        return get_wgan_gp_path(dataset, kwargs['lambda_term'], kwargs['critic_iter'])
    else:
        raise Exception(f'Unknown gen type passed to train_gen: {gen_type}')


def train_gen(gen_type, dataset, **kwargs):
    if gen_type == 'mnist_vae':
        train_mnist_vae(**kwargs)
    elif gen_type == 'wgan_gp':
        train_wgan_gp(dataset, **kwargs)
    else:
        raise Exception(f'Unknown gen type passed to train_gen: {gen_type}')


def gen_encode_dataset(gen_type, weights_path, encode_path, dataset, *args, **kwargs):
    if gen_type == 'mnist_vae':
        if dataset != 'mnist':
            raise Exception("Can't encode non-MNIST dataset")
        mnist_vae_encode_dataset(weights_path, *args, save_path=encode_path, **kwargs)
    elif gen_type == 'wgan_gp':
        wgan_gp_encode_dataset(dataset, weights_path, *args, encode_path=encode_path, **kwargs)
    else:
        raise Exception(f'Unknown gen type passed to train_gen: {gen_type}')


def run_sub_experiment(cfg_file, high_performance=False):
    with open(cfg_file) as f:
        cfg = json.load(f)

    executor = get_executor()
    cache_func = c_func(executor)
    cache_adv_ds = c_adv_dataset(executor)

    in_dataset = cfg['in_dataset']
    out_dataset = cfg['out_dataset']

    eval_vae_path = get_mnist_vae_path(beta=20., id='eval')
    eval_bg_vae_path = get_mnist_vae_path(beta=20., mutation_rate=0.3, id='eval')

    # Train generative functions
    jobs = []
    with executor.batch():
        for gen_cfg in cfg['generators']:
            gen_path = get_gen_path(gen_cfg['type'], in_dataset, **gen_cfg['kwargs'])

            jobs.append(cache_func(train_gen, gen_cfg['type'], in_dataset,
                                   **gen_cfg['kwargs']))

        jobs.append(cache_func(eval_vae_path, train_mnist_vae, beta=20., id='eval'))
        jobs.append(
            cache_func(eval_bg_vae_path,
                       train_mnist_vae,
                       beta=20.,
                       mutation_rate=0.3,
                       id='eval'))
    [job.result() for job in jobs if job]

    # Train classifiers
    jobs = []
    with executor.batch():
        for cls_cfg in cfg['classifiers']:
            cls_path = get_cls_path(cls_cfg['type'], in_dataset, id=cls_cfg['name'])
            jobs.append(
                cache_func(cls_path,
                           train_cls_func,
                           cls_cfg['type'],
                           cls_path,
                           in_dataset,
                           64,
                           cls_kwargs=cls_cfg.get('cls_kwargs'),
                           train_kwargs=cls_cfg.get('train_kwargs')))
    [job.result() for job in jobs if job]

    # Create adversarial datasets
    jobs = []
    with executor.batch():
        for cls_cfg in cfg['classifiers']:
            cls_path = get_cls_path(cls_cfg['type'], in_dataset, id=cls_cfg['name'])
            for adv_attack in cfg['adv_attacks']:
                jobs.append(
                    cache_adv_ds(in_dataset,
                                 adv_attack,
                                 cls_cfg['type'],
                                 cls_path,
                                 id=cls_cfg['name']))
    [job.result() for job in jobs if job]

    # Train OOD detectors
    jobs = []
    with executor.batch():
        for cls_cfg in cfg['classifiers']:
            cls_type = cls_cfg['type']
            id = cls_cfg['name']
            cls_path = get_cls_path(cls_cfg['type'], in_dataset, id=cls_cfg['name'])
            odin_path = get_best_odin_path(cls_type, in_dataset, out_dataset, id=id)
            llr_path = get_llr_path(in_dataset, out_dataset, 0.3, id=id)
            mahalanobis_ood_path = get_best_mahalanobis_ood_path(cls_type,
                                                                 in_dataset,
                                                                 out_dataset,
                                                                 id=id)
            jobs.append(
                cache_func(odin_path,
                           train_multiple_odin,
                           in_dataset,
                           out_dataset,
                           cls_type,
                           cls_path,
                           id=id))
            jobs.append(
                cache_func(llr_path,
                           train_llr_ood,
                           in_dataset,
                           out_dataset,
                           'mnist_vae',
                           eval_vae_path,
                           eval_bg_vae_path,
                           0.3,
                           id=id))
            jobs.append(
                cache_func(mahalanobis_ood_path,
                           train_multiple_mahalanobis_ood,
                           in_dataset,
                           out_dataset,
                           cls_type,
                           cls_path,
                           id=id))
    [job.result() for job in jobs if job]

    # Train ADV detectors
    jobs = []
    with executor.batch():
        for cls_cfg in cfg['classifiers']:
            cls_type = cls_cfg['type']
            id = cls_cfg['name']
            for adv_attack in cfg['adv_attacks']:
                density_path = get_density_path(cls_type, in_dataset, adv_attack, id=id)
                lid_path = get_best_lid_path(cls_type, in_dataset, adv_attack, id=id)
                mahalanobis_adv_path = get_best_mahalanobis_adv_path(cls_type,
                                                                     in_dataset,
                                                                     adv_attack,
                                                                     id=id)
                jobs.append(
                    cache_func(density_path,
                               train_density_adv,
                               in_dataset,
                               cls_type,
                               cls_path,
                               adv_attack,
                               id=id))
                jobs.append(
                    cache_func(lid_path,
                               train_multiple_lid_adv,
                               in_dataset,
                               cls_type,
                               cls_path,
                               adv_attack,
                               id=id))
                jobs.append(
                    cache_func(mahalanobis_adv_path,
                               train_multiple_mahalanobis_adv,
                               in_dataset,
                               cls_type,
                               cls_path,
                               adv_attack,
                               id=id))
    [job.result() for job in jobs if job]

    # Encode datasets
    for gen_cfg in cfg['generators']:
        if not 'encode_path' in gen_cfg:
            continue

        encode_path = Path(gen_cfg['encode_path'])
        if not encode_path.exists():
            if not high_performance:
                raise Exception(f'High-performance required to run to encode dataset.')

            gen_path = get_gen_path(gen_cfg['type'], in_dataset, gen_cfg['kwargs'])
            print('Running get_encode_dataset, ',
                  (gen_cfg['type'], gen_path, encode_path, in_dataset))
            gen_encode_dataset(gen_cfg['type'], gen_path, encode_path, in_dataset,
                               **gen_cfg.get('encode_kwargs', {}))
        else:
            gen_path = get_gen_path(gen_cfg['type'], in_dataset, gen_cfg['kwargs'])
            print('Found cached get_encode_dataset, ',
                  (gen_cfg['type'], gen_path, encode_path, in_dataset),
                  gen_cfg.get('encode_kwargs'))

    # Create contrastive examples
    for cls_cfg in cfg['classifiers']:
        cls_type = cls_cfg['type']
        id = cls_cfg['name']
        cls_path = get_cls_path(cls_cfg['type'], in_dataset, id=cls_cfg['name'])
        for contrastive_method in cfg['contrastive_methods']:
            for gen_cfg in cfg['generators']:
                gen_path = get_gen_path(gen_cfg['type'], in_dataset, **gen_cfg['kwargs'])
                contrastive_path = get_contrastive_dataset_path(
                    contrastive_method,
                    in_dataset,
                    cls_type,
                    gen_cfg['type'],
                    classifier_id=id,
                    subset=cfg['contrastive_subset'])

                if not contrastive_path.exists():
                    if not high_performance:
                        raise Exception(
                            f'Could not find contrastive dataset for method {contrastive_method} with generator {gen_cfg["type"]} at {contrastive_path}. Please run this expeirment in a high-performance setting and set high_performance=True.'
                        )
                    print(
                        'Running make_contrastive_dataset, ',
                        (contrastive_method, in_dataset, cls_type, cls_path, gen_cfg['type'],
                         gen_path), {
                             'classifier_id': id,
                             'subset': cfg['contrastive_subset'],
                             **cfg.get('contrastive_kwargs', {})
                         })
                    make_contrastive_dataset(contrastive_method,
                                             in_dataset,
                                             cls_type,
                                             cls_path,
                                             gen_cfg['type'],
                                             gen_path,
                                             classifier_id=id,
                                             subset=cfg['contrastive_subset'],
                                             **cfg.get('contrastive_kwargs', {}))
                else:
                    print(
                        'Found cached make_contrastive_dataset, ',
                        (contrastive_method, in_dataset, cls_type, cls_path, gen_cfg['type'],
                         gen_path), {
                             'classifier_id': id,
                             'subset': cfg['contrastive_subset'],
                             **cfg.get('contrastive_kwargs', {})
                         })

    # Evaluate contrastive examples
    for cls_cfg in cfg['classifiers']:
        cls_type = cls_cfg['type']
        id = cls_cfg['name']
        cls_path = get_cls_path(cls_cfg['type'], in_dataset, id=cls_cfg['name'])
        print(f'\n\nResults for {id}:')

        results = {}
        for contrastive_method in cfg['contrastive_methods']:
            results[contrastive_method] = {}
            for gen_cfg in cfg['generators']:
                results[contrastive_method][gen_cfg['type']] = {}
                for adv_attack in cfg['adv_attacks']:
                    contrastive_ds_path = get_contrastive_dataset_path(
                        contrastive_method,
                        in_dataset,
                        cls_type,
                        gen_cfg['type'],
                        subset=cfg['contrastive_subset'],
                        classifier_id=id)
                    contrastive_res_path = get_eval_res_path(contrastive_method,
                                                             cls_type,
                                                             in_dataset,
                                                             out_dataset,
                                                             gen_cfg['type'],
                                                             adv_attack,
                                                             classifier_id=id,
                                                             subset=cfg['contrastive_subset'])
                    if not contrastive_res_path.exists():
                        contrastive_res_path.parent.mkdir(parents=True, exist_ok=True)
                        print(f'Evaluating:')
                        print(
                            f'Contrastive method: {contrastive_method} with {gen_cfg["type"]}')
                        print(f'Adversarial attack: {adv_attack}')
                        eval_contrastive_ds(contrastive_method,
                                            contrastive_ds_path,
                                            cls_type,
                                            cls_path,
                                            in_dataset,
                                            out_dataset,
                                            gen_cfg['type'],
                                            adv_attack,
                                            classifier_id=id,
                                            subset=cfg['contrastive_subset'],
                                            verbose=True)

                    with open(contrastive_res_path) as in_file:
                        results[contrastive_method][gen_cfg['type']][adv_attack] = json.load(
                            in_file)

        _grid_output(cfg['adv_attacks'], cfg['contrastive_methods'], cfg['generators'],
                     results)


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
