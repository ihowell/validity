import fire
import numpy as np
import pathlib
import torch

from tabulate import tabulate
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score

from validity.adv_dataset import load_adv_dataset
from validity.datasets import load_datasets
from validity.detectors.load import load_ood_detectors, load_adv_detectors
from validity.detectors.lid import LIDDetector
from validity.detectors.mahalanobis import MahalanobisDetector
from validity.detectors.llr import LikelihoodRatioDetector
from validity.detectors.odin import ODINDetector
from validity.util import np_loader

from validity.generators.mnist_vae import train as train_mnist_vae

MNIST_VAE_KWARGS = {'beta': 20.}
MNIST_BG_VAE_KWARGS = {'beta': 20., 'mutation_rate': 0.3}


def joint_ood_adv(net_type,
                  in_ds_name,
                  out_ds_names,
                  adv_attacks,
                  batch_size=64,
                  data_root='./datasets/'):
    """
    Assumpmtions:
    * The first dataset name in `out_ds_names` is the ood dataset that detectors were trained on.
    """
    torch.cuda.manual_seed(0)
    if type(adv_attacks) == 'str':
        adv_attacks = [adv_attacks]

    # Load in-distribution dataset
    _, in_test_ds = load_datasets(in_ds_name, data_root=data_root)

    in_test_loader = torch.utils.data.DataLoader(in_test_ds,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    # Load adversarial datasets
    adv_datasets = {}
    for adv_attack in adv_attacks:
        _, adv_data, _ = load_adv_dataset(in_ds_name, adv_attack, net_type)
        adv_datasets[adv_attack] = adv_data[500:]

    # Load OOD datasets
    ood_loaders = {}
    for out_ds_name in out_ds_names:
        _, out_test_ds = load_datasets(out_ds_name, data_root=data_root)
        ood_loaders[out_ds_name] = torch.utils.data.DataLoader(out_test_ds,
                                                               batch_size=batch_size,
                                                               shuffle=False)

    # Load detectors
    ood_detectors = load_ood_detectors(net_type, in_ds_name, out_ds_names[0])
    adv_detectors = []
    for adv_attack in adv_attacks:
        adv_detectors = adv_detectors + load_adv_detectors(net_type, in_ds_name, adv_attack)

    detectors = ood_detectors + adv_detectors

    # Original dataset
    orig_res = {}
    orig_preds = {}
    for detector_name, detector in detectors:
        preds = []
        for batch, _ in tqdm(in_test_loader, desc=f'{detector_name}, on {in_ds_name}'):
            preds.append(detector.predict(batch))
        preds = np.concatenate(preds)
        labels = np.zeros(preds.shape[0])

        orig_res[detector_name] = {}
        orig_res[detector_name]['accuracy'] = accuracy_score(labels, preds)
        orig_preds[detector_name] = preds

    # Evaluate all on ood datasets
    ood_res = {}
    ood_preds = {}
    for detector_name, detector in detectors:
        ood_res[detector_name] = {}
        ood_preds[detector_name] = {}
        preds = []
        for batch, _ in tqdm(out_test_loader, desc=f'{detector_name}, ood {out_ds_name}'):
            preds.append(detector.predict(batch))
        preds = np.concatenate(preds)
        labels = np.ones(preds.shape[0])

        ood_res[detector_name]['accuracy'] = accuracy_score(labels, preds)
        ood_preds[detector_name] = preds

    # Evaluate all on adv datasets
    adv_res = {}
    adv_preds = {}
    for detector_name, detector in detectors:
        adv_res[detector_name] = {}
        adv_preds[detector_name] = {}
        for adv_attack, adv_dataset in adv_datasets.items():
            preds = []
            for batch, _ in tqdm(np_loader(adv_data, True),
                                 desc=f'{detector_name}, adversarial {adv_attack}'):
                preds.append(detector.predict(batch.cuda()))
            preds = np.concatenate(preds)
            labels = np.ones(preds.shape[0])

            adv_res[detector_name][adv_attack] = {}
            adv_res[detector_name][adv_attack]['accuracy'] = accuracy_score(labels, preds)
            adv_preds[detector_name][adv_attack] = preds

    # Print results
    results = [['Method', in_ds_name, out_ds_name] + adv_attacks]
    for detector_name, detector in detectors:
        row = [
            orig_res[detector_name]['accuracy'],
            ood_res[detector_name]['accuracy'],
        ]
        for adv_attack in adv_attacks:
            row.append(adv_res[detector_name][adv_attack]['accuracy'])
        row = [f'{x:.4f}' for x in row]
        row = [detector_name] + row
        results.append(row)

    print(tabulate(results, tablefmt='tsv'))
    print('')
    headers = ['OOD Method', 'Adv Method', in_ds_name, out_ds_name
               ] + adv_attacks + ['Combined']

    results = []
    for ood_method, _ in ood_detectors:
        for adv_method, _ in adv_detectors:
            overall_correct = []

            preds = np.logical_or(orig_preds[ood_method], orig_preds[adv_method])
            orig_correct = np.where(preds == 0., 1., 0.)
            overall_correct.append(orig_correct)
            orig_acc = orig_correct.mean()

            preds = np.logical_or(ood_preds[ood_method], ood_preds[adv_method])
            ood_correct = np.where(preds == 1., 1., 0.)
            overall_correct.append(ood_correct)
            ood_acc = ood_correct.mean()

            adv_acc = []
            for adv_attack in adv_attacks:
                preds = np.logical_or(adv_preds[ood_method][adv_attack],
                                      adv_preds[adv_method][adv_attack])
                adv_correct = np.where(preds == 1., 1., 0.)
                overall_correct.append(adv_correct)
                adv_acc.append(adv_correct.mean())

            correct = np.concatenate(overall_correct)
            acc = correct.mean()

            accuracies = [orig_acc, ood_acc] + adv_acc + [acc]
            row = [ood_method, adv_method] + [f'{x:.4f}' for x in accuracies]
            results.append(row)

    print(tabulate(results, headers=headers, tablefmt='tsv'))


if __name__ == '__main__':
    fire.Fire(joint_ood_adv)
