import json

import fire
import numpy as np
from pathlib import Path
import torch

from tabulate import tabulate
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score

from validity.adv_dataset import load_adv_dataset
from validity.classifiers.load import load_cls
from validity.datasets import load_datasets
from validity.detectors.load import load_detectors
from validity.detectors.lid import LIDDetector
from validity.detectors.llr import LikelihoodRatioDetector
from validity.detectors.mahalanobis import MahalanobisDetector
from validity.detectors.odin import ODINDetector
from validity.util import np_loader, NPZDataset


def eval_contrastive_ds(contrastive_method,
                        contrastive_ds_path,
                        cls_type,
                        cls_weights_path,
                        in_ds_name,
                        out_ds_name,
                        adv_attack,
                        data_root='./datasets/',
                        adv_step=True,
                        batch_size=64,
                        verbose=False):
    torch.cuda.manual_seed(0)

    # Load datasets
    dataset = NPZDataset(contrastive_ds_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load detectors
    ood_detectors, adv_detectors = load_detectors(in_ds_name,
                                                  out_ds_name,
                                                  cls_type,
                                                  adv_attack,
                                                  adv_step=adv_step)

    cls = load_cls(cls_type, cls_weights_path, in_ds_name)
    if verbose:
        print(f'Evaluating on classifier:')
    cls_preds = []
    for data, labels in tqdm(loader, disable=not verbose):
        cls_preds.append(
            torch.where(cls(data.cuda()).argmax(-1) == labels.cuda(), 1.,
                        0.).cpu().detach().numpy())
    cls_preds = np.concatenate(cls_preds)

    # Evaluate all on ood datasets
    if verbose:
        print(f'Evaluating on {len(ood_detectors)} ood detectors:')
    ood_preds = {}
    for detector_name, detector in ood_detectors:
        preds = []
        for batch, _ in tqdm(loader, disable=not verbose):
            preds.append(detector.predict(batch))
        preds = np.concatenate(preds)
        ood_preds[detector_name] = preds

    if verbose:
        print(f'Evaluating on {len(adv_detectors)} adv detectors')
    adv_preds = {}
    for detector_name, detector in adv_detectors:
        preds = []
        for batch, _ in tqdm(loader, disable=not verbose):
            preds.append(detector.predict(batch))
        preds = np.concatenate(preds)
        adv_preds[detector_name] = preds

    results = {}
    for ood_name, ood_pred in ood_preds.items():
        results[ood_name] = {}
        for adv_name, adv_pred in adv_preds.items():
            valid = 1. - np.logical_or(np.logical_or(ood_pred, adv_pred), cls_preds
                                       == 0.).mean()
            results[ood_name][adv_name] = valid

    # Print results
    print('Adv Step optimization:', adv_step)
    print(f'Target Class Valid: {float(cls_preds.mean()) * 100:.2f}%')
    result_table = [['', ''], ['', '']]

    adv_validity = {}
    for adv_name, _ in adv_detectors:
        result_table[0].append(adv_name)
        res = 1 - adv_preds[adv_name].mean()
        adv_validity[adv_name] = res
        result_table[1].append(f'{res*100:.2f}%')

    ood_validity = {}
    for ood_name in results:
        res = 1 - ood_preds[ood_name].mean()
        ood_validity[ood_name] = res
        res_row = [ood_name, f'{res * 100:.2f}%']
        for adv_name in results[ood_name]:
            res_row.append(f'{results[ood_name][adv_name] * 100:.2f}%')
        result_table.append(res_row)

    print(tabulate(result_table))

    with open(
            get_eval_res_path(contrastive_method, cls_type, in_ds_name, out_ds_name,
                              adv_attack), 'w') as out_file:
        json.dump(
            {
                'target_class_validity': float(cls_preds.mean()),
                'validity': results,
                'adv_validity': adv_validity,
                'ood_validity': ood_validity,
            }, out_file)


def get_eval_res_path(contrastive_method, cls_type, in_ds_name, out_ds_name, adv_attack):
    return Path(
        f'data/eval_{contrastive_method}_{cls_type}_{in_ds_name}_{out_ds_name}_{adv_attack}.json'
    )


if __name__ == '__main__':
    fire.Fire(eval_contrastive_ds)
