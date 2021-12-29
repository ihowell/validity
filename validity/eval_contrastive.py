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
from validity.detectors.lid import load_best_lid, load_lid, LIDDetector
from validity.detectors.mahalanobis import MahalanobisDetector, load_best_mahalanobis_adv, load_best_mahalanobis_ood, load_mahalanobis_ood, load_mahalanobis_adv
from validity.detectors.odin import ODINDetector, load_best_odin, load_odin
from validity.util import np_loader


def main(contrastive_ds_path,
         cls_type,
         in_ds_name,
         out_ds_name,
         adv_attack,
         data_root='./datasets/',
         adv_step=True):
    torch.cuda.manual_seed(0)

    # Load datasets
    dataset = NPZDataset('contrastive/xgems_mnist.npz')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load detectors
    llr_ood = get_llr_detector(in_ds_name, out_ds_name, batch_size)

    lid_adv = load_best_lid(cls_type, in_ds_name, adv_attack)
    if adv_step:
        odin_ood = load_best_odin(cls_type, in_ds_name, out_ds_name)
        mahalanobis_ood = load_best_mahalanobis_ood(cls_type, in_ds_name, out_ds_name)

        mahalanobis_adv = load_best_mahalanobis_adv(cls_type, in_ds_name, adv_attack)
    else:
        odin_ood = load_odin(cls_type, in_ds_name, out_ds_name, 0, 1000.)
        mahalanobis_ood = load_mahalanobis_ood(cls_type, in_ds_name, out_ds_name, 0)

        mahalanobis_adv = load_mahalanobis_adv(cls_type, in_ds_name, adv_attack, 0)

    ood_detectors = [
        (f'ODIN OOD {out_ds_name}', odin_ood),
        (f'Mahalanobis OOD {out_ds_name}', mahalanobis_ood),
        (f'LLR OOD {out_ds_name}', llr_ood),
    ]
    adv_detectors = [
        (f'LID Adv {adv_attack}', lid_adv),
        (f'Mahalanobis Adv {adv_attack}', mahalanobis_adv),
    ]

    # Evaluate all on ood datasets
    ood_preds = {}
    for detector_name, detector in ood_detectors:
        preds = []
        for batch, _ in tqdm(loader):
            preds.append(detector.predict(batch))
        preds = np.concatenate(preds)
        ood_preds[detector_name] = preds

    adv_preds = {}
    for detector_name, detector in adv_detectors:
        preds = []
        for batch, _ in tqdm(loader):
            preds.append(detector.predict(batch))
        preds = np.concatenate(preds)
        adv_preds[detector_name] = preds

    results = {}
    for ood_name, ood_pred in ood_preds:
        results[ood_name] = {}
        for adv_name, adv_pred in adv_preds:
            valid = 1. - np.logical_or(ood_pred, adv_pred).mean()
            results[ood_name][adv_name] = valid

    # Print results
    print('Adv Step optimization:', adv_step)
    print('')
    print('% Valid:')
    results = [['']]

    for adv_name, _ in adv_detectors:
        results[0].append(adv_name)

    for ood_name, _ in ood_detectors:
        res_row = [ood_name]
        for adv_name, _ in adv_detectors:
            res_row.append(f'{results[ood_name][adv_name]:.4f}')

    print(tabulate(results))


if __name__ == '__main__':
    fire.Fire(main)
