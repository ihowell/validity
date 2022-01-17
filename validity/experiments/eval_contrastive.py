import fire
import numpy as np
import pathlib
import torch

from tabulate import tabulate
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score

from validity.adv_dataset import load_adv_dataset
from validity.classifiers import load_cls
from validity.datasets import load_datasets
from validity.detectors.load import load_detectors
from validity.detectors.lid import LIDDetector
from validity.detectors.llr import LikelihoodRatioDetector
from validity.detectors.mahalanobis import MahalanobisDetector
from validity.detectors.odin import ODINDetector
from validity.util import np_loader, NPZDataset


def main(contrastive_ds_path,
         cls_type,
         cls_weights_path,
         in_ds_name,
         out_ds_name,
         adv_attack,
         data_root='./datasets/',
         adv_step=True,
         batch_size=64):
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
    print(f'Evaluating on classifier:')
    cls_preds = []
    for data, labels in tqdm(loader):
        cls_preds.append(
            torch.where(cls(data.cuda()).argmax(-1) == labels.cuda(), 1.,
                        0.).cpu().detach().numpy())
    cls_preds = np.concatenate(cls_preds)

    # Evaluate all on ood datasets
    print(f'Evaluating on {len(ood_detectors)} ood detectors:')
    ood_preds = {}
    for detector_name, detector in ood_detectors:
        preds = []
        for batch, _ in tqdm(loader):
            preds.append(detector.predict(batch))
        preds = np.concatenate(preds)
        ood_preds[detector_name] = preds

    print(f'Evaluating on {len(adv_detectors)} adv detectors')
    adv_preds = {}
    for detector_name, detector in adv_detectors:
        preds = []
        for batch, _ in tqdm(loader):
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
    print('')
    print(f'% Initial Valid: {float(cls_preds.mean()):.4f}')
    result_table = [['', ''], ['', '']]

    for adv_name, _ in adv_detectors:
        result_table[0].append(adv_name)
        res = 1 - adv_preds[adv_name].mean()
        result_table[1].append(f'{res:.4f}')

    for ood_name in results:
        res = 1 - ood_preds[ood_name].mean()
        res_row = [ood_name, f'{res:.4f}']
        for adv_name in results[ood_name]:
            res_row.append(f'{results[ood_name][adv_name]:.4f}')
        result_table.append(res_row)

    print(tabulate(result_table))


if __name__ == '__main__':
    fire.Fire(main)
