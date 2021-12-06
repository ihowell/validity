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


def main(net_type,
         in_ds_name,
         out_ds_name,
         adv_attack,
         data_root='./datasets/',
         adv_step=True):
    torch.cuda.manual_seed(0)

    # Load datasets
    _, in_test_ds = load_datasets(in_ds_name)
    in_test_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=1, shuffle=False)

    _, out_test_ds = load_datasets(out_ds_name)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=1, shuffle=False)

    orig_data, adv_data, noise_data = load_adv_dataset(in_ds_name, adv_attack, net_type)
    orig_data = orig_data[500:]
    adv_data = adv_data[500:]
    noise_data = noise_data[500:]
    clean_data = np.concatenate([orig_data, noise_data])

    # Load detectors
    llr_ood = get_llr_detector(in_ds_name, out_ds_name, batch_size)

    lid_adv = load_best_lid(net_type, in_ds_name, adv_attack)
    if adv_step:
        odin_ood = load_best_odin(net_type, in_ds_name, out_ds_name)
        mahalanobis_ood = load_best_mahalanobis_ood(net_type, in_ds_name, out_ds_name)

        mahalanobis_adv = load_best_mahalanobis_adv(net_type, in_ds_name, adv_attack)
    else:
        odin_ood = load_odin(net_type, in_ds_name, out_ds_name, 0, 1000.)
        mahalanobis_ood = load_mahalanobis_ood(net_type, in_ds_name, out_ds_name, 0)

        mahalanobis_adv = load_mahalanobis_adv(net_type, in_ds_name, adv_attack, 0)

    detectors = [
        # OOD
        (f'ODIN OOD {out_ds_name}', odin_ood),
        (f'Mahalanobis OOD {out_ds_name}', mahalanobis_ood),
        (f'LLR OOD {out_ds_name}', llr_ood),

        # ADV
        (f'LID Adv {adv_attack}', lid_adv),
        (f'Mahalanobis Adv {adv_attack}', mahalanobis_adv),
    ]

    # Evaluate all on ood datasets
    ood_res = {}
    for detector_name, detector in detectors:
        in_preds = []
        in_probs = []
        out_preds = []
        out_probs = []
        for batch, _ in tqdm(in_test_loader, desc=f'{detector_name}, ood in distribution'):
            preds = detector.predict(batch)
            in_preds.append(preds)
            probs = detector.predict_proba(batch)[:, 1]
            in_probs.append(probs)
        for batch, _ in tqdm(out_test_loader, desc=f'{detector_name}, ood out distribution'):
            preds = detector.predict(batch)
            out_preds.append(preds)
            probs = detector.predict_proba(batch)[:, 1]
            out_probs.append(probs)
        in_preds = np.concatenate(in_preds)
        in_probs = np.concatenate(in_probs)
        out_preds = np.concatenate(out_preds)
        out_probs = np.concatenate(out_probs)

        preds = np.concatenate([in_preds, out_preds])
        probs = np.concatenate([in_probs, out_probs])
        labels = np.concatenate([np.zeros(in_preds.shape[0]), np.ones(out_preds.shape[0])])

        fpr, tpr, thresholds = roc_curve(labels, probs)

        ood_res[detector_name] = {}
        ood_res[detector_name]['auc_score'] = auc(fpr, tpr)
        ood_res[detector_name]['tpr'] = np.mean(np.where(out_preds == 1., 1., 0.))
        ood_res[detector_name]['tnr'] = np.mean(np.where(in_preds == 0., 1., 0.))
        ood_res[detector_name]['accuracy'] = accuracy_score(labels, preds)
        ood_res[detector_name]['precision'] = precision_score(labels, preds)

    # Evaluate all on adv datasets
    adv_res = {}
    for detector_name, detector in detectors:
        clean_preds = []
        clean_probs = []
        adv_preds = []
        adv_probs = []
        for batch, _ in tqdm(np_loader(clean_data, False),
                             desc=f'{detector_name}, clean data'):
            clean_preds.append(detector.predict(batch.cuda()))
            clean_probs.append(detector.predict_proba(batch.cuda())[:, 1])
        for batch, _ in tqdm(np_loader(adv_data, True),
                             desc=f'{detector_name}, adversarial data'):
            adv_preds.append(detector.predict(batch.cuda()))
            adv_probs.append(detector.predict_proba(batch.cuda())[:, 1])
        clean_preds = np.concatenate(clean_preds)
        clean_probs = np.concatenate(clean_probs)
        adv_preds = np.concatenate(adv_preds)
        adv_probs = np.concatenate(adv_probs)

        preds = np.concatenate([clean_preds, adv_preds])
        probs = np.concatenate([clean_probs, adv_probs])
        labels = np.concatenate([np.zeros(clean_preds.shape[0]), np.ones(adv_preds.shape[0])])

        fpr, tpr, thresholds = roc_curve(labels, probs)

        adv_res[detector_name] = {}
        adv_res[detector_name]['auc_score'] = auc(fpr, tpr)
        adv_res[detector_name]['tpr'] = np.mean(np.where(adv_preds == 1., 1., 0.))
        adv_res[detector_name]['tnr'] = np.mean(np.where(clean_preds == 0., 1., 0.))
        adv_res[detector_name]['accuracy'] = accuracy_score(labels, preds)
        adv_res[detector_name]['precision'] = precision_score(labels, preds)

    # Print results
    print('Adv Step optimization:', adv_step)
    results = [['', '', 'OOD', '', '', 'ADV', ''],
               ['Method', 'AUC', 'TPR', 'TNR', 'AUC', 'TPR', 'TNR']]
    for detector_name, detector in detectors:
        row = [
            ood_res[detector_name]['auc_score'],
            ood_res[detector_name]['tpr'],
            ood_res[detector_name]['tnr'],
            adv_res[detector_name]['auc_score'],
            adv_res[detector_name]['tpr'],
            adv_res[detector_name]['tnr'],
        ]
        row = [f'{x:.4f}' for x in row]
        row = [detector_name] + row
        results.append(row)

    print(tabulate(results))


if __name__ == '__main__':
    fire.Fire(main)
