import pathlib
import multiprocessing as mp

import numpy as np
import fire
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KernelDensity

from validity.adv_dataset import load_adv_datasets
from validity.classifiers.load import load_cls
from validity.datasets import get_dataset_info, load_datasets
from validity.util import np_loader

BANDWIDTHS = {'mnist': 1.20, 'cifar10': 0.26, 'svhn': 1.00}


class DensityDetector(nn.Module):

    @classmethod
    def load(cls, saved_dict):
        detector = cls(**saved_dict['args'])
        return detector

    def __init__(self,
                 classifier_path=None,
                 num_labels=None,
                 train_ds_name=None,
                 _kdes=None,
                 _lr=None):
        super().__init__()
        self.classifier_path = classifier_path
        self.num_labels = num_labels
        self.train_ds_name = train_ds_name

        self.kdes = _kdes
        self.lr = _lr

        self.classifier = load_cls(classifier_path)
        self.classifier.eval()

    def get_save_dict(self):
        return {
            'args': {
                'classifier_path': self.classifier_path,
                'num_labels': self.num_labels,
                'train_ds_name': self.train_ds_name,
                '_kdes': self.kdes,
                '_lr': self.lr
            }
        }

    def train_kdes(self, in_train_loader):
        self.kdes = {}
        for label in range(self.num_labels):
            activations = []
            for data, labels in tqdm(in_train_loader, desc=f'Training kde {label}'):
                data = data.cuda()
                labels = labels.cuda()

                idx = torch.where(labels == label)[0]
                data = torch.index_select(data, 0, idx)
                if data.shape[0] == 0:
                    continue
                activ = self.classifier.penultimate_forward(data)[1]
                activations.append(activ.detach().cpu().numpy())
            activations = np.concatenate(activations)
            activations = np.reshape(activations, (activations.shape[0], -1))
            self.kdes[label] = KernelDensity(
                kernel='gaussian', bandwidth=BANDWIDTHS[self.train_ds_name]).fit(activations)

    def train_lr(self, in_train_loader, out_train_loader, noise_train_loader):
        clean_densities = self.score_loader(in_train_loader)
        adv_densities = self.score_loader(out_train_loader)
        noise_densities = self.score_loader(noise_train_loader)

        densities = np.concatenate([clean_densities, noise_densities, adv_densities])
        labels = np.concatenate([
            np.zeros(clean_densities.shape[0]),
            np.ones(noise_densities.shape[0] + adv_densities.shape[0])
        ])

        self.lr = LogisticRegressionCV(n_jobs=-1).fit(densities, labels)

    def evaluate(self, in_test_loader, out_test_loader, noise_test_loader):
        clean_densities = self.score_loader(in_test_loader)
        adv_densities = self.score_loader(out_test_loader)
        noise_densities = self.score_loader(noise_test_loader)

        densities = np.concatenate([clean_densities, noise_densities, adv_densities])
        labels = np.concatenate([
            np.zeros(clean_densities.shape[0]),
            np.ones(noise_densities.shape[0] + adv_densities.shape[0])
        ])

        # Evaluate regressor
        print('Evaluating regressor')
        pred_probs = self.lr.predict_proba(densities)[:, 1]
        preds = self.lr.predict(densities)

        res = {}
        fpr, tpr, thresholds = roc_curve(labels, pred_probs)
        res['plot'] = (fpr, tpr)
        res['auc_score'] = auc(fpr, tpr)
        for f, t in zip(fpr, tpr):
            if t >= 0.95:
                res['fpr_at_tpr_95'] = f
                break

        res['accuracy'] = accuracy_score(labels, preds)
        res['precision'] = precision_score(labels, preds)
        res['recall'] = recall_score(labels, preds)
        return res

    def score_loader(self, loader):
        activations = []
        for data, _ in tqdm(loader, desc='Computing activations'):
            data = data.cuda()
            activ = self.classifier.penultimate_forward(data)[1]
            activations.append(activ.detach().cpu().numpy())
        activations = np.concatenate(activations)
        activations = np.reshape(activations, (activations.shape[0], -1))
        densities = []
        for label, kde in self.kdes.items():
            print(f'Computing density score for label: {label}')
            p = mp.Pool()
            dens = p.map(kde.score_samples,
                         [activations[i:i + 1] for i in range(activations.shape[0])])
            dens = np.concatenate(dens)
            densities.append(dens)
        densities = np.stack(densities, -1)
        return densities


def train_density_adv(dataset, net_type, weights_path, adv_attack, cuda_idx=0, id=None):
    print(f'train_density_adv {locals()=}')
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(cuda_idx)

    ds_info = get_dataset_info(dataset)

    data_dict = load_adv_datasets(dataset, adv_attack, net_type, classifier_id=id)
    clean_train, clean_test = data_dict['clean']
    adv_train, adv_test = data_dict['adv']
    noise_train, noise_test = data_dict['noise']

    detector = DensityDetector(classifier_path=weights_path,
                               num_labels=ds_info.num_labels,
                               train_ds_name=dataset)
    detector.cuda()

    print('Training KDEs')
    in_train_ds, _ = load_datasets(dataset)
    in_train_loader = torch.utils.data.DataLoader(in_train_ds, batch_size=64, shuffle=True)
    detector.train_kdes(in_train_loader)

    print('Training LR')
    in_train_loader = np_loader(clean_train, True)
    out_train_loader = np_loader(adv_train, False)
    noise_train_loader = np_loader(noise_train, False)
    detector.train_lr(in_train_loader, out_train_loader, noise_train_loader)

    print('Evaluating')
    in_test_loader = np_loader(clean_test, True)
    out_test_loader = np_loader(adv_test, False)
    noise_test_loader = np_loader(noise_test, False)
    results = detector.evaluate(in_test_loader, out_test_loader, noise_test_loader)

    save_path = get_density_path(net_type, dataset, adv_attack, id=id)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector.get_save_dict(), save_path)

    res_save_path = f'density_{net_type}_{dataset}_{adv_attack}'
    if id:
        res_save_path = f'{res_save_path}_{id}'
    res_save_path = pathlib.Path('adv') / f'{res_save_path}_res.pt'
    torch.save(results, res_save_path)

    for result_name, result in results.items():
        if type(result) in [dict, tuple, list]:
            continue
        if type(result) is np.ndarray:
            if np.flatten(result).shape == [1]:
                result = np.flatten(result)[0]
            else:
                continue
        print(f'{result_name:20}: {result:.4f}')
    return results


def get_density_path(net_type, dataset, adv_attack, id=None):
    save_path = f'density_{net_type}_{dataset}_{adv_attack}'
    if id:
        save_path = f'{save_path}_{id}'
    return pathlib.Path('adv') / f'{save_path}.pt'


def load_density_adv(net_type, dataset, adv_attack, id=None):
    save_path = get_density_path(net_type, dataset, adv_attack, id=id)
    if not save_path.exists():
        return False
    save_dict = torch.load(save_path, map_location=torch.device('cpu'))
    return DensityDetector.load(save_dict)


if __name__ == '__main__':
    fire.Fire()
