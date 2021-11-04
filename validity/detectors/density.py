import json
import copy
import pathlib
import pickle
import multiprocessing as mp

import matplotlib.pyplot as plt

import numpy as np
import fire
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score

from validity.adv_dataset import load_adv_dataset

BANDWIDTHS = {'mnist': 1.20, 'cifar10': 0.26, 'svhn': 1.00}


class DensityDetector:
    def __init__(self, model=None, num_labels=None, train_ds_name=None):
        self.model = model
        self.num_labels = num_labels
        self.train_ds_name = train_ds_name

        self.kdes = None
        self.lr = None

    def train_kdes(self, in_train_loader):
        self.model.eval()

        self.kdes = {}
        for label in range(self.num_labels):
            activations = []
            for data, labels in tqdm(in_train_loader,
                                     desc=f'Training kde {label}'):
                data = data.cuda()
                labels = labels.cuda()

                idx = torch.where(labels == label)[0]
                data = torch.index_select(data, 0, idx)
                if data.shape[0] == 0:
                    continue
                activ = self.model.penultimate_forward(data)[1]
                activations.append(activ.detach().cpu().numpy())
            activations = np.concatenate(activations)
            activations = np.reshape(activations, (activations.shape[0], -1))
            self.kdes[label] = KernelDensity(
                kernel='gaussian',
                bandwidth=BANDWIDTHS[self.train_ds_name]).fit(activations)

    def train_lr(self, in_train_loader, out_train_loader, noise_train_loader):
        clean_densities = self.score_loader(in_train_loader)
        adv_densities = self.score_loader(out_train_loader)
        noise_densities = self.score_loader(noise_train_loader)

        densities = np.concatenate(
            [clean_densities, noise_densities, adv_densities])
        labels = np.concatenate([
            np.zeros(clean_densities.shape[0]),
            np.ones(noise_densities.shape[0] + adv_densities.shape[0])
        ])

        self.lr = LogisticRegressionCV(n_jobs=-1).fit(densities, labels)

    def evaluate(self, in_test_loader, out_test_loader, noise_test_loader):
        clean_densities = self.score_loader(in_test_loader)
        adv_densities = self.score_loader(out_test_loader)
        noise_densities = self.score_loader(noise_test_loader)

        densities = np.concatenate(
            [clean_densities, noise_densities, adv_densities])
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
            activ = self.model.penultimate_forward(data)[1]
            activations.append(activ.detach().cpu().numpy())
        activations = np.concatenate(activations)
        activations = np.reshape(activations, (activations.shape[0], -1))
        densities = []
        for label, kde in self.kdes.items():
            print(f'Computing density score for label: {label}')
            p = mp.Pool()
            dens = p.map(
                kde.score_samples,
                [activations[i:i + 1] for i in range(activations.shape[0])])
            dens = np.concatenate(dens)
            densities.append(dens)
        densities = np.stack(densities, -1)
        return densities


def train_density_adv(net_type,
                      weights_path,
                      adv_attack,
                      cuda_idx=0,
                      data_root='./datasets/'):
    from validity.classifiers.resnet import ResNet34
    from validity.adv_dataset import load_adv_dataset
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(cuda_idx)

    network = ResNet34(
        10,
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)))
    network.load_state_dict(
        torch.load(weights_path, map_location=f'cuda:{cuda_idx}'))
    network.cuda()

    clean_data, adv_data, noise_data = load_adv_dataset(
        'cifar10', adv_attack, 'resnet')
    idx = np.arange(clean_data.shape[0])
    np.random.shuffle(idx)
    pivot = int(clean_data.shape[0] * 0.1)
    train_idx, test_idx = idx[:pivot], idx[pivot:]

    clean_train_data = np.take(clean_data, train_idx, axis=0)
    clean_test_data = np.take(clean_data, test_idx, axis=0)
    adv_train_data = np.take(adv_data, train_idx, axis=0)
    adv_test_data = np.take(adv_data, test_idx, axis=0)
    noise_train_data = np.take(noise_data, train_idx, axis=0)
    noise_test_data = np.take(noise_data, test_idx, axis=0)

    class np_loader:
        def __init__(self, ds, label_is_ones, batch_size=64):
            self.ds = ds
            self.label_is_ones = label_is_ones
            self.batch_size = batch_size

        def __iter__(self):
            if self.label_is_ones:
                label = np.ones(self.batch_size)
            else:
                label = np.zeros(self.batch_size)

            for i in range(self.ds.shape[0] // self.batch_size):
                batch = self.ds[i * self.batch_size:(i + 1) * self.batch_size]
                yield torch.tensor(batch), torch.tensor(label)

    detector = DensityDetector(model=network,
                               num_labels=10,
                               train_ds_name='cifar10')

    in_train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor()),
                                                  batch_size=64,
                                                  shuffle=True)
    detector.train_kdes(in_train_loader)

    in_train_loader = np_loader(clean_train_data, True)
    out_train_loader = np_loader(adv_train_data, False)
    noise_train_loader = np_loader(noise_train_data, False)

    detector.train_lr(in_train_loader, out_train_loader, noise_train_loader)

    in_test_loader = np_loader(clean_test_data, True)
    out_test_loader = np_loader(adv_test_data, False)
    noise_test_loader = np_loader(noise_test_data, False)

    results = detector.evaluate(in_test_loader, out_test_loader,
                                noise_test_loader)

    save_path = pathlib.Path('density', f'{net_type}_cifar10_{adv_attack}.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector, save_path)

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


if __name__ == '__main__':
    fire.Fire()
