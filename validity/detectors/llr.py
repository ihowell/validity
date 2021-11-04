import json
import copy
import pathlib
import pickle

import matplotlib.pyplot as plt

import numpy as np
import fire
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegressionCV

from validity.adv_dataset import load_adv_dataset

BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}


class LikelihoodRatioDetector:
    def __init__(self, ll_est=None, bg_ll_est=None):
        self.ll_est = ll_est
        self.bg_ll_est = bg_ll_est

    def train(self, in_train_loader, out_train_loader, noise_train_loader):
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
                activ = self.model.penultimate_forward(data)[1]
                activations.append(activ)
            activations = torch.cat(activations)
            activations = torch.reshape(activations,
                                        (activations.shape[0], -1))
            self.kdes[label] = KernelDensity(
                kernel='gaussian',
                bandwidth=BANDWIDTHS[train_ds_name]).fit(activations)

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

    def score_loader(self, loader):
        activations = []
        for data, _ in tqdm(loader, desc='Scoring'):
            data = data.cuda()
            activ = self.model.penultimate_forward(data)[1]
            activations.append(activ)
        activations = torch.cat(activations)
        activations = torch.reshape(activations, (activations.shape[0], -1))
        densities = []
        for label, kde in self.kdes.items():
            densities.append(kde.score_samples(activations))
        densities = np.concatenate(densities, axis=1)
        return densities

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


def train_density_adv(weights_path, data_root='./datasets', cuda_idx=0):
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

    clean_data, adv_data, noisy_data = load_adv_dataset(
        'cifar10', 'fgsm', 'resnet')
    idx = np.random.shuffle(np.arrange(clean_data.shape[0]))
    pivot = int(clean_data.shape[0] * 0.1)
    train_idx, test_idx = idx[:pivot], idx[pivot:]

    clean_train_data = np.take_along_axis(clean_data, train_idx, 0)
    clean_test_data = np.take_along_axis(clean_data, test_idx, 0)
    adv_train_data = np.take_along_axis(adv_data, train_idx, 0)
    adv_test_data = np.take_along_axis(adv_data, test_idx, 0)
    noise_train_data = np.take_along_axis(noise_data, train_idx, 0)
    noise_test_data = np.take_along_axis(noise_data, test_idx, 0)

    class np_loader:
        def __init__(self, ds, label_is_ones):
            self.ds = ds
            self.label_is_ones = label_is_ones

        def __iter__(self):
            if self.label_is_ones:
                label = np.ones(64)
            else:
                label = np.zeros(64)

            for i in range(self.ds.shape[0] // 64):
                batch = self.ds[i * 64:(i + 1) * 64]
                yield torch.tensor(batch), torch.tensor(label)

    detector = DensityDetector(model=network, num_classes=10)

    in_train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor()),
                                                  batch_size=64,
                                                  shuffle=False)

    in_train_loader = np_loader(clean_train_data, True)
    out_train_loader = np_loader(adv_train_data, False)
    noise_train_loader = np_loader(noisy_train_data, False)

    detector.train(in_train_loader, out_train_loader, noise_train_loader)

    in_test_loader = np_loader(clean_test_data, True)
    out_test_loader = np_loader(adv_test_data, False)
    noise_test_loader = np_loader(noisy_test_data, False)

    results = detector.evaluate(in_test_loader, out_test_loader,
                                noise_test_loader)

    save_path = pathlib.Path('density', 'resnet34', 'cifar10', 'adv.pt')
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
    fire.Fire(get_llr_detector)
