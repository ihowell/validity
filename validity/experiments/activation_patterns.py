import csv

import fire
import numpy as np
import pathlib
import torch

from tabulate import tabulate
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from validity.adv_dataset import load_adv_dataset
from validity.classifiers import load_cls
from validity.datasets import load_datasets
from validity.util import np_loader


def activation_test(cls_type,
                    cls_weights_path,
                    in_ds_name,
                    out_ds_name,
                    adv_attacks,
                    data_root='./datasets/',
                    batch_size=64):
    torch.cuda.manual_seed(0)
    if type(adv_attacks) == 'str':
        adv_attacks = [adv_attacks]

    cls = load_cls(cls_type, cls_weights_path, in_ds_name)
    cls = cls.cuda()
    cls.eval()

    # Load datasets
    _, in_test_ds = load_datasets(in_ds_name)
    _, out_test_ds = load_datasets(out_ds_name)
    in_test_loader = torch.utils.data.DataLoader(in_test_ds,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    in_features = []
    for batch, _ in tqdm(in_test_loader, desc='In test set'):
        _, features = cls.post_relu_features(batch.cuda())
        features = [f.flatten(1) for f in features]
        features = [f.cpu().detach().numpy() for f in features]
        features = np.concatenate(features, axis=1)
        in_features.append(features)
    in_features = np.concatenate(in_features)

    out_features = []
    for batch, _ in tqdm(out_test_loader, desc='Out test set'):
        _, features = cls.post_relu_features(batch.cuda())
        features = [f.flatten(1) for f in features]
        features = [f.cpu().detach().numpy() for f in features]
        features = np.concatenate(features, axis=1)
        out_features.append(features)
    out_features = np.concatenate(out_features)

    adv_feature_means = []
    for adv_attack in adv_attacks:
        _, adv_data, _ = load_adv_dataset(in_ds_name, adv_attack, cls_type)
        adv_loader = np_loader(adv_data, False)

        adv_features = []
        for batch, _ in tqdm(adv_loader, desc=f'{adv_attack} test set'):
            _, features = cls.post_relu_features(batch.cuda())
            features = [f.flatten(1) for f in features]
            features = [f.cpu().detach().numpy() for f in features]
            features = np.concatenate(features, axis=1)
            adv_features.append(features)
        adv_features = np.concatenate(adv_features)

        adv_features = np.where(adv_features > 0., 1., 0.)
        adv_feature_means.append(np.mean(adv_features, axis=0))

    in_features = np.where(in_features > 0., 1., 0.)
    out_features = np.where(out_features > 0., 1., 0.)

    print(f'{in_features.shape=}')
    print(f'{out_features.shape=}')

    in_feature_mean = np.mean(in_features, axis=0)
    out_feature_mean = np.mean(out_features, axis=0)

    print(f'{in_feature_mean.shape=}')
    print(f'{out_feature_mean.shape=}')

    print(f'{np.sum(np.where(in_feature_mean == 1., 1., 0.))=}')
    print(f'{np.sum(np.where(in_feature_mean == 0., 1., 0.))=}')

    with open('activ_means.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Position', 'In Activation', 'Out Activation'] + adv_attacks +
                        ['Out Diff'] + [f'{a} Diff' for a in adv_attacks])
        for i in range(in_feature_mean.shape[0]):
            row = [i, in_feature_mean[i], out_feature_mean[i]]
            for adv_feature_mean in adv_feature_means:
                row.append(adv_feature_mean[i])
            row.append(out_feature_mean[i] - in_feature_mean[i])
            for adv_feature_mean in adv_feature_means:
                row.append(adv_feature_mean[i] - in_feature_mean[i])
            writer.writerow(row)


def activation_detector(cls_type,
                        cls_weights_path,
                        in_ds_name,
                        out_ds_name,
                        adv_attacks,
                        data_root='./datasets/',
                        batch_size=64,
                        sample_variances=1):
    torch.cuda.manual_seed(0)
    if type(adv_attacks) == 'str':
        adv_attacks = [adv_attacks]

    cls = load_cls(cls_type, cls_weights_path, in_ds_name)
    cls = cls.cuda()
    cls.eval()

    # Load datasets
    _, in_test_ds = load_datasets(in_ds_name)
    _, out_test_ds = load_datasets(out_ds_name)
    in_test_loader = torch.utils.data.DataLoader(in_test_ds,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    for adv_attack in adv_attacks:
        clean_data, adv_data, _ = load_adv_dataset(in_ds_name, adv_attack, cls_type)
        clean_loader = np_loader(clean_data, False)
        adv_loader = np_loader(adv_data, True)

        clean_features = []
        for batch, _ in tqdm(clean_loader, desc=f'Clean test set'):
            _, features = cls.post_relu_features(batch.cuda())
            features = [f.flatten(1) for f in features]
            features = [f.cpu().detach().numpy() for f in features]
            features = np.concatenate(features, axis=1)
            clean_features.append(features)
        clean_features = np.concatenate(clean_features)

        adv_features = []
        for batch, _ in tqdm(adv_loader, desc=f'{adv_attack} test set'):
            _, features = cls.post_relu_features(batch.cuda())
            features = [f.flatten(1) for f in features]
            features = [f.cpu().detach().numpy() for f in features]
            features = np.concatenate(features, axis=1)
            adv_features.append(features)
        adv_features = np.concatenate(adv_features)

        clean_train_features = np.where(clean_features > 0., 1., 0.)
        adv_train_features = np.where(adv_features > 0., 1., 0.)

        def split_data(a, val_prop=0.1, test_prop=0.1):
            pivot_1 = int(len(a) * (1 - val_prop - test_prop))
            pivot_2 = int(len(a) * (1 - test_prop))
            return a[:pivot_1], a[pivot_1:pivot_2], a[pivot_2:]

        clean_train, clean_validate, clean_test = split_data(clean_features)
        adv_train, adv_validate, adv_test = split_data(adv_features)

        in_feature_mean = np.mean(clean_train, axis=0)
        in_variance = in_feature_mean * (1 - in_feature_mean)
        unique_variances = np.sort(np.unique(in_variance))

        mesh_fpr = []
        mesh_tpr = []
        mesh_variance = []
        for v in tqdm(unique_variances, desc='Variances'):
            idx = np.where(in_variance <= v)[0]

            clean_bits_flipped = np.sum(clean_validate[:, idx], axis=1)
            adv_bits_flipped = np.sum(adv_validate[:, idx], axis=1)

            scores = np.concatenate([clean_bits_flipped, adv_bits_flipped])
            labels = np.concatenate(
                [np.zeros(clean_bits_flipped.shape[0]),
                 np.ones(adv_bits_flipped.shape[0])])

            fpr, tpr, thresholds = roc_curve(labels, scores)
            auc_score = auc(fpr, tpr)
            mesh_fpr.append(fpr)
            mesh_tpr.append(tpr)
            mesh_variance.append(np.ones(fpr.shape[0]) * v)

        np.savez(f'activ_{adv_attack}_mesh', {
            'fpr': mesh_fpr,
            'tpr': mesh_tpr,
            'variance': mesh_variance
        })


def viz_adv_detection(adv_attack, sub_sample=10000):
    ds = np.load(f'activ_{adv_attack}_mesh.npz')
    tpr = ds['arr_0'].item()['tpr']
    fpr = ds['arr_0'].item()['fpr']
    variance = ds['arr_0'].item()['variance']

    tpr = np.concatenate(tpr)
    fpr = np.concatenate(fpr)
    variance = np.concatenate(variance)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(ds['tpr'][::sub_sample], ds['fpr'][::sub_sample],
                    ds['variance'][::sub_sample])
    ax.set_xlabel('TPR')
    ax.set_ylabel('FPR')
    ax.set_zlabel('Variance')
    plt.show()


def viz(adv_attack, variances_to_plot=None):
    ds = np.load(f'activ_{adv_attack}_mesh.npz', allow_pickle=True)

    tpr = ds['arr_0'].item()['tpr']
    fpr = ds['arr_0'].item()['fpr']
    variance = ds['arr_0'].item()['variance']

    if variances_to_plot is None:
        plot_idx = range(len(tpr))
    else:
        plot_idx = [
            int(len(variance) / variances_to_plot * i) for i in range(variances_to_plot)
        ]

    fig = plt.figure()
    ax = fig.add_subplot()

    for idx in plot_idx:
        ax.plot(fpr[idx], tpr[idx], label=variance[idx][0])

    best_auc = 0.
    best_idx = 0.
    for i in tqdm(range(len(tpr))):
        auc_score = auc(fpr[i], tpr[i])
        if auc_score > best_auc:
            best_auc = auc_score
            best_idx = i

    print(f'Best variance: {variance[best_idx][0]:.3f}')
    print(f'Best AUC: {best_auc:.3f}')

    ax.plot(fpr[best_idx], tpr[best_idx], label=variance[best_idx][0])

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fire.Fire()
