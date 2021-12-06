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

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegressionCV

from validity.datasets import load_datasets

BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}


class LikelihoodRatioDetector:
    def __init__(self, ll_est=None, bg_ll_est=None):
        self.ll_est = ll_est
        self.bg_ll_est = bg_ll_est

    def train(self, in_train_loader, out_train_loader):
        self.ll_est.eval()
        self.bg_ll_est.eval()

        llr_in = []
        for data, _ in tqdm(in_train_loader, desc=f'In data likelihood'):
            llr = self.ll_est.log_prob(data.cuda()) - self.bg_ll_est.log_prob(data.cuda())
            llr_in.append(llr.cpu().detach().numpy())
        llr_in = np.concatenate(llr_in)

        llr_out = []
        for data, _ in tqdm(out_train_loader, desc=f'Out data likelihood'):
            llr = self.ll_est.log_prob(data.cuda()) - self.bg_ll_est.log_prob(data.cuda())
            llr_out.append(llr.cpu().detach().numpy())
        llr_out = np.concatenate(llr_out)

        llr_in = np.expand_dims(llr_in, -1)
        llr_out = np.expand_dims(llr_out, -1)

        llr = np.concatenate([llr_in, llr_out])
        labels = np.concatenate([np.zeros_like(llr_in), np.ones_like(llr_out)])

        self.lr = LogisticRegressionCV(n_jobs=-1).fit(llr, labels)

    def evaluate(self, in_test_loader, out_test_loader):
        self.ll_est.eval()
        self.bg_ll_est.eval()
        llr_in = []
        for data, _ in tqdm(in_test_loader, desc=f'In data likelihood'):
            llr = self.ll_est.log_prob(data.cuda()) - self.bg_ll_est.log_prob(data.cuda())
            llr_in.append(llr.cpu().detach().numpy())
        llr_in = np.concatenate(llr_in)

        llr_out = []
        for data, _ in tqdm(out_test_loader, desc=f'Out data likelihood'):
            llr = self.ll_est.log_prob(data.cuda()) - self.bg_ll_est.log_prob(data.cuda())
            llr_out.append(llr.cpu().detach().numpy())
        llr_out = np.concatenate(llr_out)

        llr_in = np.expand_dims(llr_in, -1)
        llr_out = np.expand_dims(llr_out, -1)

        llr = np.concatenate([llr_in, llr_out])
        labels = np.concatenate([np.zeros_like(llr_in), np.ones_like(llr_out)])

        # Evaluate regressor
        print('Evaluating regressor')
        pred_probs = self.lr.predict_proba(llr)[:, 1]
        preds = self.lr.predict(llr)

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


def train_llr_ood(in_dataset,
                  out_dataset,
                  net_type,
                  foreground_path,
                  background_path,
                  data_root='./datasets',
                  cuda_idx=0):
    from validity.generators.nvae.model import load_nvae
    from validity.generators.mnist_vae import MnistVAE
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(cuda_idx)

    # load models
    if net_type == 'nvae':
        foreground = load_nvae(foreground_path, batch_size=1)
        background = load_nvae(background_path, batch_size=1)
    elif net_type == 'mnist':
        foreground = MnistVAE()
        foreground.load_state_dict(torch.load(foreground_path))
        background = MnistVAE()
        background.load_state_dict(torch.load(background_path))

    foreground = foreground.cuda()
    background = background.cuda()
    foreground.eval()
    background.eval()

    detector = LikelihoodRatioDetector(foreground, background)

    # load datasets
    in_train_ds, in_test_ds = load_datasets(in_dataset)
    out_train_ds, out_test_ds = load_datasets(out_dataset)

    in_train_loader = torch.utils.data.DataLoader(in_train_ds, batch_size=64, shuffle=True)
    in_val_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=64, shuffle=True)
    out_train_loader = torch.utils.data.DataLoader(out_train_ds, batch_size=64, shuffle=True)
    out_val_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=64, shuffle=True)

    # train detector
    detector.train(in_train_loader, out_train_loader)

    # evaluate detector
    results = detector.evaluate(in_val_loader, out_val_loader)

    save_path = pathlib.Path('ood', f'llr_{in_dataset}_{out_dataset}_ood.pt')
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
