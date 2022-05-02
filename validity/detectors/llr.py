import pathlib

import numpy as np
import fire
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegressionCV

from validity.datasets import load_datasets, load_detector_datasets
from validity.generators.load import load_gen


class LikelihoodRatioDetector(nn.Module):

    @classmethod
    def load(cls, saved_dict):
        llr = cls(**saved_dict['args'])
        return llr

    def __init__(self, ll_est_path=None, bg_ll_est_path=None, _lr=None):
        super().__init__()
        self.ll_est_path = ll_est_path
        self.bg_ll_est_path = bg_ll_est_path
        self.lr = _lr

        self.ll_est = load_gen(self.ll_est_path)
        self.bg_ll_est = load_gen(self.bg_ll_est_path)

    def get_args(self):
        return {
            'type': 'llr',
            'args': {
                'll_est_path': self.ll_est_path,
                'bg_ll_est_path': self.bg_ll_est_path,
                '_lr': self.lr
            }
        }

    def predict(self, x):
        self.ll_est.eval()
        self.bg_ll_est.eval()
        x = x.type(torch.float)
        x = x.cuda()
        llr = self.ll_est.log_prob(x) - self.bg_ll_est.log_prob(x)
        llr = llr.cpu().detach().numpy()
        llr = np.expand_dims(llr, -1)
        return self.lr.predict(llr)

    def predict_proba(self, x):
        self.ll_est.eval()
        self.bg_ll_est.eval()
        x = x.type(torch.float)
        x = x.cuda()
        llr = self.ll_est.log_prob(x) - self.bg_ll_est.log_prob(x)
        llr = llr.cpu().detach().numpy()
        llr = np.expand_dims(llr, -1)
        return self.lr.predict_proba(llr)

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
                  foreground_path,
                  background_path,
                  mutation_rate,
                  data_root='./datasets',
                  cuda_idx=0,
                  id=None):
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(cuda_idx)

    detector = LikelihoodRatioDetector(foreground_path, background_path)
    detector.cuda()

    # load datasets
    _, _, in_ood_train_ds, in_test_ds = load_detector_datasets(in_dataset, data_root=data_root)
    _, _, out_ood_train_ds, out_test_ds = load_detector_datasets(out_dataset,
                                                                 data_root=data_root)

    in_train_loader = torch.utils.data.DataLoader(in_ood_train_ds, batch_size=64, shuffle=True)
    in_test_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=64, shuffle=True)
    out_train_loader = torch.utils.data.DataLoader(out_ood_train_ds,
                                                   batch_size=64,
                                                   shuffle=True)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=64, shuffle=True)

    # train detector
    detector.train(in_train_loader, out_train_loader)

    # evaluate detector
    results = detector.evaluate(in_test_loader, out_test_loader)

    save_path = get_llr_path(in_dataset, out_dataset, mutation_rate, id=id)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector.get_args(), save_path)

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


def evaluate_llr(in_dataset, out_dataset, mutation_rate, id=None):
    llr = load_llr(in_dataset, out_dataset, mutation_rate, id=id)

    _, in_test_ds = load_datasets(in_dataset)
    _, out_test_ds = load_datasets(out_dataset)

    in_test_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=64, shuffle=True)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=64, shuffle=True)

    in_preds = [llr.predict(data) for data, _ in in_test_loader]
    out_preds = [llr.predict(data) for data, _ in out_test_loader]
    in_preds = np.concatenate(in_preds)
    out_preds = np.concatenate(out_preds)

    in_correct = np.where(in_preds == 0., 1., 0.)
    out_correct = np.where(out_preds == 1., 1., 0.)
    correct = np.concatenate([in_correct, out_correct])

    acc = correct.mean()
    print(f'Accuracy: {acc:.4f}')


def get_llr_path(in_dataset, out_dataset, mutation_rate, id=None):
    save_name = f'llr_{in_dataset}_{out_dataset}_{mutation_rate}'
    if id:
        save_name = f'{save_name}_{id}'
    return pathlib.Path('ood') / f'{save_name}.pt'


def get_best_llr_path(in_dataset, out_dataset, id=None):
    save_name = f'llr_{in_dataset}_{out_dataset}'
    if id:
        save_name = f'{save_name}_{id}'
    return pathlib.Path('ood') / f'{save_name}_best.pt'


def load_llr(in_dataset, out_dataset, mutation_rate, id=None):
    save_path = get_llr_path(in_dataset, out_dataset, mutation_rate, id=id)
    if not save_path.exists():
        return False
    saved_dict = torch.load(save_path)
    return LikelihoodRatioDetector.load(saved_dict)


def load_best_llr(in_dataset, out_dataset, id=None):
    save_path = get_best_llr_path(in_dataset, out_dataset, id=id)
    if not save_path.exists():
        return False
    saved_dict = torch.load(save_path)
    return LikelihoodRatioDetector.load(saved_dict)


if __name__ == '__main__':
    fire.Fire()
