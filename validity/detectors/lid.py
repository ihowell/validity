import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fire
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate

from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from scipy.spatial.distance import cdist
from validity.adv_dataset import load_adv_datasets

from validity.datasets import load_datasets
from validity.classifiers.load import load_cls
from validity.util import np_loader


class LIDDetector(nn.Module):

    @classmethod
    def load(cls, saved_dict):
        detector = cls(**saved_dict['args'])
        return detector

    def __init__(self,
                 classifier_path=None,
                 k=None,
                 dataset=None,
                 estimate_size=128,
                 _num_outputs=None,
                 _lr=None):
        super().__init__()
        self.classifier_path = classifier_path
        self.k = k
        self.dataset = dataset
        self.estimate_size = estimate_size

        self.num_outputs = _num_outputs
        self.lr = _lr

        # Don't need to save to disk
        self.classifier = load_cls(self.classifier_path)
        self.classifier.eval()
        self.criterion = nn.CrossEntropyLoss()

        self._estimate_loader = None

    def get_save_dict(self):
        return {
            'args': {
                'classifier_path': self.classifier_path,
                'k': self.k,
                'dataset': self.dataset,
                'estimate_size': self.estimate_size,
                '_num_outputs': self.num_outputs,
                '_lr': self.lr
            }
        }

    def sample_estimate(self):
        if self._estimate_loader is None:
            ds, _ = load_datasets(self.dataset)
            self._estimate_loader = DataLoader(ds, batch_size=self.estimate_size, shuffle=True)
        return next(iter(self._estimate_loader))

    def predict(self, x):
        assert self.lr is not None
        x = x.type(torch.float)
        if self.num_outputs is None:
            self.set_num_outputs(x)
        estimate, _ = self.sample_estimate()
        scores = self.score(estimate.cuda(), x)
        return -self.lr.predict(scores) + 1.

    def predict_proba(self, x):
        assert self.lr is not None
        x = x.type(torch.float)
        if self.num_outputs is None:
            self.set_num_outputs(x)
        estimate, _ = self.sample_estimate()
        scores = self.score(estimate.cuda(), x)
        return -self.lr.predict_proba(scores) + 1.

    def set_num_outputs(self, data):
        temp_list = self.classifier.feature_list(data.cuda())[1]
        self.num_outputs = len(temp_list)

    def train(self, in_train_loader, out_train_loader, noise_train_loader):
        if self.num_outputs is None:
            self.set_num_outputs(next(iter(in_train_loader))[0])

        # Compute LID scores for in and out distributions
        print('Computing LID')
        LID_in = []
        LID_out = []
        LID_noise = []
        out_test_iter = iter(out_train_loader)
        noise_test_iter = iter(noise_train_loader)
        for in_data, _ in tqdm(in_train_loader):
            LID_in.append(self.score(in_data, in_data))
            out_data = next(out_test_iter)[0]
            LID_out.append(self.score(in_data, out_data))
            noise_data = next(noise_test_iter)[0]
            LID_noise.append(self.score(in_data, noise_data))
        LID_in = np.concatenate(LID_in)
        LID_out = np.concatenate(LID_out)
        LID_noise = np.concatenate(LID_noise)

        # Create train/validation and test sets
        LID_in_val = LID_in[:int(LID_in.shape[0] * 0.1)]
        LID_in_test = LID_in[int(LID_in.shape[0] * 0.1):]
        LID_out_val, LID_out_test = LID_out[:int(LID_out.shape[0] *
                                                 0.1)], LID_out[int(LID_out.shape[0] * 0.1):]
        LID_noise_val, LID_noise_test = LID_noise[:int(LID_noise.shape[0] * 0.1
                                                       )], LID_noise[int(LID_noise.shape[0] *
                                                                         0.1):]

        LID_train = np.concatenate([LID_in_val, LID_noise_val, LID_out_val])
        labels_train = np.concatenate([
            np.ones(LID_in_val.shape[0] + LID_noise_val.shape[0]),
            np.zeros(LID_out_val.shape[0])
        ])

        # Train regressor
        print('Training regressor')
        self.lr = LogisticRegressionCV(n_jobs=-1).fit(LID_train, labels_train)

    def evaluate(self, in_test_loader, out_test_loader, noise_test_loader):
        assert self.num_outputs is not None

        LID_in = []
        LID_out = []
        LID_noise = []
        out_test_iter = iter(out_test_loader)
        noise_test_iter = iter(noise_test_loader)
        for in_data, _ in tqdm(in_test_loader):
            LID_in.append(self.score(in_data, in_data))
            out_data = next(out_test_iter)[0]
            LID_out.append(self.score(in_data, out_data))
            noise_data = next(noise_test_iter)[0]
            LID_noise.append(self.score(in_data, noise_data))
        LID_in = np.concatenate(LID_in)
        LID_out = np.concatenate(LID_out)
        LID_noise = np.concatenate(LID_noise)

        LID_test = np.concatenate([LID_in, LID_noise, LID_out])
        labels_test = np.concatenate(
            [np.ones(LID_in.shape[0] + LID_noise.shape[0]),
             np.zeros(LID_out.shape[0])])

        # Evaluate regressor
        print('Evaluating regressor')
        pred_probs = self.lr.predict_proba(LID_test)[:, 1]
        preds = self.lr.predict(LID_test)

        res = {}
        fpr, tpr, thresholds = roc_curve(labels_test, pred_probs)
        res['plot'] = (fpr, tpr)
        res['auc_score'] = auc(fpr, tpr)
        for f, t in zip(fpr, tpr):
            if t >= 0.95:
                res['fpr_at_tpr_95'] = f
                break

        res['accuracy'] = accuracy_score(labels_test, preds)
        res['precision'] = precision_score(labels_test, preds)
        res['recall'] = recall_score(labels_test, preds)
        return res

    def score(self, estimate_data, data):
        data = data.cuda()
        data.requires_grad = True
        estimate_data = estimate_data.cuda()

        LID = []

        # Activations
        _, out_features = self.classifier.feature_list(data)
        _, est_features = self.classifier.feature_list(estimate_data)
        X_act = []
        Est_act = []
        for i in range(self.num_outputs):
            out_features[i] = out_features[i].view(out_features[i].size(0),
                                                   out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act.append(out_features[i].cpu().numpy().reshape((out_features[i].size(0), -1)))

            est_features[i] = est_features[i].view(est_features[i].size(0),
                                                   est_features[i].size(1), -1)
            est_features[i] = torch.mean(est_features[i].data, 2)
            Est_act.append(est_features[i].cpu().numpy().reshape(
                (est_features[i].size(0), -1)))

        # LID
        LID_list = []
        for j in range(self.num_outputs):
            lid_score = mle_batch(Est_act[j], X_act[j], k=self.k)
            lid_score = lid_score.reshape((lid_score.shape[0], -1))
            LID_list.append(lid_score)

        LID = np.concatenate(LID_list, axis=1)
        return LID


# lid of a batch of query points X
def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: -k / (np.sum(np.log(v / v[-1])) + 1e-8)
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def train_lid_adv(dataset,
                  net_type,
                  weights_path,
                  adv_attack,
                  batch_size=128,
                  cuda_idx=0,
                  k=10,
                  id=None):
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    detector = LIDDetector(classifier_path=weights_path, k=k, dataset=dataset)
    detector.cuda()

    data_dict = load_adv_datasets(dataset, adv_attack, net_type, classifier_id=id)
    clean_train, clean_test = data_dict['clean']
    adv_train, adv_test = data_dict['adv']
    noise_train, noise_test = data_dict['noise']

    in_train_loader = np_loader(clean_train, True)
    out_train_loader = np_loader(adv_train, False)
    noise_train_loader = np_loader(noise_train, True)
    detector.train(in_train_loader, out_train_loader, noise_train_loader)

    in_test_loader = np_loader(clean_test, True)
    out_test_loader = np_loader(adv_test, False)
    noise_test_loader = np_loader(noise_test, True)
    results = detector.evaluate(in_test_loader, out_test_loader, noise_test_loader)

    save_path = get_lid_path(net_type, dataset, adv_attack, k, id=id)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector.get_save_dict(), save_path)

    save_res_path = f'lid_{net_type}_{dataset}_{adv_attack}_{k}'
    if id:
        save_res_path = f'{save_res_path}_{id}'
    save_res_path = pathlib.Path('adv') / f'{save_res_path}_res.pt'
    torch.save(results, save_res_path)

    print(f'K = {k}:')
    for result_name, result in results.items():
        if type(result) in [dict, tuple, list]:
            continue
        if type(result) is np.ndarray:
            if np.flatten(result).shape == [1]:
                result = np.flatten(result)[0]
            else:
                continue
        print(f'{result_name:20}: {result:.4f}')
    return detector, results


def train_multiple_lid_adv(dataset,
                           net_type,
                           weights_path,
                           adv_attack,
                           batch_size=128,
                           cuda_idx=0,
                           latex_print=False,
                           id=None):
    k_list = range(10, 100, 10)
    result_table = [['K', 'AUC Score', 'FPR at TPR=0.95']]

    best_detector = None
    best_auc = None

    for k in k_list:
        detector, res = train_lid_adv(dataset,
                                      net_type,
                                      weights_path,
                                      adv_attack,
                                      batch_size=batch_size,
                                      cuda_idx=cuda_idx,
                                      k=k,
                                      id=id)
        fpr, tpr = res['plot']
        plt.plot(fpr, tpr, label=str(k))
        row = [k, res['auc_score'], res['fpr_at_tpr_95']]
        result_table.append(row)

        if best_detector is None:
            best_detector = detector
            best_auc = res['auc_score']
        elif res['auc_score'] > best_auc:
            best_detector = detector
            best_auc = res['auc_score']

    save_path = get_best_lid_path(net_type, dataset, adv_attack, id=id)
    torch.save(best_detector.get_save_dict(), save_path)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('adv/lid_results.png')

    if latex_print:
        print(' & '.join(result_table[0]) + '\\\\')
        for row in result_table[1:]:
            magnitude = str(row[0])
            scores = [f'{x:.4f}' for x in row[1:]]
            new_row = [magnitude] + scores
            print(' & '.join(new_row) + '\\\\')
    else:
        print(tabulate(result_table))


def get_lid_path(net_type, dataset, adv_attack, k, id=None):
    save_path = f'lid_{net_type}_{dataset}_{adv_attack}_{k}'
    if id:
        save_path = f'{save_path}_{id}'
    return pathlib.Path('adv') / f'{save_path}.pt'


def load_lid(net_type, dataset, adv_attack, k, id=None):
    save_path = get_lid_path(net_type, dataset, adv_attack, k, id=id)
    if not save_path.exists():
        return None
    save_dict = torch.load(save_path, map_location=torch.device('cpu'))
    return LIDDetector.load(save_dict)


def get_best_lid_path(net_type, dataset, adv_attack, id=None):
    save_path = f'lid_{net_type}_{dataset}_{adv_attack}'
    if id:
        save_path = f'{save_path}_{id}'
    return pathlib.Path('adv') / f'{save_path}_best.pt'


def load_best_lid(net_type, dataset, adv_attack, id=None):
    save_path = get_best_lid_path(net_type, dataset, adv_attack, id=id)
    if not save_path.exists():
        return False
    save_dict = torch.load(save_path, map_location=torch.device('cpu'))
    return LIDDetector.load(save_dict)


if __name__ == '__main__':
    fire.Fire()
