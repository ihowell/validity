import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fire
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from scipy.spatial.distance import cdist


class LIDDetector:
    def __init__(self, model=None, net_type=None, k=None):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.net_type = net_type
        self.k = k

        self.feature_list = None
        self.num_outputs = None
        self.sample_mean = None
        self.precision = None
        self.lr = None

    def evaluate(self, in_test_loader, out_test_loader, noise_test_loader):
        self.model.eval()

        if self.num_outputs is None:
            for data, label in in_test_loader:
                temp_list = self.model.feature_list(data.cuda())[1]
                break
            self.num_outputs = len(temp_list)

        # Compute LID scores for in and out distributions
        print('Computing LID')
        print('Scoring in dataset')
        LID_in = self.score_for_loader(in_test_loader)
        print('Scoring out of dataset')
        LID_out = self.score_for_loader(out_test_loader)
        print('Scoring noise')
        LID_noise = self.score_for_loader(noise_test_loader)

        # Create train/validation and test sets
        LID_in_val, LID_in_test = LID_in[:500], LID_in[500:]
        LID_out_val, LID_out_test = LID_out[:500], LID_out[500:]
        LID_noise_val, LID_noise_test = LID_noise[:500], LID_noise[500:]

        LID_val = np.concatenate([LID_in_val, LID_noise_val, LID_out_val])
        LID_test = np.concatenate([LID_in_test, LID_noise_test, LID_out_test])
        labels_val = np.concatenate(
            [np.ones(LID_in_val.shape[0] + LID_noise_val.shape[0]),
             np.zeros(LID_out_val.shape[0])])
        labels_test = np.concatenate(
            [np.ones(LID_in_test.shape[0] + LID_noise_test.shape[0]),
             np.zeros(LID_out_test.shape[0])])

        # Train regressor
        print('Training regressor')
        self.lr = LogisticRegressionCV(n_jobs=-1).fit(LID_val, labels_val)

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

    def score_for_loader(self, data_loader):
        return np.concatenate([self.score(data) for data, _ in data_loader])

    def score(self, data):
        data = data.cuda()
        data.requires_grad = True

        LID = []

        # Activations
        output, out_features = self.model.feature_list(data)
        X_act = []
        for i in range(self.num_outputs):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act.append(out_features[i].cpu().numpy().reshape((out_features[i].size(0), -1)))

        # LID
        LID_list = []
        for j in range(self.num_outputs):
            lid_score = mle_batch(X_act[j], X_act[j], k=self.k)
            lid_score = lid_score.reshape((lid_score.shape[0], -1))

            LID_list.append(lid_score)

        LID = np.concatenate(LID_list, axis=1)
        # LID_concat = LID_list[0]
        # for i in range(1, self.num_outputs):
        #     LID_concat = np.concatenate((LID_concat, LID_list[i]), axis=1)

        # LID.extend(LID_concat)

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
    f = lambda v: -k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def train_lid_adv(net_type, weights_path, adv_attack, data_root='./datasets', cuda_idx=0, k=10):
    from validity.classifiers.resnet import ResNet34
    from validity.adv_dataset import load_adv_dataset
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    network = ResNet34(10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    network.load_state_dict(torch.load(weights_path, map_location=f'cuda:{cuda_idx}'))
    network.cuda()

    clean_data, adv_data, noisy_data = load_adv_dataset('cifar10', adv_attack, 'resnet')

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

    detector = LIDDetector(model=network, net_type='resnet', k=k)

    in_test_loader = np_loader(clean_data, True)
    out_test_loader = np_loader(adv_data, False)
    noise_test_loader = np_loader(noisy_data, True)
    results = detector.evaluate(in_test_loader, out_test_loader, noise_test_loader)

    save_path = pathlib.Path('lid', f'{net_type}_cifar10_{k}.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector, save_path)
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
    return results


def train_multiple_lid_adv(net_type, weights_path, adv_attack, data_root='./datasets/', cuda_idx=0):
    k_list = range(10, 100, 10)
    result_table = [['K', 'FPR at TPR=0.95', 'AUC Score']]

    for k in k_list:
        res = train_lid_adv(net_type, weights_path, adv_attack, data_root=data_root, cuda_idx=cuda_idx, k=k)
        fpr, tpr = res['plot']
        plt.plot(fpr, tpr, label=str(k))
        row = [k, res['fpr_at_tpr_95'], res['auc_score']]
        result_table.append(row)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()

    plt.savefig('adv_results.png')

    print(tabulate(result_table))


if __name__ == '__main__':
    fire.Fire()
