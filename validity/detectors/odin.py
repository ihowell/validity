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
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score


class ODINDetector:
    def __init__(self, network=None, noise_magnitude=None, temper=None):
        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        self.noise_magnitude = noise_magnitude
        self.temper = temper

        self.lr = None

    def predict(self, inputs):
        assert self.lr is not None
        score = self.score(inputs)
        return -self.lr.predict(score) + 1.

    def predict_proba(self, inputs):
        assert self.lr is not None
        score = self.score(inputs)
        return -self.lr.predict_proba(score) + 1.

    def evaluate(self, in_loader, out_loader):
        self.network.eval()
        score_in = []
        for data, _ in tqdm(in_loader, desc='Test in loader'):
            score_in.append(self.score(data))
        score_in = np.concatenate(score_in)
        val_score_in, test_score_in = score_in[:1000], score_in[1000:]

        score_out = []
        for data, _ in tqdm(out_loader, desc='Test out loader'):
            score_out.append(self.score(data))
        score_out = np.concatenate(score_out)
        val_score_out, test_score_out = score_out[:1000], score_out[1000:]

        val_scores = np.concatenate([val_score_in, val_score_out])
        test_scores = np.concatenate([test_score_in, test_score_out])
        val_labels = np.concatenate(
            [np.ones(val_score_in.shape[0]),
             np.zeros(val_score_out.shape[0])])
        test_labels = np.concatenate(
            [np.ones(test_score_in.shape[0]),
             np.zeros(test_score_out.shape[0])])

        res = {}

        self.lr = LogisticRegressionCV(n_jobs=-1).fit(val_scores, val_labels)
        test_probs = self.lr.predict_proba(test_scores)[:, 1]
        test_preds = self.lr.predict(test_scores)

        fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
        res['plot'] = (fpr, tpr)
        res['auc_score'] = auc(fpr, tpr)
        for f, t in zip(fpr, tpr):
            if t >= 0.95:
                res['fpr_at_tpr_95'] = f
                break

        res['accuracy'] = accuracy_score(test_labels, test_preds)
        res['precision'] = precision_score(test_labels, test_preds)
        res['recall'] = recall_score(test_labels, test_preds)
        return res

    def score(self, images):
        images = images.cuda()
        inputs = Variable(images, requires_grad=True)
        outputs = self.network(inputs)

        if self.noise_magnitude != 0.:
            # Using temperature scaling
            outputs = outputs / self.temper
            labels = outputs.data.max(1)[1]
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            tempInputs = torch.add(inputs.data, gradient, alpha=-self.noise_magnitude)
            outputs = self.network(tempInputs)

        outputs = outputs / self.temper
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1, keepdim=True)
        return soft_out.data.cpu()


def train_odin(in_dataset,
               out_dataset,
               net_type,
               weights_path,
               data_root='./datasets/',
               cuda_idx=0,
               magnitude=1e-2,
               temperature=1000.):
    from validity.classifiers.resnet import ResNet34
    from validity.classifiers.mnist import MnistClassifier
    from validity.datasets import load_datasets
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    if net_type == 'mnist':
        network = MnistClassifier()
        network.load_state_dict(torch.load(weights_path, map_location=f'cuda:0'))
    elif net_type == 'resnet':
        network = ResNet34(
            10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        network.load_state_dict(torch.load(weights_path, map_location=f'cuda:0'))
    network = network.cuda()

    odin = ODINDetector(network, magnitude, temperature)

    _, in_test_ds = load_datasets(in_dataset)
    in_test_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=64, shuffle=True)
    _, out_test_ds = load_datasets(out_dataset)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=64, shuffle=True)

    results = odin.evaluate(in_test_loader, out_test_loader)

    save_path = pathlib.Path(
        'ood', f'odin_{net_type}_{in_dataset}_{out_dataset}_{magnitude}_{temperature}.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(odin, save_path)

    res_save_path = pathlib.Path(
        'ood', f'odin_{net_type}_{in_dataset}_{out_dataset}_{magnitude}_{temperature}_res.pt')
    torch.save(results, res_save_path)

    print(f'Magnitude {magnitude} temperature {temperature}:')
    for result_name, result in results.items():
        if type(result) in [dict, tuple, list]:
            continue
        if type(result) is np.ndarray:
            if np.flatten(result).shape == [1]:
                result = np.flatten(result)[0]
            else:
                continue
        print(f'{result_name:20}: {result:.4f}')
    return odin, results


def train_multiple_odin(in_dataset,
                        out_dataset,
                        net_type,
                        weights_path,
                        data_root='./datasets/',
                        cuda_idx=0,
                        latex_print=False):
    magnitudes = [
        0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.007, 0.01, 0.014, 0.02, 0.05
    ]
    result_table = [['Magnitude', 'AUC Score', 'FPR at TPR=0.95']]

    best_detector = None
    best_auc = None

    for magnitude in magnitudes:
        detector, res = train_odin(in_dataset, out_dataset, net_type, weights_path, data_root,
                                   cuda_idx, magnitude)
        fpr, tpr = res['plot']
        plt.plot(fpr, tpr, label=str(magnitude))
        row = [magnitude, res['auc_score'], res['fpr_at_tpr_95']]
        result_table.append(row)

        if best_detector is None:
            best_detector = detector
            best_auc = res['auc_score']
        elif res['auc_score'] > best_auc:
            best_detector = detector
            best_auc = res['auc_score']

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('ood/odin_results.png')

    save_path = pathlib.Path('ood', f'odin_{net_type}_{in_dataset}_{out_dataset}_best.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_detector, save_path)

    if latex_print:
        print(' & '.join(result_table[0]) + '\\\\')
        for row in result_table[1:]:
            magnitude = str(row[0])
            scores = [f'{x:.4f}' for x in row[1:]]
            new_row = [magnitude] + scores
            print(' & '.join(new_row) + '\\\\')
    else:
        print(tabulate(result_table))


def load_odin(net_type, in_dataset, out_dataset, magnitude, temperature):
    save_path = pathlib.Path(
        'ood', f'odin_{net_type}_{in_dataset}_{out_dataset}_{magnitude}_{temperature}.pt')
    assert save_path.exists(), f'{save_path} does not exist'
    return torch.load(save_path)


def load_best_odin(net_type, in_dataset, out_dataset):
    save_path = pathlib.Path('ood', f'odin_{net_type}_cifar10_svhn_best.pt')
    assert save_path.exists(), f'{save_path} does not exist'
    return torch.load(save_path)


if __name__ == '__main__':
    fire.Fire()
