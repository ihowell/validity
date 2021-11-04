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

        self.threshold = None

    def set_threshold(self, threshold):
        self.threshold = threshold

    def forward(self, inputs):
        assert self.threshold is not None
        score = self.score(inputs)
        return score > threshold

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
        test_labels = np.concatenate([
            np.ones(test_score_in.shape[0]),
            np.zeros(test_score_out.shape[0])
        ])

        res = {}

        fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
        best_th = thresholds[0]
        best_acc = (1. - fpr[0] + tpr[0]) / 2.
        for fr, tr, th in zip(fpr, tpr, thresholds):
            acc = (1. - fr + tr) / 2.
            if best_acc < acc:
                best_acc = acc
                best_th = th
        self.threshold = best_th

        fpr, tpr, thresholds = roc_curve(test_labels, test_scores)
        res['plot'] = (fpr, tpr)
        res['auc_score'] = auc(fpr, tpr)
        for f, t in zip(fpr, tpr):
            if t >= 0.95:
                res['fpr_at_tpr_95'] = f
                break

        res['threshold'] = self.threshold
        res['accuracy'] = accuracy_score(test_labels,
                                         test_scores > self.threshold)
        res['precision'] = precision_score(test_labels,
                                           test_scores > self.threshold)
        res['recall'] = recall_score(test_labels, test_scores > self.threshold)
        return res

    def score(self, images):
        images = images.cuda()

        inputs = Variable(images, requires_grad=True)

        batch_outputs = self.network(inputs)

        # Using temperature scaling
        outputs = batch_outputs / self.temper
        labels = outputs.data.max(1)[1]
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # if True:
        #     gradient.index_copy_(
        #         1,
        #         torch.LongTensor([0]).cuda(),
        #         gradient.index_select(1,
        #                               torch.LongTensor([0]).cuda()) / (0.2023))
        #     gradient.index_copy_(
        #         1,
        #         torch.LongTensor([1]).cuda(),
        #         gradient.index_select(1,
        #                               torch.LongTensor([1]).cuda()) / (0.1994))
        #     gradient.index_copy_(
        #         1,
        #         torch.LongTensor([2]).cuda(),
        #         gradient.index_select(1,
        #                               torch.LongTensor([2]).cuda()) / (0.2010))

        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,
                               gradient,
                               alpha=-self.noise_magnitude)

        outputs = self.network(tempInputs)
        outputs = outputs / self.temper
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1, keepdim=True)
        return soft_out.data.cpu()


def train_odin(location,
               data_root='./datasets/',
               cuda_idx=0,
               magnitude=1e-2,
               temperature=1000.):
    from validity.classifiers.resnet import ResNet34
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    network = ResNet34(
        10,
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)))
    weights = torch.load(location, map_location=f'cuda:{cuda_idx}')
    network.load_state_dict(weights)
    network.cuda()

    odin = ODINDetector(network, magnitude, temperature)

    # Note: The transform is the per channel mean and std dev of
    # cifar10 training set.

    in_test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor()),
                                                 batch_size=64,
                                                 shuffle=True)

    out_dataset = datasets.SVHN(root=data_root,
                                split='test',
                                download=True,
                                transform=transforms.ToTensor())
    out_test_loader = torch.utils.data.DataLoader(out_dataset,
                                                  batch_size=64,
                                                  shuffle=True)
    results = odin.evaluate(in_test_loader, out_test_loader)
    save_path = pathlib.Path('odin', 'resnet34', 'cifar10', 'ood.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(odin, save_path)
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
    return results


def train_multiple_odin(weights_path, data_root='./datasets/', cuda_idx=0):
    magnitudes = [
        0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2
    ]
    result_table = [['Magnitude', 'FPR at TPR=0.95', 'AUC Score']]

    for magnitude in magnitudes:
        res = train_odin(weights_path, data_root, cuda_idx, magnitude)
        fpr, tpr = res['plot']
        plt.plot(fpr, tpr, label=str(magnitude))
        row = [magnitude, res['fpr_at_tpr_95'], res['auc_score']]
        result_table.append(row)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()

    plt.savefig('results.png')

    print(tabulate(result_table))


if __name__ == '__main__':
    fire.Fire()
