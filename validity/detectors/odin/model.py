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
        self.sample_mean = None
        self.precision = None
        self.lr = None

        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        self.noise_magnitude = noise_magnitude
        self.temper = temper

    def forward(self, inputs):
        score = self.score(inputs)
        return self.lr.predict_proba(score)

    def train(self, in_loader, out_loader):
        score_in = []
        for data, _ in tqdm(in_loader, desc='in loader'):
            score_in.append(self.score(data))
        score_in = np.concatenate(score_in)

        score_out = []
        for data, _ in tqdm(out_loader, desc='out loader'):
            score_out.append(self.score(data))
        score_out = np.concatenate(score_out)

        # plt.hist(score_in, bins=100, alpha=0.5, label='in', density=True)
        # plt.hist(score_out, bins=100, alpha=0.5, label='out', density=True)
        # plt.show()

        # Resample to smaller set to even sets
        min_size = min(score_in.shape[0], score_out.shape[0])
        score_in = score_in[:min_size]
        score_out = score_out[:min_size]

        scores = np.concatenate([score_in, score_out])
        labels = np.concatenate(
            [np.zeros(score_in.shape[0]),
             np.ones(score_out.shape[0])])

        self.lr = LogisticRegressionCV(n_jobs=-1).fit(scores, labels)

    def evaluate(self, in_loader, out_loader):
        score_in = []
        for data, _ in tqdm(in_loader, desc='Test in loader'):
            score_in.append(self.score(data))
        score_in = np.concatenate(score_in)

        score_out = []
        for data, _ in tqdm(out_loader, desc='Test out loader'):
            score_out.append(self.score(data))
        score_out = np.concatenate(score_out)

        min_size = min(score_in.shape[0], score_out.shape[0])
        score_in = score_in[:min_size]
        score_out = score_out[:min_size]

        scores = np.concatenate([score_in, score_out])
        labels = np.concatenate(
            [np.zeros(score_in.shape[0]),
             np.ones(score_out.shape[0])])

        res = {}

        fpr, tpr, thresholds = roc_curve(labels, -scores)
        res['auc_score'] = auc(fpr, tpr)
        for f, t in zip(fpr, tpr):
            if t >= 0.95:
                res['fpr_at_tpr_95'] = f
                break
        res['plot'] = (fpr, tpr)

        # acc = accuracy_score(labels, scores)
        # precision = precision_score(labels, scores)
        # recall = recall_score(labels, scores)

        # print(f'Num in dist = {in_preds.shape[0]}')
        # print(f'Num out dist = {out_preds.shape[0]}')
        # print(f'AUC = {auc_score:0.4f}')
        # print(f'Accuracy = {acc:0.4f}')
        # print(f'Precision = {precision:0.4f}')
        # print(f'Recall = {recall:0.4f}')
        # print(f'Mean in dist = {np.mean(in_preds):0.4f}')
        # print(f'Mean out dist = {np.mean(out_preds):0.4f}')
        # plt.plot(fpr, tpr)
        # plt.show()
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
        if gradient.shape[1] == 3:
            gradient[0][0] = (gradient[0][0]) / 0.2023
            gradient[0][1] = (gradient[0][1]) / 0.1994
            gradient[0][2] = (gradient[0][2]) / 0.2010

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
               data_root='./',
               cuda_idx=0,
               magnitude=1e-2,
               temperature=1000.):
    from validity.classifiers.resnet import ResNet34
    network = ResNet34(10)
    weights = torch.load(location, map_location=torch.device('cpu'))
    network.load_state_dict(weights)
    network = network.cuda()

    odin = ODINDetector(network, magnitude, temperature)

    # Note: The transform is the per channel mean and std dev of
    # cifar10 training set.
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    in_train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=in_transform),
                                                  batch_size=64,
                                                  shuffle=True)
    out_train_loader = torch.utils.data.DataLoader(datasets.SVHN(
        root=data_root, split='train', download=True, transform=in_transform),
                                                   batch_size=64,
                                                   shuffle=True)

    odin.train(in_train_loader, out_train_loader)
    torch.save(odin, 'odin.pt')

    in_test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=in_transform),
                                                 batch_size=64,
                                                 shuffle=True)

    out_dataset = datasets.SVHN(root=data_root,
                                split='test',
                                download=True,
                                transform=in_transform)
    out_test_loader = torch.utils.data.DataLoader(out_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

    results = odin.evaluate(in_test_loader, out_test_loader)
    print(f'Magnitude {magnitude} temperature {temperature}:')
    print(f'FPR: {results["fpr_at_tpr_95"]}')
    print(f'AUC: {results["auc_score"]}')
    return results


def train_multiple_odin(weights_path, data_root='./', cuda_idx=0):
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
