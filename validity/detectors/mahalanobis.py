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

from validity.util import np_loader


class MahalanobisDetector:
    def __init__(self, model=None, num_classes=None, noise_magnitude=None, net_type=None):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.noise_magnitude = noise_magnitude
        self.num_classes = num_classes
        self.net_type = net_type

        self.feature_list = None
        self.num_outputs = None
        self.sample_mean = None
        self.precision = None
        self.lr = None

    def predict(self, inputs):
        assert self.lr is not None
        Mahalanobis = []
        for layer_idx in range(self.num_outputs):
            Mahalanobis.append(self.score(inputs, layer_idx))
        Mahalanobis = np.stack(Mahalanobis, axis=1)
        return -self.lr.predict(Mahalanobis) + 1.

    def predict_proba(self, inputs):
        assert self.lr is not None
        Mahalanobis = []
        for layer_idx in range(self.num_outputs):
            Mahalanobis.append(self.score(inputs, layer_idx))
        Mahalanobis = np.stack(Mahalanobis, axis=1)
        return -self.lr.predict_proba(Mahalanobis) + 1.

    def train(self, in_train_loader):
        # Set network to evaluation mode
        self.model.eval()

        if self.num_outputs is None:
            for data, label in in_train_loader:
                temp_list = self.model.feature_list(data.cuda())[1]
                break
            self.num_outputs = len(temp_list)
            self.feature_list = np.array([t.shape[1] for t in temp_list])

        # Compute sample mean and precision
        print('Calculating sample statistics')
        self.sample_estimator(in_train_loader)

    def evaluate(self, in_test_loader, out_test_loader, noise_test_loader=None):
        self.model.eval()

        if self.num_outputs is None:
            for data, label in in_test_loader:
                temp_list = self.model.feature_list(data.cuda())[1]
                break
            self.num_outputs = len(temp_list)

        # Compute Mahalanobis scores for in and out distributions
        print('Computing Mahalanobis distances')
        print('Scoring in dataset')
        Mahalanobis_in = self.score_for_loader(in_test_loader)
        print('Scoring out of dataset')
        Mahalanobis_out = self.score_for_loader(out_test_loader)
        if noise_test_loader:
            print('Scoring noise')
            Mahalanobis_noise = self.score_for_loader(noise_test_loader)

        # Create train/validation and test sets
        Mahalanobis_in_val, Mahalanobis_in_test = \
            Mahalanobis_in[:500], Mahalanobis_in[500:]
        Mahalanobis_out_val, Mahalanobis_out_test = \
            Mahalanobis_out[:500], Mahalanobis_out[500:]
        if noise_test_loader:
            Mahalanobis_noise_val, Mahalanobis_noise_test = \
                Mahalanobis_noise[:500], Mahalanobis_noise[500:]

        Mahalanobis_val = np.concatenate([Mahalanobis_in_val, Mahalanobis_out_val])
        Mahalanobis_test = np.concatenate([Mahalanobis_in_test, Mahalanobis_out_test])
        labels_val = np.concatenate(
            [np.ones(Mahalanobis_in_val.shape[0]),
             np.zeros(Mahalanobis_out_val.shape[0])])
        labels_test = np.concatenate(
            [np.ones(Mahalanobis_in_test.shape[0]),
             np.zeros(Mahalanobis_out_test.shape[0])])
        if noise_test_loader:
            Mahalanobis_val = np.concatenate([Mahalanobis_val, Mahalanobis_noise_val])
            Mahalanobis_test = np.concatenate([Mahalanobis_test, Mahalanobis_noise_test])
            labels_val = np.concatenate([labels_val, np.ones(Mahalanobis_noise_val.shape[0])])
            labels_test = np.concatenate(
                [labels_test, np.ones(Mahalanobis_noise_test.shape[0])])

        # Train regressor
        print('Training regressor')
        self.lr = LogisticRegressionCV(n_jobs=-1).fit(Mahalanobis_val, labels_val)

        # Evaluate regressor
        print('Evaluating regressor')
        test_probs = self.lr.predict_proba(Mahalanobis_test)[:, 1]
        preds = self.lr.predict(Mahalanobis_test)

        res = {}
        fpr, tpr, thresholds = roc_curve(labels_test, test_probs)
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
        Mahalanobis = []
        for layer_idx in range(self.num_outputs):
            M = np.concatenate([self.score(data, layer_idx) for data, _ in data_loader])
            Mahalanobis.append(M)
        Mahalanobis = np.stack(Mahalanobis, axis=1)
        return Mahalanobis

    def score(self, data, layer_index):
        data = data.cuda()
        data.requires_grad = True

        out_features = self.model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = []
        for i in range(self.num_classes):
            batch_sample_mean = self.sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[layer_index]),
                                       zero_f.t()).diag()
            gaussian_score.append(term_gau.view(-1, 1))
        gaussian_score = torch.cat(gaussian_score, 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = self.sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(self.precision[layer_index])),
                                   zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data.data, gradient, alpha=-self.noise_magnitude)

        with torch.no_grad():
            noise_out_features = self.model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0),
                                                     noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = []
        for i in range(self.num_classes):
            batch_sample_mean = self.sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[layer_index]),
                                       zero_f.t()).diag()
            noise_gaussian_score.append(term_gau.view(-1, 1))
        noise_gaussian_score = torch.cat(noise_gaussian_score, 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        return noise_gaussian_score.cpu().numpy()

    def sample_estimator(self, in_train_loader):
        group_lasso = EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_sample_per_class = np.empty(self.num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(self.num_outputs):
            temp_list = []
            for j in range(self.num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in in_train_loader:
            total += data.size(0)
            data = data.cuda()
            with torch.no_grad():
                output, out_features = self.model.feature_list(data)

                # get hidden features
                for i in range(self.num_outputs):
                    out_features[i] = out_features[i].view(out_features[i].size(0),
                                                           out_features[i].size(1), -1)
                    out_features[i] = torch.mean(out_features[i].data, 2)

                # compute the accuracy
                pred = output.data.max(1)[1]
                equal_flag = pred.eq(target.cuda()).cpu()
                correct += equal_flag.sum()

                # construct the sample matrix
                for i in range(data.size(0)):
                    label = target[i]
                    if num_sample_per_class[label] == 0:
                        out_count = 0
                        for out in out_features:
                            list_features[out_count][label] = out[i].view(1, -1)
                            out_count += 1
                    else:
                        out_count = 0
                        for out in out_features:
                            list_features[out_count][label] \
                                = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                            out_count += 1
                    num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in self.feature_list:
            temp_list = torch.Tensor(self.num_classes, int(num_feature)).cuda()
            for j in range(self.num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(self.num_outputs):
            X = 0
            for i in range(self.num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            precision.append(temp_precision)

        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
        self.sample_mean = sample_class_mean
        self.precision = precision


def train_mahalanobis_ood(in_dataset,
                          out_dataset,
                          net_type,
                          weights_path,
                          data_root='./datasets/',
                          cuda_idx=0,
                          magnitude=1e-2):
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

    if in_dataset == 'mnist':
        num_labels = 10
    elif in_dataset == 'cifar10':
        num_labels = 10

    detector = MahalanobisDetector(model=network,
                                   num_classes=num_labels,
                                   noise_magnitude=magnitude,
                                   net_type=net_type)

    in_train_ds, in_test_ds = load_datasets(in_dataset)
    in_train_loader = torch.utils.data.DataLoader(in_train_ds, batch_size=64, shuffle=True)

    detector.train(in_train_loader)

    in_test_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=64, shuffle=True)

    _, out_test_ds = load_datasets(out_dataset)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=64, shuffle=True)
    results = detector.evaluate(in_test_loader, out_test_loader)

    save_path = pathlib.Path(
        'ood', f'mahalanobis_ood_{net_type}_{in_dataset}_{out_dataset}_{magnitude}.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector, save_path)

    res_save_path = pathlib.Path(
        'ood', f'mahalanobis_ood_{net_type}_{in_dataset}_{out_dataset}_{magnitude}_res.pt')
    torch.save(results, res_save_path)

    print(f'Magnitude {magnitude}:')
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


def train_mahalanobis_adv(dataset,
                          net_type,
                          weights_path,
                          adv_attack,
                          data_root='./datasets',
                          cuda_idx=0,
                          magnitude=1e-2):
    from validity.classifiers.resnet import ResNet34
    from validity.classifiers.mnist import MnistClassifier
    from validity.adv_dataset import load_adv_dataset
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

    if dataset == 'mnist':
        num_labels = 10
    elif dataset == 'cifar10':
        num_labels = 10

    clean_data, adv_data, noisy_data = load_adv_dataset(dataset, adv_attack, net_type)

    detector = MahalanobisDetector(model=network,
                                   num_classes=num_labels,
                                   noise_magnitude=magnitude,
                                   net_type=net_type)

    train_ds, _ = load_datasets(dataset)
    in_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=False)

    detector.train(in_train_loader)

    in_test_loader = np_loader(clean_data, True)
    out_test_loader = np_loader(adv_data, False)
    noise_test_loader = np_loader(noisy_data, True)
    results = detector.evaluate(in_test_loader, out_test_loader, noise_test_loader)

    save_path = pathlib.Path(
        'adv', f'mahalanobis_adv_{net_type}_{dataset}_{adv_attack}_{magnitude}.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector, save_path)
    print(f'Magnitude {magnitude}:')
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


def train_multiple_mahalanobis_ood(in_dataset,
                                   out_dataset,
                                   net_type,
                                   weights_path,
                                   data_root='./datasets/',
                                   cuda_idx=0,
                                   latex_print=False):
    magnitudes = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    result_table = [['Magnitude', 'AUC Score', 'FPR at TPR=0.95']]

    best_detector = None
    best_auc = None

    for magnitude in magnitudes:
        detector, res = train_mahalanobis_ood(in_dataset, out_dataset, net_type, weights_path,
                                              data_root, cuda_idx, magnitude)
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
    plt.savefig('ood/mahalanobis_ood_results.png')

    save_path = pathlib.Path('ood',
                             f'mahalanobis_ood_{net_type}_{in_dataset}_{out_dataset}_best.pt')
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


def train_multiple_mahalanobis_adv(dataset,
                                   net_type,
                                   weights_path,
                                   adv_attack,
                                   data_root='./datasets/',
                                   cuda_idx=0,
                                   latex_print=False):
    magnitudes = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0028, 0.005, 0.01]
    result_table = [['Magnitude', 'AUC Score', 'FPR at TPR=0.95']]

    best_detector = None
    best_auc = None

    for magnitude in magnitudes:
        detector, res = train_mahalanobis_adv(dataset,
                                              net_type,
                                              weights_path,
                                              adv_attack,
                                              data_root=data_root,
                                              cuda_idx=cuda_idx,
                                              magnitude=magnitude)
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

    save_path = pathlib.Path('adv',
                             f'mahalanobis_adv_{net_type}_{dataset}_{adv_attack}_best.pt')
    torch.save(best_detector, save_path)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('adv/mahalanobis_adv_results.png')

    if latex_print:
        print(' & '.join(result_table[0]) + '\\\\')
        for row in result_table[1:]:
            magnitude = str(row[0])
            scores = [f'{x:.4f}' for x in row[1:]]
            new_row = [magnitude] + scores
            print(' & '.join(new_row) + '\\\\')
    else:
        print(tabulate(result_table))


def load_mahalanobis_ood(net_type, in_dataset, out_dataset, magnitude):
    save_path = pathlib.Path(
        'ood', f'mahalanobis_ood_{net_type}_{in_dataset}_{out_dataset}_{magnitude}.pt')
    assert save_path.exists(), f'{save_path} does not exist'
    return torch.load(save_path)


def load_mahalanobis_adv(net_type, dataset, adv_attack, magnitude):
    save_path = pathlib.Path(
        'adv', f'mahalanobis_adv_{net_type}_{dataset}_{adv_attack}_{magnitude}.pt')
    assert save_path.exists(), f'{save_path} does not exist'
    return torch.load(save_path)


def load_best_mahalanobis_ood(net_type, in_dataset, out_dataset):
    save_path = pathlib.Path('ood',
                             f'mahalanobis_ood_{net_type}_{in_dataset}_{out_dataset}_best.pt')
    assert save_path.exists(), f'{save_path} does not exist'
    return torch.load(save_path)


def load_best_mahalanobis_adv(net_type, dataset, adv_attack):
    save_path = pathlib.Path('adv',
                             f'mahalanobis_adv_{net_type}_{dataset}_{adv_attack}_best.pt')
    assert save_path.exists(), f'{save_path} does not exist'
    return torch.load(save_path)


if __name__ == '__main__':
    fire.Fire()
