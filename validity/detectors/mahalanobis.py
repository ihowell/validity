import pathlib
from pkgutil import get_data
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fire
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate
from torch.autograd import Variable

from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from validity.adv_dataset import load_adv_datasets
from validity.classifiers.load import load_cls
from validity.datasets import load_datasets, load_detector_datasets, get_dataset_info
from validity.util import np_loader


class MahalanobisDetector(nn.Module):

    @classmethod
    def load(cls, saved_dict):
        detector = cls(**saved_dict['args'])
        detector.load_state_dict(saved_dict['state_dict'])
        return detector

    def __init__(self,
                 classifier_path=None,
                 num_classes=None,
                 noise_magnitude=None,
                 sample_mean=None,
                 precision=None,
                 lr=None,
                 sc=None):
        """
        The following are used only when loading the classifier. Do not set yourself.
        - `sample_mean`
        - `precision`
        - `lr`
        - `sc`
        """
        super().__init__()
        self.classifier_path = classifier_path
        self.num_classes = num_classes
        self.noise_magnitude = noise_magnitude

        self.sample_mean = sample_mean
        self.precision = precision
        self.lr = lr
        self.sc = sc

        # Do not need to be saved to disk
        self.classifier = load_cls(self.classifier_path)
        self.classifier.cuda()
        self.criterion = nn.CrossEntropyLoss()

        self.feature_list = None
        self.num_outputs = None

    def get_save_dict(self):
        return {
            'args': {
                'classifier_path': self.classifier_path,
                'num_classes': self.num_classes,
                'noise_magnitude': self.noise_magnitude,
                'sample_mean': self.sample_mean,
                'precision': self.precision,
                'lr': self.lr,
                'sc': self.sc,
            },
            'state_dict': self.state_dict(),
        }

    def predict(self, inputs):
        assert self.lr is not None and self.sc is not None
        inputs = inputs.type(torch.float)
        Mahalanobis = []
        if self.num_outputs is None:
            self.set_num_outputs(inputs)
        for layer_idx in range(self.num_outputs):
            Mahalanobis.append(self.score(inputs, layer_idx))
        Mahalanobis = np.stack(Mahalanobis, axis=1)
        scaled = self.sc.transform(Mahalanobis)
        return -self.lr.predict(scaled) + 1.

    def predict_proba(self, inputs):
        assert self.lr is not None
        inputs = inputs.type(torch.float)
        Mahalanobis = []
        if self.num_outputs is None:
            self.set_num_outputs(inputs)
        for layer_idx in range(self.num_outputs):
            Mahalanobis.append(self.score(inputs, layer_idx))
        Mahalanobis = np.stack(Mahalanobis, axis=1)
        scaled = self.sc.transform(Mahalanobis)
        return -self.lr.predict_proba(scaled) + 1.

    def set_num_outputs(self, data):
        temp_list = self.classifier.feature_list(data.cuda())[1]
        self.num_outputs = len(temp_list)
        self.feature_list = np.array([t.shape[1] for t in temp_list])

    def validate(self, in_loader, out_loader, noise_loader=None):
        self.classifier.eval()

        if self.num_outputs is None:
            self.set_num_outputs(next(iter(in_loader))[0])

        # Compute Mahalanobis scores for in and out distributions
        print('Computing Mahalanobis distances')
        print('Scoring in dataset')
        Mahalanobis_in = self.score_for_loader(in_loader)
        print('Scoring out of dataset')
        Mahalanobis_out = self.score_for_loader(out_loader)
        if noise_loader:
            print('Scoring noise')
            Mahalanobis_noise = self.score_for_loader(noise_loader)

        Mahalanobis_train = np.concatenate([Mahalanobis_in, Mahalanobis_out])
        labels = np.concatenate(
            [np.ones(Mahalanobis_in.shape[0]),
             np.zeros(Mahalanobis_out.shape[0])])
        if noise_loader:
            Mahalanobis_train = np.concatenate([Mahalanobis_train, Mahalanobis_noise])
            labels = np.concatenate([labels, np.ones(Mahalanobis_noise.shape[0])])

        self.sc = StandardScaler()
        scaled_data_val = self.sc.fit_transform(Mahalanobis_train)

        # Train regressor
        print('Training regressor')
        self.lr = LogisticRegressionCV(n_jobs=-1).fit(scaled_data_val, labels)

    def evaluate(self, in_test_loader, out_test_loader, noise_test_loader=None):
        print('Evaluating regressor')
        Mahalanobis_in = self.score_for_loader(in_test_loader)
        Mahalanobis_out = self.score_for_loader(out_test_loader)
        if noise_test_loader:
            Mahalanobis_noise = self.score_for_loader(noise_test_loader)

        Mahalanobis_test = np.concatenate([Mahalanobis_in, Mahalanobis_out])
        labels_test = np.concatenate(
            [np.ones(Mahalanobis_in.shape[0]),
             np.zeros(Mahalanobis_out.shape[0])])
        if noise_test_loader:
            Mahalanobis_test = np.concatenate([Mahalanobis_test, Mahalanobis_noise])
            labels_test = np.concatenate([labels_test, np.ones(Mahalanobis_noise.shape[0])])

        scaled_data_test = self.sc.transform(Mahalanobis_test)

        res = {}
        test_probs = self.lr.predict_proba(scaled_data_test)[:, 1]
        preds = self.lr.predict(scaled_data_test)

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
        self.classifier.eval()
        if self.num_outputs is None:
            self.set_num_outputs(next(iter(data_loader))[0])

        Mahalanobis = []
        for layer_idx in range(self.num_outputs):
            M = np.concatenate([self.score(data, layer_idx) for data, _ in data_loader])
            Mahalanobis.append(M)
        Mahalanobis = np.stack(Mahalanobis, axis=1)
        return Mahalanobis

    def score(self, data, layer_index):
        self.classifier.eval()
        if self.num_outputs is None:
            self.set_num_outputs(data)

        data = data.cuda()
        data.requires_grad = True

        out_features = self.classifier.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        sample_mean = self.sample_mean.to(device=out_features.get_device())
        precision = self.precision.to(device=out_features.get_device())

        # compute Mahalanobis score
        gaussian_score = []
        for i in range(self.num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]),
                                       zero_f.t()).diag()
            gaussian_score.append(term_gau.view(-1, 1))
        gaussian_score = torch.cat(gaussian_score, 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data.data, gradient, alpha=-self.noise_magnitude)

        with torch.no_grad():
            noise_out_features = self.classifier.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0),
                                                     noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = []
        for i in range(self.num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]),
                                       zero_f.t()).diag()
            noise_gaussian_score.append(term_gau.view(-1, 1))
        noise_gaussian_score = torch.cat(noise_gaussian_score, 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        return noise_gaussian_score.cpu().numpy()

    def sample_estimator(self, in_train_loader):
        """
        The estimates of mean and precision are generated using the same dataset that trained the classifier
        """
        print('Sampling training set statistics')
        self.classifier.eval()
        if self.num_outputs is None:
            self.set_num_outputs(next(iter(in_train_loader))[0])

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
                output, out_features = self.classifier.feature_list(data)

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
                          magnitude=1e-2,
                          classifier_id=None):
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    ds_info = get_dataset_info(in_dataset)

    detector = MahalanobisDetector(classifier_path=weights_path,
                                   num_classes=ds_info.num_labels,
                                   noise_magnitude=magnitude)

    cls_train_ds, _, in_val_ds, in_test_ds = load_detector_datasets(in_dataset,
                                                                    data_root=data_root)
    _, _, out_val_ds, out_test_ds = load_detector_datasets(out_dataset)

    cls_train_loader = torch.utils.data.DataLoader(cls_train_ds, batch_size=64, shuffle=True)
    in_val_loader = torch.utils.data.DataLoader(in_val_ds, batch_size=64, shuffle=True)
    out_val_loader = torch.utils.data.DataLoader(out_val_ds, batch_size=64, shuffle=True)
    in_test_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=64, shuffle=True)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=64, shuffle=True)

    detector.sample_estimator(cls_train_loader)
    detector.validate(in_val_loader, out_val_loader)
    results = detector.evaluate(in_test_loader, out_test_loader)

    save_path = get_mahalanobis_ood_path(net_type,
                                         in_dataset,
                                         out_dataset,
                                         magnitude,
                                         classifier_id=classifier_id)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector.get_save_dict(), save_path)

    res_save_path = f'mahalanobis_{net_type}_{in_dataset}_{out_dataset}_{magnitude}'
    if classifier_id:
        res_save_path = f'{res_save_path}_{classifier_id}'
    res_save_path = pathlib.Path('ood') / f'{res_save_path}_res.pt'
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
                          cuda_idx=0,
                          magnitude=1e-2,
                          batch_size=64,
                          classifier_id=None):

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(cuda_idx)

    ds_info = get_dataset_info(dataset)

    data_dict = load_adv_datasets(dataset, adv_attack, net_type, classifier_id=classifier_id)
    clean_val_data, clean_test_data = data_dict['clean']
    adv_val_data, adv_test_data = data_dict['adv']
    noise_val_data, noise_test_data = data_dict['noise']

    detector = MahalanobisDetector(classifier_path=weights_path,
                                   num_classes=ds_info.num_labels,
                                   noise_magnitude=magnitude)

    cls_train_ds, _ = load_datasets(dataset)
    cls_train_loader = torch.utils.data.DataLoader(cls_train_ds,
                                                   batch_size=batch_size,
                                                   shuffle=False)

    in_val_loader = np_loader(clean_val_data, True)
    adv_val_loader = np_loader(adv_val_data, False)
    noise_val_loader = np_loader(noise_val_data, True)
    in_test_loader = np_loader(clean_test_data, True)
    adv_test_loader = np_loader(adv_test_data, False)
    noise_test_loader = np_loader(noise_test_data, True)

    detector.sample_estimator(cls_train_loader)
    detector.validate(in_val_loader, adv_val_loader, noise_val_loader)
    results = detector.evaluate(in_test_loader, adv_test_loader, noise_test_loader)

    save_path = get_mahalanobis_adv_path(net_type,
                                         dataset,
                                         adv_attack,
                                         magnitude,
                                         classifier_id=classifier_id)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector.get_save_dict(), save_path)

    res_save_path = f'mahalanobis_{net_type}_{dataset}_{adv_attack}_{magnitude}'
    if classifier_id:
        res_save_path = f'{res_save_path}_{classifier_id}'
    res_save_path = pathlib.Path('adv') / f'{res_save_path}_res.pt'
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


def train_multiple_mahalanobis_ood(in_dataset,
                                   out_dataset,
                                   net_type,
                                   weights_path,
                                   data_root='./datasets/',
                                   cuda_idx=0,
                                   latex_print=False,
                                   classifier_id=None):
    magnitudes = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    result_table = [['Magnitude', 'Accuracy', 'AUC Score', 'FPR at TPR=0.95']]

    best_detector = None
    best_auc = None

    for magnitude in magnitudes:
        detector, res = train_mahalanobis_ood(in_dataset,
                                              out_dataset,
                                              net_type,
                                              weights_path,
                                              data_root,
                                              cuda_idx,
                                              magnitude,
                                              classifier_id=classifier_id)
        fpr, tpr = res['plot']
        plt.plot(fpr, tpr, label=str(magnitude))
        row = [magnitude, res['accuracy'], res['auc_score'], res['fpr_at_tpr_95']]
        result_table.append(row)

        if best_detector is None:
            best_detector = detector
            best_auc = res['auc_score']
        elif res['auc_score'] > best_auc:
            best_detector = detector
            best_auc = res['auc_score']

    img_path = 'mahalanobis_ood'
    if classifier_id:
        img_path += f'_{classifier_id}'
    img_path = f'ood/{img_path}_results.png'
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(img_path)

    save_path = get_best_mahalanobis_ood_path(net_type,
                                              in_dataset,
                                              out_dataset,
                                              classifier_id=classifier_id)
    torch.save(best_detector.get_save_dict(), save_path)

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
                                   cuda_idx=0,
                                   latex_print=False,
                                   classifier_id=None):
    magnitudes = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0028, 0.005, 0.01]
    result_table = [['Magnitude', 'Accuracy', 'AUC Score', 'FPR at TPR=0.95']]

    best_detector = None
    best_auc = None

    for magnitude in magnitudes:
        detector, res = train_mahalanobis_adv(dataset,
                                              net_type,
                                              weights_path,
                                              adv_attack,
                                              cuda_idx=cuda_idx,
                                              magnitude=magnitude,
                                              classifier_id=classifier_id)
        fpr, tpr = res['plot']
        plt.plot(fpr, tpr, label=str(magnitude))
        row = [magnitude, res['accuracy'], res['auc_score'], res['fpr_at_tpr_95']]
        result_table.append(row)

        if best_detector is None:
            best_detector = detector
            best_auc = res['auc_score']
        elif res['auc_score'] > best_auc:
            best_detector = detector
            best_auc = res['auc_score']

    save_path = get_best_mahalanobis_adv_path(net_type,
                                              dataset,
                                              adv_attack,
                                              classifier_id=classifier_id)
    torch.save(best_detector.get_save_dict(), save_path)

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


def evaluate_best_mahalanobis_ood(net_type, in_dataset, out_dataset, classifier_id=None):
    detector = load_best_mahalanobis_ood(net_type,
                                         in_dataset,
                                         out_dataset,
                                         classifier_id=classifier_id)
    detector.eval()

    _, in_test_ds = load_datasets(in_dataset)
    _, out_test_ds = load_datasets(out_dataset)

    in_test_loader = torch.utils.data.DataLoader(in_test_ds, batch_size=64, shuffle=True)
    out_test_loader = torch.utils.data.DataLoader(out_test_ds, batch_size=64, shuffle=True)

    in_preds = [detector.predict(data) for data, _ in in_test_loader]
    out_preds = [detector.predict(data) for data, _ in out_test_loader]
    in_preds = np.concatenate(in_preds)
    out_preds = np.concatenate(out_preds)

    in_correct = np.where(in_preds == 0., 1., 0.)
    out_correct = np.where(out_preds == 1., 1., 0.)
    correct = np.concatenate([in_correct, out_correct])

    acc = correct.mean()
    print(f'Accuracy: {acc:.4f}')


def get_mahalanobis_ood_path(net_type, in_dataset, out_dataset, magnitude, classifier_id=None):
    save_path = f'mahalanobis_{net_type}_{in_dataset}_{out_dataset}_{magnitude}'
    if classifier_id:
        save_path = f'{save_path}_{classifier_id}'
    return pathlib.Path('ood') / f'{save_path}.pt'


def get_best_mahalanobis_ood_path(net_type, in_dataset, out_dataset, classifier_id=None):
    save_path = f'mahalanobis_{net_type}_{in_dataset}_{out_dataset}'
    if classifier_id:
        save_path = f'{save_path}_{classifier_id}'
    return pathlib.Path('ood') / f'{save_path}_best.pt'


def get_mahalanobis_adv_path(net_type, dataset, adv_attack, magnitude, classifier_id=None):
    save_path = f'mahalanobis_{net_type}_{dataset}_{adv_attack}_{magnitude}'
    if classifier_id:
        save_path = f'{save_path}_{classifier_id}'
    return pathlib.Path('adv') / f'{save_path}.pt'


def get_best_mahalanobis_adv_path(net_type, dataset, adv_attack, classifier_id=None):
    save_path = f'mahalanobis_{net_type}_{dataset}_{adv_attack}'
    if classifier_id:
        save_path = f'{save_path}_{classifier_id}'
    return pathlib.Path('adv') / f'{save_path}_best.pt'


def load_mahalanobis_ood(net_type, in_dataset, out_dataset, magnitude, classifier_id=None):
    save_path = get_mahalanobis_ood_path(net_type,
                                         in_dataset,
                                         out_dataset,
                                         magnitude,
                                         classifier_id=classifier_id)
    if not save_path.exists():
        return False
    save_dict = torch.load(save_path, map_location=torch.device('cpu'))
    return MahalanobisDetector.load(save_dict)


def load_best_mahalanobis_ood(net_type, in_dataset, out_dataset, classifier_id=None):
    save_path = get_best_mahalanobis_ood_path(net_type,
                                              in_dataset,
                                              out_dataset,
                                              classifier_id=classifier_id)
    if not save_path.exists():
        return False
    save_dict = torch.load(save_path, map_location=torch.device('cpu'))
    return MahalanobisDetector.load(save_dict)


def load_mahalanobis_adv(net_type, dataset, adv_attack, magnitude, classifier_id=None):
    save_path = get_mahalanobis_adv_path(net_type,
                                         dataset,
                                         adv_attack,
                                         magnitude,
                                         classifier_id=classifier_id)
    if not save_path.exists():
        return False
    save_dict = torch.load(save_path, map_location=torch.device('cpu'))
    return MahalanobisDetector.load(save_dict)


def load_best_mahalanobis_adv(net_type, dataset, adv_attack, classifier_id=None):
    save_path = get_best_mahalanobis_adv_path(net_type,
                                              dataset,
                                              adv_attack,
                                              classifier_id=classifier_id)
    if not save_path.exists():
        return False
    save_dict = torch.load(save_path, map_location=torch.device('cpu'))
    return MahalanobisDetector.load(save_dict)


if __name__ == '__main__':
    fire.Fire()
