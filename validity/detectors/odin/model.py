import torch
import torch.nn as nn
import numpy as np
import fire
from tqdm import tqdm

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.linear_model import LogisticRegressionCV


class ODINDetector:
    def __init__(self, network, noise_magnitude, temper):
        self.sample_mean = None
        self.precision = None
        self.lr = None

        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        self.noise_magnitude = noise_magnitude
        self.temper = temper

    def forward(self, inputs):
        score = self.mahalanobis_score(inputs)
        return self.lr.predict_proba(score)

    def train(self, in_loader, out_loader):
        mahalanobis_in = []
        for data, _ in tqdm(in_loader, desc='in loader'):
            mahalanobis_in.append(self.mahalanobis_score(data))
        mahalanobis_in = np.concatenate(mahalanobis_in)

        mahalanobis_out = []
        for data, _ in tqdm(out_loader, desc='out loader'):
            mahalanobis_out.append(self.mahalanobis_score(data))
        mahalanobis_out = np.concatenate(mahalanobis_out)

        # Resample to smaller set to even sets
        min_size = min(mahalanobis_in.shape[0], mahalanobis_out.shape[0])
        mahalanobis_in = mahalanobis_in[:min_size]
        mahalanobis_out = mahalanobis_out[:min_size]

        scores = np.concatenate([mahalanobis_in, mahalanobis_out])
        labels = np.concatenate([
            np.zeros(mahalanobis_in.shape[0]),
            np.ones(mahalanobis_out.shape[0])
        ])

        self.lr = LogisticRegressionCV(n_jobs=-1).fit(scores, labels)

    def mahalanobis_score(self, images):
        inputs = Variable(images, requires_grad=True)

        # Currently assumes batch_size=1
        # assert inputs.shape[0] == 1

        outputs = self.network(inputs)

        # Using temperature scaling
        outputs = outputs / self.temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = torch.argmax(outputs, 1)
        labels = Variable(torch.LongTensor(maxIndexTemp))

        loss = self.criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if gradient.shape[1] == 3:
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
            gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
            gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)

        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -self.noise_magnitude, gradient)
        outputs = self.network(Variable(tempInputs))
        outputs = outputs / self.temper
        # Calculating the confidence after adding perturbations
        # nnOutputs = outputs.data.cpu()
        nnOutputs = outputs
        nnOutputs = nnOutputs - torch.max(nnOutputs, 1, keepdim=True)[0]
        nnOutputs = nn.Softmax(1)(nnOutputs)
        return nnOutputs

def main(location):
    from validity.classifiers.resnet import ResNet34
    network = ResNet34(10)
    weights = torch.load(location, map_location=torch.device('cpu'))
    network.load_state_dict(weights)
    odin = ODINDetector(network, 1e-2, 1000.)

    in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    in_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./', train=True, download=True,
            transform=in_transform),
        batch_size=64, shuffle=True)
    out_train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root='./', split='train', download=True,
            transform=in_transform),
        batch_size=64, shuffle=True)

    odin.train(in_train_loader, out_train_loader)



if __name__ == '__main__':
    fire.Fire(main)
