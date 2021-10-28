import torch.nn as nn
import numpy as np

from sklearn.linear_model import LogisticRegressionCV


class ODINDetector:
    def __init__(self, sample_mean, precision, network, noise_magnitude,
                 temper):
        self.sample_mean = sample_mean
        self.precision = precision
        self.network = network
        self.lr = None
        self.criterion = nn.CrossEntropyLoss()
        self.noise_magnitude = noise_magnitude
        self.temper = temper

    def forward(inputs):
        score = self.mahalanobis_score(inputs)
        return self.lr.predict_proba(score)

    def train(in_inputs, out_inputs):
        mahalanobis_in = self.mahalanobis_score(in_inputs)
        mahalanobis_out = self.mahalanobis_score(out_inputs)

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

    def mahalanobis_score(inputs):
        # Currently assumes batch_size=1
        assert inputs.shape[0] == 1

        outputs = network(inputs)

        # Using temperature scaling
        outputs = outputs / self.temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs, 1)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
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
        tempInputs = torch.add(inputs.data, -self.noiseMagnitude, gradient)
        outputs = self.network(Variable(tempInputs))
        outputs = outputs / self.temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.amax(nnOutputs, 1)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), 1)
        return nnOutputs


if __name__ == '__main__':
    from validity.classifiers.resnet import ResNet34
    network = ResNet34(10)
    weights = torch.load(
        '/mnt/d/research/classifiers/cifar10/resnet_cifar10.pth')
    network.load_state_dict(weights)
