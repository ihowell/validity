import inspect
import time
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from validity.classifiers import load_cls
from validity.datasets import load_datasets
from validity.generators.load import load_gen, load_encoded_ds
from validity.util import ZipDataset, get_executor, EarlyStopping


def am(generator,
       classifier,
       x_start,
       y_target,
       z_start=None,
       latent_coef=1e-1,
       max_steps=1e6,
       writer=None,
       strategy=None,
       seed=None,
       lr=5e-2,
       stopping_patience=2e3,
       **kwargs):
    """Performs activation maximization using the generator as an
    approximation of the data manifold.

    Args:
        x_start (tf.Tensor): (1HWC)
        y_target (tf.Tensor): (1)

    """
    max_steps = int(max_steps)
    torch.autograd.set_detect_anomaly(True)
    x_start = x_start.cuda()
    if z_start is not None:
        z_start = z_start.cuda()
    else:
        z_start = generator.encode(x_start)

    latent_coef = torch.tensor(latent_coef).cuda()

    z = z_start.detach().clone().requires_grad_(True)
    x_hat = generator.decode(z)
    y_start = classifier(x_hat).argmax(-1)

    optimizer = optim.SGD([z], lr=lr)
    early_stopping = EarlyStopping(patience=stopping_patience)

    for step in range(max_steps):
        optimizer.zero_grad()
        x = generator.decode(z)
        logits = classifier(x)

        latent_distance = (torch.sum((z_start - z)**2, dim=-1) + 1e-10).sqrt()
        class_logits = []
        for i in range(y_target.size(0)):
            class_logits.append(logits[i, y_target[i]])
        class_logits = torch.stack(class_logits)

        loss = -class_logits + latent_coef * latent_distance
        loss = loss.mean()
        loss.backward()

        if writer:
            writer.add_scalar('am/loss', loss.mean(), step)
            writer.add_scalar('am/logit', class_logits.mean(), step)
            writer.add_scalar('am/latent dist', latent_distance.mean(), step)
            writer.add_scalar('am/classification', torch.argmax(logits, dim=1)[0], step)
            writer.add_scalar('am/logit orig', logits[0, y_start[0].item()], step)
            writer.add_scalar('am/logit probe', logits[0, y_target[0].item()], step)

            sorted_logits = logits.sort(dim=1)[0]
            marginal = sorted_logits[:, -1] - sorted_logits[:, -2]
            writer.add_scalar('am/marginal', marginal[0], step)

            img = torch.cat([x_start, x], 3)
            writer.add_images('am/example', torch.tensor(img), step)

        optimizer.step()
        if early_stopping(loss):
            break

    return generator.decode(z)


def run_am(dataset,
           classifier_net_type,
           classifier_weights_path,
           generator_net_type,
           generator_weights_path,
           data_root='./datasets/',
           cuda_idx=0,
           batch_size=1,
           seed=1,
           **kwargs):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, test_ds = load_datasets(dataset)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier.eval()

    generator = load_gen(generator_net_type, generator_weights_path, dataset)
    generator.eval()

    for data, label in loader:
        break
    target_label = (label + 1) % 10

    print('label', label)
    print('target label', target_label)

    writer = SummaryWriter()
    x_hat = am(generator, classifier, data, target_label, writer=writer, **kwargs)

    img = x_hat.cpu().detach()
    img = torch.cat([data, img], 3)
    img = make_grid(img, nrow=3).numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    fire.Fire()
