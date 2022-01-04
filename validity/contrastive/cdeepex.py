# Follows nocedal2004numerical, page 515 for the implementation of the
# augmented lagrangian method
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
from tensorboardX import SummaryWriter

from validity.classifiers import load_cls
from validity.datasets import load_datasets
from validity.generators.load import load_gen
from validity.util import EarlyStopping


def cdeepex(generator,
            classifier,
            x_start,
            y_probe,
            num_classes,
            writer=None,
            z_start=None,
            inner_iters=100,
            outer_iters=10000,
            tb_writer=None,
            strategy=None,
            seed=None,
            **kwargs):
    """Performs activation maximization using the generator as an
    approximation of the data manifold.

    Args:
        x_start (tf.Tensor): (1CHW)
        y_target (tf.Tensor): ()

    """
    x_start = x_start.cuda()
    if z_start is not None:
        z_0 = z_start.cuda()
    else:
        z_0 = generator.encode(x_start).clone().detach()

    z = z_0.clone().detach().requires_grad_()
    del_z_0 = (x_start - generator.decode(z_0)).clone().detach()
    y_true = classifier(x_start).argmax(-1)
    y_probe = y_probe.cuda()

    c = 1.
    lam = torch.tensor([1.]).cuda()
    mu_1 = torch.tensor([1.]).cuda()
    mu_2 = torch.tensor([1.]).cuda()
    beta = 1.01
    gamma = 0.24
    del_x_threshold = 1e-1
    del_x_patience = 300

    print(f'{y_true=}')
    print(f'{y_probe=}')
    y_prime_idx = torch.tensor([i for i in range(num_classes)
                                if i not in [y_true, y_probe]]).cuda()

    def h(z, y_1, y_2):
        img = generator.decode(z)
        logits = classifier(img)
        return (logits.index_select(1, y_1) - logits.index_select(1, y_2)).squeeze(-1)

    steps_under_threshold = 0
    for i in range(outer_iters):
        old_z = z.clone().detach()
        early_stopping = EarlyStopping(patience=10000, verbose=False)
        optimizer = optim.SGD([z], lr=1e-4)
        for j in range(inner_iters):
            optimizer.zero_grad()

            # loss = loss_c(z, lam, mu_1, mu_2, c)
            obj = (z - z_0).norm(dim=-1)
            probe = lam * h(z, y_true, y_probe) + c / 2 * h(z, y_true, y_probe).norm(dim=-1)

            vals = mu_1 + c * h(z, y_true, y_prime_idx)
            constraint_1 = 1 / (2 * c) * (torch.maximum(torch.zeros_like(vals).cuda(),
                                                        vals).square() - mu_1.square()).sum()
            vals = mu_2 + c * h(z, y_probe, y_prime_idx)
            constraint_2 = 1 / (2 * c) * (torch.maximum(torch.zeros_like(vals).cuda(),
                                                        vals).square() - mu_2.square()).sum()

            loss = obj + probe + constraint_1 + constraint_2

            loss.backward()
            optimizer.step()
            if early_stopping(loss.mean()):
                break

        lam = (lam + c * h(z, y_true, y_probe)).detach()
        vals = mu_1 + c * h(z, y_true, y_prime_idx)
        mu_1 = torch.maximum(torch.zeros_like(vals).cuda(), vals).detach()
        vals = mu_2 + c * h(z, y_probe, y_prime_idx)
        mu_2 = torch.maximum(torch.zeros_like(vals).cuda(), vals).detach()

        del_x = generator.decode(z) - generator.decode(old_z)
        del_x_norm = del_x.norm()

        if h(z, y_true, y_probe) > gamma * h(old_z, y_true, y_probe):
            c = c * beta

        logits = classifier(generator.decode(z))

        if writer:
            writer.add_scalar('cdeepex/loss', loss, i)
            writer.add_scalar('cdeepex/obj', obj, i)
            writer.add_scalar('cdeepex/probe', probe, i)
            writer.add_scalar('cdeepex/constraint_1', constraint_1, i)
            writer.add_scalar('cdeepex/constraint_2', constraint_2, i)
            writer.add_scalar('cdeepex/lam', lam.mean(), i)
            writer.add_scalar('cdeepex/mu_1', mu_1.mean(), i)
            writer.add_scalar('cdeepex/mu_2', mu_2.mean(), i)
            writer.add_scalar('cdeepex/penalty', c, i)
            writer.add_scalar('cdeepex/del x', del_x_norm, i)
            writer.add_scalar('logits/h', h(z, y_true, y_probe), i)
            writer.add_scalar('logits/logit orig', logits.index_select(1, y_true), i)
            writer.add_scalar('logits/logit target', logits.index_select(1, y_probe), i)
            writer.add_scalar('logits/max y prime', h(z, y_probe, y_prime_idx).max(), i)

            img = torch.cat([x_start, generator.decode(z_0), generator.decode(z)], 3)[0]
            writer.add_image('cdeepex/example', torch.tensor(img), i)

        print(f'Min loss: {loss}')

        if del_x_norm < del_x_threshold:
            steps_under_threshold += 1
            if steps_under_threshold >= del_x_patience:
                break
        else:
            steps_under_threshold = 0

    return generator.decode(z)


def run_cdeepex(dataset,
                classifier_net_type,
                classifier_weights_path,
                generator_net_type,
                generator_weights_path,
                data_root='./datasets/',
                cuda_idx=0,
                seed=1):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, test_ds = load_datasets(dataset)
    in_test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)

    if dataset == 'mnist':
        num_classes = 10

    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier.eval()

    generator = load_gen(generator_net_type, generator_weights_path)
    generator.eval()

    data, label = next(iter(in_test_loader))
    target_label = (label + 1) % 10

    print('label', label)
    print('target label', target_label)

    writer = SummaryWriter()
    x_hat = cdeepex(generator, classifier, data, target_label, num_classes, writer)
    img = x_hat.cpu().detach()
    img = torch.cat([data, img], 3)[0].numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    fire.Fire(run_cdeepex)
