import inspect
import time
import sys
import json
import math
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

from validity.classifiers.load import load_cls
from validity.datasets import load_datasets
from validity.generators.load import load_gen, load_encoded_ds
from validity.util import get_executor

IMPROVE_EPS = 5e-3


def xgems(generator,
          classifier,
          x_init,
          y_probe_init,
          batch_size=None,
          z_init=None,
          class_coef=1.0,
          writer=None,
          strategy=None,
          seed=None,
          max_steps=1e5,
          lr=1e-2,
          stopping_patience=2e3,
          disable_tqdm=True):
    """Performs activation maximization using the generator as an
    approximation of the data manifold.

    Args:
        x_start (tf.Tensor): (NHWC)
        y_target (tf.Tensor): (N)

    """
    max_steps = int(max_steps)
    torch.autograd.set_detect_anomaly(True)

    N = x_init.size(0)
    z_res = [None] * N

    if batch_size is None:
        batch_size = N

    if z_init is None:
        z_init = []
        for i in range(math.ceil(x_init.size(0) / batch_size)):
            z_init.append(generator.encode(x_init[i * batch_size:(i + 1) * batch_size].cuda()))
        z_init = torch.cat(z_init).detach_()
    else:
        z_init = z_init.cuda()

    def data_gen():
        for i in range(x_init.size(0)):
            yield i, x_init[i], y_probe_init[i], z_init[i]

    data_gen_itr = iter(data_gen())
    data = [data for _, data in zip(range(batch_size), data_gen_itr)]
    active_indices, x_start, y_probe, z_start = zip(*data)

    active_indices = torch.tensor(active_indices).cuda()
    x_start = torch.stack(x_start).cuda()
    y_probe = torch.stack(y_probe).cuda()
    z_start = torch.stack(z_start).cuda()

    class_coef = torch.tensor(class_coef).cuda()

    z = z_start.detach().clone().requires_grad_(True)
    x_hat = generator.decode(z)
    y_start = classifier(x_hat).argmax(-1)
    n = x_start.size(0)

    optimizer = optim.SGD([z], lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = torch.ones(z.size(0)).cuda() * float('Inf')
    steps_since_best_loss = torch.zeros(z.size(0)).cuda()

    for step in tqdm(range(max_steps), disable=disable_tqdm):
        optimizer.zero_grad()
        x = generator.decode(z)
        logits = classifier(x)

        decode_loss = torch.mean((x_start - x)**2, (1, 2, 3))
        if logits.size(0) != y_probe.size(0):
            print(f'{z.shape=}')
            print(f'{x.shape=}')
            print(f'{logits.shape=}')
            print(f'{y_probe.shape=}')
        class_loss = criterion(logits, y_probe)

        loss = decode_loss + class_coef * class_loss

        if writer:
            writer.add_scalar('xgem/loss', loss.mean(), step)
            writer.add_scalar('xgem/class loss', class_loss.mean(), step)
            writer.add_scalar('xgem/decode loss', decode_loss.mean(), step)
            writer.add_scalar('xgem/logit orig', logits[0, y_start[0].item()], step)
            writer.add_scalar('xgem/logit probe', logits[0, y_probe[0].item()], step)

            sorted_logits = logits.sort(dim=1)[0]
            marginal = sorted_logits[:, -1] - sorted_logits[:, -2]
            writer.add_scalar('xgem/marginal', marginal[0], step)

            img = torch.cat([x_start, x], 3)
            writer.add_images('xgem/xgem', torch.tensor(img), step)

        loss.sum().backward()
        if writer:
            writer.add_scalar('xgem/z grad max', z.grad.max(), step)
        optimizer.step()

        improved_loss = torch.where(best_loss > 0., (1 - IMPROVE_EPS) * best_loss,
                                    best_loss - IMPROVE_EPS)
        best_loss = torch.where(loss < improved_loss, loss, best_loss).detach()
        steps_since_best_loss = torch.where(loss < improved_loss,
                                            torch.tensor(0.).cuda(), steps_since_best_loss + 1)

        removals = steps_since_best_loss >= stopping_patience

        if not removals.any():
            continue

        removal_idx = torch.where(removals)[0]

        for i in removal_idx:
            print(f'Storing {int(active_indices[i])}')
            if z_res[int(active_indices[i])] is None:
                z_res[int(active_indices[i])] = z[i].detach().clone()

        idx_to_keep = torch.tensor([i for i in range(n) if i not in removal_idx]).cuda()

        if idx_to_keep.size(0) == 0:
            break

        z = z[idx_to_keep].detach_().requires_grad_()
        z_start = z_start[idx_to_keep].detach_()
        x_start = x_start[idx_to_keep].detach_()
        y_start = y_start[idx_to_keep].detach_()
        y_probe = y_probe[idx_to_keep].detach_()
        active_indices = active_indices[idx_to_keep].detach_()
        steps_since_best_loss = steps_since_best_loss[idx_to_keep].detach_()
        best_loss = best_loss[idx_to_keep].detach_()

        n = z.size(0)
        optimizer = optim.SGD([z], lr=lr)

    for i, idx in enumerate(active_indices):
        idx = int(idx)
        if z_res[idx] is None:
            z_res[idx] = z[i].detach().clone()

    z_res = torch.stack(z_res)
    return generator.decode(z_res)


def run_xgems(dataset,
              classifier_net_type,
              classifier_weights_path,
              generator_net_type,
              generator_weights_path,
              class_coef=5.0,
              data_root='./datasets/',
              cuda_idx=0,
              batch_size=1,
              seed=1):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, test_ds = load_datasets(dataset)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    classifier = load_cls(classifier_weights_path)
    classifier.eval()

    generator = load_gen(generator_net_type, generator_weights_path, dataset)
    generator.eval()

    for data, label in loader:
        break
    target_label = (label + 1) % 10

    print('label', label)
    print('target label', target_label)

    writer = SummaryWriter()
    x_hat = xgems(generator,
                  classifier,
                  data,
                  target_label,
                  class_coef=class_coef,
                  writer=writer)


if __name__ == '__main__':
    fire.Fire()
