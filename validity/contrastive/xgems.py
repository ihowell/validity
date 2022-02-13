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
from validity.generators.load import load_gen, load_encoded_ds
from validity.util import ZipDataset, get_executor


def xgems(generator,
          classifier,
          x_start,
          y_target,
          z_start=None,
          class_coef=1.0,
          writer=None,
          strategy=None,
          seed=None,
          **kwargs):
    """Performs activation maximization using the generator as an
    approximation of the data manifold.

    Args:
        x_start (tf.Tensor): (1HWC)
        y_target (tf.Tensor): (1)

    """
    x_start = x_start.cuda()
    if z_start is not None:
        if type(z_start) is list:
            zs_init = [z.cuda() for z in z_start]
        else:
            zs_init = z_start.cuda()
    else:
        zs_init = generator.encode(x_start)

    class_coef = torch.tensor(class_coef).cuda()

    if type(zs_init) is list:
        zs = [z.detach().clone().requires_grad_(True) for z in zs_init]
    else:
        zs = zs_init.detach().clone().requires_grad_(True)
    x_reencode_start = generator.decode(zs)
    #initial_log_p = initial_log_p.clone().detach()
    criterion = nn.CrossEntropyLoss()
    y_start = torch.argmax(classifier(x_reencode_start), 1)

    optim_lr = 1e-2
    weight_decay = 1e-5
    if type(zs) is list:
        optimizer = optim.Adam(zs[0:1], lr=optim_lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD([zs], lr=optim_lr)  #, weight_decay=weight_decay)

    for step in range(2000):
        optimizer.zero_grad()
        x = generator.decode(zs)
        logits = classifier(x)
        decode_loss = torch.mean((x_start - x)**2, (1, 2, 3))
        class_loss = criterion(logits, y_target.cuda())
        loss = decode_loss + class_coef * class_loss
        # loss = class_loss + 1e-4 * torch.relu(initial_log_p - log_p)
        if writer:
            # writer.add_scalar('xgem/log_p', log_p, step)
            writer.add_scalar('xgem/loss', loss.mean(), step)
            writer.add_scalar('xgem/class loss', class_loss.mean(), step)
            writer.add_scalar('xgem/decode loss', decode_loss.mean(), step)
            # writer.add_scalar('xgem/classification', torch.argmax(logits, dim=1)[0], step)
            writer.add_scalar('xgem/logit orig', logits[0, y_start[0].item()], step)
            writer.add_scalar('xgem/logit probe', logits[0, y_target[0].item()], step)

            sorted_logits = logits.sort(dim=1)[0]
            marginal = sorted_logits[:, -1] - sorted_logits[:, -2]
            writer.add_scalar('xgem/marginal', marginal[0], step)

            img = torch.cat([x_start, x], 3)
            writer.add_images('xgem/xgem', torch.tensor(img), step)

        loss.sum().backward()
        if writer:
            writer.add_scalar('xgem/zs grad max', zs.grad.max(), step)
        optimizer.step()

    return generator.decode(zs)


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
    x_hat = xgems(generator,
                  classifier,
                  data,
                  target_label,
                  class_coef=class_coef,
                  writer=writer)


if __name__ == '__main__':
    fire.Fire()
