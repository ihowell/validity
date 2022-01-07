# Follows nocedal2004numerical, page 515 for the implementation of the
# augmented lagrangian method
import inspect
import time
import sys
import json
import time
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
            inner_iters=10000,
            outer_iters=10000,
            tb_writer=None,
            strategy=None,
            seed=None,
            del_x_threshold=1e-1,
            del_x_patience=10,
            **kwargs):
    """Performs activation maximization using the generator as an
    approximation of the data manifold.

    Args:
        x_start (tf.Tensor): (1CHW)
        y_target (tf.Tensor): ()

    """
    n = x_start.size(0)
    x_start = x_start.cuda()
    if z_start is not None:
        z_0 = z_start.cuda()
    else:
        z_0 = generator.encode(x_start).clone().detach()

    z_res = [None] * n

    z = z_0.clone().detach().requires_grad_()
    del_z_0 = (x_start - generator.decode(z_0)).clone().detach()
    y_true = classifier(x_start).argmax(-1)
    y_probe = y_probe.cuda()

    active_indices = torch.arange(n).cuda()

    c = torch.tensor([1.] * n).cuda()
    lam = torch.tensor([1.] * n).cuda()
    mu_1 = torch.ones((n, num_classes - 2)).cuda()
    mu_2 = torch.ones((n, num_classes - 2)).cuda()
    beta = 1.01
    gamma = 0.24

    y_prime_idx = torch.tensor(
        [[i for i in range(num_classes) if i not in [y_true[j], y_probe[j]]]
         for j in range(n)]).cuda()

    def h(z, y_1, y_2):
        img = generator.decode(z)
        logits = classifier(img)

        vals = []
        for i in range(y_2.size(0)):
            vals.append(
                (logits[i].index_select(0, y_1[i]) - logits[i].index_select(0, y_2[i])))
        vals = torch.stack(vals)
        return vals

    steps_under_threshold = [0] * n
    for i in range(outer_iters):
        old_z = z.clone().detach()
        early_stopping = EarlyStopping(patience=100, verbose=False)
        optimizer = optim.SGD([z], lr=1e-3)
        for j in range(inner_iters):
            optimizer.zero_grad()

            obj = (z - z_0).norm(dim=-1)
            probe = lam * h(z, y_true, y_probe).squeeze(-1) + c / 2 * h(
                z, y_true, y_probe).norm(p=2, dim=-1)

            vals = mu_1 + c.unsqueeze(-1) * h(z, y_true, y_prime_idx)
            constraint_1 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                            mu_1.square())
            constraint_1 = 1 / (2 * c) * constraint_1.sum(dim=1)

            vals = mu_2 + c.unsqueeze(-1) * h(z, y_probe, y_prime_idx)
            constraint_2 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                            mu_2.square())
            constraint_2 = 1 / (2 * c) * constraint_2.sum(dim=1)

            loss = obj + probe + constraint_1 + constraint_2
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            if early_stopping(loss.mean()):
                break

        lam = (lam + c * h(z, y_true, y_probe).squeeze(-1)).detach()
        vals = mu_1 + c.unsqueeze(-1) * h(z, y_true, y_prime_idx)
        mu_1 = torch.maximum(torch.zeros_like(vals).cuda(), vals).detach()
        vals = mu_2 + c.unsqueeze(-1) * h(z, y_probe, y_prime_idx)
        mu_2 = torch.maximum(torch.zeros_like(vals).cuda(), vals).detach()
        c = c * torch.where(
            h(z, y_true, y_probe) > gamma * h(old_z, y_true, y_probe), beta, 1.).squeeze(-1)

        del_x = generator.decode(z) - generator.decode(old_z)
        del_x_norm = del_x.norm(p=2, dim=(1, 2, 3))

        logits = classifier(generator.decode(z))

        if writer:
            writer.add_scalar('cdeepex/loss', loss, i)
            writer.add_scalar('cdeepex/obj', obj.mean(), i)
            writer.add_scalar('cdeepex/probe', probe.mean(), i)
            writer.add_scalar('cdeepex/constraint_1', constraint_1.mean(), i)
            writer.add_scalar('cdeepex/constraint_2', constraint_2.mean(), i)
            writer.add_scalar('cdeepex/lam', lam.mean(), i)
            writer.add_scalar('cdeepex/mu_1', mu_1.mean(), i)
            writer.add_scalar('cdeepex/mu_2', mu_2.mean(), i)
            writer.add_scalar('cdeepex/penalty', c.mean(), i)
            writer.add_scalar('cdeepex/del x', del_x_norm.mean(), i)
            writer.add_scalar('logits/h', h(z, y_true, y_probe).mean(), i)
            writer.add_scalar('logits/logit orig', logits[0].index_select(0, y_true[0]), i)
            writer.add_scalar('logits/logit target', logits[0].index_select(0, y_probe[0]), i)
            writer.add_scalar('logits/max y prime', h(z, y_probe, y_prime_idx).max(), i)

            img = torch.cat([x_start, generator.decode(z_0), generator.decode(z)], 3)
            writer.add_images('cdeepex/example', torch.tensor(img), i)

        print(f'step: {i} inner_steps: {j} min loss: {float(loss.mean()):.4f} '
              f'del x: {float(del_x_norm.mean()):.4f}')

        for i in reversed(range(del_x_norm.size(0))):
            if del_x_norm[i] < del_x_threshold:
                steps_under_threshold[i] += 1
                if steps_under_threshold[i] >= del_x_patience:
                    print(f'Storing {i}')
                    # Store result
                    z_res[int(active_indices[i])] = z[i].clone().detach()

                    z = torch.cat([z[:i], z[i + 1:]]).detach()
                    active_indices = torch.cat([active_indices[:i],
                                                active_indices[i + 1:]]).detach()
                    z_0 = torch.cat([z_0[:i], z_0[i + 1:]]).detach()
                    del_z_0 = torch.cat([del_z_0[:i], del_z_0[i + 1:]]).detach()
                    y_true = torch.cat([y_true[:i], y_true[i + 1:]]).detach()
                    y_probe = torch.cat([y_probe[:i], y_probe[i + 1:]]).detach()
                    y_prime_idx = torch.cat([y_prime_idx[:i], y_prime_idx[i + 1:]]).detach()
                    lam = torch.cat([lam[:i], lam[i + 1:]]).detach()
                    mu_1 = torch.cat([mu_1[:i], mu_1[i + 1:]]).detach()
                    mu_2 = torch.cat([mu_2[:i], mu_2[i + 1:]]).detach()
                    c = torch.cat([c[:i], c[i + 1:]]).detach()
                    x_start = torch.cat([x_start[:i], x_start[i + 1:]]).detach()
            else:
                steps_under_threshold[i] = 0

        if active_indices.size(0) == 0:
            break

    z = torch.stack(z_res)

    return generator.decode(z)


def run_cdeepex(dataset,
                classifier_net_type,
                classifier_weights_path,
                generator_net_type,
                generator_weights_path,
                data_root='./datasets/',
                cuda_idx=0,
                seed=0):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, test_ds = load_datasets(dataset)
    #in_test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=True)

    if dataset == 'mnist':
        num_classes = 10

    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier.eval()

    generator = load_gen(generator_net_type, generator_weights_path)
    generator.eval()

    # data, label = next(iter(in_test_loader))

    data, label = zip(*[test_ds[i] for i in range(128)])
    data = torch.stack(data)
    label = torch.tensor(label)
    target_label = (label + 1) % 10

    print('label', label)
    print('target label', target_label)

    z_start = generator.encode(data.cuda()).clone().detach()
    start = time.time()
    writer = SummaryWriter()
    x_hat = cdeepex(generator,
                    classifier,
                    data,
                    target_label,
                    num_classes,
                    writer,
                    z_start=z_start)
    # for i in range(4):
    #     writer = SummaryWriter()
    #     x_hat = cdeepex(generator,
    #                     classifier,
    #                     data[i].unsqueeze(0),
    #                     target_label[i].unsqueeze(0),
    #                     num_classes,
    #                     writer,
    #                     z_start=z_start[i].unsqueeze(0))
    finish = time.time()
    print(f'Completion time {finish-start:.1f} sec')
    img = x_hat.cpu().detach()
    img = torch.cat([data, img], 3)[0].numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    fire.Fire(run_cdeepex)
