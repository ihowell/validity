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
            inner_patience=100,
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

    assert torch.where(y_true == y_probe)[0].size(0) == 0

    active_indices = torch.arange(n).cuda()

    c = torch.tensor([1.] * n).cuda()
    lam = torch.tensor([1.] * n).cuda()
    mu_1 = torch.ones((n, num_classes - 2)).cuda()
    mu_2 = torch.ones((n, num_classes - 2)).cuda()
    beta = 1.01
    gamma = 0.24

    y_prime_idx = [[i for i in range(num_classes) if i not in [y_true[j], y_probe[j]]]
                   for j in range(n)]
    y_prime_idx = torch.tensor(y_prime_idx).cuda()

    def h(z, y_1, y_2):
        img = generator.decode(z)
        logits = classifier(img)

        vals = []
        for i in range(y_2.size(0)):
            vals.append(
                (logits[i].index_select(0, y_1[i]) - logits[i].index_select(0, y_2[i])))
        vals = torch.stack(vals)
        return vals

    optimizer = optim.SGD([z], lr=1e-3)
    steps_under_threshold = torch.zeros(n).cuda()
    old_z = z.clone().detach()
    best_loss = None
    inner_steps = torch.zeros(n).cuda()
    outer_steps = torch.zeros(n).cuda()
    steps_since_best_loss = torch.zeros(n).cuda()

    for i in range(outer_iters * inner_iters):
        optimizer.zero_grad()

        obj = (z - z_0).norm(dim=-1)
        probe = lam * h(z, y_true, y_probe).squeeze(-1) + c / 2 * h(z, y_true, y_probe).norm(
            p=2, dim=-1)

        vals = mu_1 + c.unsqueeze(-1) * h(z, y_true, y_prime_idx)
        constraint_1 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                        mu_1.square())
        constraint_1 = 1 / (2 * c) * constraint_1.sum(dim=1)

        vals = mu_2 + c.unsqueeze(-1) * h(z, y_probe, y_prime_idx)
        constraint_2 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                        mu_2.square())
        constraint_2 = 1 / (2 * c) * constraint_2.sum(dim=1)

        loss = obj + probe + constraint_1 + constraint_2
        loss.mean().backward()
        optimizer.step()

        if best_loss is None:
            best_loss = loss.detach()
        else:
            improved_loss = torch.where(best_loss > 0., 0.995 * best_loss, best_loss - 0.005)
            steps_since_best_loss = torch.where(loss < improved_loss,
                                                torch.tensor(0.).cuda(),
                                                steps_since_best_loss + 1)
            best_loss = torch.where(loss < improved_loss, loss, best_loss)

        # TODO: Fix the bottom test. It should be tracking the total
        # number of inner steps.
        updates = torch.logical_or(steps_since_best_loss >= inner_patience,
                                   inner_steps >= inner_iters)
        inner_steps += 1
        if not updates.any():
            continue

        update_idx = torch.where(updates)[0]

        # Reset counters for updates
        for idx in update_idx:
            outer_steps[idx] += 1
            inner_steps[idx] = 0
            steps_since_best_loss[idx] = 0
            best_loss[idx] = float("Inf")

        # Get only necessary subsets of tensors to update outer
        # iteration parameters
        s_z = z[update_idx]
        s_old_z = old_z[update_idx]
        s_lam = lam[update_idx]
        s_c = c[update_idx]
        s_y_true = y_true[update_idx]
        s_y_probe = y_probe[update_idx]
        s_y_prime_idx = y_prime_idx[update_idx]
        s_mu_1 = mu_1[update_idx]
        s_mu_2 = mu_2[update_idx]
        s_c = c[update_idx]

        # Update outer iteration parameters
        n_lam = s_lam + s_c * h(s_z, s_y_true, s_y_probe).squeeze(-1)
        vals = s_mu_1 + s_c.unsqueeze(-1) * h(s_z, s_y_true, s_y_prime_idx)
        n_mu_1 = torch.maximum(torch.zeros_like(vals).cuda(), vals)
        vals = s_mu_2 + s_c.unsqueeze(-1) * h(s_z, s_y_probe, s_y_prime_idx)
        n_mu_2 = torch.maximum(torch.zeros_like(vals).cuda(), vals)
        n_c = s_c * torch.where(
            h(s_z, s_y_true, s_y_probe) > gamma * h(s_old_z, s_y_true, s_y_probe), beta,
            1.).squeeze(-1)

        del_x = generator.decode(z) - generator.decode(old_z)
        del_x_norm = del_x.norm(p=2, dim=(1, 2, 3))

        for j, idx in enumerate(update_idx):
            old_z[idx] = z[idx].detach()

            lam[idx] = n_lam[j].detach()
            c[idx] = n_c[j].detach()
            mu_1[idx] = n_mu_1[j].detach()
            mu_2[idx] = n_mu_2[j].detach()

        logits = classifier(generator.decode(z))

        if writer:
            writer.add_scalar('cdeepex/loss', loss.mean(), i)
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

        print(f'step: {i} min loss: {float(loss.mean()):.4f} '
              f'del x: {float(del_x_norm.mean()):.4f} '
              f'updates: {list(update_idx.cpu().detach().numpy())}')

        # Calculate outer steps under threshold for exiting optimization
        updated_steps = torch.where(torch.logical_and(del_x_norm < del_x_threshold, updates),
                                    steps_under_threshold + 1,
                                    torch.tensor(0.).cuda())
        steps_under_threshold = torch.where(updates, updated_steps, steps_under_threshold)

        idx_to_remove = torch.where(
            torch.logical_or(steps_under_threshold >= del_x_patience,
                             outer_steps >= outer_iters))[0]

        if idx_to_remove.size(0) == 0:
            continue

        for i in idx_to_remove:
            print(f'Storing {int(active_indices[i])}')
            z_res[int(active_indices[i])] = z[i].clone().detach()

        idx_to_keep = torch.tensor([i for i in range(n) if i not in idx_to_remove]).cuda()
        n = idx_to_keep.size(0)
        if idx_to_keep.size(0) == 0:
            break

        z = z[idx_to_keep].detach()
        active_indices = active_indices[idx_to_keep].detach()
        z_0 = z_0[idx_to_keep].detach()
        del_z_0 = del_z_0[idx_to_keep].detach()
        y_true = y_true[idx_to_keep].detach()
        y_probe = y_probe[idx_to_keep].detach()
        y_prime_idx = y_prime_idx[idx_to_keep].detach()
        lam = lam[idx_to_keep].detach()
        mu_1 = mu_1[idx_to_keep].detach()
        mu_2 = mu_2[idx_to_keep].detach()
        c = c[idx_to_keep].detach()
        x_start = x_start[idx_to_keep].detach()
        steps_under_threshold = steps_under_threshold[idx_to_keep].detach()
        old_z = old_z[idx_to_keep].detach()
        best_loss = best_loss[idx_to_keep].detach()
        inner_steps = outer_steps[idx_to_keep].detach()
        outer_steps = outer_steps[idx_to_keep].detach()
        steps_since_best_loss = steps_since_best_loss[idx_to_keep].detach()

        optimizer = optim.SGD([z], lr=1e-3)

    z = torch.stack(z_res)

    return generator.decode(z)


def run_cdeepex(dataset,
                classifier_net_type,
                classifier_weights_path,
                generator_net_type,
                generator_weights_path,
                batch_size=32,
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

    data, label = zip(*[test_ds[i] for i in range(batch_size)])
    data = torch.stack(data)
    label = torch.tensor(label)
    target_label = (label + 1) % 10

    print('label', label)
    print('target label', target_label)

    z_start = generator.encode(data.cuda()).clone().detach()
    writer = SummaryWriter()
    start = time.time()
    print('Starting cdeepex')
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
