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
from validity.generators.load import load_gen, load_encoded_ds
from validity.util import EarlyStopping

IMPROVE_EPS = 1e-3


def cdeepex(generator,
            classifier,
            x_start,
            y_probe,
            num_classes,
            writer=None,
            z_start=None,
            inner_iters=100000,
            inner_patience=100,
            outer_iters=10000,
            tb_writer=None,
            strategy=None,
            seed=None,
            del_x_threshold=1e-1,
            del_x_patience=5,
            min_c=1e3,
            max_c=1e4,
            lr=1e-5,
            grad_eps=1.,
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
        z_0 = generator.encode(x_start).detach().clone()

    z_res = [None] * n

    z = z_0.detach().clone().requires_grad_()
    best_z = z.detach().clone()
    del_z_0 = (x_start - generator.decode(z_0)).detach().clone()
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

    def h(logits, y_1, y_2):
        vals = []
        for i in range(y_2.size(0)):
            vals.append((logits[i, y_1[i]] - logits[i, y_2[i]]))
        vals = torch.stack(vals)
        return vals

    def loss_fn(z, logits=None, writer=None):
        if logits is None:
            img = generator.decode(z)
            logits = classifier(img)

        obj = (z - z_0).square().sum(dim=-1)
        h_true_probe = h(logits, y_true, y_probe)
        eq_constraint = lam * h_true_probe.squeeze(-1) + c / 2 * h_true_probe.square().sum(
            dim=-1)

        vals = mu_1 + c.unsqueeze(-1) * h(logits, y_true, y_prime_idx)
        ineq_constraint_1 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                             mu_1.square())
        ineq_constraint_1 = 1 / (2 * c) * ineq_constraint_1

        vals = mu_2 + c.unsqueeze(-1) * h(logits, y_probe, y_prime_idx)
        ineq_constraint_2 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                             mu_2.square())
        ineq_constraint_2 = 1 / (2 * c) * ineq_constraint_2

        loss = obj + eq_constraint + ineq_constraint_1.sum(dim=1) + ineq_constraint_2.sum(
            dim=1)

        if writer:
            if type(writer) is list:
                for idx in range(n):
                    step = i
                    writer[idx].add_scalar('cdeepex_inner/loss', loss[idx], step)
                    writer[idx].add_scalar('cdeepex_inner/obj', obj[idx], step)
                    writer[idx].add_scalar('cdeepex_inner/eq_constraint', eq_constraint[idx],
                                           step)
                    writer[idx].add_scalar('cdeepex_inner/ineq_constraint_1_mean',
                                           ineq_constraint_1[idx].mean(), step)
                    writer[idx].add_scalar('cdeepex_inner/ineq_constraint_2_mean',
                                           ineq_constraint_2[idx].mean(), step)
                    for j in range(ineq_constraint_1[idx].size(0)):
                        writer[idx].add_scalar(f'ineq_constraint_1/{j}',
                                               ineq_constraint_1[idx, j], step)
                        writer[idx].add_scalar(f'ineq_constraint_2/{j}',
                                               ineq_constraint_2[idx, j], step)
            else:
                writer.add_scalar('cdeepex_inner/loss', loss.mean(), i)
                writer.add_scalar('cdeepex_inner/obj', obj.mean(), i)
                writer.add_scalar('cdeepex_inner/probe', probe.mean(), i)
                writer.add_scalar('cdeepex_inner/constraint_1', constraint_1.mean(), i)
                writer.add_scalar('cdeepex_inner/constraint_2', constraint_2.mean(), i)

                writer.add_scalar('logits/h', h(logits, y_true, y_probe).mean(), i)
                writer.add_scalar('logits/logit orig', logits[0, y_true[0]], i)
                writer.add_scalar('logits/logit target', logits[0, y_probe[0]], i)
                writer.add_scalar('logits/max y prime',
                                  h(logits, y_probe, y_prime_idx).max(), i)
                img = torch.cat([generator.decode(z_0), generator.decode(z)], 3)
                writer.add_images('cdeepex/example', torch.tensor(img), i)

        return loss, eq_constraint, ineq_constraint_1, ineq_constraint_2

    optimizer = optim.SGD([z], lr=lr)
    steps_under_threshold = torch.zeros(n).cuda()
    old_z = z.detach().clone()
    best_loss = None
    inner_steps = torch.zeros(n).cuda()
    outer_steps = torch.zeros(n).cuda()
    steps_since_best_loss = torch.zeros(n).cuda()

    for i in tqdm(range(outer_iters * inner_iters)):
        optimizer.zero_grad()

        loss, _, _, _ = loss_fn(z, writer=writer)
        loss.sum().backward()
        grad_norm = z.grad.norm(p=2)
        writer[0].add_scalar('cdeepex_inner/grad_norm', grad_norm, i)
        optimizer.step()

        x_temp = generator.decode(z)
        logits = classifier(x_temp)

        if writer:
            if type(writer) is list:
                tmp_img = torch.cat([x_start, x_temp], 3)
                for idx in range(n):
                    step = i
                    writer[idx].add_scalar('cdeepex_inner/h',
                                           h(logits, y_true, y_probe)[idx], step)
                    writer[idx].add_scalar('logits/logit orig', logits[idx, y_true[idx]], step)
                    writer[idx].add_scalar('logits/logit target', logits[idx, y_probe[idx]],
                                           step)
                    writer[idx].add_scalar(
                        'logits/marginal',
                        logits[idx, y_probe[idx]] - logits[idx, y_true[idx]], step)
                    writer[idx].add_scalar(
                        'logits/orig marginal',
                        (logits[idx, y_true[idx]] - logits[idx, y_prime_idx].max()), step)
                    writer[idx].add_scalar(
                        'logits/probe marginal',
                        (logits[idx, y_probe[idx]] - logits[idx, y_prime_idx].max()), step)
                    writer[idx].add_image('cdeepex/example', torch.tensor(tmp_img[idx]), step)

        if best_loss is None:
            best_loss = loss.detach()
        else:
            improved_loss = torch.where(best_loss > 0., (1 - IMPROVE_EPS) * best_loss,
                                        best_loss - IMPROVE_EPS)
            steps_since_best_loss = torch.where(loss < improved_loss,
                                                torch.tensor(0.).cuda(),
                                                steps_since_best_loss + 1)
            best_z = torch.where(loss < improved_loss, z, best_z).detach()
            best_loss = torch.where(loss < improved_loss, loss, best_loss).detach()

        updates = torch.logical_or(steps_since_best_loss >= inner_patience,
                                   inner_steps >= inner_iters)

        # updates = grad_norm < grad_eps

        inner_steps += 1
        if not updates.any():
            continue

        update_idx = torch.where(updates)[0]

        # Update z to be the best
        with torch.no_grad():
            for j, idx in enumerate(update_idx):
                z[idx] = best_z[idx].detach().clone()

            # Get only necessary subsets of tensors to update outer
            # iteration parameters
            s_z = z[update_idx]
            s_img = generator.decode(s_z)
            s_logits = classifier(s_img)
            s_old_z = old_z[update_idx]
            s_old_img = generator.decode(s_old_z)
            s_old_logits = classifier(s_old_img)
            s_lam = lam[update_idx]
            s_c = c[update_idx]
            s_y_true = y_true[update_idx]
            s_y_probe = y_probe[update_idx]
            s_y_prime_idx = y_prime_idx[update_idx]
            s_mu_1 = mu_1[update_idx]
            s_mu_2 = mu_2[update_idx]
            s_c = c[update_idx]

            old_loss, _, _, _ = loss_fn(s_old_z)
            assert (best_loss <= old_loss).all()

            # Update outer iteration parameters.
            # dimitri1999nonlinear, Section 4.2.2
            n_lam = s_lam + s_c * h(s_logits, s_y_true, s_y_probe).squeeze(-1)
            vals = s_mu_1 + s_c.unsqueeze(-1) * h(s_logits, s_y_true, s_y_prime_idx)
            n_mu_1 = torch.maximum(torch.zeros_like(vals).cuda(), vals)
            vals = s_mu_2 + s_c.unsqueeze(-1) * h(s_logits, s_y_probe, s_y_prime_idx)
            n_mu_2 = torch.maximum(torch.zeros_like(vals).cuda(), vals)
            n_c = s_c * torch.where(
                h(s_logits, s_y_true, s_y_probe).norm(p=2, dim=-1) > gamma *
                h(s_old_logits, s_y_true, s_y_probe).norm(p=2, dim=-1), beta, 1.).squeeze(-1)
            n_c = torch.minimum(n_c, torch.ones_like(n_c).cuda() * max_c)

            del_x = x_temp - generator.decode(old_z)
            del_x_norm = del_x.norm(p=2, dim=(1, 2, 3))

            for j, idx in enumerate(update_idx):
                # Update multipliers
                lam[idx] = n_lam[j].detach()
                mu_1[idx] = n_mu_1[j].detach()
                mu_2[idx] = n_mu_2[j].detach()
                c[idx] = n_c[j].detach()

            update_h = h(s_logits, s_y_true, s_y_probe)
            n_loss, eq_constraint, ineq_constraint_1, ineq_constraint_2 = loss_fn(
                s_z, logits=s_logits)
            for j, idx in enumerate(update_idx):
                outer_steps[idx] += 1
                inner_steps[idx] = 0
                steps_since_best_loss[idx] = 0
                old_z[idx] = z[idx].detach().clone()
                best_loss[idx] = n_loss[j].detach()

                valid_sat = update_h[j] < 0.
                eq_con_sat = lam[idx].abs() < 0.1 and 0. < eq_constraint[idx] < 0.1
                ineq_con_1_sat = (mu_1[idx] == 0.).all() and (ineq_constraint_1[idx]
                                                              == 0.).all()
                ineq_con_2_sat = (mu_1[idx] == 0.).all() and (ineq_constraint_1[idx]
                                                              == 0.).all()

                if valid_sat and eq_con_sat and ineq_con_1_sat and ineq_con_2_sat:
                    # print(f'feasible lam: {lam[idx]} {i=} {outer_steps[idx]=}')
                    img = torch.cat([x_start, x_temp], 3)
                    writer[idx].add_image('cdeepex/feasible', img[idx], outer_steps[idx])

            if writer:
                img = torch.cat([x_start, x_temp], 3)
                if type(writer) is list:
                    for j, idx in enumerate(update_idx):
                        step = outer_steps[idx]
                        writer[idx].add_scalar('cdeepex/lam', lam[idx], step)
                        for k in range(mu_1.size(1)):
                            writer[idx].add_scalar(f'mu_1/{k}', mu_1[idx, k], step)
                            writer[idx].add_scalar(f'mu_2/{k}', mu_2[idx, k], step)
                        writer[idx].add_scalar('cdeepex/penalty', c[idx], step)
                        writer[idx].add_scalar('cdeepex/del x', del_x_norm[idx], step)
                        writer[idx].add_scalar('cdeepex/n_loss', n_loss[idx], step)
                else:
                    writer.add_scalar('cdeepex/lam', lam.mean(), step)
                    writer.add_scalar('cdeepex/mu_1', mu_1.mean(), step)
                    writer.add_scalar('cdeepex/mu_2', mu_2.mean(), step)
                    writer.add_scalar('cdeepex/penalty', c.mean(), step)
                    writer.add_scalar('cdeepex/del x', del_x_norm.mean(), step)

            # print(f'step: {i} min loss: {float(loss.mean()):.4f} '
            #       f'del x: {float(del_x_norm.mean()):.4f} '
            #       f'updates: {list(update_idx.cpu().detach().numpy())}')

            # Calculate outer steps under threshold for exiting optimization
            updated_steps = torch.where(
                torch.logical_and(del_x_norm < del_x_threshold, updates),
                steps_under_threshold + 1,
                torch.tensor(0.).cuda())
            steps_under_threshold = torch.where(updates, updated_steps, steps_under_threshold)

            idx_to_remove = torch.where(
                torch.logical_and(
                    torch.logical_or(steps_under_threshold >= del_x_patience,
                                     outer_steps >= outer_iters), c >= min_c))[0]
            #torch.logical_and(
            #    constraint_1.mean(dim=1) == 0.,
            #    constraint_2.mean(dim=1) == 0.))))[0]

            if idx_to_remove.size(0) == 0:
                continue

            for i in idx_to_remove:
                print(f'Storing {int(active_indices[i])}')
                z_res[int(active_indices[i])] = z[i].detach().clone()

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

            if type(writer) is list:
                new_writer = [writer[int(idx)] for idx in idx_to_keep]
                writer = new_writer

            optimizer = optim.SGD([z], lr=lr)

    z = torch.stack(z_res)

    return generator.decode(z)


def run_cdeepex(dataset,
                classifier_net_type,
                classifier_weights_path,
                generator_net_type,
                generator_weights_path,
                batch_size=1,
                data_root='./datasets/',
                lr=1e-5,
                cuda_idx=0,
                seed=0,
                all_labels=False,
                id=None,
                target_label=None,
                encode_dir=None):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, test_ds = load_datasets(dataset)
    encoded_test_ds = load_encoded_ds(dataset, generator_net_type, encode_dir=encode_dir)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    encode_loader = torch.utils.data.DataLoader(encoded_test_ds,
                                                batch_size=batch_size,
                                                shuffle=False)

    if dataset == 'mnist':
        num_classes = 10

    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier.eval()

    generator = load_gen(generator_net_type, generator_weights_path, dataset)
    generator.eval()

    data, label = next(iter(loader))
    # z_start, _ = next(iter(encode_loader))
    z_start = generator.encode(data.cuda()).clone().detach()

    print('label', label)
    if all_labels:
        pred_labels = classifier(data.cuda()).argmax(-1)
        data = data.repeat_interleave(num_classes - 1, dim=0)
        z_start = z_start.repeat_interleave(num_classes - 1, dim=0)
        target_label = [[i for i in range(num_classes) if i != j] for j in pred_labels]
        target_label = torch.tensor(target_label).reshape((-1, ))
    elif target_label is not None:
        target_label = torch.tensor([target_label])
    else:
        target_label = (label + 1) % 10

    writer_paths = [f'runs/cdeepex_{i}' for i in range(target_label.size(0))]
    if id:
        writer_paths = [f'{p}_{id}' for p in writer_paths]
    writers = [SummaryWriter(p) for p in writer_paths]
    start = time.time()
    print('Starting cdeepex')
    x_hat = cdeepex(generator,
                    classifier,
                    data,
                    target_label,
                    num_classes,
                    writer=writers,
                    lr=lr,
                    z_start=z_start)
    finish = time.time()
    print(f'Completion time {finish-start:.1f} sec')
    # print(f'{target_label=}')
    pred_labels = classifier(x_hat).argmax(dim=-1)
    # print(f'{pred_labels=}')

    img = x_hat.cpu().detach()
    img = torch.cat([data, img], 3)[0].numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    fire.Fire(run_cdeepex)
