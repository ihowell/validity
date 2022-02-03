# Follows nocedal2004numerical, page 515 for the implementation of the
# augmented lagrangian method
import inspect
import time
import sys
import json
import math
from pathlib import Path
from typing import Generator

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

IMPROVE_EPS = 1e-4
EQ_EPS = 1e-1


def cdeepex(generator,
            classifier,
            x_init,
            y_probe_init,
            num_classes,
            batch_size=None,
            writer=None,
            z_init=None,
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
            lr_end=None,
            grad_eps=1.,
            **kwargs):
    """Performs activation maximization using the generator as an
    approximation of the data manifold.

    Args:
        x_init (tf.Tensor): (1CHW)
        y_target (tf.Tensor): ()

    """
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
    z_0 = torch.stack(z_start).cuda()

    n = x_start.size(0)

    z = z_0.detach().clone().requires_grad_()
    best_z = z.detach().clone()
    del_z_0 = (x_start - generator.decode(z_0)).detach().clone()
    y_true = classifier(x_start).argmax(-1)

    assert torch.where(y_true == y_probe)[0].size(0) == 0

    if lr_end is not None:
        ln_lr_start = torch.log(torch.tensor(lr))
        ln_lr_end = torch.log(torch.tensor(lr_end))

    c = torch.ones(n).cuda()
    lam = torch.ones(n).cuda()
    mu_1 = torch.ones((n, num_classes - 2)).cuda()
    mu_2 = torch.ones((n, num_classes - 2)).cuda()
    beta = 1.01
    gamma = 0.24

    y_prime_idx = [[i for i in range(num_classes) if i not in [y_true[j], y_probe[j]]]
                   for j in range(n)]
    y_prime_idx = torch.tensor(y_prime_idx).cuda()

    steps_under_threshold = torch.zeros(n).cuda()
    old_z = z.detach().clone()
    inner_steps = torch.zeros(n).cuda()
    outer_steps = torch.zeros(n).cuda()
    steps_since_best_loss = torch.zeros(n).cuda()

    img = generator.decode(z)
    logits = classifier(img)
    loss, _, _, _ = loss_fn(z, z_0, logits, y_true, y_probe, y_prime_idx, lam, mu_1, mu_2, c)
    best_loss = loss.detach().clone()

    def inf_gen():
        i = 0
        while True:
            yield i
            i += 1

    perf_time = time.time()
    perf_iter = 0
    p_bar = tqdm(total=N)
    for i in inf_gen():
        curr_time = time.time()
        if curr_time - perf_time > 1.0:
            p_bar.set_description(
                f'Itr {i} at {float(i - perf_iter) / (curr_time - perf_time):.2f} it/s')
            perf_time = curr_time
            perf_iter = i

        if z.grad is not None:
            z.grad.detach_()
            z.grad.zero_()

        img = generator.decode(z)
        logits = classifier(img)
        loss, _, _, _ = loss_fn(z,
                                z_0,
                                logits,
                                y_true,
                                y_probe,
                                y_prime_idx,
                                lam,
                                mu_1,
                                mu_2,
                                c,
                                writer=writer,
                                active_indices=active_indices,
                                step=i)
        loss.sum().backward()
        grad_norm = z.grad.norm(p=2, dim=1)

        if lr_end is not None:
            used_lr = torch.exp(ln_lr_start + (ln_lr_end - ln_lr_start) * (c - 1) /
                                (max_c - 1))
        else:
            used_lr = torch.tensor(lr).cuda()

        with torch.no_grad():
            z.add_(-z.grad * used_lr.unsqueeze(-1))

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
                    for j, logit in enumerate(logits[idx]):
                        writer[idx].add_scalar(f'logits/{j}', logit, step)

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
                    writer[idx].add_scalar('cdeepex_inner/grad_norm', grad_norm[idx], step)

        improved_loss = torch.where(best_loss > 0., (1 - IMPROVE_EPS) * best_loss,
                                    best_loss - IMPROVE_EPS)
        steps_since_best_loss = torch.where(loss < improved_loss,
                                            torch.tensor(0.).cuda(), steps_since_best_loss + 1)

        best_z = torch.where((loss < improved_loss).unsqueeze(-1), z, best_z).detach()
        best_loss = torch.where(loss < improved_loss, loss, best_loss).detach()

        updates = steps_since_best_loss >= inner_patience

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
            s_z_0 = z_0[update_idx]
            s_old_z = old_z[update_idx]
            s_lam = lam[update_idx]
            s_c = c[update_idx]
            s_y_true = y_true[update_idx]
            s_y_probe = y_probe[update_idx]
            s_y_prime_idx = y_prime_idx[update_idx]
            s_mu_1 = mu_1[update_idx]
            s_mu_2 = mu_2[update_idx]

            s_img = generator.decode(s_z)
            s_logits = classifier(s_img)
            s_old_img = generator.decode(s_old_z)
            s_old_logits = classifier(s_old_img)

            old_loss, _, _, _ = loss_fn(s_old_z, s_z_0, s_old_logits, s_y_true, s_y_probe,
                                        s_y_prime_idx, s_lam, s_mu_1, s_mu_2, s_c)
            # assert torch.logical_or(best_loss[update_idx] <= old_loss,
            #                         torch.isclose(best_loss[update_idx], old_loss)).all()

            # Update outer iteration parameters.
            # dimitri1999nonlinear, Section 4.2.2
            n_lam = s_lam + s_c * h(s_logits, s_y_true, s_y_probe).squeeze(-1)
            vals = s_mu_1 + s_c.unsqueeze(-1) * h(s_logits, s_y_prime_idx, s_y_true)
            n_mu_1 = torch.maximum(torch.zeros_like(vals).cuda(), vals)
            vals = s_mu_2 + s_c.unsqueeze(-1) * h(s_logits, s_y_prime_idx, s_y_probe)
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
                s_z, s_z_0, s_logits, s_y_true, s_y_probe, s_y_prime_idx, s_lam, s_mu_1,
                s_mu_2, s_c)
            for j, idx in enumerate(update_idx):
                outer_steps[idx] += 1
                inner_steps[idx] = 0
                steps_since_best_loss[idx] = 0
                old_z[idx] = z[idx].detach().clone()
                best_loss[idx] = n_loss[j].detach()

            eq_con_sat = h(s_logits, s_y_true, s_y_probe) < EQ_EPS
            ineq_con_1_sat = h(s_logits, s_y_true, s_y_prime_idx) >= 0.
            ineq_con_2_sat = h(s_logits, s_y_probe, s_y_prime_idx) >= 0.
            feasible = torch.logical_and(
                eq_con_sat,
                torch.logical_and(ineq_con_1_sat.all(dim=-1), ineq_con_2_sat.all(dim=-1)))

            is_feasible = torch.zeros(n).bool().cuda()

            for j, f_idx in enumerate(torch.where(feasible)[0]):
                # print(f'feasible lam: {lam[idx]} {i=} {outer_steps[idx]=}')
                idx = update_idx[f_idx]
                is_feasible[idx] = True
                img = torch.cat([x_start, x_temp], 3)
                active_idx = active_indices[idx]
                if writer and type(writer) is list:
                    writer[active_idx].add_image('cdeepex/feasible', img[idx],
                                                 outer_steps[idx])
                z_res[int(active_indices[idx])] = s_z[j].detach().clone()

            if writer:
                img = torch.cat([x_start, x_temp], 3)
                if type(writer) is list:
                    for j, idx in enumerate(update_idx):
                        step = outer_steps[idx]
                        active_idx = active_indices[idx]
                        writer[active_idx].add_scalar('cdeepex/lam', lam[idx], step)
                        writer[active_idx].add_scalar('cdeepex/penalty', c[idx], step)
                        writer[active_idx].add_scalar('cdeepex/del x', del_x_norm[idx], step)
                        writer[active_idx].add_scalar('cdeepex/n_loss', n_loss[j], step)
                        writer[active_idx].add_scalar('cdeepex_inner/lam', lam[idx], i)
                        writer[active_idx].add_scalar('cdeepex_inner/penalty', c[idx], i)
                        writer[active_idx].add_scalar('cdeepex_inner/del x', del_x_norm[idx],
                                                      i)
                        writer[active_idx].add_scalar('cdeepex_inner/n_loss', n_loss[j], i)
                        for k in range(mu_1.size(1)):
                            writer[active_idx].add_scalar(f'mu_1/{k}', mu_1[idx, k], step)
                            writer[active_idx].add_scalar(f'mu_2/{k}', mu_2[idx, k], step)

            # Calculate outer steps under threshold for exiting optimization
            updated_steps = torch.where(
                torch.logical_and(del_x_norm < del_x_threshold, updates),
                steps_under_threshold + 1,
                torch.tensor(0.).cuda())
            steps_under_threshold = torch.where(updates, updated_steps, steps_under_threshold)

            to_remove = torch.logical_and(
                torch.logical_or(steps_under_threshold >= del_x_patience,
                                 outer_steps >= outer_iters), c >= min_c)

            idx_to_remove = torch.where(torch.logical_or(is_feasible, to_remove))[0]

            if idx_to_remove.size(0) == 0:
                continue

            p_bar.update(idx_to_remove.size(0))

            for i in idx_to_remove:
                print(f'Storing {int(active_indices[i])}')
                if z_res[int(active_indices[i])] is None:
                    z_res[int(active_indices[i])] = z[i].detach().clone()

            idx_to_keep = torch.tensor([i for i in range(n) if i not in idx_to_remove]).cuda()

            data_to_add = [data for _, data in zip(range(idx_to_remove.size(0)), data_gen_itr)]
            n_add = len(data_to_add)

            if idx_to_keep.size(0) + n_add == 0:
                break

            if n_add > 0:
                idx_to_add, x_start_to_add, y_probe_to_add, z_0_to_add = zip(*data_to_add)

                idx_to_add = torch.tensor(idx_to_add).cuda()
                x_start_to_add = torch.stack(x_start_to_add).cuda()
                y_probe_to_add = torch.stack(y_probe_to_add).cuda()
                z_0_to_add = torch.stack(z_0_to_add).cuda()

                del_z_0_to_add = (x_start_to_add -
                                  generator.decode(z_0_to_add)).detach().clone()
                logits_to_add = classifier(x_start_to_add)
                y_true_to_add = logits_to_add.argmax(-1)
                y_prime_idx_to_add = [[
                    i for i in range(num_classes)
                    if i not in [y_true_to_add[j], y_probe_to_add[j]]
                ] for j in range(n_add)]
                y_prime_idx_to_add = torch.tensor(y_prime_idx_to_add).cuda()
                loss_to_add, _, _, _ = loss_fn(z_0_to_add, z_0_to_add, logits_to_add,
                                               y_true_to_add, y_probe_to_add,
                                               y_prime_idx_to_add,
                                               torch.ones(n_add).cuda(),
                                               torch.ones((n_add, num_classes - 2)).cuda(),
                                               torch.ones((n_add, num_classes - 2)).cuda(),
                                               torch.ones(n_add).cuda())

                # yapf: disable
                z              = torch.cat([z[idx_to_keep],              z_0_to_add]).detach_().clone().requires_grad_()
                z_0            = torch.cat([z_0[idx_to_keep],            z_0_to_add]).detach_().clone()
                best_z         = torch.cat([best_z[idx_to_keep],         z_0_to_add]).detach_().clone()
                old_z          = torch.cat([old_z[idx_to_keep],          z_0_to_add]).detach_().clone()
                del_z_0        = torch.cat([del_z_0[idx_to_keep],        del_z_0_to_add]).detach_()
                active_indices = torch.cat([active_indices[idx_to_keep], idx_to_add]).detach_()
                x_start        = torch.cat([x_start[idx_to_keep],        x_start_to_add]).detach_()
                y_true         = torch.cat([y_true[idx_to_keep],         y_true_to_add]).detach_()
                y_probe        = torch.cat([y_probe[idx_to_keep],        y_probe_to_add]).detach_()
                y_prime_idx    = torch.cat([y_prime_idx[idx_to_keep],    y_prime_idx_to_add]).detach_()
                best_loss      = torch.cat([best_loss[idx_to_keep],      loss_to_add]).detach_()

                lam                   = torch.cat([lam[idx_to_keep],                   torch.ones(n_add).cuda()]).detach_()
                mu_1                  = torch.cat([mu_1[idx_to_keep],                  torch.ones((n_add, num_classes - 2)).cuda()]).detach_()
                mu_2                  = torch.cat([mu_2[idx_to_keep],                  torch.ones((n_add, num_classes - 2)).cuda()]).detach_()
                c                     = torch.cat([c[idx_to_keep],                     torch.ones(n_add).cuda()]).detach_()
                steps_under_threshold = torch.cat([steps_under_threshold[idx_to_keep], torch.zeros(n_add).cuda()]).detach_()
                inner_steps           = torch.cat([outer_steps[idx_to_keep],           torch.zeros(n_add).cuda()]).detach_()
                outer_steps           = torch.cat([outer_steps[idx_to_keep],           torch.zeros(n_add).cuda()]).detach_()
                steps_since_best_loss = torch.cat([steps_since_best_loss[idx_to_keep], torch.zeros(n_add).cuda()]).detach_()
                # yapf: enable
            else:
                # yapf: disable
                z              = z[idx_to_keep].detach_().clone().requires_grad_()
                z_0            = z_0[idx_to_keep].detach_().clone()
                best_z         = best_z[idx_to_keep].detach_().clone()
                old_z          = old_z[idx_to_keep].detach_().clone()
                del_z_0        = del_z_0[idx_to_keep].detach_()
                active_indices = active_indices[idx_to_keep].detach_()
                x_start        = x_start[idx_to_keep].detach_()
                y_true         = y_true[idx_to_keep].detach_()
                y_probe        = y_probe[idx_to_keep].detach_()
                y_prime_idx    = y_prime_idx[idx_to_keep].detach_()
                best_loss      = best_loss[idx_to_keep].detach_()

                lam                   = lam[idx_to_keep].detach_()
                mu_1                  = mu_1[idx_to_keep].detach_()
                mu_2                  = mu_2[idx_to_keep].detach_()
                c                     = c[idx_to_keep].detach_()
                steps_under_threshold = steps_under_threshold[idx_to_keep].detach_()
                inner_steps           = outer_steps[idx_to_keep].detach_()
                outer_steps           = outer_steps[idx_to_keep].detach_()
                steps_since_best_loss = steps_since_best_loss[idx_to_keep].detach_()
                # yapf: enable

            n = z.size(0)

    res = torch.stack(z_res)
    decoded = []
    for i in range(math.ceil(res.size(0) / batch_size)):
        decoded.append(generator.decode(res[i * batch_size:(i + 1) * batch_size].cuda()))
    return torch.cat(decoded)


def h(logits, y_1, y_2):
    vals = []
    for i in range(y_2.size(0)):
        vals.append(logits[i, y_1[i]] - logits[i, y_2[i]])
    vals = torch.stack(vals)
    return vals


def loss_fn(z,
            z_0,
            logits,
            y_true,
            y_probe,
            y_prime_idx,
            lam,
            mu_1,
            mu_2,
            c,
            writer=None,
            active_indices=None,
            step=None):
    obj = (z - z_0).square().sum(dim=-1)
    h_true_probe = h(logits, y_true, y_probe)
    eq_constraint = lam * h_true_probe.squeeze(-1) + c / 2 * h_true_probe.square()

    h_prime_true = h(logits, y_prime_idx, y_true)

    vals = mu_1 + c.unsqueeze(-1) * h_prime_true
    ineq_constraint_1 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                         mu_1.square())
    ineq_constraint_1 = 1 / (2 * c.unsqueeze(-1)) * ineq_constraint_1

    h_prime_probe = h(logits, y_prime_idx, y_probe)
    vals = mu_2 + c.unsqueeze(-1) * h_prime_probe
    ineq_constraint_2 = (torch.maximum(torch.zeros_like(vals).cuda(), vals).square() -
                         mu_2.square())
    ineq_constraint_2 = 1 / (2 * c.unsqueeze(-1)) * ineq_constraint_2

    loss = obj + eq_constraint + ineq_constraint_1.sum(dim=1) + ineq_constraint_2.sum(dim=1)

    if writer:
        if type(writer) is list:
            for i in range(loss.size(0)):
                idx = active_indices[i]
                writer[idx].add_scalar('cdeepex_inner/loss', loss[i], step)
                writer[idx].add_scalar('cdeepex_inner/obj', obj[i], step)
                writer[idx].add_scalar('cdeepex_inner/eq_constraint', eq_constraint[i], step)
                writer[idx].add_scalar('cdeepex_inner/ineq_constraint_1_mean',
                                       ineq_constraint_1[i].mean(), step)
                writer[idx].add_scalar('cdeepex_inner/ineq_constraint_2_mean',
                                       ineq_constraint_2[i].mean(), step)
                for j in range(ineq_constraint_1[i].size(0)):
                    writer[idx].add_scalar(f'ineq_constraint_1/{j}', ineq_constraint_1[i, j],
                                           step)
                    writer[idx].add_scalar(f'ineq_constraint_2/{j}', ineq_constraint_2[i, j],
                                           step)
                    writer[idx].add_scalar(f'ineq_constraint_1/h_{j}', h_prime_true[i, j],
                                           step)
                    writer[idx].add_scalar(f'ineq_constraint_2/h_{j}', h_prime_probe[i, j],
                                           step)

    return loss, eq_constraint, ineq_constraint_1, ineq_constraint_2


def run_cdeepex(dataset,
                classifier_net_type,
                classifier_weights_path,
                generator_net_type,
                generator_weights_path,
                total_data=1,
                data_root='./datasets/',
                lr=1e-5,
                cuda_idx=0,
                seed=0,
                all_labels=False,
                id=None,
                target_label=None,
                encode_dir=None,
                log_tensorboard=True,
                **kwargs):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, test_ds = load_datasets(dataset)
    encoded_test_ds = load_encoded_ds(dataset, generator_net_type, encode_dir=encode_dir)

    if dataset == 'mnist':
        num_classes = 10

    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier.eval()

    generator = load_gen(generator_net_type, generator_weights_path, dataset)
    generator.eval()

    data = [data for _, data in zip(range(total_data), test_ds)]
    data, label = zip(*data)
    data = torch.stack(data)
    label = torch.tensor(label)
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

    writers = None
    if log_tensorboard:
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
                    z_start=z_start,
                    **kwargs)
    finish = time.time()
    print(f'Completion time {finish-start:.1f} sec')
    # print(f'{target_label=}')
    pred_labels = classifier(x_hat).argmax(dim=-1)
    print(f'{pred_labels=}')

    img = x_hat.cpu().detach()
    img = torch.cat([data, img], 3)
    img = make_grid(img, nrow=3).numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    fire.Fire(run_cdeepex)
