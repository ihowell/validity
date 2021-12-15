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

from validity.classifiers import MnistClassifier, ResNet34
from validity.datasets import load_datasets
from validity.generators.bern_vae import BernVAE
from validity.generators.nvae.model import load_nvae
from validity.util import EarlyStopping


def cdeepex(encode,
            decode,
            classifier,
            x_start,
            y_probe,
            num_classes,
            writer,
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
    y_true = classifier(x_start).argmax(-1)
    y_probe = y_probe.cuda()
    z_0 = encode(x_start).clone().detach()
    z = z_0.clone().detach().requires_grad_()
    del_z_0 = (x_start - decode(z_0)[0]).clone().detach()
    c = 1.
    lam = torch.tensor([1.]).cuda()
    mu_1 = torch.tensor([1.]).cuda()
    mu_2 = torch.tensor([1.]).cuda()
    beta = 1.01
    gamma = 0.24
    del_x_threshold = 1e-2
    del_x_patience = 20

    y_prime_idx = torch.tensor([i for i in range(num_classes)
                                if i not in [y_true, y_probe]]).cuda()

    # def loss_c(z, lam, mu_1, mu_2, c):

    #     return l

    def h(z, y_1, y_2):
        img = decode(z)[0] + del_z_0
        logits = classifier(img)
        return (logits.index_select(1, y_1) - logits.index_select(1, y_2)).squeeze(-1)

    steps_under_threshold = 0
    for i in range(outer_iters):
        old_z = z.clone().detach()
        early_stopping = EarlyStopping(patience=10, verbose=False)
        optimizer = optim.Adam([z], weight_decay=1e-5)
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

        del_x = decode(z)[0] - decode(old_z)[0]
        del_x_norm = del_x.norm()

        if h(z, y_true, y_probe) > gamma * h(old_z, y_true, y_probe):
            c = c * beta

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

        img = torch.cat([decode(z_0)[0], decode(z)[0]], 3)[0]
        writer.add_image('cdeepex/example', torch.tensor(img), i)
        print(f'Min loss: {loss}')
        if del_x_norm < del_x_threshold:
            steps_under_threshold += 1
            if steps_under_threshold >= del_x_patience:
                break
        else:
            steps_under_threshold = 0

    return decode(z_0)[0], decode(z)[0]


def run_xgems(dataset,
              classifier_net_type,
              classifier_weights_path,
              generator_net_type,
              generator_weights_path,
              class_coef=1.0,
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

    if classifier_net_type == 'mnist':
        classifier = MnistClassifier()
        classifier.load_state_dict(torch.load(classifier_weights_path, map_location=f'cuda:0'))
    elif classifier_net_type == 'cifar10':
        classifier = ResNet34(
            10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        classifier.load_state_dict(
            torch.load(classifier_weights_path, map_location=f'cuda:{cuda_idx}'))

    classifier = classifier.cuda()
    classifier.eval()

    if generator_net_type == 'mnist':
        generator = BernVAE()
        generator.load_state_dict(torch.load(generator_weights_path))

        def encode(x):
            return generator.encode(x)

        def decode(z):
            x_hat = generator.decode(z)
            # x_hat_binarized = x_hat.bernoulli()
            # log_p = generator.log_prob(x_hat_binarized)
            return x_hat, None  # log_p

    elif generator_net_type == 'nvae':
        generator = load_nvae(generator_weights_path, batch_size=1)

        def encode(x):
            z, combiner_cells_s = generator.encode(x)
            return [z] + combiner_cells_s

        def decode(zs):
            z, combiner_cells_s = zs[0], zs[1:]
            logits, log_p = generator.decode(z, combiner_cells_s, 1.)
            return generator.decoder_output(logits).sample(), log_p

    generator = generator.cuda()
    generator.eval()

    for data, label in in_test_loader:
        break

    target_label = (label + 1) % 10

    print('label', label)
    print('target label', target_label)

    writer = SummaryWriter()
    x, x_hat = cdeepex(encode, decode, classifier, data, target_label, num_classes, writer)
    # plt.imshow(np.transpose(torch.cat([x, x_hat], 3)[0].cpu().detach().numpy(), (1, 2, 0)))
    # plt.show()

    # sample_per_class = {}
    # for i in range(info.features['label'].num_classes):
    #     sample_per_class[i] = None

    # for batch in ds_dict['test']:
    #     for img, label in zip(batch['image'], batch['label']):
    #         if sample_per_class[label.numpy()] is None:
    #             sample_per_class[label.numpy()] = img
    #             if all([x is not None for x in sample_per_class.values()]):
    #                 done_processing = True
    #                 break
    #     if done_processing:
    #         break

    # for y_start, x_orig in sample_per_class.items():
    #     for y_target in sample_per_class:
    #         x_start = tf.expand_dims(x_orig, 0)

    #         x_reencode_start, img = xgems(generator.encode,
    #                                       generator.decode,
    #                                       classifier,
    #                                       x_start,
    #                                       y_start,
    #                                       y_target,
    #                                       class_coef=class_coef,
    #                                       tb_writer=tb_writer,
    #                                       strategy=strategy,
    #                                       seed=seed,
    #                                       **kwargs)

    #         tf.print('Original class:', y_start)
    #         tf.print('Target class:', y_target)
    #         tf.print('Final class:', tf.argmax(classifier(img)['logits'], axis=1))

    #         img = tf.concat([x_reencode_start, img], axis=2)
    #         img = tf.cast(img * 255, tf.uint8)
    #         png_tensor = tf.io.encode_png(img[0])

    #         img_path = f'{int(y_start)}_to_{int(y_target)}.png'
    #         if seed is not None:
    #             img_path = f'{int(y_start)}_to_{int(y_target)}_seed_{seed}.png'
    #         img_path = Path(output_dir, 'images', img_path)
    #         img_path.parent.mkdir(exist_ok=True, parents=True)
    #         with open(img_path, 'wb') as png:
    #             png.write(png_tensor.numpy())


if __name__ == '__main__':
    fire.Fire(run_xgems)
