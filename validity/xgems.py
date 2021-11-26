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
import numpy as np
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from validity.generators.nvae.model import load_nvae
from validity.classifiers import ResNet34


def xgems(encode,
          decode,
          classifier,
          x_start,
          y_target,
          class_coef,
          writer,
          tb_writer=None,
          strategy=None,
          seed=None,
          **kwargs):
    """Performs activation maximization using the generator as an
    approximation of the data manifold.

    Args:
        x_start (tf.Tensor): (1HWC)
        y_target (tf.Tensor): ()

    """
    x_start = x_start.cuda()
    class_coef = torch.tensor(class_coef).cuda()
    zs_init = encode(x_start)
    zs = [z.detach().clone().requires_grad_(True) for z in zs_init]
    x_reencode_start, initial_log_p = decode(zs)
    initial_log_p = initial_log_p.clone().detach()
    criterion = nn.CrossEntropyLoss()
    y_start = torch.argmax(classifier(x_reencode_start), 1)[0]

    # prefix = f'{y_start}_to_{y_target}'

    def loss_fn(z):
        x = decode(z)

        y = classifier(x)
        class_loss = criterion(logits=logits, labels=tf.expand_dims(y_target, 0))

        decode_loss = tf.reduce_mean((x_start - x)**2, (1, 2, 3))

        # if strategy == 'crs_only':
        #     loss = class_loss
        # elif strategy == 'latent_distance':
        #     decode_loss = tf.reduce_mean((z_init - z)**2, range(1, len(z.shape)))
        #     loss = decode_loss + class_coef * class_loss
        # else:
        loss = decode_loss + class_coef * class_loss

        sm = tf.nn.softmax(logits[0])
        diff = sm[y_start] - sm[y_target]

        sorted_logits = logits.sort(dim=1)
        marginal = sorted_logits[-1] - sorted_logits[-2]

        return {
            'class_prediction': diff,
            'class_loss': class_loss,
            'decode_loss': decode_loss,
            'loss': loss,
            'logits': logits,
            'marginal': marginal,
            'path_viz': tf.concat([x_start, x], 2)
        }

    optimizer = optim.Adam(zs[0:1], lr=1e-3)

    for step in range(15000):
        optimizer.zero_grad()
        x, log_p = decode(zs)
        logits = classifier(x)
        decode_loss = torch.mean((x_start - x)**2, (1, 2, 3))
        class_loss = criterion(logits, y_target.cuda())
        # loss = decode_loss + class_coef * class_loss
        loss = class_loss + 1e-4 * torch.relu(initial_log_p - log_p)
        writer.add_scalar('xgem/log_p', log_p, step)
        writer.add_scalar('xgem/loss', loss, step)
        writer.add_scalar('xgem/class loss', class_loss, step)
        writer.add_scalar('xgem/decode loss', decode_loss, step)
        writer.add_scalar('xgem/classification', torch.argmax(logits, dim=1)[0], step)
        sorted_logits = logits.sort(dim=1)[0]
        marginal = sorted_logits[:, -1] - sorted_logits[:, -2]
        writer.add_scalar('xgem/marginal', marginal[0], step)
        x_diff = (x_start - x) * 2 + 0.5
        img = torch.cat([x_start, x, x_diff], 3)[0]
        writer.add_image('xgem/xgem', torch.tensor(img), step)
        loss.backward()
        optimizer.step()

    # z = am(loss_fn, z, tb_writer=tb_writer, prefix=prefix, **kwargs)

    return x_reencode_start, decode(zs)


def run_xgems(dataset, weights_path, generator_checkpoint, class_coef=1.0, data_root='./datasets/', cuda_idx=0):
    torch.cuda.manual_seed(0)
    in_test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=data_root,
                                                                  train=False,
                                                                  download=True,
                                                                  transform=transforms.ToTensor()),
                                                 batch_size=1,
                                                 shuffle=False)
    classifier = ResNet34(10, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    classifier.load_state_dict(torch.load(weights_path, map_location=f'cuda:{cuda_idx}'))
    classifier = classifier.cuda()
    classifier.eval()

    checkpoint = torch.load(generator_checkpoint, map_location='cpu')
    args = checkpoint['args']
    nvae = load_nvae(generator_checkpoint, batch_size=1)
    nvae = nvae.cuda()
    nvae.eval()

    for data, label in in_test_loader:
        break

    target_label = (label + 1) % 10

    def encode(x):
        z, combiner_cells_s = nvae.encode(x)
        return [z] + combiner_cells_s

    def decode(zs):
        z, combiner_cells_s = zs[0], zs[1:]
        logits, log_p = nvae.decode(z, combiner_cells_s, 1.)
        return nvae.decoder_output(logits).sample(), log_p

    print('label', label)
    print('target label', target_label)

    writer = SummaryWriter()
    x, x_hat = xgems(encode, decode, classifier, data, target_label, class_coef, writer)
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
