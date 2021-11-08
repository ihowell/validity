import inspect
import time
import sys
import json
from pathlib import Path

import fire
import torch
import numpy as np


def xgems(encoder,
          decoder,
          classifier,
          x_start,
          y_start,
          y_target,
          class_coef,
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
    encoder = vae_wrapper(encoder)
    decoder = vae_wrapper(decoder)

    z_init = encoder(x_start)
    z = tf.Variable(initial_value=z_init, trainable=True)
    x_reencode_start = decoder(z)

    prefix = f'{y_start}_to_{y_target}'

    def loss_fn(z):
        x = decoder(z)

        y = classifier(x, False)

        logits = y['logits']
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.expand_dims(y_target, 0))

        decode_loss = tf.reduce_mean((x_start - x)**2, (1, 2, 3))

        if strategy == 'crs_only':
            loss = class_loss
        elif strategy == 'latent_distance':
            decode_loss = tf.reduce_mean((z_init - z)**2, range(1, len(z.shape)))
            loss = decode_loss + class_coef * class_loss
        else:
            loss = decode_loss + class_coef * class_loss

        sm = tf.nn.softmax(logits[0])
        diff = sm[y_start] - sm[y_target]

        except_target = tf.concat([sm[:y_target], sm[y_target + 1:]], 0)
        marginal = tf.reduce_max(except_target) - sm[y_target]

        return {
            'class_prediction': diff,
            'class_loss': class_loss,
            'decode_loss': decode_loss,
            'loss': loss,
            'logits': logits,
            'marginal': marginal,
            'path_viz': tf.concat([x_start, x], 2)
        }

    optimizer = optim.Adam(z)

    for step in range(200):
        optimizer.zero_grad()
        x = decoder(z)
        output = model(x)
        loss = decode_loss + class_coef * class_loss
        loss.backward()
        optimizer.step()

    # z = am(loss_fn, z, tb_writer=tb_writer, prefix=prefix, **kwargs)

    return x_reencode_start, decoder(z)


def run_xgems(dataset,
              classifier_dir,
              generator_dir,
              output_dir,
              batch_size=128,
              class_coef=1.0,
              strategy=None,
              seed=None,
              **kwargs):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    args = locals()
    args.update(kwargs)
    del args['kwargs']
    print(args)
    with open('args.json', 'w') as out_file:
        json.dump(args, out_file)

    # Initialize the variables in the network
    ds_dict, info = get_dataset(dataset, batch_size, image_size=(32, 32), seed=seed)

    tb_writer = tf.summary.create_file_writer(output_dir)

    # Initialize variables in models
    for batch in ds_dict['train']:
        classifier = load_classifier(classifier_dir, batch)
        generator = load_generator(generator_dir, batch)
        break

    sample_per_class = {}
    for i in range(info.features['label'].num_classes):
        sample_per_class[i] = None

    for batch in ds_dict['test']:
        for img, label in zip(batch['image'], batch['label']):
            if sample_per_class[label.numpy()] is None:
                sample_per_class[label.numpy()] = img
                if all([x is not None for x in sample_per_class.values()]):
                    done_processing = True
                    break
        if done_processing:
            break

    for y_start, x_orig in sample_per_class.items():
        for y_target in sample_per_class:
            x_start = tf.expand_dims(x_orig, 0)

            x_reencode_start, img = xgems(generator.encode,
                                          generator.decode,
                                          classifier,
                                          x_start,
                                          y_start,
                                          y_target,
                                          class_coef=class_coef,
                                          tb_writer=tb_writer,
                                          strategy=strategy,
                                          seed=seed,
                                          **kwargs)

            tf.print('Original class:', y_start)
            tf.print('Target class:', y_target)
            tf.print('Final class:', tf.argmax(classifier(img)['logits'], axis=1))

            img = tf.concat([x_reencode_start, img], axis=2)
            img = tf.cast(img * 255, tf.uint8)
            png_tensor = tf.io.encode_png(img[0])

            img_path = f'{int(y_start)}_to_{int(y_target)}.png'
            if seed is not None:
                img_path = f'{int(y_start)}_to_{int(y_target)}_seed_{seed}.png'
            img_path = Path(output_dir, 'images', img_path)
            img_path.parent.mkdir(exist_ok=True, parents=True)
            with open(img_path, 'wb') as png:
                png.write(png_tensor.numpy())


if __name__ == '__main__':
    fire.Fire(run_xgems)
