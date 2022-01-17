import torch

from validity.util import NPZDataset

from .mnist_vae import MnistVAE
from .nvae.model import load_nvae
from .wgan_gp import WGAN_GP


def load_gen(gen_type, weights_path, dataset):
    if dataset == 'mnist':
        num_channels = 1
    elif dataset == 'fmnist':
        num_channels = 1

    if gen_type == 'mnist_vae':
        generator = MnistVAE()
        generator.load_state_dict(torch.load(weights_path))

    elif gen_type == 'nvae':
        generator = load_nvae(weights_path, batch_size=1)

        def encode(x):
            z, combiner_cells_s = generator.encode(x)
            return [z] + combiner_cells_s

        def decode(zs):
            z, combiner_cells_s = zs[0], zs[1:]
            logits, log_p = generator.decode(z, combiner_cells_s, 1.)
            return generator.decoder_output(logits).sample(), log_p

    elif gen_type == 'wgan_gp':
        generator = WGAN_GP(num_channels=num_channels)
        generator.load_state_dict(torch.load(weights_path))

    generator = generator.cuda()
    return generator


def load_encoded_ds(dataset, gen_type):
    return NPZDataset(f'data/{gen_type}_encode_{dataset}_test.npz')
