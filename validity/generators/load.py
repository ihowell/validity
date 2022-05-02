from pathlib import Path

import torch

from validity.util import NPZDataset

from .mnist_vae import MnistVAE
from .wgan_gp import WGAN_GP


def load_gen(weights_path):
    print(f'{weights_path=}')
    saved_dict = torch.load(weights_path)
    print(f'{saved_dict.keys()=}')
    gen_type = saved_dict['type']

    if gen_type == 'mnist_vae':
        gen = MnistVAE.load(saved_dict)
    elif gen_type == 'wgan_gp':
        gen = WGAN_GP.load(saved_dict)
    else:
        raise Exception(f'Unknown generator type: {gen_type}')
    return gen


def load_encoded_ds(dataset, gen_type, encode_dir=None, id=None):
    ds_path = get_encoded_ds_path(dataset, gen_type, encode_dir=encode_dir, id=id)
    return NPZDataset(ds_path)


def get_encoded_ds_path(dataset, gen_type, encode_dir=None, id=None):
    if encode_dir is None:
        encode_dir = 'data'
    save_path = f'{gen_type}_encode_{dataset}_test'
    if id:
        save_path += f'_{id}'
    return Path(f'{encode_dir}/{save_path}.npz')
