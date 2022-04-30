import torch
import fire

from validity.generators.wgan_gp import WGAN_GP
from validity.generators.mnist_vae import MnistVAE


def convert_wgan_gp(old_path, new_path):
    gen = WGAN_GP(num_channels=1)
    gen.load_state_dict(torch.load(old_path))
    torch.save(gen.get_args(), new_path)


def convert_mnist_vae(old_path, beta, new_path):
    gen = MnistVAE(beta=beta)
    gen.load_state_dict(torch.load(old_path))
    torch.save(gen.get_args(), new_path)


if __name__ == '__main__':
    fire.Fire()