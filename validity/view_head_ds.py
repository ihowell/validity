import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.load('data/am_wgan_gp_mnist.npz')
    img = make_grid(torch.tensor(data['arr_0']), nrow=9)
    img = np.transpose(img, (1, 2, 0))

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
