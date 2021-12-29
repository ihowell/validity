import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import torch.autograd as autograd
import fire
import numpy as np
import matplotlib.pyplot as plt
import submitit
from tqdm import tqdm
from torchvision import transforms, datasets

from tensorboardX import SummaryWriter

from validity.datasets import load_datasets
from validity.util import EarlyStopping, loop_gen


class WGAN_GP(nn.Module):
    def __init__(self, critic_iter=5, lambda_term=10):
        super().__init__()
        self.critic_iter = critic_iter
        self.lambda_term = lambda_term

        self.generator = nn.Sequential(
            nn.Linear(128, 4 * 4 * 256),
            nn.ReLU(True),
            nn.Unflatten(-1, [256, 4, 4]),
            # 256x4x4
            nn.ConvTranspose2d(256, 128, 4),
            nn.ReLU(True),
            # 128x7x7
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            # 64x14x14
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x14x14
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x7x7
            nn.Conv2d(128, 256, 5, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x4x4
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 1),
        )

    def encode(self, x, attempts=8, writer=None):
        z = torch.randn((x.size(0), attempts, 128), requires_grad=True, device='cuda')
        opt = optim.SGD([z], lr=10)
        #opt = optim.Adam([z], lr=1, betas=(0.5, 0.9))
        early_stopping = EarlyStopping(patience=1000, verbose=False)
        best_z = None
        best_loss = None
        for step in range(int(1e7)):
            opt.zero_grad()
            x_hat = self.generator(z.flatten(0, 1))
            x_hat = x_hat.reshape([x.size(0), attempts] + list(x_hat.shape[1:]))

            loss = (x.unsqueeze(1) - x_hat).square().sum((-3, -2, -1)).sqrt()
            loss.mean().backward()
            opt.step()
            loss, loss_idx = loss.min(1)
            if best_loss is None:
                best_loss = loss.detach()
                best_z = []
                for i in range(x.size(0)):
                    best_z.append(z[i][loss_idx[i]])
                best_z = torch.stack(best_z).detach()
            else:
                for i in range(x.size(0)):
                    if loss[i] < best_loss[i]:
                        best_loss[i] = loss[i].detach()
                        best_z[i] = z[i][loss_idx[i]].detach()

            if writer:
                writer.add_scalar('encode/best_loss', best_loss.mean(), step)
                x_hat = self.generator(best_z)
                img = torch.cat([x, x_hat], 2)
                writer.add_images('encode/img', img, step)

            if early_stopping(best_loss.mean().cpu().detach().numpy()):
                break
            if best_loss.mean() < 1.:
                break

        return best_z

    def decode(self, z):
        return self.generator(z)

    def get_train_optimizers(self):
        return optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0., 0.9)), \
            optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0., 0.9))

    def sample(self, batch_size):
        z = torch.randn((batch_size, 128)).cuda()
        fake_data = self.generator(z)
        return fake_data

    def train_step(self, train_loader, optim_disc, optim_gen):
        for i in range(self.critic_iter):
            optim_disc.zero_grad()
            real_data, label = next(train_loader)
            real_data = real_data.cuda()
            batch_size = real_data.size(0)

            z = torch.randn((batch_size, 128)).cuda()
            fake_data = self.generator(z)

            real_logits = self.discriminator(real_data)
            fake_logits = self.discriminator(fake_data)

            epsilon = torch.rand(batch_size).reshape((batch_size, 1, 1, 1)).cuda()
            interpolated_data = epsilon * real_data + (1. - epsilon) * fake_data
            interpolated_logits = self.discriminator(interpolated_data)
            gradients = autograd.grad(outputs=interpolated_logits.mean(),
                                      inputs=interpolated_data,
                                      create_graph=True)[0]
            grad_penalty = gradients.square().sum((2, 3)).sqrt()
            grad_penalty = (grad_penalty - 1.).square() * self.lambda_term
            discriminator_loss = fake_logits.mean() - real_logits.mean() + grad_penalty.mean()
            Wasserstein_D = real_logits.mean() - fake_logits.mean()
            discriminator_loss.backward()
            optim_disc.step()

        optim_gen.zero_grad()
        z = torch.randn((batch_size, 128)).cuda()
        fake_data = self.generator(z)
        fake_logits = self.discriminator(fake_data)
        generator_loss = -fake_logits.mean()
        generator_loss.backward()
        optim_gen.step()

        return {
            'discriminator_loss': discriminator_loss.mean(),
            'generator_loss': generator_loss.mean(),
            'Wasserstein_D': Wasserstein_D.mean(),
            'logits real': real_logits.mean(),
            'logits fake': fake_logits.mean(),
            'grad_penalty': grad_penalty,
        }


def train(generator_iters=200000, batch_size=64, data_root='./datasets/', cuda_idx=0):
    gan = WGAN_GP()
    gan = gan.cuda()
    gan.train()

    train_set, _ = load_datasets('mnist')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader = loop_gen(train_loader)

    save_name = f'mnist2'
    writer = SummaryWriter('gan/' + save_name)
    save_path = f'gan/{save_name}.pt'

    optim_disc, optim_gen = gan.get_train_optimizers()

    step = 0
    for _ in tqdm(range(generator_iters)):
        metrics = gan.train_step(train_loader, optim_disc, optim_gen)
        for metric_name, metric in metrics.items():
            if '_img' in metric_name:
                metric_name = ''.join(metric_name.split('_img'))
                writer.add_images(f'train/{metric_name}', metric, step)
            else:
                writer.add_scalar(f'train/{metric_name}', metric.mean(), step)

        imgs = gan.sample(3)
        writer.add_images(f'train/samples', imgs, step)

        torch.save(gan.state_dict(), save_path)
        step += 1


def test_encode(gan_path, batch_size=64, data_root='./datasets/', cuda_idx=0, seed=0):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    gan = WGAN_GP()
    gan.load_state_dict(torch.load(gan_path, map_location=f'cuda:{cuda_idx}'))
    gan = gan.cuda()
    gan.eval()

    train_set, _ = load_datasets('mnist')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter('gan/mnist_encode_1')

    for data, label in train_loader:
        data = data
        z = gan.encode(data.cuda(), writer=writer)
        break


def encode_dataset(dataset,
                   gan_path,
                   shards=20,
                   batch_size=64,
                   data_root='./datasets/',
                   cuda_idx=0,
                   seed=0):
    executor = submitit.AutoExecutor(folder='logs')
    executor.update_parameters(timeout_min=7 * 24 * 60,
                               gpus_per_node=1,
                               slurm_partition='gpu',
                               slurm_gres='gpu',
                               slurm_mem_per_cpu='32G',
                               slurm_array_parallelism=100)
    jobs = []
    with executor.batch():
        for i in range(shards):
            jobs.append(
                executor.submit(_encode_dataset_job, dataset, gan_path, i, shards, batch_size,
                                data_root, cuda_idx, seed))
    [job.result() for job in jobs]

    test_data = []
    test_labels = []
    for i in range(shards):
        _file = np.load(f'data/tmp/wgan_gp_encode_{dataset}_test_{i}_{shards}.npz')
        test_data.append(_file['arr_0'])
        test_labels.append(_file['arr_1'])
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)

    np.savez(f'data/wgan_gp_encode_{dataset}_test.npz', test_data, test_labels)


def _encode_dataset_job(dataset,
                        gan_path,
                        shard_idx,
                        shards,
                        batch_size=64,
                        data_root='./datasets/',
                        cuda_idx=0,
                        seed=0):
    #import torch.backends.cudnn as cudnn
    #cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    gan = WGAN_GP()
    gan.load_state_dict(torch.load(gan_path, map_location=f'cuda:{cuda_idx}'))
    gan = gan.cuda()
    gan.eval()

    _, test_set = load_datasets('mnist')
    test_set = torch.utils.data.Subset(test_set, range(17))

    n = len(test_set)
    shard_lower = (n * shard_idx) // shards
    shard_upper = (n * (shard_idx + 1)) // shards
    test_set = torch.utils.data.Subset(test_set, range(shard_lower, shard_upper))
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_data = []
    test_labels = []
    for data, label in tqdm(loader):
        z = gan.encode(data.cuda())
        test_data.append(z.cpu().numpy())
        test_labels.append(label.numpy())
        if len(test_data) >= 2:
            break

    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)
    pathlib.Path('data/tmp').mkdir(exist_ok=True, parents=True)
    np.savez(f'data/tmp/wgan_gp_encode_{dataset}_test_{shard_idx}_{shards}.npz', test_data,
             test_labels)


if __name__ == '__main__':
    fire.Fire()
