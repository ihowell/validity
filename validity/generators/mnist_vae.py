import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import fire
from tqdm import tqdm
from torchvision import transforms, datasets
import numpy as np

from tensorboardX import SummaryWriter

from validity.datasets import load_datasets
from validity.util import EarlyStopping

SIG_EPSILON = 1e-2
CLIP_GRAD_VALUE = 10.


class MnistVAE(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta
        self.tied_log_sig_x = nn.Parameter(torch.tensor([0.], requires_grad=True))

        self.enc_transform = transforms.Resize((64, 64))

        self.enc_layers = nn.ModuleList([
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Flatten(),
        ])
        self.linear_z_mu = nn.Linear(51200, 400)
        self.linear_z_sig = nn.Linear(51200, 400)

        self.latent_shape = (400, )

        self.dec_layers = nn.ModuleList([
            nn.Linear(400, 28 * 28 * 128),
            nn.Unflatten(-1, (128, 28, 28)),
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding='same'),
        ])

    def encode(self, x):
        # Encode to a random code
        mu_z, log_sig_z = self._encode(x)
        # sig_z = torch.exp(log_sig_z) + SIG_EPSILON
        # pz = dist.normal.Normal(mu_z, sig_z)
        # pz = dist.independent.Independent(pz, 1)
        # z = pz.rsample()
        return mu_z

    def decode(self, z):
        # Decode code to a random sample
        mu_x_hat = self._decode(z)
        # sig_x_hat = self.tied_log_sig_x.exp() + SIG_EPSILON
        # px_hat = dist.normal.Normal(mu_x_hat, sig_x_hat)
        # px_hat = dist.independent.Independent(px_hat, 3)
        # x_hat = px_hat.rsample().clamp(0., 1.)
        return mu_x_hat.clamp(0., 1.)

    def _encode(self, x):
        out = x
        for layer in self.enc_layers:
            out = layer(out)

        mu_z = self.linear_z_mu(out)
        log_sig_z = self.linear_z_sig(out)
        return mu_z, log_sig_z

    def _decode(self, z):
        out = z
        for layer in self.dec_layers:
            out = layer(out)

        return out

    def forward(self, x):
        n = x.size(0)

        # encode
        mu_z, log_sig_z = self._encode(x)
        sig_z = torch.exp(log_sig_z) + SIG_EPSILON
        pz = dist.normal.Normal(mu_z, sig_z)
        pz = dist.independent.Independent(pz, 1)
        z = pz.rsample()

        # decode
        mu_x_hat = self._decode(z)
        sig_x_hat = self.tied_log_sig_x.exp() + SIG_EPSILON
        px_hat = dist.normal.Normal(mu_x_hat, sig_x_hat)
        px_hat = dist.independent.Independent(px_hat, 3)
        posterior_nll = -px_hat.log_prob(x)
        x_hat = px_hat.rsample()

        # prior z
        mu_p = torch.zeros_like(mu_z).cuda()
        prior = dist.normal.Normal(mu_p, 1.)
        prior = dist.independent.Independent(prior, 1)

        # kl divergence
        posterior = pz
        kl = dist.kl.kl_divergence(posterior, prior)

        if not kl.isfinite().all():
            raise Exception('KL divergence non-finite')

        if not posterior_nll.isfinite().all():
            raise Exception('Posterior non-finite')

        # elbo
        nelbo = kl + posterior_nll
        loss = self.beta * kl + posterior_nll

        metrics = {
            'nelbo': nelbo,
            'kl': kl,
            'posterior_nll': posterior_nll,
            'reconstruction_img': torch.cat([x, x_hat], dim=2)[:3].clamp(0., 1.),
            'mu_z': mu_z.mean(),
            'sig_z': sig_z.mean(),
            'mu_x_hat': mu_x_hat.mean(),
            'sig_x_hat': sig_x_hat.mean(),
        }

        return loss, metrics

    def sample(self, num_samples):
        latent_shape = [num_samples] + list(self.latent_shape)
        mu_p = torch.zeros(latent_shape).cuda()
        sig_p = torch.ones(latent_shape).cuda()
        prior = dist.normal.Normal(mu_p, sig_p)
        prior = dist.independent.Independent(prior, 1)
        z = prior.rsample()

        mu_x_hat = self._decode(z)
        # sig_x_hat = F.softplus(log_sig_x_hat) + SIG_EPSILON
        sig_x_hat = self.tied_log_sig_x.exp() + SIG_EPSILON
        # sig_x_hat = log_sig_x_hat.sigmoid() + SIG_EPSILON
        px_hat = dist.normal.Normal(mu_x_hat, sig_x_hat)
        px_hat = dist.independent.Independent(px_hat, 3)
        x_hat = px_hat.rsample().clamp(0., 1.)
        return x_hat

    def log_prob(self, x):
        n = x.size(1)
        loss, metrics = self.forward(x)
        return -1. * metrics['nelbo']

    def latent_log_prob(self, z):
        n = z.size(0)

        # prior z
        mu_p = torch.zeros([n] + list(self.latent_shape)).cuda()
        sig_p = torch.ones([n] + list(self.latent_shape)).cuda()
        prior = dist.multivariate_normal.MultivariateNormal(mu_p, sig_p)
        prior = dist.independent.Independent(prior, 1)

        return prior.log_prob(z)


def train(beta=1.,
          max_epochs=1000,
          batch_size=64,
          data_root='./datasets/',
          cuda_idx=0,
          mutation_rate=None,
          anneal_epochs=None,
          warm_epochs=None):
    beta = float(beta)
    vae = MnistVAE(beta=beta)
    vae = vae.cuda()

    train_set, _ = load_datasets('mnist')
    train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    save_name = f'vae_mnist_{beta}'
    if mutation_rate:
        save_name += f'_bg_{mutation_rate}'
    if anneal_epochs:
        save_name += f'_anneal_{anneal_epochs}'
    if warm_epochs:
        save_name += f'_warm_{warm_epochs}'

    writer = SummaryWriter('models/' + save_name)
    save_path = f'models/{save_name}.pt'

    optimizer = optim.Adam(vae.parameters(), weight_decay=1e-4)

    early_stopping = EarlyStopping(vae.state_dict(), save_path, patience=50)

    step = 0

    for epoch in range(max_epochs):
        if warm_epochs:
            if epoch < warm_epochs:
                vae.beta = 0.
                early_stopping.reset()
            elif epoch < warm_epochs + (anneal_epochs or 0):
                vae.beta = beta / float(anneal_epochs) * float(epoch - warm_epochs)
                early_stopping.reset()
            elif epoch == anneal_epochs:
                vae.beta = beta
                early_stopping.reset()
        elif anneal_epochs:
            if epoch < anneal_epochs:
                vae.beta = beta / float(anneal_epochs) * float(epoch)
                early_stopping.reset()
            elif epoch == anneal_epochs:
                vae.beta = beta
                early_stopping.reset()

        print(f'Epoch {epoch}')
        vae.train()
        for data, _ in tqdm(train_loader):
            optimizer.zero_grad()
            data = data.cuda()

            if mutation_rate:
                m = torch.bernoulli(mutation_rate * torch.ones_like(data))
                r = torch.rand(data.shape).cuda()
                data = torch.where(m == 1., r, data)

            loss, metrics = vae(data)
            writer.add_scalar('train/loss', loss.mean(), step)
            for metric_name, metric in metrics.items():
                if '_img' in metric_name:
                    metric_name = ''.join(metric_name.split('_img'))
                    writer.add_images(f'train/{metric_name}', metric, step)
                else:
                    writer.add_scalar(f'train/{metric_name}', metric.mean(), step)

            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), CLIP_GRAD_VALUE)
            max_grad = torch.tensor([a.grad.max() for a in vae.parameters()
                                     if a is not None]).max()
            writer.add_scalar('train/max_grad', max_grad, step)

            optimizer.step()
            step += 1

        with torch.no_grad():
            x_hat = vae.sample(3)
        writer.add_images('samples', x_hat, step)

        vae.eval()
        for data, _ in tqdm(val_loader):
            val_loss = []
            with torch.no_grad():
                data = data.cuda()
                loss, metrics = vae(data)
            writer.add_scalar('val/loss', loss.mean(), step)
            for metric_name, metric in metrics.items():
                if '_img' in metric_name:
                    metric_name = ''.join(metric_name.split('_img'))
                    writer.add_images(f'val/{metric_name}', metric, step)
                else:
                    writer.add_scalar(f'val/{metric_name}', metric.mean(), step)

            step += 1

            val_loss.append(loss.mean())

        val_loss = torch.tensor(val_loss).mean()

        if early_stopping(val_loss):
            break


def train_multiple_bg(mutation_rates, **kwargs):
    #print(f'{kwargs=}')
    for mutation_rate in mutation_rates:
        train(mutation_rate=mutation_rate, **kwargs)


def encode_dataset(weights_path, batch_size=512, data_root='./datasets/', cuda_idx=0, seed=0):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    vae = MnistVAE()
    vae.load_state_dict(torch.load(weights_path))
    vae = vae.cuda()
    vae.eval()

    _, test_set = load_datasets('mnist')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_data = []
    test_labels = []
    for data, label in tqdm(test_loader):
        data = data
        z = vae.encode(data.cuda())
        test_data.append(z.cpu().detach().numpy())
        test_labels.append(label.detach().numpy())
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)

    pathlib.Path('data').mkdir(exist_ok=True)
    np.savez(f'data/vae_encode_mnist_test.npz', test_data, test_labels)


def get_save_path(beta=1., mutation_rate=None, anneal_epochs=None, warm_epochs=None):
    save_name = f'vae_mnist_{beta}'
    if mutation_rate:
        save_name += f'_bg_{mutation_rate}'
    if anneal_epochs:
        save_name += f'_anneal_{anneal_epochs}'
    if warm_epochs:
        save_name += f'_warm_{warm_epochs}'
    save_path = f'models/{save_name}.pt'
    return save_path


if __name__ == '__main__':
    fire.Fire()
