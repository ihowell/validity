import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import fire
from tqdm import tqdm
from torchvision import transforms, datasets

from tensorboardX import SummaryWriter

from validity.util import EarlyStopping
from .mixture import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic

EPSILON = torch.tensor(1e-6)
SIG_EPSILON = 6e-3
LOG_EPSILON = torch.log(EPSILON).cuda()
SIG_MAX = torch.log(torch.tensor([1e5])).cuda()
CLIP_GRAD_VALUE = 2e-2

LN_2 = torch.log(torch.tensor(2.))
LOG2T = torch.log2(torch.tensor(1 - 2 * EPSILON))

CHANNEL_MULT = 2


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


def get_skip_connection(C, stride, affine, channel_mult):
    if stride == 1:
        return Identity()
    elif stride == 2:
        return FactorizedReduce(C, int(channel_mult * C))
    elif stride == -1:
        return nn.Sequential(UpSample(), nn.Conv2d(C, int(C / channel_mult), 1,
                                                   padding='same'))


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(c, c // r,
                                                  bias=False), nn.ReLU(inplace=True),
                                        nn.Linear(c // r, c, bias=False), nn.Sigmoid())

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class EncResCell(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels_in = channels_in
        self.layers = nn.ModuleList([
            # nn.BatchNorm2d(channels_in),
            nn.SiLU(),
            nn.Conv2d(channels_in, channels_in, 3, padding='same'),
            # nn.BatchNorm2d(channels_in),
            nn.SiLU(),
            nn.Conv2d(channels_in, channels_in, 3, padding='same'),
            #SE_Block(channels_in),
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = x + 0.1 * out
        return out


class DecResCell(nn.Module):
    def __init__(self, channels, mult, use_se):
        super().__init__()

        self.channels = channels

        self.layers = nn.ModuleList([
            #nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, padding='same'),
            #nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, mult * channels, 5, groups=channels, padding='same'),
            #nn.BatchNorm2d(mult * channels),
            nn.SiLU(),
            nn.Conv2d(mult * channels, channels, 1, padding='same'),
        ])

        self.use_se = use_se
        if self.use_se:
            self.se = SE_Block(channels)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        if self.use_se:
            out = self.se(out)

        out = x + 0.1 * out
        return out


class MnistVAE(nn.Module):
    def __init__(self, beta=1., num_logistic=10):
        super().__init__()
        self.beta = beta
        self.num_logistic = num_logistic

        self.tied_log_sig_z = nn.Parameter(torch.tensor([0.], requires_grad=True))

        self.enc_transform = transforms.Resize((64, 64))

        activ = nn.SiLU

        self.enc_layers = nn.ModuleList([
            nn.Conv2d(1, 8, 5, stride=2),
            activ(),
            EncResCell(8),
            activ(),
            nn.Conv2d(8, 16, 5, stride=2),
            activ(),
            EncResCell(16),
            activ(),
            EncResCell(16),
            activ(),
            nn.Conv2d(16, 32, 5, stride=2),
            activ(),
            EncResCell(32),
            activ(),
            EncResCell(32),
            activ(),
            EncResCell(32),
            activ(),
            nn.Conv2d(32, 64, 5, padding='same'),
            activ(),
            EncResCell(64),
            activ(),
            EncResCell(64),
            activ(),
            EncResCell(64),
            activ(),
            EncResCell(64),
            activ(),
            nn.Conv2d(64, 128, 5, padding='same'),
            activ(),
            EncResCell(128),
            activ(),
            EncResCell(128),
            activ(),
            EncResCell(128),
            activ(),
            EncResCell(128),
            activ(),
        ])
        self.linear_mu = nn.Linear(3200, 300)

        self.latent_shape = (300, )

        self.dec_layers = nn.ModuleList([
            nn.Linear(300, 1600),
            nn.Unflatten(-1, (64, 5, 5)),
            activ(),
            DecResCell(64, 4, True),
            activ(),
            DecResCell(64, 4, True),
            activ(),
            DecResCell(64, 4, True),
            activ(),
            DecResCell(64, 4, True),
            activ(),
            nn.ConvTranspose2d(64, 128, 5, stride=2),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            nn.ConvTranspose2d(128, 128, 5, stride=2, output_padding=1, padding=1),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, True),
            activ(),
            DecResCell(128, 4, False),
            activ(),
            nn.Conv2d(128, 3 * self.num_logistic, 3, padding='same'),
        ])

    def encode(self, x):
        # Encode to a random code
        mu_z = self._encode(x)
        sig_z = torch.exp(self.tied_log_sig_z) + SIG_EPSILON
        pz = dist.normal.Normal(mu_z, sig_z)
        pz = dist.independent.Independent(pz, 1)
        z = pz.rsample()
        return mu_z

    def decode(self, z):
        # Decode code to a random sample
        mu_x_hat = self._decode(z)
        x_hat = x_hat.view(num_samples, self.num_logistic * 3, -1)
        x_hat = sample_from_discretized_mix_logistic(x_hat)
        x_hat = x_hat.view(num_samples, 1, 28, 28)
        return x_hat

    def _encode(self, x):
        out = x
        out = self.enc_transform(out)
        for layer in self.enc_layers:
            out = layer(out)

        out = torch.flatten(out, 1)
        mu_z = self.linear_mu(out)
        return mu_z

    def _decode(self, z):
        out = z
        for layer in self.dec_layers:
            out = layer(out)
        return out

    def forward(self, x):
        n = x.size(0)

        # encode
        mu_z = self._encode(x)
        sig_z = torch.exp(self.tied_log_sig_z) + SIG_EPSILON
        pz = dist.normal.Normal(mu_z, sig_z)
        pz = dist.independent.Independent(pz, 1)
        z = pz.rsample()

        # decode
        x_hat = self._decode(z)
        x_hat = x_hat.view(n, self.num_logistic * 3, -1)
        posterior_nll = discretized_mix_logistic_loss(x_hat, x.view(n, -1, 1), reduce=False)
        posterior_nll = posterior_nll.sum([1, 2])
        x_hat = sample_from_discretized_mix_logistic(x_hat)
        x_hat = x_hat.view(n, 1, 28, 28)

        # prior z
        mu_p = torch.zeros_like(mu_z).cuda()
        prior = dist.normal.Normal(mu_p, 1.)
        prior = dist.independent.Independent(prior, 1)

        # kl divergence
        posterior = pz
        kl = dist.kl.kl_divergence(posterior, prior)

        # elbo
        nelbo = kl + posterior_nll
        loss = self.beta * kl + posterior_nll

        # bpd
        D = torch.tensor(x.shape[1:]).prod()
        bits_per_dim = posterior_nll / (D * LN_2)

        metrics = {
            'nelbo': nelbo,
            'bits_per_dim': bits_per_dim,
            'kl': kl,
            'posterior_nll': posterior_nll,
            'reconstruction_img': torch.cat([x, x_hat], dim=2)[:3]
        }

        return loss, metrics

    def sample(self, num_samples):
        latent_shape = [num_samples] + list(self.latent_shape)
        mu_p = torch.zeros(latent_shape).cuda()
        sig_p = torch.ones(latent_shape).cuda()
        prior = dist.normal.Normal(mu_p, sig_p)
        prior = dist.independent.Independent(prior, 1)
        z = prior.rsample()

        x_hat = self._decode(z)
        x_hat = x_hat.view(num_samples, self.num_logistic * 3, -1)
        x_hat = sample_from_discretized_mix_logistic(x_hat)
        x_hat = x_hat.view(num_samples, 1, 28, 28)
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
          mutation_rate=None):
    beta = float(beta)
    vae = MnistVAE(beta=beta)
    vae = vae.cuda()

    dataset = datasets.MNIST(root=data_root,
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    save_name = f'mnist_{beta}'
    if mutation_rate:
        save_name = f'mnist_background_{beta}_{mutation_rate}'

    writer = SummaryWriter('vae/' + save_name)
    save_path = f'vae/{save_name}.pt'

    optimizer = optim.Adam(vae.parameters(), weight_decay=1e-5)

    early_stopping = EarlyStopping(vae.state_dict(), save_path, patience=50)

    step = 0

    for epoch in range(max_epochs):
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
            nn.utils.clip_grad_value_(vae.parameters(), CLIP_GRAD_VALUE)
            # max_grad = torch.tensor([a.grad.max() for a in vae.parameters()
            #                          if a is not None]).max()
            # writer.add_scalar('train/max_grad', max_grad, step)

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


if __name__ == '__main__':
    fire.Fire()
