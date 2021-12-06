import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import fire
from tqdm import tqdm
from torchvision import transforms, datasets

from tensorboardX import SummaryWriter


class MnistVAE(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta

        self.enc_transform = transforms.Resize((64, 64))

        self.enc_layers = nn.ModuleList([
            nn.Conv2d(1, 8, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding='same'),
        ])
        self.linear_mu = nn.Linear(1600, 300)
        self.linear_log_sig = nn.Linear(1600, 300)

        self.latent_shape = (300, )

        self.dec_layers = nn.ModuleList([
            nn.Linear(300, 1600),
            nn.Unflatten(-1, (64, 5, 5)),
            nn.ConvTranspose2d(64, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 5, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding='same'),
            nn.ReLU(),
        ])
        self.dec_mu = nn.Conv2d(8, 1, 3, padding='same')

    def encode(self, x):
        out = x
        out = self.enc_transform(out)
        for layer in self.enc_layers:
            out = layer(out)

        out = torch.flatten(out, 1)
        mu_z = self.linear_mu(out)
        log_sig_z = self.linear_log_sig(out)
        return mu_z, log_sig_z

    def decode(self, z):
        out = z
        for layer in self.dec_layers:
            out = layer(out)
        mu_x = self.dec_mu(out)
        return mu_x

    def forward(self, x):
        # encode
        mu_z, log_sig_z = self.encode(x)
        mu_z = mu_z.unsqueeze(-1)
        sig_z = F.softplus(log_sig_z).unsqueeze(-1).unsqueeze(-1) + 1e-3
        pz = dist.multivariate_normal.MultivariateNormal(mu_z, sig_z)
        pz = dist.independent.Independent(pz, 1)
        z = pz.rsample()

        # decode
        mu_x_hat = self.decode(z.squeeze(-1))
        px_hat = dist.bernoulli.Bernoulli(logits=mu_x_hat)
        px_hat = dist.independent.Independent(px_hat, 3)
        # x_hat = mu_x_hat
        x_hat = px_hat.sample()

        # posterior nll
        posterior_nll = -1. * px_hat.log_prob(torch.where(x > 0.02, 1., 0.))

        # prior z
        mu_p = torch.zeros_like(mu_z).cuda()
        sig_p = torch.ones_like(log_sig_z).unsqueeze(-1).unsqueeze(-1)
        prior = dist.multivariate_normal.MultivariateNormal(mu_p, sig_p)
        prior = dist.independent.Independent(prior, 1)

        # kl divergence
        posterior = pz
        kl = dist.kl.kl_divergence(posterior, prior)

        # elbo
        nelbo = self.beta * kl + posterior_nll

        return nelbo, kl, posterior_nll, x_hat

    def sample(self, num_samples):
        latent_shape = [num_samples] + list(self.latent_shape)
        mu_p = torch.zeros(latent_shape).unsqueeze(-1).cuda()
        sig_p = torch.ones(latent_shape).unsqueeze(-1).unsqueeze(-1).cuda()
        prior = dist.multivariate_normal.MultivariateNormal(mu_p, sig_p)
        prior = dist.independent.Independent(prior, 1)
        z = prior.rsample()

        mu_x_hat = self.decode(z.squeeze(-1))
        px_hat = dist.bernoulli.Bernoulli(logits=mu_x_hat)
        px_hat = dist.independent.Independent(px_hat, 3)
        x_hat = px_hat.sample()
        return x_hat

    def log_prob(self, x):
        n = x.size(1)
        nelbo, kl, posterior_nll, x_hat = self.forward(x)
        return -1. * nelbo

    def latent_log_prob(self, z):
        n = z.size(0)

        # prior z
        mu_p = torch.zeros([n] + list(self.latent_shape) + [1]).cuda()
        sig_p = torch.ones([n] + list(self.latent_shape) + [1, 1]).cuda()
        prior = dist.multivariate_normal.MultivariateNormal(mu_p, sig_p)
        prior = dist.independent.Independent(prior, 1)

        return prior.log_prob(z)


def train(beta=1.,
          max_epochs=1000,
          batch_size=64,
          data_root='./datasets/',
          cuda_idx=0,
          mutation_rate=None):
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

    optimizer = optim.Adam(vae.parameters())

    step = 0
    best_val_loss = None
    best_epoch = 0
    patience = 20
    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')
        for data, _ in tqdm(train_loader):
            optimizer.zero_grad()
            data = data.cuda()

            if mutation_rate:
                m = torch.bernoulli(mutation_rate * torch.ones_like(data))
                r = torch.rand(data.shape).cuda()
                data = torch.where(m == 1., r, data)

            nelbo, kl, posterior_nll, x_hat = vae(data)
            writer.add_scalar('train/nelbo', nelbo.mean(), step)
            writer.add_scalar('train/kl', kl.mean(), step)
            writer.add_scalar('train/posterior_nll', posterior_nll.mean(), step)
            writer.add_images('train/reconstruction',
                              torch.cat([data, x_hat], dim=2)[:3], step)
            loss = nelbo.mean()
            loss.backward()
            optimizer.step()
            step += 1

        with torch.no_grad():
            x_hat = vae.sample(3)
        writer.add_images('samples', x_hat, step)

        for data, _ in tqdm(val_loader):
            val_loss = []
            with torch.no_grad():
                data = data.cuda()
                nelbo, kl, posterior_nll, x_hat = vae(data)
                writer.add_scalar('val/nelbo', nelbo.mean(), step)
                writer.add_scalar('val/kl', kl.mean(), step)
                writer.add_scalar('val/posterior_nll', posterior_nll.mean(), step)
                writer.add_images('val/reconstruction',
                                  torch.cat([data, x_hat], dim=2)[:3], step)
                step += 1

                val_loss.append(nelbo.mean())

        val_loss = torch.tensor(val_loss).mean()
        if best_val_loss is None:
            best_val_loss = val_loss
            continue

        if val_loss < 0.:
            raise Exception('Unhandled negative loss')

        if val_loss < best_val_loss * (1 - 0.005):
            print(f'New best val loss: {val_loss:0.3f}')
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(vae.state_dict(), save_path)

        if epoch - best_epoch >= patience:
            break


if __name__ == '__main__':
    fire.Fire()
