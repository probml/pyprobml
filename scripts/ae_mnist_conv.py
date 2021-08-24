import superimport

import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from einops import rearrange
from argparse import ArgumentParser

class ConvAEModule(nn.Module):
    def __init__(self, input_shape,
                 encoder_conv_filters,
                 decoder_conv_t_filters,
                 latent_dim,
                 deterministic=False):
        super(ConvAEModule, self).__init__()
        self.input_shape = input_shape

        self.latent_dim = latent_dim
        self.deterministic = deterministic

        all_channels = [self.input_shape[0]] + encoder_conv_filters

        self.enc_convs = nn.ModuleList([])

        # encoder_conv_layers
        for i in range(len(encoder_conv_filters)):
            self.enc_convs.append(nn.Conv2d(all_channels[i], all_channels[i + 1],
                                            kernel_size=3, stride=2, padding=1))
            if not self.latent_dim == 2:
                self.enc_convs.append(nn.BatchNorm2d(all_channels[i + 1]))
            self.enc_convs.append(nn.LeakyReLU())

        self.flatten_out_size = self.flatten_enc_out_shape(input_shape)

        if self.latent_dim == 2:
            self.mu_linear = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.mu_linear = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )

        if self.latent_dim == 2:
            self.log_var_linear = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.log_var_linear = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )

        if self.latent_dim == 2:
            self.decoder_linear = nn.Linear(self.latent_dim, self.flatten_out_size)
        else:
            self.decoder_linear = nn.Sequential(
                nn.Linear(self.latent_dim, self.flatten_out_size),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )

        all_t_channels = [encoder_conv_filters[-1]] + decoder_conv_t_filters

        self.dec_t_convs = nn.ModuleList([])

        num = len(decoder_conv_t_filters)

        # decoder_trans_conv_layers
        for i in range(num - 1):
            self.dec_t_convs.append(nn.UpsamplingNearest2d(scale_factor=2))
            self.dec_t_convs.append(nn.ConvTranspose2d(all_t_channels[i], all_t_channels[i + 1],
                                                       3, stride=1, padding=1))
            if not self.latent_dim == 2:
                self.dec_t_convs.append(nn.BatchNorm2d(all_t_channels[i + 1]))
            self.dec_t_convs.append(nn.LeakyReLU())

        self.dec_t_convs.append(nn.UpsamplingNearest2d(scale_factor=2))
        self.dec_t_convs.append(nn.ConvTranspose2d(all_t_channels[num - 1], all_t_channels[num],
                                                   3, stride=1, padding=1))
        self.dec_t_convs.append(nn.Sigmoid())

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample
    
    def _run_step(self, x):
      mu, log_var = self.encode(x)
      std = torch.exp(0.5*log_var)
      p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
      q = torch.distributions.Normal(mu, std)
      z = self.reparameterize(mu,log_var)
      recon = self.decode(z)
      return z, recon, p, q

    def flatten_enc_out_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        for l in self.enc_convs:
            x = l(x)
        self.shape_before_flattening = x.shape
        return int(np.prod(self.shape_before_flattening))

    def encode(self, x):
        for l in self.enc_convs:
            x = l(x)
        x = x.view(x.size()[0], -1)  # flatten
        mu = self.mu_linear(x)
        log_var = self.log_var_linear(x)
        return mu, log_var

    def decode(self, z):
        z = self.decoder_linear(z)
        recon = z.view(z.size()[0], *self.shape_before_flattening[1:])
        for l in self.dec_t_convs:
            recon = l(recon)
        return recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        if self.deterministic:
            return self.decode(mu), mu, None
        else:
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var

class ConvAE(LightningModule):
    def __init__(self,input_shape,
                      encoder_conv_filters,
                      decoder_conv_t_filters,
                      latent_dim,
                      kl_coeff=0.1,
                      lr = 0.001):
        super(ConvAE, self).__init__()
        self.kl_coeff = kl_coeff
        self.lr = lr 
        self.vae = ConvAEModule(input_shape, encoder_conv_filters, decoder_conv_t_filters, latent_dim)

    def step(self, batch, batch_idx):
      x, y = batch
      z, x_hat, p, q = self.vae._run_step(x)

      loss = F.binary_cross_entropy(x_hat, x, reduction='sum')

      logs = {
          "loss": loss,
      }
      return loss, logs
    
    def training_step(self, batch, batch_idx):
      loss, logs = self.step(batch, batch_idx)
      self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
      return loss

    def validation_step(self, batch, batch_idx):
      loss, logs = self.step(batch, batch_idx)
      self.log_dict({f"val_{k}": v for k, v in logs.items()})
      return loss

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    parser = ArgumentParser(description='Hyperparameters for our experiments')
    parser.add_argument('--bs', type=int, default=500, help="batch size")
    parser.add_argument('--epochs', type=int, default=50, help="num epochs")
    parser.add_argument('--latent-dim', type=int, default=20, help="size of latent dim for our vae")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--kl-coeff', type=int, default=0.75, help="kl coeff aka beta term in the elbo loss function")
    hparams = parser.parse_args()

    ae = ConvAE((1, 28, 28), 
                encoder_conv_filters=[28,64,64],
                decoder_conv_t_filters=[64,28,1],
                latent_dim=hparams.latent_dim, kl_coeff=hparams.kl_coeff, lr=hparams.lr)

    mnist_full = MNIST(".", train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
    dm = DataLoader(mnist_full, batch_size=hparams.bs)
    trainer = Trainer(gpus=1, weights_summary='full', max_epochs=hparams.epochs)
    trainer.fit(ae, dm)

    torch.save(ae.state_dict(), f"ae-mnist-conv-latent-dim-{hparams.latent_dim}.ckpt")
