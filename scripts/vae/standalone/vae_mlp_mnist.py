"""
Install pytorch lightning and einops

pip install pytorch_lightning einops
"""

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from einops import rearrange

class VAE(nn.Module):

  def __init__(self, n_z,
                model_name="vae"):
    super().__init__()
    self.encoder = nn.Sequential(
          nn.Linear(28*28, 512),
          nn.ReLU()
    )
    self.model_name = model_name
    self.fc_mu = nn.Linear(512, n_z)
    self.fc_var = nn.Linear(512, n_z)
    self.decoder = nn.Sequential(
        nn.Linear(n_z, 512),
        nn.ReLU(),
        nn.Linear(512, 28*28),
        nn.Sigmoid()
    )

  def forward(self, x):
    # in lightning, forward defines the prediction/inference actions
    x = self.encoder(x)
    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    p, q, z = self.sample(mu, log_var)
    return self.decoder(z)
  
  def _run_step(self, x):
    x = self.encoder(x)
    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    p, q, z = self.sample(mu, log_var)
    return z, self.decoder(z), p, q

  def encode(self, x):
    x = self.encoder(x)
    mu = self.fc_mu(x)
    return mu

  def sample(self, mu, log_var):
    std = torch.exp(log_var / 2)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    return p, q, z

class BasicVAEModule(pl.LightningModule):

    def __init__(self,
                 n_z=2,
                 kl_coeff=0.1,
                 lr=0.001):
      super().__init__()
      self.vae = VAE(n_z)
      self.kl_coeff = kl_coeff
      self.lr = lr

    def forward(self, x):
      return self.vae(x)

    def step(self, batch, batch_idx):
      x, y = batch
      z, x_hat, p, q = self.vae._run_step(x)

      recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')

      log_qz = q.log_prob(z)
      log_pz = p.log_prob(z)

      kl = log_qz - log_pz
      kl = kl.sum() # I tried sum, here
      kl *= self.kl_coeff

      loss = kl + recon_loss

      logs = {
          "recon_loss": recon_loss,
          "kl": kl,
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
    parser.add_argument('--latent-dim', type=int, default=12, help="size of latent dim for our vae")
    parser.add_argument('--epochs', type=int, default=50, help="num epochs")
    parser.add_argument('--gpus', type=int, default=1, help="gpus, if no gpu set to 0, to run on all  gpus set to -1")
    parser.add_argument('--bs', type=int, default=500, help="batch size")
    hparams = parser.parse_args()

    mnist_full = MNIST(".", download=True, train=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                        lambda x: rearrange(x, 'c h w -> (c h w)')]))
    dm = DataLoader(mnist_full, batch_size=hparams.bs, shuffle=True)
    vae = BasicVAEModule(hparams.latent_dim)

    trainer = pl.Trainer(gpus=hparams.gpus, weights_summary='full', max_epochs=hparams.epochs)
    trainer.fit(vae, dm)
    torch.save(vae.state_dict(), "vae-mnist-mlp.ckpt")
