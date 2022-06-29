import superimport

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from einops import rearrange
from argparse import ArgumentParser

class AE(nn.Module):

  def __init__(self, n_z,
                model_name="vae"):
    super().__init__()
    self.encoder = nn.Sequential(
          nn.Linear(28*28, 512),
          nn.ReLU()
    )
    self.model_name = model_name
    self.fc_mu = nn.Linear(512, n_z)
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
    return self.decoder(mu)

  def encode(self, x):
    x = self.encoder(x)
    mu = self.fc_mu(x)
    return mu

class BasicAEModule(LightningModule):

    def __init__(self,
                 n_z=2,
                 kl_coeff=0.1,
                 lr=0.001):
      super().__init__()
      self.vae = AE(n_z)
      self.kl_coeff = kl_coeff
      self.lr = lr

    def forward(self, x):
      return self.vae(x)

    def step(self, batch, batch_idx):
      x, y = batch
      x = rearrange(x, 'b c h w -> b (c h w)')
      x_hat= self.vae(x)

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
    parser.add_argument('--latent-dim', type=int, default=20, help="size of latent dim for our vae")
    parser.add_argument('--epochs', type=int, default=50, help="num epochs")
    parser.add_argument('--gpus', type=int, default=1, help="gpus, if no gpu set to 0, to run on all  gpus set to -1")
    parser.add_argument('--bs', type=int, default=500, help="batch size")
    hparams = parser.parse_args()

    mnist_full = MNIST(".", download=True, train=True,
                         transform=transforms.Compose([transforms.ToTensor()]))
    dm = DataLoader(mnist_full, batch_size=hparams.bs, shuffle=True)
    ae = BasicAEModule(hparams.latent_dim)

    trainer = Trainer(gpus=hparams.gpus, weights_summary='full', max_epochs=hparams.epochs)
    trainer.fit(ae, dm)
    torch.save(ae.state_dict(), f"ae-mnist-mlp-latent-dim-{hparams.latent_dim}.ckpt")
