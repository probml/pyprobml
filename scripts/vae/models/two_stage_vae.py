import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
import pytorch_lightning as pl

def kl_divergence(mean, logvar):
  return -0.5 * torch.mean(1 + logvar - torch.square(mean) - torch.exp(logvar))

def loss(config, x, x_hat, z, mu, logvar):
  recons_loss = F.mse_loss(x_hat, x, reduction='mean')

  kld_loss = kl_divergence(mu, logvar)

  loss = recons_loss + config["kl_coeff"] * kld_loss
  return loss 

class Encoder(nn.Module):

  def __init__(self, 
               input_dim: int = 256,
               hidden_dims: Optional[list] = None,
               latent_dim: int = 64):
    super(Encoder, self).__init__()

    if hidden_dims is None:
      hidden_dims = [250, 200, 150]

    modules = [] 
    for hidden_dim in hidden_dims:
      modules.append(
        nn.Sequential(
          nn.Linear(input_dim, hidden_dim),
          nn.LeakyReLU())
      )
      input_dim = hidden_dim

    self.encoder = nn.Sequential(*modules)
    self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
    self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

  def forward(self, x):
    x = self.encoder(x)
    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    return mu, log_var

class Decoder(nn.Module):

  def __init__(self,
               latent_dim: int = 64,
               hidden_dims: Optional[list] = None,
               output_dim: int = 256):
    super(Decoder, self).__init__()

    if hidden_dims is None:
      hidden_dims = [250, 200, 150]
      hidden_dims.reverse()

    modules = [] 
    for hidden_dim in hidden_dims:
      modules.append(
        nn.Sequential(
          nn.Linear(latent_dim, hidden_dim),
          nn.LeakyReLU())
      )
      latent_dim = hidden_dim
    
    self.decoder = nn.Sequential(*modules)
    self.final_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim)) 

  def forward(self, z):
    result = self.decoder(z)
    result = self.final_layer(result)
    return result

class Stage2VAE(nn.Module):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        name: str,
        loss: Callable,
        encoder: Callable,
        decoder: Callable,
        first_stage: pl.LightningModule,
        **kwargs
    ):

        super(Stage2VAE, self).__init__()

        self.name = name
        self.loss = loss
        self.kwargs = kwargs
        self.encoder = encoder
        self.decoder = decoder
        self.first_stage = first_stage

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample(mu, log_var)
        return z, self.decoder(z), mu, log_var
    
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_loss(self, x):
      z = self.first_stage.det_encode(x).detach()
      
      u, z_hat, mu, logvar = self._run_step(z)
      loss = self.loss(z, z_hat, z , mu, logvar)

      return loss

    def step_sample(self, batch, batch_idx):
      x, y = batch
      z, x_hat = self._run_step(x)
      return z, x_hat
