import torch
from torch import nn
from typing import Callable

class VAE(nn.Module):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        loss: Callable,
        encoder: Callable,
        decoder: Callable,
        **kwargs
    ):

        super(VAE, self).__init__()

        self.loss = loss
        self.kwargs = kwargs
        self.encoder = encoder
        self.decoder = decoder

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
        z, x_hat, mu, logvar = self._run_step(x)

        loss = self.loss(x, x_hat, z , mu, logvar)

        return loss

    def step_sample(self, batch, batch_idx):
      x, y = batch
      z, x_hat = self._run_step(x)
      return z, x_hat
