# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor
    
def kl_divergence(mean, logvar):
  return -0.5 * torch.mean(1 + logvar - torch.square(mean) - torch.exp(logvar))

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

def loss(config, x, x_hat, z, mu, logvar):
    
  log_sigma = torch.tensor(((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log())
  log_sigma = softclip(log_sigma, -6)
  recons_loss = gaussian_nll(x_hat, log_sigma, x).mean()
  
  kld_loss = kl_divergence(mu, logvar)

  loss = recons_loss + config["kl_coeff"] * kld_loss
  return loss 

class Encoder(nn.Module):

  def __init__(self, 
                in_channels: int = 3, 
                hidden_dims: Optional[list] = None,
                latent_dim: int = 256):
    super(Encoder, self).__init__()

    modules = []
    if hidden_dims is None:
        hidden_dims = [32, 64, 128, 256, 512]

    # Build Encoder
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size= 3, stride= 2, padding  = 1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_channels = h_dim

    self.encoder = nn.Sequential(*modules)
    self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
    self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

  def forward(self, x):
    x = self.encoder(x)
    x = torch.flatten(x, start_dim=1)
    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    return mu, log_var

class Decoder(nn.Module):

  def __init__(self,
               hidden_dims: Optional[list] = None,
               latent_dim: int = 256):
    super(Decoder, self).__init__()

    # Build Decoder
    modules = []

    if hidden_dims is None:
        hidden_dims = [32, 64, 128, 256, 512]
        hidden_dims.reverse()
      
    self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)

    for i in range(len(hidden_dims) - 1):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU())
        )

    self.decoder = nn.Sequential(*modules)
    self.final_layer = nn.Sequential(
                          nn.ConvTranspose2d(hidden_dims[-1],
                                              hidden_dims[-1],
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1),
                          nn.BatchNorm2d(hidden_dims[-1]),
                          nn.LeakyReLU(),
                          nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                    kernel_size= 3, padding= 1),
                          nn.Sigmoid())

  def forward(self, z):
    result = self.decoder_input(z)
    result = result.view(-1, 512, 2, 2)
    result = self.decoder(result)
    result = self.final_layer(result)
    return result