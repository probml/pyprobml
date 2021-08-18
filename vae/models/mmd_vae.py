# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def compute_kernel(x1: torch.Tensor,
                  x2: torch.Tensor,
                 kernel_type: str = 'rbf') -> torch.Tensor:
    # Convert the tensors into row and column vectors
    D = x1.size(1)
    N = x1.size(0)

    x1 = x1.unsqueeze(-2) # Make it into a column tensor
    x2 = x2.unsqueeze(-3) # Make it into a row tensor

    """
    Usually the below lines are not required, especially in our case,
    but this is useful when x1 and x2 have different sizes
    along the 0th dimension.
    """
    x1 = x1.expand(N, N, D)
    x2 = x2.expand(N, N, D)

    if kernel_type == 'rbf':
        result = compute_rbf(x1, x2)
    elif kernel_type == 'imq':
        result = compute_inv_mult_quad(x1, x2)
    else:
        raise ValueError('Undefined kernel type.')

    return result

def compute_rbf(x1: torch.Tensor,
                x2: torch.Tensor,
                latent_var: float = 2.,
                eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the RBF Kernel between x1 and x2.
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    sigma = (2. / z_dim) * latent_var

    result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    return result

def compute_inv_mult_quad(x1: torch.Tensor,
                         x2: torch.Tensor,
                         latent_var: float = 2.,
                         eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
    given by
            k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    C = 2 * z_dim * latent_var
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

    # Exclude diagonal elements
    result = kernel.sum() - kernel.diag().sum()

    return result

def MMD(prior_z:torch.Tensor, z: torch.Tensor):

    prior_z__kernel = compute_kernel(prior_z, prior_z)
    z__kernel = compute_kernel(z, z)
    priorz_z__kernel = compute_kernel(prior_z, z)

    mmd = prior_z__kernel.mean() + \
            z__kernel.mean() - \
            2 * priorz_z__kernel.mean()
    return mmd

def loss(config, x, x_hat, z, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    mmd = MMD(torch.randn_like(z), z)

    loss = recon_loss + \
            config["beta"] * mmd
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
    return mu, torch.zeros_like(mu)

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
