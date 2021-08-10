import torch
from torch import nn 
import numpy as np
from torch import Tensor 
from typing import Callable
import torch.nn.functional as F 
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):

    def __init__(self, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.disc = nn.Sequential(
            self._make_disc_block(image_channels, feature_maps, batch_norm=False),
            self._make_disc_block(feature_maps, feature_maps * 2),
            self._make_disc_block(feature_maps * 2, feature_maps * 4),
            self._make_disc_block(feature_maps * 4, feature_maps * 8),
            self._make_disc_block(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True),
        )

    @staticmethod
    def _make_disc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        batch_norm: bool = True,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return disc_block

    def forward(self, x: Tensor) -> Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)

class Discriminator(nn.Module):

    def __init__(self, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.disc = nn.Sequential(
            self._make_disc_block(image_channels, feature_maps, batch_norm=False),
            self._make_disc_block(feature_maps, feature_maps * 2),
            self._make_disc_block(feature_maps * 2, feature_maps * 4),
            self._make_disc_block(feature_maps * 4, feature_maps * 8),
            self._make_disc_block(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True),
        )

    @staticmethod
    def _make_disc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        batch_norm: bool = True,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            disc_block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)),
                nn.Sigmoid(),
            )

        return disc_block

    def forward(self, x: Tensor) -> Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)

class Generator(nn.Module):

    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.gen = nn.Sequential(
            self._make_gen_block(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )

    @staticmethod
    def _make_gen_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return gen_block

    def forward(self, noise: Tensor) -> Tensor:
        return self.gen(noise)

def get_noise(real: Tensor,  configs:dict) -> Tensor:
    batch_size = len(real)
    device = real.device
    noise = torch.randn(batch_size, configs["latent_dim"], device=device)
    noise = noise.view(*noise.shape, 1, 1)
    return noise 

def get_sample(generator: Callable, real: Tensor, configs: dict) -> Tensor:
    noise = get_noise(real, configs)
    fake = generator(noise)

    return fake

def instance_noise(configs:dict, epoch_num: int, real: Tensor):
    if configs["loss_params"]["instance_noise"]:
        if configs["instance_noise_params"]["gamma"] is not None:
            noise_level = (configs["instance_noise_params"]["gamma"])**epoch_num*configs["instance_noise_params"]["noise_level"]
        else:
            noise_level = configs["instance_noise_params"]["noise_level"]
        real = real + noise_level*torch.randn_like(real)
    return real 

def top_k(configs:dict, epoch_num: int, preds: Tensor):
    if configs["loss_params"]["top_k"]:
        if configs["top_k_params"]["gamma"] is not None:
            k = int((configs["top_k_params"]["gamma"])**epoch_num*configs["top_k_params"]["k"])
        else:
            k= configs["top_k_params"]["k"]
        preds = torch.topk(preds, k )[0]
    return preds

def compute_gradient_penalty(discriminator: Callable, real_samples: Tensor, fake_samples: Tensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    device = real_samples.device
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.to(device)

    interpolates.requires_grad_(True)
    if interpolates.grad is not None:
      interpolates.grad.zero_()
    d_interpolates = discriminator(interpolates)
    d_interpolates.backward(torch.ones_like(d_interpolates).to(device))
    gradients = interpolates.grad.data

    gradients = gradients.view(gradients.size(0), -1).to(device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def disc_loss(configs: dict, discriminator: Callable, generator: Callable, epoch_num: int, real: Tensor) -> Tensor:
    # Train with real
    real = instance_noise(configs, epoch_num, real)
    real_pred = discriminator(real).mean()

    # Train with fake
    fake = get_sample(generator, real, configs["loss_params"])
    fake = instance_noise(configs, epoch_num, fake)
    fake_pred = discriminator(fake)
    fake_pred = top_k(configs, epoch_num, fake_pred).mean()
    
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real, fake)

    disc_loss = -real_pred + fake_pred + configs["loss_params"]["lambda_gp"] * gradient_penalty
    clip_value = 0.01
    for p in discriminator.parameters():
        p.data.clamp_(-clip_value, clip_value)
    return disc_loss

def gen_loss(configs: dict, discriminator:Callable, generator:Callable, epoch_num: int, real: Tensor) -> Tensor:
    # Train with fake
    fake = get_sample(generator, real, configs["loss_params"])
    fake = instance_noise(configs, epoch_num, fake)
    fake_pred = discriminator(fake)
    fake_pred = top_k(configs, epoch_num, fake_pred).mean()

    return -fake_pred 
