import torch
from torch import nn 
from torch import Tensor 
from typing import Callable
import torch.nn.functional as F 

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

def get_sample(discriminator: Callable, generator: Callable, real: Tensor, configs: dict) -> Tensor:
    noise = get_noise(real, configs)
    noise_t = noise.clone()
    noise_t.requires_grad_(True)
    if noise_t.grad is not None:
      noise_t.grad.zero_()
    fake = generator(noise_t)
    fake_pred = discriminator(fake)
    fake_pred.backward(torch.ones_like(fake_pred ).to(noise.device))
    noise_t = noise_t + configs["alpha"]*noise_t.grad.data
    fake = generator(noise_t)
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
        if configs["loss_params"]["top_k"] is not None:
            k = int((configs["loss_params"]["top_k"])**epoch_num*configs["loss_params"]["top_k"])
        else:
            k= configs["loss_params"]["top_k"]
        preds = torch.topk(preds, k )
    return preds 

def get_fake_preds(configs: dict, discriminator: Callable, generator: Callable, epoch_num: int, real: Tensor):

    fake = get_sample(discriminator, generator, real, configs["loss_params"])
    fake = instance_noise(configs, epoch_num, fake)
    fake_pred = discriminator(fake)
    fake_pred = top_k(configs, epoch_num, fake_pred)
    return fake_pred

def disc_loss(configs: dict, discriminator: Callable, generator: Callable, epoch_num: int, real: Tensor) -> Tensor:
    # Train with real
    real = instance_noise(configs, epoch_num, real)
    real_pred = discriminator(real)
    real_gt = torch.ones_like(real_pred)
    real_loss = F.binary_cross_entropy(real_pred, real_gt)

    # Train with fake
    fake_pred = get_fake_preds(configs, discriminator, generator, epoch_num, real)
    fake_gt = torch.zeros_like(fake_pred)
    fake_loss = F.binary_cross_entropy(fake_pred, fake_gt)

    disc_loss = real_loss + fake_loss

    return disc_loss

def gen_loss(configs: dict, discriminator:Callable, generator:Callable, epoch_num: int, real: Tensor) -> Tensor:
    # Train with fake
    fake_pred = get_fake_preds(configs, discriminator, generator, epoch_num, real)
    fake_gt = torch.ones_like(fake_pred)
    gen_loss = F.binary_cross_entropy(fake_pred, fake_gt)

    return gen_loss
