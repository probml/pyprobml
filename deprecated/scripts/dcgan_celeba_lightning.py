# -*- coding: utf-8 -*-
"""
Author: Ang Ming Liang

Please run the following command before running the script

wget -q https://raw.githubusercontent.com/sayantanauddy/vae_lightning/main/data.py
or curl https://raw.githubusercontent.com/sayantanauddy/vae_lightning/main/data.py > data.py

Then, make sure to get your kaggle.json from kaggle.com then run 

mkdir /root/.kaggle 
cp kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
rm kaggle.json

to copy kaggle.json into a folder first 
"""

import superimport

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import  LightningModule, Trainer
from einops import rearrange
from tqdm import tqdm
from data import CelebADataset,  CelebADataModule
from torch import Tensor
from argparse import ArgumentParser
from typing import Any, Optional
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from argparse import ArgumentParser

class DCGANGenerator(nn.Module):

    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
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
        use_relu: bool = False
    ) -> nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Relu() if use_relu else nn.Mish(),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return gen_block

    def forward(self, noise: Tensor) -> Tensor:
        return self.gen(noise)


class DCGANDiscriminator(nn.Module):

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

class DCGAN(LightningModule):
    """
    DCGAN implementation.
    Example::
        from pl_bolts.models.gans import DCGAN
        m = DCGAN()
        Trainer(gpus=2).fit(m)
    Example CLI::
        # mnist
        python dcgan_module.py --gpus 1
        # cifar10
        python dcgan_module.py --gpus 1 --dataset cifar10 --image_channels 3
    """

    def __init__(
        self,
        beta1: float = 0.5,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        image_channels: int = 3,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        topk: Optional[int] = 144,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            beta1: Beta1 value for Adam optimizer
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            image_channels: Number of channels of the images from the dataset
            latent_dim: Dimension of the latent space
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

        self.criterion = nn.BCELoss()
        self.noise_factor=0
        self.topk= topk

    def _get_generator(self) -> nn.Module:
        generator = DCGANGenerator(self.hparams.latent_dim, self.hparams.feature_maps_gen, self.hparams.image_channels)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self) -> nn.Module:
        discriminator = DCGANDiscriminator(self.hparams.feature_maps_disc, self.hparams.image_channels)
        discriminator.apply(self._weights_init)
        return discriminator

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, 0.999)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, noise: Tensor) -> Tensor:
        """
        Generates an image given input noise
        Example::
            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

    def _disc_step(self, real: Tensor) -> Tensor:
        disc_loss = self._get_disc_loss(real)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, real: Tensor) -> Tensor:
        gen_loss = self._get_gen_loss(real)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _get_disc_loss(self, real: Tensor, smooth=0) -> Tensor:
        # Train with real
        real = real + self.noise_factor*torch.rand_like(real)
        real_pred = self.discriminator(real)
        real_gt = smooth*torch.rand_like(real_pred)+(1-smooth)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = smooth*torch.rand_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_loss(self, real: Tensor) -> Tensor:
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        topk_predictions = torch.topk( fake_pred , self.topk )[0]
        fake_gt = torch.ones_like(topk_predictions)
        gen_loss = self.criterion(topk_predictions, fake_gt)

        return gen_loss

    def _get_fake_pred(self, real: Tensor) -> Tensor:
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise) 
        fake = fake + self.noise_factor*torch.rand_like(real)
        fake_pred = self.discriminator(fake)

        return fake_pred

    def _get_noise(self, n_samples: int, latent_dim: int) -> Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--latent_dim", default=100, type=int)
        parser.add_argument("--learning_rate", default=0.0002, type=float)
        parser.add_argument("--topk", default=10, type=float)
        return parser

def plt_image_generated(m, size, threshold=1, fname="generated.png"):
    plt.figure(figsize=(size,size))
    values = truncnorm.rvs(-threshold, threshold, size=(64, 100))
    z = torch.from_numpy(values).float()
    imgs = rearrange(make_grid(m(z)), 'c h w -> h w c').detach().numpy()
    plt.imshow(imgs)
    plt.savefig(fname)

def test_scaling(dm):
    # Making sure the scalling is between 0-1
    for batch in tqdm(dm.train_dataloader()):
        x, y = batch
        assert 1 >= x.max()
        assert 0 <= x.min()
        assert torch.any(x < 1)
        assert torch.any(x > 0)

def ewa(
        averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: torch.LongTensor
    , smooth=0.9) -> torch.FloatTensor:
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97
        """
        alpha = smooth/ (num_averaged + 1)
        return averaged_model_parameter*(1-alpha) + model_parameter * alpha
        
if __name__ == "__main__":
    parser = ArgumentParser(description='Hyperparameters for our experiments')
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--image-size', type=int, default=64, help="Image size")
    parser.add_argument('--crop-size', type=int, default=128, help="Crop size")
    parser.add_argument('--bs', type=int, default=144, help="batch size")
    parser.add_argument('--data-path', type=str, default="kaggle", help="batch size") 
    parser.add_argument('--gpus', type=int, default=1, help="gpu use") 
    parser.add_argument('--epochs', type=int, default=50, help="Num of epochs") 
    parser = DCGAN.add_model_specific_args(parser)

    # Hyperparameters
    hparams = parser.parse_args()
    
    SEED = hparams.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False

    IMAGE_SIZE = hparams.image_size
    BATCH_SIZE = hparams.bs
    CROP = hparams.crop_size
    DATA_PATH = hparams.data_path

    trans = []
    trans.append(transforms.RandomHorizontalFlip())
    if CROP > 0:
        trans.append(transforms.CenterCrop(CROP))
    trans.append(transforms.Resize(IMAGE_SIZE))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    ds = CelebADataset(root='kaggle', split='test', target_type='attr', download=True)
    dm = CelebADataModule(data_dir=DATA_PATH,
                                target_type='attr',
                                train_transform=transform,
                                val_transform=transform,
                                download=True,
                                batch_size=BATCH_SIZE,
                                num_workers=1)

    dm.prepare_data() # force download now
    dm.setup() # force make data loaders now

    m = DCGAN()
    checkpoint_callback = ModelCheckpoint(monitor='loss/gen_epoch',
                                      dirpath='./checkpoints',
                                      filename='sample-celeba-{epoch:02d}-{gan_loss:.2f}',
                                      save_top_k=3)
    runner = Trainer(
                    logger=None,
                    gpus = hparams.gpus,
                    max_epochs = hparams.epochs,
                    callbacks=[checkpoint_callback])
    runner.fit(m, datamodule=dm)

    torch.save(m.state_dict(), "dcgan.ckpt")
    plt_image_generated(m, 10)
