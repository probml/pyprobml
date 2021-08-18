import torch
from torch import nn, Tensor
from typing import Any, Callable
from pytorch_lightning import LightningModule


class GAN(LightningModule):
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
        name: str,
        generator: Callable, 
        discriminator: Callable,
        gen_loss: Callable,
        disc_loss: Callable,
        sampling: Callable,
        config: dict
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
        
        self.name = name
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = config["learning_rate"]
        self.beta1 = config["beta1"] 
        self.sampling = sampling
        self.gen_loss = lambda num, real : gen_loss(discriminator, generator, num, real)
        self.disc_loss = lambda num, real : disc_loss(discriminator, generator, num, real)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr = self.learning_rate
        betas = (self.beta1, 0.999)
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
        noise = noise.to(self.device)
        noise = noise.view(*noise.shape, 1, 1)
        if self.sampling is not None:
            noise = self.sampling(noise)
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
        disc_loss = self.disc_loss(self.trainer.current_epoch+1, real)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, real: Tensor) -> Tensor:
        gen_loss = self.gen_loss(self.trainer.current_epoch+1, real)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _get_noise(self, n_samples: int, latent_dim: int) -> Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)
    
    def load_model(self):
        try:
            self.load_state_dict(torch.load(f"{self.name}_celeba.ckpt"))
        except  FileNotFoundError:
            print(f"Please train the model using python run.py -c ./configs/{self.model.name}.yaml")
    