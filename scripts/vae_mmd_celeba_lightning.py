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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from data import  CelebADataModule
from argparse import ArgumentParser
from einops import rearrange

IMAGE_SIZE = 64
CROP = 128
DATA_PATH = "kaggle"

trans = []
trans.append(transforms.RandomHorizontalFlip())
if CROP > 0:
  trans.append(transforms.CenterCrop(CROP))
trans.append(transforms.Resize(IMAGE_SIZE))
trans.append(transforms.ToTensor())
transform = transforms.Compose(trans)

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
    sigma = 2. * z_dim * latent_var

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

class VAE(LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        input_height: int,
        hidden_dims = None,
        in_channels = 3,
        enc_out_dim: int = 512,
        beta: float = 1,
        latent_dim: int = 256,
        lr: float = 1e-3
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.beta = beta
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

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

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

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

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def encode(self, x):
      x = self.encoder(x)
      x = torch.flatten(x, start_dim=1)
      mu = self.fc_mu(x)
      return mu

    def forward(self, x):
      z = self.encode(x)
      return self.decode(z)

    def _run_step(self, x):
      z = self.encode(x)

      return z, self.decode(z)

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat = self._run_step(x)
        
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        mmd = MMD(torch.randn_like(z), z)

        loss = recon_loss + \
               self.beta * mmd

        logs = {
            "recon_loss": recon_loss,
            "mmd": mmd,
        }
        return loss, logs

    def decode(self, z):
      result = self.decoder_input(z)
      result = result.view(-1, 512, 2, 2)
      result = self.decoder(result)
      result = self.final_layer(result)
      return result

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
    parser.add_argument('--latent-dim', type=int, default=128, help="size of latent dim for our vae")
    parser.add_argument('--epochs', type=int, default=50, help="num epochs")
    parser.add_argument('--gpus', type=int, default=1, help="gpus, if no gpu set to 0, to run on all  gpus set to -1")
    parser.add_argument('--bs', type=int, default=256, help="batch size")
    parser.add_argument('--beta', type=int, default=1, help="kl coeff aka beta term in the elbo loss function")
    parser.add_argument('--lr', type=int, default=1e-3, help="learning rate")
    hparams = parser.parse_args()
    
    m = VAE(input_height=IMAGE_SIZE, latent_dim=hparams.latent_dim, beta=hparams.beta, lr=hparams.lr)
    dm = CelebADataModule(data_dir=DATA_PATH,
                                target_type='attr',
                                train_transform=transform,
                                val_transform=transform,
                                download=True,
                                batch_size=hparams.bs)
    trainer = Trainer(gpus=1, weights_summary='full', max_epochs=10, auto_lr_find=True)

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(m, dm)

    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    m.lr = new_lr

    trainer= Trainer(gpus = hparams.gpus,
                    max_epochs = hparams.epochs)
    trainer.fit(m, datamodule=dm)
    torch.save(m.state_dict(), "mmd-vae-celeba-conv.ckpt")
