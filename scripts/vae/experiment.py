import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class VAEModule(LightningModule):
    """
    Standard lightning training code.
    """

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        latent_dim: int = 256
    ):

        super(VAEModule, self).__init__()

        self.lr = lr
        self.model = model
        self.latent_dim = latent_dim

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
    
    def det_encode(self, x):
        x = x.to(self.device)
        mu, _ = self.model.encoder(x)
        return mu

    def stoch_encode(self, x):
        x = x.to(self.device)
        mu, log_var = self.model.encoder(x)
        z = self.model.sample(mu, log_var)
        return z

    def decode(self, z):
        return self.model.decoder(z)
    
    def get_samples(self, num):
        z = torch.randn(num, self.latent_dim)
        z = z.to(self.device)
        return self.model.decoder(z)

    def step(self, batch, batch_idx):
        x, y = batch

        loss = self.model.compute_loss(x)

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
