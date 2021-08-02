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
        lr: float = 1e-3
    ):

        super(VAEModule, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.model = model

    def forward(self, x):
        return self.model(x)

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