import torch
import warnings
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
        self.model_name = model.name
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
    
    def load_model(self):
        try:
            self.load_state_dict(torch.load(f"{self.model.name}_celeba_conv.ckpt"))
        except  FileNotFoundError:
            print(f"Please train the model using python run.py -c ./configs/{self.model.name}.yaml")

class VAE2stageModule(LightningModule):
    """
    Standard lightning training code.
    """

    def __init__(
        self,
        stage1,
        stage2,
        lr: float = 1e-3,
        latent_dim: int = 256
    ):

        super(VAE2stageModule, self).__init__()

        self.lr = lr
        self.stage1 = stage1
        self.stage2 = stage2
        self.model_name = stage2.model_name
        self.latent_dim = latent_dim

    @staticmethod
    def load_model_from_checkpoint(vae):
        try:
            vae.load_state_dict(torch.load(f"{vae.model.name}_celeba_conv.ckpt"))
        except  FileNotFoundError:
            print(f"Please train the model using python run.py -c ./configs/{vae.model.name}.yaml") 

    def load_model(self):
        self.load_model_from_checkpoint(self.stage1)
        self.load_model_from_checkpoint(self.stage2)

    def forward(self, x):
        u = self.stoch_encode(x)
        return self.decode(u)
    
    def det_encode(self, x):
        x = x.to(self.device)
        u = self.stage2.det_encode(self.stage1.det_encode(x))
        return u

    def stoch_encode(self, x):
        x = x.to(self.device)
        u = self.stage2.stoch_encode(self.stage1.stoch_encode(x))
        return u

    def decode(self, u):
        return self.stage1.decode(self.stage2.decode(u))
    
    def get_samples(self, num):
        u = torch.randn(num, self.latent_dim)
        u = u.to(self.device)
        return self.decode(u)

class VQVAEModule(LightningModule):
    """
    Standard lightning training code.
    """

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        latent_dim: int = 256
    ):

        super(VQVAEModule, self).__init__()

        self.lr = lr
        self.model = model
        self.model_name = model.name
        self.latent_dim = latent_dim

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
    
    def encode(self, x):
        x = x.to(self.device)
        z = self.model.encoder(x)[0]
        return z

    def qunatize_encode(self, x):
        x = x.to(self.device)
        z = self.model.encoder(x)[0]
        quantized_inputs, _ = self.model.vq_layer(z)
        return quantized_inputs

    def decode(self, z):
        return self.model.decoder(z)
    
    def get_samples(self, num):
        # Warning these numbers are hardcoded for the default archiecture
        warnings.warn("Sampling does not work yet, we need to sample from a pixel cnn prior", RuntimeWarning, stacklevel=2)
        z = torch.randn(num, self.latent_dim, 16, 16)
        z = z.to(self.device)
        quantized_inputs, _ = self.model.vq_layer(z)
        return self.model.decoder(quantized_inputs)

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
    
    def load_model(self):
        try:
            self.load_state_dict(torch.load(f"{self.model.name}_celeba_conv.ckpt"))
        except  FileNotFoundError:
            print(f"Please train the model using python run.py -c ./configs/{self.model.name}.yaml")
