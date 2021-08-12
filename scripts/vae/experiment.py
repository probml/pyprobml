import torch
import warnings
from pytorch_lightning import LightningModule
from torch.nn import functional as F


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
        warnings.warn("Sampling does not work yet, we need to sample from a pixel cnn prior", RuntimeWarning,
                      stacklevel=2)
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
        except FileNotFoundError:
            print(f"Please train the model using python run.py -c ./configs/{self.model.name}.yaml")


class PixelCNNModule(LightningModule):

    def __init__(self, pixel_cnn, vq_vae, height=None, width=None, lr=1e-3):
        super().__init__()

        self.model = pixel_cnn
        self.encoder = vq_vae.model.encoder
        self.decoder = vq_vae.model.decoder
        self.vector_quantizer = vq_vae.model.vq_layer
        self.lr = lr
        self.height, self.width = height, width
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        encoding_one_hot = F.one_hot(x, num_classes=self.vector_quantizer.K)
        encoding_one_hot = encoding_one_hot.view(-1, self.height, self.width, self.vector_quantizer.K).permute(0, 3, 1,
                                                                                                               2).float()
        output = self.model(encoding_one_hot)  # 256x512x16x16
        return output

    def training_step(self, batch, batch_idx):
        x_train, _ = batch  # 256x3x16x16
        with torch.no_grad():
            encoded_outputs = self.encoder(x_train)[0]  # go through encoder
            codebook_indices = self.vector_quantizer.get_codebook_indices(encoded_outputs)  # BHW X 1

        output = self(codebook_indices)
        loss = self.loss_fn(output, codebook_indices.view(-1, self.height, self.width))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def sample(self, inputs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        with torch.no_grad():
            inputs_ohe = F.one_hot(inputs.long(), num_classes=self.vector_quantizer.K).to(device)
            inputs_ohe = inputs_ohe.view(-1, self.height, self.width, self.vector_quantizer.K).permute(0, 3, 1,
                                                                                                       2).float()
            x = self.model(inputs_ohe)
            dist = torch.distributions.Categorical(logits=x.permute(0, 2, 3, 1))
            sampled = dist.sample()

        self.model.train()
        return sampled

    def get_priors(self, batch):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        priors = torch.zeros(size=(batch,) + (1, self.height, self.width), device=device)
        # Iterate over the priors because generation has to be done sequentially pixel by pixel.
        for row in range(self.height):
            for col in range(self.width):
                # Feed the whole array and retrieving the pixel value probabilities for the next
                # pixel.
                probs = self.sample(priors.view((-1, 1)))
                # Use the probabilities to pick pixel values and append the values to the priors.
                priors[:, 0, row, col] = probs[:, row, col]

        priors = priors.squeeze()
        return priors

    def generate_samples_from_priors(self, priors):
        priors = priors.to("cpu")
        priors_ohe = F.one_hot(priors.view(-1, 1).long(), num_classes=self.vector_quantizer.K).squeeze().float()
        quantized = torch.matmul(priors_ohe, self.vector_quantizer.embedding.weight)  # [BHW, D]
        quantized = quantized.view(-1, self.height, self.width, self.vector_quantizer.D).permute(0, 3, 1, 2)
        with torch.no_grad():
            return self.decoder(quantized)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path))

        except FileNotFoundError:
            print(f"Please train the model using python run.py -c ./configs/{self.model.name}.yaml")

    def save(self, path="./pixelcnn_model.ckpt"):
        torch.save(self.state_dict(), path)