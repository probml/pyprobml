
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Callable

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def get_codebook_indices(self, latents:Tensor) -> Tensor:
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        return encoding_inds

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        encoding_inds = self.get_codebook_indices(latents)

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.nn.functional.one_hot(encoding_inds, num_classes=self.K).float().to(device)

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)

class Encoder(nn.Module):

  def __init__(self, 
                in_channels: int = 3, 
                hidden_dims: Optional[list] = None,
                latent_dim: int = 256):
    super(Encoder, self).__init__()

    modules = []
    if hidden_dims is None:
        hidden_dims = [128, 256]

    # Build Encoder
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU())
        )
        in_channels = h_dim

    modules.append(
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                        kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
    )

    for _ in range(6):
        modules.append(ResidualLayer(in_channels, in_channels))
    modules.append(nn.LeakyReLU())

    modules.append(
        nn.Sequential(
            nn.Conv2d(in_channels, latent_dim,
                        kernel_size=1, stride=1),
            nn.LeakyReLU())
    )

    self.encoder = nn.Sequential(*modules)

  def forward(self, x):
    result = self.encoder(x)
    return [result]

class Decoder(nn.Module):

  def __init__(self,
               hidden_dims: Optional[list] = None,
               latent_dim: int = 256):
    super(Decoder, self).__init__()

    modules = []

    if hidden_dims is None:
        hidden_dims = [128, 256]

    modules.append(
        nn.Sequential(
            nn.Conv2d(latent_dim,
                        hidden_dims[-1],
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.LeakyReLU())
    )

    for _ in range(6):
        modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

    modules.append(nn.LeakyReLU())

    hidden_dims.reverse()

    for i in range(len(hidden_dims) - 1):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),
                nn.LeakyReLU())
        )

    modules.append(
        nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                                out_channels=3,
                                kernel_size=4,
                                stride=2, padding=1),
            nn.Tanh()))

    self.decoder = nn.Sequential(*modules)

  def forward(self, z):
    result = self.decoder(z)
    return result

def loss(config, x_hat, x, vq_loss):

    recons_loss = F.mse_loss(x_hat, x)

    loss = recons_loss + vq_loss
    return loss

class VQVAE(nn.Module):

    def __init__(self,
                 name: str, 
                 loss: Callable, 
                 encoder: Callable,
                 decoder: Callable,
                 config: dict) -> None:
        super(VQVAE, self).__init__()

        self.name = name
        self.loss = loss
        self.encoder = encoder
        self.decoder = decoder
        self.vq_layer = VectorQuantizer(config["num_embeddings"],
                                        config["embedding_dim"],
                                        config["beta"])

    def forward(self, x: Tensor):
        encoding = self.encoder(x)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return self.decoder(quantized_inputs)

    def _run_step(self, x: Tensor):
        encoding = self.encoder(x)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return self.decoder(quantized_inputs), x, vq_loss
    
    def compute_loss(self, x):
        x_hat, x, vq_loss = self._run_step(x)

        loss = self.loss(x_hat, x, vq_loss)

        return loss
