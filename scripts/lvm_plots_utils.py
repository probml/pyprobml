from typing import Callable, Tuple
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision.utils import make_grid
from scipy.stats import truncnorm
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_interpolation(interpolation):
  """
  interpolation: can accept either string or function
  """
  if interpolation=="spherical":
    return slerp
  elif interpolation=="linear":
    return lerp 
  elif callable(interpolation):
    return interpolation

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif torch.allclose(low, high):
        return low
    omega = torch.arccos(torch.dot(low/torch.norm(low), high/torch.norm(high)))
    so = torch.sin(omega)
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega)/so * high

def make_imrange(arr: list):
    interpolation = torch.stack(arr)
    imgs = rearrange(make_grid(interpolation,11), 'c h w -> h w c')
    imgs = imgs.cpu().detach().numpy() if torch.cuda.is_available() else imgs.detach().numpy()
    return imgs

def get_imrange(G:Callable[[torch.tensor], torch.tensor], start:torch.tensor,
               end:torch.tensor, nums:int=8, interpolation="spherical") -> torch.tensor:
    """
    Decoder must produce a 3d vector to be appened togther to form a new grid
    """
    val = 0 
    arr2 = []
    inter = get_interpolation(interpolation)
    for val in torch.linspace(0, 1, nums):
        new_z = torch.unsqueeze(inter(val, start, end),0)
        arr2.append(G(new_z))
    return make_imrange(arr2) 

def get_random_samples(decoder: Callable[[torch.tensor], torch.tensor], truncation_threshold=1, latent_dim=20) -> torch.tensor:
  """
  Decoder must produce a 4d vector to be feed into make_grid
  """
  values = truncnorm.rvs(-truncation_threshold, truncation_threshold, size=(64, latent_dim))
  z = torch.from_numpy(values).float()
  z = z.to(device)
  imgs = rearrange(make_grid(decoder(z)), 'c h w -> h w c').cpu().detach().numpy()
  return imgs

def get_grid_samples(decoder:Callable[[torch.tensor], torch.tensor], latent_size:int = 2, size:int=10, max_z:float = 3.1) -> torch.tensor:
    """
    Decoder must produce a 3d vector to be appened togther to form a new grid
    """
    arr = []
    for i in range(0, size):
        z1 = (((i / (size-1)) * max_z)*2) - max_z
        for j in range(0, size):
            z2 = (((j / (size-1)) * max_z)*2) - max_z
            z_ = torch.tensor([[z1, z2]+(latent_size-2)*[0]], device=device)
            decoded = decoder(z_)
            arr.append(decoded)
    return torch.stack(arr)

def plot_embeddings_tsne(batch: Tuple, encoder:Callable[[torch.tensor], torch.tensor]):
  """
  Given a batch and an encoder it plots TSNE embedding of the latent space
  """
  X_data, y_data = batch
  X_data = X_data.to(device)
  np.random.seed(42)
  tsne = TSNE()
  X_data_2D = tsne.fit_transform(encoder(X_data).cpu().detach().numpy())
  X_data_2D = (X_data_2D - X_data_2D.min()) / (X_data_2D.max() - X_data_2D.min())

  # adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
  plt.figure(figsize=(10, 8))
  cmap = plt.cm.tab10
  plt.scatter(X_data_2D[:, 0], X_data_2D[:, 1], c=y_data, s=10, cmap=cmap)
  image_positions = np.array([[1., 1.]])
  for index, position in enumerate(X_data_2D):
      dist = np.sum((position - image_positions) ** 2, axis=1)
      if np.min(dist) > 0.02: # if far enough from other images
          image_positions = np.r_[image_positions, [position]]
          imagebox = matplotlib.offsetbox.AnnotationBbox(
              matplotlib.offsetbox.OffsetImage(rearrange(X_data[index].cpu(), "c h w -> (c h) w"), cmap="binary"),
              position, bboxprops={"edgecolor": tuple(cmap([y_data[index]])[0]), "lw": 2})
          plt.gca().add_artist(imagebox)
  plt.axis("off")
