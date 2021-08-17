import umap
from typing import Callable, Tuple
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision.utils import make_grid
from scipy.stats import truncnorm
from scipy.stats import norm
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

def get_embedder(encoder, X_data, y_data=None, use_embedder="TSNE"):
  X_data_2D = encoder(X_data)
  if X_data_2D.shape[-1] == 2:
    return X_data_2D
  if use_embedder=="UMAP":
    umap_fn = umap.UMAP()
    X_data_2D = umap_fn.fit_transform(X_data_2D.cpu().detach().numpy(), y_data)
  elif use_embedder=="TSNE":
    tsne = TSNE()
    X_data_2D = tsne.fit_transform(X_data_2D.cpu().detach().numpy())
  return X_data_2D

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

def get_random_samples(decoder: Callable[[torch.tensor], torch.tensor],
                       truncation_threshold=1, latent_dim=20, num_images=64,
                       num_images_per_row=8) -> torch.tensor:
  """
  Decoder must produce a 4d vector to be feed into make_grid
  """
  values = truncnorm.rvs(-truncation_threshold, truncation_threshold, size=(num_images, latent_dim))
  z = torch.from_numpy(values).float()
  z = z.to(device)
  imgs = rearrange(make_grid(decoder(z), num_images_per_row), 'c h w -> h w c').cpu().detach().numpy()
  return imgs

def get_grid_samples(decoder:Callable[[torch.tensor], torch.tensor],
                     latent_size:int = 2, size:int=10, max_z:float = 3.1) -> torch.tensor:
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
  
def plot_scatter_plot(batch, encoder, use_embedder="TSNE", min_distance =0.03):
  """
  Plots scatter plot of embeddings
  """
  X_data, y_data = batch
  X_data = X_data.to(device)
  np.random.seed(42)
  X_data_2D = get_embedder(encoder, X_data, y_data, use_embedder)
  X_data_2D = (X_data_2D - X_data_2D.min()) / (X_data_2D.max() - X_data_2D.min())

  # adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
  fig = plt.figure(figsize=(10, 8))
  cmap = plt.cm.tab10
  plt.scatter(X_data_2D[:, 0], X_data_2D[:, 1], c=y_data, s=10, cmap=cmap)
  image_positions = np.array([[1., 1.]])
  for index, position in enumerate(X_data_2D):
      dist = np.sum((position - image_positions) ** 2, axis=1)
      if np.min(dist) > 0.04: # if far enough from other images
          image_positions = np.r_[image_positions, [position]]
          if X_data[index].shape[0] == 3: 
            imagebox = matplotlib.offsetbox.AnnotationBbox(
              matplotlib.offsetbox.OffsetImage(rearrange(X_data[index].cpu(), "c h w -> h w c"), cmap="binary"),
              position, bboxprops={"edgecolor": tuple(cmap([y_data[index]])[0]), "lw": 2})
          elif X_data[index].shape[0] == 1: 
            imagebox = matplotlib.offsetbox.AnnotationBbox(
              matplotlib.offsetbox.OffsetImage(rearrange(X_data[index].cpu(), "c h w -> (c h) w"), cmap="binary"),
              position, bboxprops={"edgecolor": tuple(cmap([y_data[index]])[0]), "lw": 2})
          plt.gca().add_artist(imagebox)
  plt.axis("off")
  return fig

def plot_grid_plot(batch, encoder, use_cdf=False, use_embedder="TSNE", model_name="VAE mnist"):
    """
    This takes in images in batch, so G should produce a 3D tensor output example
    for a model that outputs images with a channel dim along with a batch dim we need 
    to rearrange the tensor as such to produce the correct shape
    def decoder(z):
        return rearrange(m.decode(z), "b c h w -> b (c h) w")
    """
    figsize = 8
    example_images, example_labels  = batch
    example_images = example_images.to(device=device)

    z_points = get_embedder(encoder, example_images, use_embedder=use_embedder)
    p_points = norm.cdf(z_points)

    fig = plt.figure(figsize=(figsize, figsize))
    if use_cdf:
        plt.scatter(p_points[:, 0] , p_points[:, 1] , cmap='rainbow' , c= example_labels
                , alpha=0.5, s=5)
    else:
        plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
                , alpha=0.5, s=2)
    plt.colorbar()
    plt.title(f"{model_name} embedding")
    return fig

def plot_grid_plot_with_sample(batch, encoder, decoder, use_embedder="TSNE", model_name="VAE mnist"):
    """
    This takes in images in batch, so G should produce a 3D tensor output example
    for a model that outputs images with a channel dim along with a batch dim we need 
    to rearrange the tensor as such to produce the correct shape
    def decoder(z):
        return rearrange(m.decode(z), "b c h w -> b (c h) w")
    """
    figsize = 8
    example_images, example_labels  = batch
    example_images = example_images.to(device=device)
    
    z_points = get_embedder(encoder, example_images, use_embedder=use_embedder)
    plt.figure(figsize=(figsize, figsize))
    #plt.scatter(z_points[:, 0] , z_points[:, 1], c='black', alpha=0.5, s=2)
    plt.scatter(z_points[:, 0] , z_points[:, 1], cmap='rainbow', c=example_labels, alpha=0.5, s=2)
    plt.colorbar()

    grid_size = 15
    grid_depth = 2
    np.random.seed(42)
    x_min = np.min(z_points[:,0]); x_max = np.max(z_points[:,0]);
    y_min = np.min(z_points[:,1]); y_max = np.max(z_points[:,1]);
    x = np.random.uniform(low=x_min, high=x_max, size=grid_size*grid_depth)
    y = np.random.uniform(low=y_min, high=y_max, size=grid_size*grid_depth)

    z_grid = np.array(list(zip(x, y)))
    t_z_grid = torch.FloatTensor(z_grid).to(device)
    reconst = decoder(t_z_grid)
    reconst = reconst.cpu().detach() if torch.cuda.is_available() else reconst.detach()
    plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'red', alpha=1, s=20)
    n = np.shape(z_grid)[0]
    for i in range(n):
        x = z_grid[i,0]
        y = z_grid[i,1]
        plt.text(x, y, i)
    plt.title(f"{model_name} embedding with samples")

    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_size*grid_depth):
        ax = fig.add_subplot(grid_depth, grid_size, i+1)
        ax.axis('off')
        #ax.text(0.5, -0.35, str(np.round(z_grid[i],1)), fontsize=8, ha='center', transform=ax.transAxes)
        ax.text(0.5, -0.35, str(i))
        ax.imshow(reconst[i,:],cmap = 'Greys')
