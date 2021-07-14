import torch
import numpy as np
from scipy.stats import norm
import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from einops import rearrange
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_interpolation(interpolation):
  if interpolation=="spherical":
    return slerp
  elif interpolation=="linear":
    return lerp 

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def imrange(G, start, end, nums=8, interpolation="spherical"):
    """
    This takes in images sequentially, so G should produce a 2D tensor output example
    for a model that outputs images with a channel dim along with a batch dim we need 
    to rearrange the tensor as such to produce the correct shape
    def decoder(z):
        return rearrange(m.decode(z), "b c h w -> (b c h) w")
    """
    val = 0 
    arr2 = []
    inter = get_interpolation(interpolation)
    for val in torch.linspace(0, 1, nums):
        new_z = torch.unsqueeze(inter(val, start, end),0)
        arr2.append(G(new_z))
    return arr2 

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

def plot_embeddings_tsne(batch, encoder):
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
              position, bboxprops={"edgecolor": cmap([y_data[index]]), "lw": 2})
          plt.gca().add_artist(imagebox)
  plt.axis("off")
  
def plot_imrange(arr, figsize=(20,200), fname="interpolation.pdf"):
    plt.figure(figsize=figsize)
    interpolation = torch.unsqueeze(torch.stack(arr),1)
    imgs = rearrange(make_grid(interpolation,11), 'c h w -> h w c')
    imgs = imgs.cpu().detach().numpy() if torch.cuda.is_available() else imgs.detach().numpy()
    plt.imshow(imgs)
    pml.savefig(f"{fname}")
    plt.show()

def grid_sampling(decoder, size=10, max_z = 3.1, model_name="VAE mnist"):
    img_it = 0
    for i in range(0, size):
        z1 = (((i / (size-1)) * max_z)*2) - max_z
        for j in range(0, size):
            z2 = (((j / (size-1)) * max_z)*2) - max_z
            z_ = torch.tensor([[z1, z2]], device=device)
            decoded = decoder(z_).cpu().detach().numpy() if torch.cuda.is_available()else decoder(z_).detach().numpy()
            plt.subplot(size, size, 1 + img_it)
            img_it +=1
            plt.imshow(decoded, cmap = plt.cm.gray)
            plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
    pml.savefig(f'Samples from {model_name}.pdf')
    plt.show()

def show_2d_embeddings(batch, encoder, use_cdf=False, model_name="VAE mnist"):
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

    z_points = encoder(example_images)
    z_points = z_points.cpu().detach()
    p_points = norm.cdf(z_points)

    plt.figure(figsize=(figsize, figsize))
    if use_cdf:
        plt.scatter(p_points[:, 0] , p_points[:, 1] , cmap='rainbow' , c= example_labels
                , alpha=0.5, s=5)
    else:
        plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
                , alpha=0.5, s=2)
    plt.colorbar()
    plt.title(f"{model_name} embedding")
    pml.savefig(f'{model_name} embedding.pdf')
    plt.show()

def sample_from_embeddings(batch, encoder, decoder, model_name="VAE mnist"):
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
    
    z_points = encoder(example_images)
    z_points = z_points.cpu().detach().numpy() if torch.cuda.is_available() else z_points.detach().numpy()

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
    pml.savefig(f'{model_name} embedding with samples.pdf')
    plt.show()

    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size*grid_depth):
        ax = fig.add_subplot(grid_depth, grid_size, i+1)
        ax.axis('off')
        #ax.text(0.5, -0.35, str(np.round(z_grid[i],1)), fontsize=8, ha='center', transform=ax.transAxes)
        ax.text(0.5, -0.35, str(i))
        ax.imshow(reconst[i,:],cmap = 'Greys')
    pml.savefig(f'{model_name} images sampled')
    plt.show()
