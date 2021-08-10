import torch
import matplotlib.pyplot as plt
from einops import rearrange
from scipy.stats import truncnorm
from torchvision.utils import make_grid

def sample_from_truncated_normal(gans, num, threshold=1, num_images_per_row=8, figsize=(10,10)):
    values = truncnorm.rvs(-threshold, threshold, size=(num, gans[0].generator.latent_dim))
    z = torch.from_numpy(values).float()
    for gan in gans:
        plotting(gan, z, num_images_per_row, figsize=figsize)

def plotting(gan, z, num_row=8, figsize=(10,10)):
    imgs = rearrange(make_grid(gan(z), num_row), 'c h w -> h w c').cpu().detach().numpy()
    plt.figure(figsize=figsize)
    plt.imshow(imgs)
    plt.title(f"{gan.name}")
    plt.savefig(f"{gan.name}.png")
    plt.show()

