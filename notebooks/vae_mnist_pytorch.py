# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/vae_mnist_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="VtUJ8nKlUrbG"
# # (Variational) autoencoders with CNNs on MNIST  using PyTorch
#
# Based on https://github.com/probml/pyprobml/blob/master/notebooks/vae_celeba_tf.ipynb. Translated to PyTorch by always-newbie161@ and murphyk@.
#
# We use a 2d latent space so we can visualize things easily.

# + [markdown] id="6cJfLsEHbZzf"
#
# # Setup

# + id="zeN7qXGPbZzf"
import os
import time
import numpy as np
np.random.seed(0)
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

import sklearn
from sklearn import metrics

import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows


# + id="7vDwd4W0bZzf"
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision.utils import make_grid
from random import randint
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + id="BvFtt39Tt7kP"
# !mkdir ./outputs

# + id="84XrFmfSVZM3"
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/vae_helpers_pytorch.py
from vae_helpers_pytorch import *

# + [markdown] id="L6gR8D0au2Iu"
# # Load Mnist

# + id="0_y-ySIVu5Re" colab={"base_uri": "https://localhost:8080/", "height": 760, "referenced_widgets": ["bd624d5d460b4feba8297333504409cc", "45c4ef47801a4949ae4b8f3b75aad104", "0675bee07a004767b7f402ae55251a16", "202e00f680384417adc399e676a1dc7a", "49c16223556b4a039064c8a0f897da87", "b1323884ede247d586ae4a04d6340db7", "7764ef6754be429fb62a50c6b62b205a", "1860c1f6fa7248a0b0dacac1f51cbfdf", "5438a1c25a5c4a2ca4047a8120517770", "b56e5296200249fcbaa9c01f5b0fba15", "6ad2fd6491854eff93b37c3047a7af40", "80fc414b17f64169b34d7c9f49dc43c6", "f611d9a969154d308eb13ddc4524314f", "410258f9cabb438a8f05f9b1797d8152", "8c468065419541658281657c27b8880f", "abaa0e5482b74593ba9e07ca8230103a", "de9f590ceed7402abc7bc2359713e0f1", "e893ffc2b92549ffa61100d759ee64df", "aa69cde0da0141918b1b63a3fdfd5bf1", "96458f5998ae4a1eb223e166ad3b0dde", "d30c31ce86bf453daccc94b27939467f", "0f49fbcc2fdf47c785fef6650c7ddb34", "f36510af78b74779b2c7bd503cc1dc9c", "cb2c0fecdba1418e9205385272e849dc", "30720691c9e24fcb8b2dcd776b159b44", "4f309d80893f41dbb66245e538dd2dfa", "9af95185c0034c3f80d3feab51b4cf07", "ee18a0666abd4849b73be3decfb9e5b3", "58a99765b38743f08b9b7804dd06f698", "1627318246c947bf9fc48c710160588a", "8ed42b43a2e14d6c9063a19130cfcf6b", "126b33c875b94852932c0f108b1b989c"]} outputId="78281ad0-b4eb-4285-9449-adf8b49bec87"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
])
# training set and train data loader
mnist_trainset = torchvision.datasets.MNIST(
    root='../input', train=True, download=True, transform=transform
)
mnist_trainloader = DataLoader(
    mnist_trainset, batch_size=32, shuffle=True
)
# validation set and validation data loader
mnist_testset = torchvision.datasets.MNIST(
    root='../input', train=False, download=True, transform=transform
)
mnist_testloader = DataLoader(
    mnist_testset, batch_size=32, shuffle=False
)

# + [markdown] id="e7ipXpm-bj1t"
# # Model

# + colab={"base_uri": "https://localhost:8080/"} id="U_lS2fjVDGpS" outputId="1552cda0-8a9e-40a8-fa5e-ee6477b00b60"
encoder_conv_filters=[32,64,64, 64]
decoder_conv_t_filters=[64,64,32,1]
Xsmall = next(iter(mnist_trainloader))[0][0,:]
models_mnist = {}
models_mnist['2d-det'] = ConvVAE(Xsmall.shape,encoder_conv_filters,decoder_conv_t_filters,
                                 2,'2d-det',device, deterministic=True).to(device)
models_mnist['2d-stoch'] = ConvVAE(Xsmall.shape,encoder_conv_filters,decoder_conv_t_filters,
                                   2,'2d-stoch',device).to(device)
summary(models_mnist['2d-det'],Xsmall.shape)



# + [markdown] id="vHgljMbKoh6l"
# # Training

# + colab={"base_uri": "https://localhost:8080/"} id="Udh9tiRZW-7D" outputId="652be055-d1e4-407d-82bf-c58d26bf7cbc"
images_to_recon = next(iter(mnist_trainloader)) 
print(images_to_recon[0].shape)
print(images_to_recon[1].shape)

# + id="4XeWma8zVkyT"
x, y = next(iter(mnist_trainloader)) # first batch
images_to_recon = x
nimages = 6
images_to_recon = images_to_recon[:nimages,:,:,:]
display_every_n_epochs = 1

def callback(model, epoch, train_epoch_loss, valid_epoch_loss):
  #display.clear_output(wait=False) # don't erase old outputs
  print(f"Training Losses on epoch {epoch}"); print(train_epoch_loss)
  print(f"Validation Losses on epoch {epoch}"); print(valid_epoch_loss)
  if epoch % display_every_n_epochs == 0:
    samples = model.generate_images(nimages)
    recon = model.reconstruct_images(images_to_recon)
    show_images(samples, ttl='samples')
    show_images(images_to_recon, ttl='input')
    show_images(recon, ttl='reconstruction')
    plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 375} id="kZ8YNy0qVukd" outputId="c4aea717-fd55-4654-fc55-843769096292"
epochs = 2
lr = 0.001
optimizer = torch.optim.Adam(models_mnist['2d-det'].parameters(), lr=lr)
train_loss, valid_loss = training(models_mnist['2d-det'], mnist_trainloader, mnist_testloader,
                     optimizer, epochs, callback=callback)
fig, axs = plot_loss_histories(train_loss, valid_loss)
plt.show()

# + [markdown] id="_jQBcYoIzzr8"
# # 2d Embeddings for mnist

# + colab={"base_uri": "https://localhost:8080/"} id="XCwpdZRI0vNd" outputId="73dd9e24-e483-435b-d92f-af65845ad7a1"
N = int(len(mnist_testset))
images_mnist = torch.zeros(N,1,32,32)
labels_mnist = torch.zeros(N)
bs = mnist_testloader.batch_size
c = 0
for i,b in enumerate(mnist_testloader):
 images_mnist[i*bs:(i+1)*bs,:,:,:] = b[0]
 labels_mnist[i*bs:(i+1)*bs] = b[1]

print(images_mnist.shape)
print(labels_mnist.shape)


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="9LgEtsZXYul4" outputId="b5e04cbe-2ace-486a-b652-cf3bb2ff929c"
# generate images from random points in latent space
def sample_from_embeddings(model, n_to_show=5000):
  figsize = 8
  np.random.seed(42)
  example_idx = np.random.choice(range(N), n_to_show)
  example_images = images_mnist[example_idx].to(device)
  example_labels = labels_mnist[example_idx]
  z_points,_ = model.encode(example_images)
  z_points = z_points.cpu().detach().numpy()

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
  #x = np.random.normal(size = grid_size * grid_depth)
  #y = np.random.normal(size = grid_size * grid_depth)
  z_grid = np.array(list(zip(x, y)))
  t_z_grid = torch.FloatTensor(z_grid).to(device)
  reconst = model.decode(t_z_grid)
  reconst = reconst.cpu().detach()
  plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'red', alpha=1, s=20)
  n = np.shape(z_grid)[0]
  for i in range(n):
    x = z_grid[i,0]
    y = z_grid[i,1]
    plt.text(x, y, i)
  plt.title(model.model_name)
  plt.show()

  fig = plt.figure(figsize=(figsize, grid_depth))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)

  for i in range(grid_size*grid_depth):
      ax = fig.add_subplot(grid_depth, grid_size, i+1)
      ax.axis('off')
      #ax.text(0.5, -0.35, str(np.round(z_grid[i],1)), fontsize=8, ha='center', transform=ax.transAxes)
      ax.text(0.5, -0.35, str(i))
      ax.imshow(reconst[i,:][0, :, :],cmap = 'Greys')

sample_from_embeddings(models_mnist['2d-det'])
sample_from_embeddings(models_mnist['2d-stoch']) 

# + colab={"base_uri": "https://localhost:8080/", "height": 987} id="SUcNoSwHY520" outputId="08d29e03-2e03-4830-96b6-4fa29c9e7cee"
# color code latent points
from scipy.stats import norm

def show_2d_embeddings(model, n_to_show=5000, use_cdf=False):
  figsize = 8

  np.random.seed(42)
  example_idx = np.random.choice(range(N), n_to_show)
  example_images = images_mnist[example_idx].to(device)
  example_labels = labels_mnist[example_idx]

  z_points,_ = model.encode(example_images)
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
  plt.title(model.model_name)
  plt.show()

show_2d_embeddings(models_mnist['2d-det'])
show_2d_embeddings(models_mnist['2d-stoch'])


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="HN4yUvRIZPrU" outputId="795da4ee-84ac-4889-8ca7-97b7a1abb6d1"
# Generate images from 2d grid

def generate_from_2d_grid(model):
  n_to_show = 5000 #500
  grid_size = 10
  figsize = 8
  np.random.seed(0)

  np.random.seed(42)
  example_idx = np.random.choice(range(N), n_to_show)
  example_images = images_mnist[example_idx].to(device)
  example_labels = labels_mnist[example_idx]
  z_points,_ = model.encode(example_images)
  z_points = z_points.cpu().detach().numpy()

  plt.figure(figsize=(figsize, figsize))
  plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
              , alpha=0.5, s=2)
  plt.colorbar()

  x_min = np.min(z_points[:,0]); x_max = np.max(z_points[:,0]);
  y_min = np.min(z_points[:,1]); y_max = np.max(z_points[:,1]);
  x = np.linspace(x_min, x_max, grid_size)
  y = np.linspace(y_min, y_max, grid_size)
  #x = norm.ppf(np.linspace(0.01, 0.99, grid_size))
  #y = norm.ppf(np.linspace(0.01, 0.99, grid_size))
  xv, yv = np.meshgrid(x, y)
  xv = xv.flatten()
  yv = yv.flatten()
  z_grid = np.array(list(zip(xv, yv)))
  t_z_grid = torch.FloatTensor(z_grid).to(device)
  reconst = model.decode(t_z_grid).cpu().detach()

  plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'black'#, cmap='rainbow' , c= example_labels
              , alpha=1, s=5)
  plt.title(model.model_name)
  plt.show()

  fig = plt.figure(figsize=(figsize, figsize))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
  for i in range(grid_size**2):
      ax = fig.add_subplot(grid_size, grid_size, i+1)
      ax.axis('off')
      ax.imshow(reconst[i,:][0, :, :], cmap = 'Greys')
  plt.show()

generate_from_2d_grid(models_mnist['2d-det'])
generate_from_2d_grid(models_mnist['2d-stoch'])

