# Library of pytorch functions related to convolutional variational autoencoders

import superimport

import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import imageio

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


### Model

# Convolutional Variational Autoencoder
class ConvVAE(nn.Module):
    def __init__(self, input_shape,
                 encoder_conv_filters,
                 decoder_conv_t_filters,
                 latent_dim,
                 model_name,
                 device,
                 kl_factor=1,
                 deterministic=False):
        super(ConvVAE, self).__init__()
        self.input_shape = input_shape

        self.latent_dim = latent_dim
        self.model_name = model_name
        self.deterministic = deterministic
        self.device = device
        self.kl_factor = 1

        all_channels = [self.input_shape[0]] + encoder_conv_filters

        self.enc_convs = nn.ModuleList([])

        # encoder_conv_layers
        for i in range(len(encoder_conv_filters)):
            self.enc_convs.append(nn.Conv2d(all_channels[i], all_channels[i + 1],
                                            kernel_size=3, stride=2, padding=1))
            if not self.latent_dim == 2:
                self.enc_convs.append(nn.BatchNorm2d(all_channels[i + 1]))
            self.enc_convs.append(nn.LeakyReLU())

        self.flatten_out_size = self.flatten_enc_out_shape(input_shape)

        if self.latent_dim == 2:
            self.mu_linear = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.mu_linear = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )

        if self.latent_dim == 2:
            self.log_var_linear = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.log_var_linear = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )

        if self.latent_dim == 2:
            self.decoder_linear = nn.Linear(self.latent_dim, self.flatten_out_size)
        else:
            self.decoder_linear = nn.Sequential(
                nn.Linear(self.latent_dim, self.flatten_out_size),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )

        all_t_channels = [encoder_conv_filters[-1]] + decoder_conv_t_filters

        self.dec_t_convs = nn.ModuleList([])

        num = len(decoder_conv_t_filters)

        # decoder_trans_conv_layers
        for i in range(num - 1):
            self.dec_t_convs.append(nn.UpsamplingNearest2d(scale_factor=2))
            self.dec_t_convs.append(nn.ConvTranspose2d(all_t_channels[i], all_t_channels[i + 1],
                                                       3, stride=1, padding=1))
            if not self.latent_dim == 2:
                self.dec_t_convs.append(nn.BatchNorm2d(all_t_channels[i + 1]))
            self.dec_t_convs.append(nn.LeakyReLU())

        self.dec_t_convs.append(nn.UpsamplingNearest2d(scale_factor=2))
        self.dec_t_convs.append(nn.ConvTranspose2d(all_t_channels[num - 1], all_t_channels[num],
                                                   3, stride=1, padding=1))
        self.dec_t_convs.append(nn.Sigmoid())

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std).to(self.device)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def flatten_enc_out_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        for l in self.enc_convs:
            x = l(x)
        self.shape_before_flattening = x.shape
        return int(np.prod(self.shape_before_flattening))

    def encode(self, x):
        for l in self.enc_convs:
            x = l(x)
        x = x.view(x.size()[0], -1)  # flatten
        mu = self.mu_linear(x)
        log_var = self.log_var_linear(x)
        return mu, log_var

    def decode(self, z):
        z = self.decoder_linear(z)
        recon = z.view(z.size()[0], *self.shape_before_flattening[1:])
        for l in self.dec_t_convs:
            recon = l(recon)
        return recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        if self.deterministic:
            return self.decode(mu), mu, None
        else:
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var

    def generate_images(self, num_images=6):
        noise = torch.normal(torch.zeros(num_images, self.latent_dim)).to(self.device)
        samples = self.decode(noise).cpu().detach()
        return samples

    def reconstruct_images(self, images):
        images = images.to(self.device)
        recon, _, _ = self.forward(images)
        recon = recon.cpu().detach()
        return recon


### Training


def train_one_epoch(model, dataloader, optimizer, is_train=True):
    if is_train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)
    running_loss_nll = 0.0
    running_loss_kl = 0.0
    running_loss = 0.0
    counter = 0  # counts batches in epoch
    criterion = nn.BCELoss(reduction='sum')
    for i, data in enumerate(dataloader):
        counter += 1
        data = data.to(model.device)
        if is_train:
            optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        nll = criterion(reconstruction, data)  # negative log likelihood
        if model.deterministic:
            loss = nll
        else:
            # KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = nll + model.kl_factor * kl_loss
            running_loss_kl += kl_loss.item()
        running_loss += loss.item()
        running_loss_nll += nll.item()
        if is_train:
            loss.backward()
            optimizer.step()

    losses = {'loss': running_loss / counter,
              'nll-loss': running_loss_nll / counter,
              'kl-loss': running_loss_kl / counter}
    return losses


def training(model, trainloader, testloader, optimizer, epochs, callback=None):
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        train_epoch_loss = train_one_epoch(model, trainloader, optimizer, is_train=True)
        valid_epoch_loss = train_one_epoch(model, testloader, optimizer, is_train=False)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        if callback: callback(model, epoch, train_epoch_loss, valid_epoch_loss)
    return train_loss, valid_loss


def save_model(model, where):
    dump = {'latent_dim': model.latent_dim,
            'input_shape': model.input_shape,
            'state_dict': model.state_dict()}
    torch.save(dump, where)


### Plotting

def list_of_dict_to_opposite(LD=None):
    # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    if LD is None:
        LD = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    DL = {k: [dic[k] for dic in LD] for k in LD[0]}
    return DL


def dict_of_list_to_opposite(DL=None):
    if DL is None:
        DL = {'a': [1, 3], 'b': [2, 4]}
    LD = [dict(zip(DL, t)) for t in zip(*DL.values())]
    return LD


def plot_loss_histories(train_loss, valid_loss):
    train_dict = list_of_dict_to_opposite(train_loss)
    valid_dict = list_of_dict_to_opposite(valid_loss)
    keys = list(train_dict.keys())
    nlosses = len(keys)
    colors = ['r', 'g', 'b', 'k']
    fig, axs = plt.subplots(ncols=nlosses, figsize=(10, 5))
    for i in range(nlosses):
        ax = axs[i]
        key = keys[i]
        ax.plot(train_dict[key], '-o', color=colors[i], label=f'train {key}')
        ax.plot(valid_dict[key], ':x', color=colors[i], label=f'valid {key}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
    return fig, axs


def show_images(images, max_images=6, ttl=''):
    if type(images) == torch.Tensor:
        images = images.cpu().detach().numpy()
    N, C, H, W = images.shape
    N = np.minimum(N, max_images)
    fig, axs = plt.subplots(1, N, figsize=(10, 5))
    for i in range(N):
        ax = axs[i]
        img = images[i, :]
        if C == 1:
            ax.imshow(img[0, :, :], cmap='gray')
        else:
            #   ax.imshow(np.transpose(np.transpose(img,0,2),0,1)[:, :, :])
            ax.imshow(np.transpose(img, (1, 2, 0))[:, :, :])
        ax.set_title(ttl)
        ax.axis('off')
    # plt.suptitle(ttl)
    plt.tight_layout()
    return fig, axs


