# Utility functions for celeba dataset

# Based on
# https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/helper_data.py

import torch
import matplotlib.pyplot as plt
import numpy as np

# https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/helper_data.py#L251
def compute_average_images(feature_idx, image_dim, data_loader, device=None, encoding_fn=None, max_batches=None):
    avg_img_with_feat = torch.zeros(image_dim, dtype=torch.float32)
    avg_img_without_feat = torch.zeros(image_dim, dtype=torch.float32)

    num_img_with_feat = 0
    num_images_without_feat = 0

    num_batches = 0
    for images, labels in data_loader:
        num_batches += 1
        idx_img_with_feat = labels[:, feature_idx].to(torch.bool)

        if encoding_fn is None:
            embeddings = images
        else:
            ####################################
            ### Get latent representation
            with torch.no_grad():
                if device is not None:
                    images = images.to(device)
                embeddings = encoding_fn(images).to('cpu')
            ####################################

        avg_img_with_feat += torch.sum(embeddings[idx_img_with_feat], axis=0)
        avg_img_without_feat += torch.sum(embeddings[~idx_img_with_feat], axis=0)
        num_img_with_feat += idx_img_with_feat.sum(axis=0)
        num_images_without_feat += (~idx_img_with_feat).sum(axis=0)

        if max_batches is not None:
            if num_batches > max_batches: break

    avg_img_with_feat /= num_img_with_feat
    avg_img_without_feat /= num_images_without_feat

    return avg_img_with_feat, avg_img_without_feat



# https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/helper_plotting.py#L172
def plot_modified_images(original, diff,
                         diff_coefficients=(0., 0.5, 1., 1.5, 2., 2.5, 3.),
                         decoding_fn=None,
                         device=None,
                         normalize=True,
                         figsize=(10, 4)):
    fig, axes = plt.subplots(nrows=2, ncols=len(diff_coefficients),
                             sharex=True, sharey=True, figsize=figsize)

    for i, alpha in enumerate(diff_coefficients):
        more = original + alpha * diff
        less = original - alpha * diff

        if decoding_fn is not None:
            ### Latent -> Original space
            with torch.no_grad():
                if device is not None:
                    more = more.to(device).unsqueeze(0)
                    less = less.to(device).unsqueeze(0)

                more = decoding_fn(more).to('cpu').squeeze(0)
                less = decoding_fn(less).to('cpu').squeeze(0)

        if normalize:   # map from -1..1 to 0..1
          more = more / 2 + 0.5
          less = less / 2 + 0.5

        smore = f'+{alpha}'
        sless = f'-{alpha}'

        axes[0][i].set_title(smore)
        axes[0][i].imshow(more.permute(1, 2, 0))

        axes[1][i].set_title(sless)
        axes[1][i].imshow(less.permute(1, 2, 0))

        axes[1][i].axison = False
        axes[0][i].axison = False

