
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from einops import rearrange

def pl1 = vutils.make_grid(model_samples, nrow=num_of_images_per_row).cpu().detach().numpy()
plt.title(title, fontsize=10)
plt.imshow(rearrange(img1, "c h w -> h w c"))
plt.axis('off')
if filename is not None:
    # plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
plt.show()

def plot_samples(vaes, num=25, figsize=(10 ,30), num_of_images_per_row=5, figdir=None):
    filename = None
    if hasattr(vaes, '__iter__'): # list of models
        for vae in vaes:
            if figdir is not None:
                filename = f'{figdir}/vae-samples-{vae.model_name}.png'
            plot_samples(vae, num, figsize, num_of_images_per_row, filename)
    else:
        vae = vaes # single model
        model_samples = vae.get_samples(num)
        title = f"Samples from {vae.model_name}"
        if figdir is not None:
            filename = f'{figdir}/vae-samples-{vae.model_name}.png'
        plot(model_samples, title, figsize, num_of_images_per_row, figdir)

def plot_reconstruction(vaes, batch, num_of_samples=5, num_of_images_per_row=5, figsize=(10, 30), figdir=None):
    x, y = batch
    img = x[:num_of_samples, :, :, :]
    filename = None
    if figdir is not None:
        filename = f'{figdir}/vae-recon-original.png'
    plot(img, "Original", figsize, num_of_images_per_row, filename)

    if hasattr(vaes, '__iter__'):
        for vae in vaes:
            title = f"Reconstruction from {vae.model_name}"
            if figdir is not None:
                filename = f'{figdir}/vae-recon-{vaes.model_name}.png'
        plot(vaes(img), title, figsize, num_of_images_per_row, filename)
        filename = f'{figdir}/vae-recon-{vae.model_name}.png'
    plot(vae(img), title, figsize, num_of_images_per_row, filename)
else:
    title = f"Reconstruction from {vaes.model_name}"
    if figdir is not None:
        foo
    plot(vaes(img), title, figsize, num_of_images_per_row)
