import os
from jax.interpreters.batching import batch
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import checkpoints

import math
import matplotlib.pyplot as plt
#from utils import save_image


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
    """Make a grid of images and Save it into an image file.
  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp - A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    scale_each (bool, optional): If ``True``, scale each image in the batch of
      images separately rather than the (min, max) over all images. Default: ``False``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the filename extension.
      If a file object was used instead of a filename, this parameter should always be used.
  """
    if not (isinstance(ndarray, jnp.ndarray) or
        (isinstance(ndarray, list) and all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError('array_like of tensors expected, got {}'.format(type(ndarray)))

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
      for x in range(xmaps):
        if k >= nmaps:
          break
        grid = jax.ops.index_update(
          grid, jax.ops.index[y * height + padding:(y + 1) * height,
                              x * width + padding:(x + 1) * width],
          ndarray[k])
        k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0, 0, 255).astype(jnp.uint8)
    plt.imshow(ndarr)
    plt.savefig(fp)
    # im = Image.fromarray(ndarr.copy())
    # im.save(fp, format=format)

class Encoder(nn.Module):
  latent_dim: int
  hidden_dims = [32, 64, 128, 256, 512]

  @nn.compact
  def __call__(self, x):
    # Build Encoder
    for h_dim in self.hidden_dims:
      x = nn.Conv(features=h_dim, kernel_size=(3, 3), strides=(2,2), padding="valid")(x)
      x = nn.GroupNorm()(x)
      x = nn.gelu(x)

    x = x.reshape((x.shape[0], -1))
    mean_x = nn.Dense(self.latent_dim, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latent_dim, name='fc2_logvar')(x)
    return mean_x, logvar_x

class Decoder(nn.Module):
  hidden_dims = [32, 64, 128, 256, 512]

  @nn.compact
  def __call__(self, z):
    shape_before_flattening, flatten_out_size = self.flatten_enc_shape()

    x = nn.Dense(flatten_out_size, name='fc1')(z)
    x = x.reshape((x.shape[0], *shape_before_flattening[1:]))
    
    hidden_dims = self.hidden_dims[::-1]
    # Build Decoder
    for h_dim in range(len(hidden_dims)-1):
      x = nn.ConvTranspose(features=hidden_dims[h_dim], kernel_size=(3, 3), strides=(2,2))(x)
      x = nn.GroupNorm()(x)
      x = nn.gelu(x)
    
    x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.sigmoid(x)
    return x
  
  def flatten_enc_shape(self):
    x = jnp.ones([1, 64, 64, 3], jnp.float32)

    # Build Encoder
    for h_dim in self.hidden_dims:
      x = nn.Conv(features=h_dim, kernel_size=(3, 3), strides=(2,2))(x)
      x = nn.gelu(x)

    return x.shape, int(np.prod(x.shape))

class VAE(nn.Module):
  latents: int

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return self.decoder(z)

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def mse(x_hat, x):
  return 0.5*jnp.sum((x_hat - x)**2)

class VAE_celeba:
    def __init__(self, figdir="results", data_dir="kaggle", image_size=64, num_epochs=30, batch_size=256, learning_rate=1e-3, kl_coeff=1, latents=256):
        
        if not os.path.exists(figdir):
            print('making directory {}'.format(figdir))
            os.mkdir(figdir)

        self.figdir = figdir
        self.data_dir = data_dir
        self.latents = latents
        self.kl_coeff = kl_coeff
        self.image_size = image_size
        self.batch_size = batch_size 
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def compute_metrics(self, recon_x, x, mean, logvar):
        mse_loss = mse(recon_x, x).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        return {
            'mse': mse_loss,
            'kld': kld_loss,
            'loss': mse_loss + self.kl_coeff*kld_loss
        }

    def model(self):
        return VAE(latents=self.latents)

    def get_samples_and_reconstruction(self, rng, state, test_ds):
      @jax.jit
      def eval(params, images, z, z_rng):
          def eval_model(vae):
              recon_images, mean, logvar = vae(images, z_rng)
              comparison = jnp.concatenate([images[:8].reshape(-1, self.image_size, self.image_size, 3),
                                          recon_images[:8].reshape(-1, self.image_size, self.image_size, 3)])
              generate_images = vae.generate(z)
              generate_images = generate_images.reshape(-1, self.image_size, self.image_size, 3)
              metrics = self.compute_metrics(recon_images, images, mean, logvar)
              return metrics, comparison, generate_images

          return nn.apply(eval_model, self.model())({'params': params})
      rng, z_key, eval_rng = random.split(rng, 3)
      z = random.normal(z_key, (64, self.latents))
      metrics, comparison, sample = eval(state.params, test_ds, z, eval_rng)
      save_image(
          comparison, f'results/reconstruction.png', nrow=8)
      save_image(sample, f'results/sample.png', nrow=8)

    def main(self):

        @jax.jit
        def train_step(state, batch, z_rng):
            def loss_fn(params):
                recon_x, mean, logvar = self.model().apply({'params': params}, batch, z_rng)
                bce_loss = mse(recon_x, batch).mean()
                kld_loss = kl_divergence(mean, logvar).mean()
                loss = bce_loss + self.kl_coeff*kld_loss
                return loss
            grads = jax.grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads)

        @jax.jit
        def eval(params, images, z, z_rng):
            def eval_model(vae):
                recon_images, mean, logvar = vae(images, z_rng)
                comparison = jnp.concatenate([images[:8].reshape(-1, self.image_size, self.image_size, 3),
                                            recon_images[:8].reshape(-1, self.image_size, self.image_size, 3)])
                generate_images = vae.generate(z)
                generate_images = generate_images.reshape(-1, self.image_size, self.image_size, 3)
                metrics = self.compute_metrics(recon_images, images, mean, logvar)
                return metrics, comparison, generate_images

            return nn.apply(eval_model, self.model())({'params': params})

        # Make sure tf does not allocate gpu memory.
        tf.config.experimental.set_visible_devices([], 'GPU')

        rng = random.PRNGKey(0)
        rng, key = random.split(rng)

        @tf.autograph.experimental.do_not_convert
        def decode_fn(s):
            img = tf.io.decode_jpeg(tf.io.read_file(s))
            img.set_shape([218, 178, 3])
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.resize(img, (self.image_size, self.image_size), antialias=True)
            return tf.cast(img, dtype = jnp.dtype("float32"))

        dataset_celeba = tf.data.Dataset.list_files(self.data_dir+'/img_align_celeba/img_align_celeba/*.jpg', shuffle=False)
        train_dataset_celeba = (dataset_celeba
                                .map(decode_fn)
                                .map(tf.image.random_flip_left_right)
                                .shuffle(self.batch_size*16)
                                .batch(self.batch_size)
                                .repeat())
        test_ds = next(iter(tfds.as_numpy(train_dataset_celeba)))
        train_ds = iter(tfds.as_numpy(train_dataset_celeba))

        init_data = jnp.ones([self.batch_size, self.image_size, self.image_size, 3], jnp.float32)

        state = train_state.TrainState.create(
            apply_fn=self.model().apply,
            params=self.model().init(key, init_data, rng)['params'],
            tx=optax.adam(self.learning_rate),
        )

        rng, z_key, eval_rng = random.split(rng, 3)
        z = random.normal(z_key, (64, self.latents))

        steps_per_epoch = 50000 // self.batch_size

        for epoch in range(self.num_epochs):
            for _ in range(steps_per_epoch):
                batch = next(train_ds)
                rng, key = random.split(rng)
                state = train_step(state, batch, key)

            metrics, comparison, sample = eval(state.params, test_ds, z, eval_rng)
            save_image(comparison, f'{self.figdir}/reconstruction_{epoch}.png', nrow=8)
            save_image(sample, f'{self.figdir}/sample_{epoch}.png', nrow=8)
            print('eval epoch: {}, loss: {:.4f}, MSE: {:.4f}, KLD: {:.4f}'.format(
                epoch + 1, metrics['loss'], metrics['mse'], metrics['kld']
            ))
        checkpoints.save_checkpoint(".", state, epoch, "celeba_vae_checkpoint_")
  
if __name__ == "__main__":
  fig_dir = "results_celeba"
  data_dir = "kaggle"
  image_size = 64
  num_epochs = 30
  batch_size = 256
  learning_rate = 1e-3
  kl_coeff = 1
  latents = 256 
  vae = VAE_celeba(
        fig_dir,
        data_dir,
        image_size,
        num_epochs,
        batch_size, 
        learning_rate, 
        kl_coeff, 
        latents)
  vae.main()
