import os 
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.stats import truncnorm
from flax.training import checkpoints
from utils import save_image

class Encoder(nn.Module):
  latent_dim: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=28, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.GroupNorm(28)(x)
    x = nn.gelu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.GroupNorm(32)(x)
    x = nn.gelu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.GroupNorm(32)(x)
    x = nn.gelu(x)
    x = x.reshape((x.shape[0], -1))
    mean_x = nn.Dense(self.latent_dim, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latent_dim, name='fc2_logvar')(x)
    return mean_x, logvar_x

class Decoder(nn.Module):

  @nn.compact
  def __call__(self, z):
    shape_before_flattening, flatten_out_size = self.flatten_enc_shape()
    #print(shape_before_flattening, flatten_out_size)
    x = nn.Dense(flatten_out_size, name='fc1')(z)
    x = nn.gelu(x)
    x = x.reshape((x.shape[0], *shape_before_flattening[1:]))
    x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.GroupNorm(32)(x)
    x = nn.gelu(x)
    x = nn.ConvTranspose(features=28, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.GroupNorm(28)(x)
    x = nn.gelu(x)
    x = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(2,2))(x)
    return x
  
  def flatten_enc_shape(self):
    x = jnp.ones([1, 32, 32, 1], jnp.float32)
    x = nn.Conv(features=28, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2,2))(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2,2))(x)
    return x.shape, int(np.prod(x.shape))

class VAE(nn.Module):
  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))

class VAE_mnist:
  def __init__(self, figdir="results", train_set_size=50000, test_set_size=10000, num_epochs=30, batch_size=256, learning_rate=1e-3, kl_coeff=1, latents=256):
      
      if not os.path.exists(figdir):
          print('making directory {}'.format(figdir))
          os.mkdir(figdir)

      self.figdir = figdir
      self.train_set_size = train_set_size
      self.latents = latents
      self.kl_coeff = kl_coeff
      self.test_set_size = test_set_size
      self.batch_size = batch_size 
      self.learning_rate = learning_rate
      self.num_epochs = num_epochs

  def model(self):
    return VAE(self.latents)
  
  def compute_metrics(self, recon_x, x, mean, logvar):
    bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    return {
        'bce': bce_loss,
        'kld': kld_loss,
        'loss': bce_loss + self.kl_coeff*kld_loss
    }

  def main(self):

    @jax.jit
    def train_step(state, batch, z_rng):
      def loss_fn(params):
        recon_x, mean, logvar = self.model().apply({'params': params}, batch, z_rng)
        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + self.kl_coeff*kld_loss
        return loss
      grads = jax.grad(loss_fn)(state.params)
      return state.apply_gradients(grads=grads)


    @jax.jit
    def eval(params, images, z, z_rng):
      def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate([images[:8].reshape(-1, 32, 32, 1),
                                      recon_images[:8].reshape(-1, 32, 32, 1)])

        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 32, 32, 1)
        metrics = self.compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

      return nn.apply(eval_model, self.model())({'params': params})

    def prepare_image(x):
      x = tf.cast(x['image'], tf.float32)
      x = tf.image.resize(x, (32, 32), "nearest")
      return x

    def get_datasets():
      """Load MNIST train and test datasets into memory."""
      ds_builder = tfds.builder('binarized_mnist')
      ds_builder.download_and_prepare()
      train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
      train_ds = train_ds.map(prepare_image)
      train_ds = train_ds.cache()
      train_ds = train_ds.repeat()
      train_ds = train_ds.shuffle(self.train_set_size)
      train_ds = train_ds.batch(self.batch_size)
      train_ds = iter(tfds.as_numpy(train_ds))

      test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
      test_ds = test_ds.map(prepare_image).batch(self.test_set_size)
      test_ds = np.array(list(test_ds)[0])
      test_ds = jax.device_put(test_ds)
      return train_ds, test_ds

    def create_train_state(key, rng, batch_size, learning_rate):
      init_data = jnp.ones([batch_size, 28, 28, 1], jnp.float32)

      state = train_state.TrainState.create(
          apply_fn=self.model().apply,
          params=self.model().init(key, init_data, rng)['params'],
          tx=optax.adam(learning_rate),
      )
      return state

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)

    train_ds, test_ds = get_datasets()

    state = create_train_state(key, rng, self.batch_size, self.learning_rate)
    rng, z_key, eval_rng = random.split(rng, 3)
    values = truncnorm.rvs(-1,1 , size=(64, self.latents))
    z = random.normal(z_key, (64, self.latents))

    steps_per_epoch = self.train_set_size // self.batch_size

    for epoch in range(self.num_epochs):
        for _ in range(steps_per_epoch):
            batch = next(train_ds)
            rng, key = random.split(rng)

            state = train_step(state, batch, key)

        metrics, comparison, sample = eval(state.params, test_ds, z, eval_rng)
        save_image(
                comparison, f'{self.figdir}/reconstruction_{epoch}.png', nrow=8)
        save_image(sample, f'{self.figdir}/sample_{epoch}.png', nrow=8)
        print('eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
            epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
        ))
    checkpoints.save_checkpoint(".", state, epoch, "mnist_vae_checkpoint_")

if __name__ == '__main__':
  fig_dir = "results_mnist"
  num_epochs = 50
  batch_size = 512
  learning_rate = 0.001
  train_set_size = 50000
  test_set_size = 10000
  kl_coeff = 1
  latents = 7
  vae = VAE_mnist(
        fig_dir,
        train_set_size,
        test_set_size,
        num_epochs,
        batch_size, 
        learning_rate, 
        kl_coeff, 
        latents)
  vae.main()
