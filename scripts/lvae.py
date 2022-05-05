# Base VAE Code: https://github.com/google/flax/tree/main/examples/vae
# Followed the original Theano repository of LVAE paper: https://github.com/casperkaae/LVAE
# Authors gave more details about the paper in issues: https://github.com/casperkaae/LVAE/issues/1
# Finally, in some parts I followed: https://github.com/AntixK/PyTorch-VAE/blob/master/models/lvae.py
# PS: Importance weighting is not implemented.
# Firat Oncel / oncelf@itu.edu.tr


from tkinter.tix import Tree
from unicodedata import name
from absl import app
from absl import flags
from flax import linen as nn
from flax.training import train_state, checkpoints
import shutil
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any, Callable, Sequence
import utils as vae_utils
import scipy.io
import os
import math
from PIL import Image
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "False"



"""
    This code is created with reference to torchvision/utils.py.
    Modify: torch.tensor -> jax.numpy.DeviceArray
    If you want to know about this file in detail, please visit the original code:
        https://github.com/pytorch/vision/blob/master/torchvision/utils.py
"""


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
        grid = grid.at[y * height + padding:(y + 1) * height,
                       x * width + padding:(x + 1) * width].set(ndarray[k])
        k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format)



FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The learning rate for the Adam optimizer.')
)
flags.DEFINE_float(
    'damping', default=0.75,
    help=('lr damping factor')
)
flags.DEFINE_integer(
    'batch_size', default=256,
    help=('Batch size for training.')
)
flags.DEFINE_integer(
    'num_epochs', default=500,
    help=('Number of training epochs.')
)
flags.DEFINE_string(
    'dataset_name', default='mnist',
    help=('Dataset Name')
)
flags.DEFINE_string(
    'load_path', default=None,
    help=('Dataset Name')
)
flags.DEFINE_integer(
    'num_layers', default=5,
    help=('Number of layers')
)
flags.DEFINE_integer(
    'nt', default=200,
    help=('KL step')
)
flags.DEFINE_integer(
    'save_every', default=25,
    help=('Save and Eval Every')
)
# Taken from paper
latent_dim = [64, 32, 16, 8, 4]
hidden_dim = [512, 256, 128, 64, 32]
input_dim = 28*28

class TrainState(train_state.TrainState):
  batch_stats: Any

class Encoder(nn.Module):
  hidden_dim: int
  latent_dim: int


  @nn.compact
  def __call__(self, x, train):
      x = nn.Dense(self.hidden_dim, name='fc1')(x)
      x = nn.BatchNorm(not train, name='bn1')(x)
      x = nn.leaky_relu(x)
      x = nn.Dense(self.hidden_dim, name='fc2')(x)
      x = nn.BatchNorm(not train, name='bn2')(x)
      x = nn.leaky_relu(x)
      mean_x =  nn.Dense(self.latent_dim, name='fc_m')(x)
      logvar_x =  nn.Dense(self.latent_dim, name='fc_var')(x)
      return x, mean_x, jnp.clip(nn.softplus(logvar_x), 0, 10)


class Ladder(nn.Module):
  hidden_dim: int
  latent_dim: int


  @nn.compact
  def __call__(self, x, train):
      x = nn.Dense(self.hidden_dim, name='fc1')(x)
      x = nn.BatchNorm(not train, name='bn1')(x)
      x = nn.leaky_relu(x)
      x = nn.Dense(self.hidden_dim, name='fc2')(x)
      x = nn.BatchNorm(not train, name='bn2')(x)
      x = nn.leaky_relu(x)
      mean_x = nn.Dense(self.latent_dim, name='fc_m')(x)
      logvar_x = nn.Dense(self.latent_dim, name='fc_var')(x)
      return mean_x, jnp.clip(nn.softplus(logvar_x), 0, 10)

class Decoder(nn.Module):
  hidden_dim: int


  @nn.compact
  def __call__(self, x, train):
      x = nn.Dense(self.hidden_dim, name='fc1')(x)
      x = nn.BatchNorm(not train, name='bn1')(x)
      x = nn.leaky_relu(x)
      x = nn.Dense(self.hidden_dim, name='fc2')(x)
      x = nn.BatchNorm(not train, name='bn2')(x)
      x = nn.leaky_relu(x)
      return x

class Final(nn.Module):
  hidden_dim: int


  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.hidden_dim, name='fc1')(x)
    return nn.sigmoid(x)

class LVAE(nn.Module):
    latent_dims: Sequence[int]
    hidden_dims: Sequence[int]
    input_dim: int
    

    def setup(self):
        self.num_rungs = len(self.latent_dims)

        assert len(self.latent_dims) == len(self.hidden_dims), "Length of the latent" \
                                                     "and hidden dims must be the same"
        # Build Encoder
        modules = []
        for i, h_dim in enumerate(self.hidden_dims):
          modules.append(Encoder(h_dim, self.latent_dims[i]))
        self.encoders = modules

        # Build Decoder
        modules = []
        for i in range(self.num_rungs-2, -1, -1):
          modules.append(Ladder(self.hidden_dims[i], self.latent_dims[i]))
        modules.append(Decoder(self.hidden_dims[0]))
        self.decoders = modules
        
        # Final layer
        self.final = Final(self.input_dim)

        
    def __call__(self, x, key_list, train, z=None, generate=False):
        encoded, decoded, m_s = [], [], []
        num_layers = self.num_rungs
        # First layer
        d, m, s = self.encoders[0](x, train)
        encoded.append((d, m, s))

        for i in range(1, num_layers):
          d, m, s = self.encoders[i](d, train)
          encoded.append((d, m, s))
        
        
        if not generate:
          for i in range(num_layers-1):
            z = reparameterize(key_list[i], m, s)
            _, mu, sigma = encoded[num_layers-2-i]
            mu_dec, sigma_dec = self.decoders[i](z, train)
            m, s = merge_gauss(mu, mu_dec, sigma, sigma_dec)
            decoded.append((mu_dec, sigma_dec))
            m_s.append((m, s))

          z = reparameterize(key_list[-1], m, s)
          dec = self.decoders[-1](z, train)
          p_x = self.final(dec)
        
        else: 
          for i in range(num_layers-1):
            mu_dec, sigma_dec = self.decoders[i](z, train)
            z = reparameterize(key_list[i], mu_dec, sigma_dec)
          dec = self.decoders[-1](z, train)
          p_x = self.final(dec)

        all_mu_sigma = [] # To calculate KL Divergence
        if not generate:
          _, mu_enc_last, sigma_enc_last = encoded[-1]
          all_mu_sigma.append((mu_enc_last, sigma_enc_last))
          all_mu_sigma.append((0., 1.)) # Standard normal
          
          for i in range(len(m_s)):
            all_mu_sigma.append(m_s[i]) # Merged
            all_mu_sigma.append(decoded[i]) # Decoder outputs

        return p_x, all_mu_sigma


def merge_gauss(mu_1, mu_2, log_var_1, log_var_2):
  p_1 = 1. / (jnp.exp(log_var_1) + 1e-7)
  p_2 = 1. / (jnp.exp(log_var_2) + 1e-7)
  mu = (mu_1 * p_1 + mu_2 * p_2)/(p_1 + p_2)
  log_var = jnp.log(1./(p_1 + p_2))
  return mu, log_var


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


@jax.jit
def kl_divergence(q_params, p_params):
  mu_q, log_var_q = q_params
  mu_p, log_var_p = p_params
  kl = (log_var_p - log_var_q) + (jnp.exp(log_var_q) + (mu_q - mu_p)**2)/(2 * jnp.exp(log_var_p) + 1e-6) - 0.5
  kl = jnp.sum(kl, axis = -1)
  return kl.mean()


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = jnp.log(logits + 1e-6)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


def compute_metrics(recon_x, x, kl_list):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  kld_loss = 0
  for i in range(0, len(kl_list), 2):
    kld_loss += kl_divergence(kl_list[i], kl_list[i+1])
  return {
      'bce': bce_loss,
      'kld': kld_loss,
      'loss': bce_loss + kld_loss
  }


def model(latent_dim, hidden_dim, input_dim):
  return LVAE(latent_dim, hidden_dim, input_dim)


@jax.jit
def train_step(state, batch, key_list, kl_weight):
  def loss_fn(params, batch_stats):
    recon_x, kl_list = model(latent_dim, hidden_dim, input_dim).apply({'params': params, 'batch_stats': batch_stats}, batch, key_list, False)
    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = 0
    for i in range(0, len(kl_list), 2):
      kld_loss += kl_divergence(kl_list[i], kl_list[i+1])

    loss = bce_loss + kl_weight*kld_loss
    return loss
  grads = jax.grad(loss_fn)(state.params, state.batch_stats)
  return state.apply_gradients(grads=grads)


@jax.jit
def eval(params, batch_stats, images, z, key_list):
  def eval_model(vae):
    recon_images, kl_list = vae(images, key_list, False)
    comparison = jnp.concatenate([images[:16].reshape(-1, 28, 28, 1), recon_images[:16].reshape(-1, 28, 28, 1)])
    generated_images, _ = vae(images, key_list, False, z, True)
    generated_images = generated_images.reshape(-1, 28, 28, 1)
    metrics = compute_metrics(recon_images, images, kl_list)
    return metrics, comparison, generated_images

  return nn.apply(eval_model, model(latent_dim, hidden_dim, input_dim))({'params': params, 'batch_stats': batch_stats})


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x/255


def prepare_dataset(dataset_name):
  if dataset_name.startswith('omni'):
      # Download https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat and place it under data
      omni_raw = scipy.io.loadmat('data/chardata.mat')
      train_data = np.array(omni_raw['data'].T.astype('float32'))
      test_data = np.array(omni_raw['testdata'].T.astype('float32'))
      train_ds = tf.data.Dataset.from_tensor_slices(train_data)
      train_ds = train_ds.cache()
      train_ds = train_ds.repeat()
      train_ds = train_ds.shuffle(train_data.shape[0])
      train_ds = train_ds.batch(FLAGS.batch_size)
      train_ds = iter(tfds.as_numpy(train_ds))
      test_ds = tf.data.Dataset.from_tensor_slices(test_data).batch(test_data.shape[0])
      test_ds = np.array(list(test_ds)[0])
      test_ds = jax.device_put(test_ds)
      train_size = train_data.shape[0]
  
  elif dataset_name.startswith('mni'):
      ds_builder = tfds.builder('mnist')
      ds_builder.download_and_prepare()
      train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
      train_ds = train_ds.map(prepare_image)
      train_ds = train_ds.cache()
      train_ds = train_ds.repeat()
      train_ds = train_ds.shuffle(50000)
      train_ds = train_ds.batch(FLAGS.batch_size)
      train_ds = iter(tfds.as_numpy(train_ds))
      test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
      test_ds = test_ds.map(prepare_image).batch(10000)
      test_ds = np.array(list(test_ds)[0])
      test_ds = jax.device_put(test_ds)
      train_size = 50000
  
  return train_ds, test_ds, train_size
    



def main(argv):
  del argv
  try:
    shutil.rmtree('{}_results'.format(FLAGS.dataset_name))
  except:
    print("Results Folder Not Found")
  os.mkdir('{}_results'.format(FLAGS.dataset_name))
  os.mkdir('{}_results/reconstruction'.format(FLAGS.dataset_name))
  os.mkdir('{}_results/sample'.format(FLAGS.dataset_name))

  try:
    os.mkdir('{}_ckpts'.format(FLAGS.dataset_name))
  except:
    print("Checkpoint Folder Already Found")
  CKPT_DIR = '{}_ckpts'.format(FLAGS.dataset_name)

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')
  rng = random.PRNGKey(0)
  init_key_list = []
  rng, key = random.split(rng)
  for i in range(FLAGS.num_layers):
    rng, key = random.split(rng)
    init_key_list.append(key)
  
  train_ds, test_ds, train_size = prepare_dataset(FLAGS.dataset_name)

  print("Number of train samples:", train_size)
  
  # Sampling z for evaluation
  eval_keys = []
  for _ in range(FLAGS.num_layers + 1):
    rng, key = random.split(rng)
    eval_keys.append(key)
  z = random.normal(eval_keys.pop(), (64, latent_dim[-1]))

  rng, key = random.split(rng)
  # Initialization
  init_data = random.uniform(key, (FLAGS.batch_size, input_dim), jnp.float32, -0.08, 0.08)
  total_steps = FLAGS.num_epochs*(train_size//FLAGS.batch_size) + FLAGS.num_epochs
  damping = FLAGS.damping

  # LR Scheduler
  piecewise_constant_decay_scheduler = optax.piecewise_constant_schedule(init_value=FLAGS.learning_rate,
                                                                  boundaries_and_scales={int(total_steps*0.1):damping,
                                                                                        int(total_steps*0.2):damping,
                                                                                        int(total_steps*0.3):damping,
                                                                                        int(total_steps*0.4):damping,
                                                                                        int(total_steps*0.5):damping,
                                                                                        int(total_steps*0.6):damping,
                                                                                        int(total_steps*0.7):damping,
                                                                                        int(total_steps*0.8):damping,
                                                                                        int(total_steps*0.9):damping})

  optimizer = optax.adam(learning_rate=piecewise_constant_decay_scheduler)
  
  # Creating train state

  state = TrainState.create(
      apply_fn=model(latent_dim, hidden_dim, input_dim).apply,
      params=model(latent_dim, hidden_dim, input_dim).init(key, init_data, init_key_list, True, z, False)['params'],
      tx=optimizer,
      batch_stats=model(latent_dim, hidden_dim, input_dim).init(key, init_data, init_key_list, True, z, False)['batch_stats']
  )

  if FLAGS.load_path is not None:
    state = checkpoints.restore_checkpoint(ckpt_dir=FLAGS.load_path, target=state)
    print("Loaded from checkpoint!")
  steps_per_epoch = train_size // FLAGS.batch_size
  kl_weight = 0
  warm_up = 1/FLAGS.nt
  


  for epoch in range(FLAGS.num_epochs):
    for _ in range(steps_per_epoch):
      key_list = []
      batch = next(train_ds)
      for _ in range(FLAGS.num_layers):
        rng, key = random.split(rng)
        key_list.append(key)
      state = train_step(state, batch, key_list, kl_weight)

    kl_weight += warm_up
    kl_weight = min(1, kl_weight)
    if not ((epoch) % FLAGS.save_every):
      metrics, comparison, sample = eval(state.params, state.batch_stats, test_ds, z, eval_keys)
      save_image(comparison, f'{FLAGS.dataset_name}_results/reconstruction/{str(epoch).zfill(4)}.png', nrow=16)
      save_image(sample, f'{FLAGS.dataset_name}_results/sample/{str(epoch).zfill(4)}.png', nrow=16)
      print('eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(epoch, metrics['loss'], metrics['bce'], metrics['kld']))
      checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=epoch, overwrite=True)


if __name__ == '__main__':
  app.run(main)
