# Load pre-trained ConvVAE model (eg trained in colab)
# See https://github.com/probml/pyprobml/blob/master/notebooks/lvm/vae_mnist_2d_tf.ipynb for training script

import superimport

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyprobml_utils as pml
import os

figdir = "../figures"
 
import tensorflow as tf
from tensorflow import keras
#import tensorflow_datasets as tfds
import pickle

folder = '/home/murphyk/Downloads'

if 0:
    with open(os.path.join(folder, 'mnist_small.pkl'), 'rb') as f:
        Xsmall = pickle.load(f)
        
    X = Xsmall[0,:,:,0].numpy();
    plt.imshow(X)
    input_shape = [28, 28, 1] # MNIST
    num_colors = input_shape[2]


with open(os.path.join(folder, 'celeba_small.pkl'), 'rb') as f:
    Xsmall = pickle.load(f)
    
X = Xsmall[0,:,:,:].numpy();
plt.imshow(X)
input_shape = [64, 64, 3] 
num_colors = input_shape[2]


from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

def make_encoder(
        input_dim,
        z_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        use_batch_norm = False,
        use_dropout= False
        ):
  encoder_input = Input(shape=input_dim, name='encoder_input')
  x = encoder_input
  n_layers_encoder = len(encoder_conv_filters)
  for i in range(n_layers_encoder):
      conv_layer = Conv2D(
          filters = encoder_conv_filters[i]
          , kernel_size = encoder_conv_kernel_size[i]
          , strides = encoder_conv_strides[i]
          , padding = 'same'
          , name = 'encoder_conv_' + str(i)
          )
      x = conv_layer(x)
      if use_batch_norm:
          x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      if use_dropout:
          x = Dropout(rate = 0.25)(x)
  shape_before_flattening = K.int_shape(x)[1:]
  x = Flatten()(x)
  mu = Dense(z_dim, name='mu')(x) # no activation
  log_var = Dense(z_dim, name='log_var')(x) # no activation
  encoder = Model(encoder_input, (mu, log_var))
  return encoder, shape_before_flattening

def make_decoder(
        shape_before_flattening,
        z_dim,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        use_batch_norm = False,
        use_dropout= False
        ):
  decoder_input = Input(shape=(z_dim,), name='decoder_input')
  x = Dense(np.prod(shape_before_flattening))(decoder_input)
  x = Reshape(shape_before_flattening)(x)
  n_layers_decoder = len(decoder_conv_t_filters)
  for i in range(n_layers_decoder):
      conv_t_layer = Conv2DTranspose(
          filters = decoder_conv_t_filters[i]
          , kernel_size = decoder_conv_t_kernel_size[i]
          , strides = decoder_conv_t_strides[i]
          , padding = 'same'
          , name = 'decoder_conv_t_' + str(i)
          )
      x = conv_t_layer(x)
      if i < n_layers_decoder - 1:
          if use_batch_norm:
              x = BatchNormalization()(x)
          x = LeakyReLU()(x)
          if use_dropout:
              x = Dropout(rate = 0.25)(x)
      # No activation fn in final layer since returns logits
      #else:
      #    x = Activation('sigmoid')(x)
  decoder_output = x
  decoder = Model(decoder_input, decoder_output)
  return decoder
  


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def sample_gauss(mean, logvar):
  eps = tf.random.normal(shape=mean.shape)
  return eps * tf.exp(logvar * .5) + mean

class ConvVAE(tf.keras.Model):
  def __init__(self,
        input_dim,
        latent_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        use_batch_norm = False,
        use_dropout= False,
        recon_loss_scaling = 1,
        kl_loss_scaling = 1,
        use_mse_loss = False
        ):
    super(ConvVAE, self).__init__()
    # Save all args so we can reconstruct this object later
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.encoder_conv_filters = encoder_conv_filters
    self.encoder_conv_kernel_size = encoder_conv_kernel_size
    self.encoder_conv_strides = encoder_conv_strides
    self.decoder_conv_t_filters = decoder_conv_t_filters
    self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
    self.decoder_conv_t_strides = decoder_conv_t_strides
    self.use_batch_norm = use_batch_norm
    self.use_dropout = use_dropout
    self.recon_loss_scaling = recon_loss_scaling
    self.kl_loss_scaling = kl_loss_scaling
    self.use_mse_loss = use_mse_loss
    
    self.inference_net, self.shape_before_flattening = make_encoder(
        input_dim,
        latent_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        use_batch_norm,
        use_dropout)
    
    self.generative_net = make_decoder(
        self.shape_before_flattening,
        latent_dim,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        use_batch_norm,
        use_dropout)

  @tf.function
  def sample(self, nsamples=1):
    eps = tf.random.normal(shape=(nsamples, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode_stochastic(self, x):
    mean, logvar = self.inference_net(x)
    return sample_gauss(mean, logvar)

  def decode(self, z, apply_sigmoid=True):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
  
  @tf.function
  def compute_loss(self, x):
    mean, logvar = self.inference_net(x)
    z = sample_gauss(mean, logvar)
    if self.use_mse_loss:
      x_probs = self.decode(z, apply_sigmoid=True)
      mse = tf.reduce_mean( (x - x_probs) ** 2, axis=[1, 2, 3])
      logpx_z = -0.5*mse # log exp(-0.5 (x-mu)^2)
    else:
      x_logit = self.decode(z, apply_sigmoid=False)
      cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x) # -sum_{c=0}^1 p_c log q_c
      logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3]) # sum over H,W,C
    logpz = log_normal_pdf(z, 0., 0.) # prior: mean=0, logvar=0
    logqz_x = log_normal_pdf(z, mean, logvar)
    kl_loss = logpz - logqz_x # MC approximation
    return -tf.reduce_mean(self.recon_loss_scaling * logpx_z + self.kl_loss_scaling * kl_loss) # -ve ELBO
          
  @tf.function
  def compute_gradients(self, x):
    with tf.GradientTape() as tape:
      loss = self.compute_loss(x)
    gradients = tape.gradient(loss, self.trainable_variables)
    return gradients

#############
    
def load_model(model_class, folder):
    with open(os.path.join(folder, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    model = model_class(*params)
    model.load_weights(os.path.join(folder, 'weights.h5'))
    return model

model = load_model(ConvVAE, folder)

# Check results match colab 
L = model.compute_loss(Xsmall)
print(L)
M, V = model.inference_net(Xsmall)
print(M)
