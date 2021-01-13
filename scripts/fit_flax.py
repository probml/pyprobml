# Fitting functions for flax models


import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


def fit_model(model, train_iter, test_iter,  rng,
        num_steps, make_optimizer, train_batch, eval_batch,
        preprocess_train_batch = None, preprocess_test_batch = None,
        print_every = 1, eval_every = 1):
  batch = next(train_iter)
  X = batch['X']
  params = model.init(rng, X)['params']
  optimizer = make_optimizer.create(params)  
  history = pd.DataFrame({'train_loss': [], 'train_accuracy': [],
                   'test_loss': [], 'test_accuracy': [], 'step': []})
  
  for step in range(num_steps):
    batch = next(train_iter)
    if preprocess_train_batch is not None:
      batch = preprocess_train_batch(batch, rng)
    optimizer, train_metrics = train_batch(model, optimizer, batch)
    if (print_every > 0) & (step % print_every == 0):
       print('train step: {:d}, loss: {:0.4f}, accuracy: {:0.2f}'.format(
              step, train_metrics['loss'], 
                 train_metrics['accuracy']))
       
    if (eval_every > 0) & (step % eval_every == 0):
      batch = next(test_iter)
      if preprocess_test_batch is not None:
        batch = preprocess_test_batch(batch, rng)
      test_metrics = eval_batch(model, optimizer.target, batch)
      history = history.append(
                {'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy'],
                'step': step},
                ignore_index=True
                )
      
  params = optimizer.target
  return params, history



def onehot(labels, num_classes):
  y = (labels[..., None] == jnp.arange(num_classes)[None])
  return y.astype(jnp.float32)

def cross_entropy_loss_onehot(logits, onehots):
  return -jnp.mean(jnp.sum(onehots * logits, axis=-1))

def compute_metrics(logits, onehots):
  loss = cross_entropy_loss_onehot(logits, onehots)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(onehots, -1))
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics

def eval_batch(model, params, batch):
  logits = model.apply({'params': params}, batch['X'])
  onehots = onehot(batch['y'], model.nclasses)
  return compute_metrics(logits, onehots)

eval_batch = jax.jit(eval_batch, static_argnums=0)

def train_batch(model, optimizer, batch):
  onehots = onehot(batch['y'], model.nclasses)
  def loss_fn(params):
    logits = model.apply({'params': params}, batch['X'])
    loss = cross_entropy_loss_onehot(logits, onehots)
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, onehots)
  return optimizer, metrics

train_batch = jax.jit(train_batch, static_argnums=0)


############ Testing

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging

class ModelTest(nn.Module):
  nhidden: int
  nclasses: int
  @nn.compact
  def __call__(self, x):
    if self.nhidden > 0:
      x = nn.Dense(self.nhidden)(x)
      x = nn.relu(x)
    x = nn.Dense(self.nclasses)(x)
    x = nn.log_softmax(x)
    return x


def fit_model_test():
  # We just check we can run the functions and that they return "something"
  N = 3; D = 5; C = 10;
  model = ModelTest(nhidden = 0, nclasses = C)
  rng = jax.random.PRNGKey(0)
  X = np.random.randn(N,D)
  y = np.random.choice(C, size=N, p=(1/C)*np.ones(C));
  batch = {'X': X, 'y': y}
  params = model.init(rng, X)['params']
  metrics = eval_batch(model, params, batch)
  make_optimizer = optim.Momentum(learning_rate=0.1, beta=0.9)
  optimizer = make_optimizer.create(params)
  optimizer, metrics = train_batch(model, optimizer, batch)
  #print(optimizer)
  num_steps = 2
  def make_iter():
    while True:
      yield batch
  train_iter = make_iter(); test_iter = make_iter();
  params, history =  fit_model(model, train_iter, test_iter,  rng,
      num_steps, make_optimizer, train_batch, eval_batch,
      print_every=1)
  print('test passed')
  
  
def main():
    fit_model_test()

if __name__ == "__main__":
    main()
    
