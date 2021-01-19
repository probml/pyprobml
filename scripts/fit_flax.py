# Fitting code for flax classifiers


import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util
import pandas as pd


def fit_model(model, train_iter, test_iter,  rng,
        num_steps, make_optimizer, train_batch, eval_batch,
        preprocess_train_batch = None, preprocess_test_batch = None,
        print_every = 1, eval_every = 1):
  batch = next(train_iter)
  if preprocess_train_batch is not None:
      batch = preprocess_train_batch(batch, rng)
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

def softmax_cross_entropy(logprobs, labels):
  # class label is last dimension (-1)
  onehots = jax.nn.one_hot(labels, logprobs.shape[-1])
  losses = -jnp.sum(onehots * logprobs, axis=-1)
  return jnp.mean(losses)

def compute_metrics(logprobs, labels):
  loss = softmax_cross_entropy(logprobs, labels)
  accuracy = jnp.mean(jnp.argmax(logprobs, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def eval_batch(model, params, batch):
  logprobs = model.apply({'params': params}, batch['X'])
  return compute_metrics(logprobs, batch['y'])

eval_batch = jax.jit(eval_batch, static_argnums=0)

def train_batch(model, optimizer, batch):
  labels = batch['y']
  def loss_fn(params):
    logprobs = model.apply({'params': params}, batch['X'])
    loss = softmax_cross_entropy(logprobs, labels)
    return loss, logprobs
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logprobs), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logprobs, labels)
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

def make_iterator_from_batch(batch):
  while True:
    yield batch

def test():
  # We just check we can run the functions and that they return "something"
  print('testing fit-flax')
  N = 3; D = 5; C = 10;
  model = ModelTest(nhidden = 0, nclasses = C)
  rng = jax.random.PRNGKey(0)
  X = np.random.randn(N,D)
  y = np.random.choice(C, size=N, p=(1/C)*np.ones(C));
  batch = {'X': X, 'y': y}
  params = model.init(rng, X)['params']

  logprobs = model.apply({'params': params}, batch['X'])
  assert logprobs.shape==(N,C)
  labels = batch['y']
  loss = softmax_cross_entropy(logprobs, labels)
  assert loss.shape==()

  metrics = eval_batch(model, params, batch)
  assert np.allclose(loss, metrics['loss'])

  make_optimizer = optim.Momentum(learning_rate=0.1, beta=0.9)
  optimizer = make_optimizer.create(params)
  optimizer, metrics = train_batch(model, optimizer, batch)
  num_steps = 2
  train_iter = make_iterator_from_batch(batch);
  test_iter = make_iterator_from_batch(batch);
  params_init = params

  params_new, history =  fit_model(model, train_iter, test_iter,  rng,
      num_steps, make_optimizer, train_batch, eval_batch,
      print_every=1)
  diff = tree_util.tree_multimap(lambda x,y: x-y, params_init, params_new)
  diff_max = tree_util.tree_map(lambda x: jnp.max(x), diff)
  assert jnp.abs(diff_max['Dense_0']['kernel']) > 0 # has changed 

  print('test passed')
  
  
def main():
    fit_model_test()

if __name__ == "__main__":
    main()
    
