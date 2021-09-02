# Generic fitting function for flax classifiers.
# murphyk@gmail.com

# We assume the model has an apply method
# that returns the *log probabities* of each class as an N*C array,
# where N is the size of the batch.

# We assume the dataset iterators return a stream of minibatches
# of dicts, with fields
# X: (N,D) containing features
# y: (N,) containing integer label


import superimport

import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util, jit
from functools import partial

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging



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


@partial(jit, static_argnums=(0,))
def eval_classifier(model, params, batch):
  logprobs = model.apply({'params': params}, batch['X'])
  return compute_metrics(logprobs, batch['y'])

@partial(jit, static_argnums=(0,))
def update_classifier(model, optimizer, batch):
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


def append_history(history, train_metrics, test_metrics):
    history['train_loss'].append(train_metrics['loss'])
    history['train_accuracy'].append(train_metrics['accuracy'])
    history['test_loss'].append(test_metrics['loss'])
    history['test_accuracy'].append(test_metrics['accuracy'])
    return history
          
def fit_model(
      model, rng, num_steps, train_iter,
      test_iter = None,
      train_fn = update_classifier,
      test_fn = eval_classifier,
      make_optimizer = None,
      preprocess_train_batch = None, preprocess_test_batch = None,
      print_every = 1, test_every = None):

  batch = next(train_iter)
  if preprocess_train_batch is not None:
      batch = preprocess_train_batch(batch, rng)
  X = batch['X']
  params = model.init(rng, X)['params']

  if make_optimizer is None:
    make_optimizer = optim.Momentum(learning_rate=0.1, beta=0.9)
  optimizer = make_optimizer.create(params) 

  history = {'train_loss': [], 'train_accuracy': [],
             'test_loss': [], 'test_accuracy': []}
  if test_iter is None:
    test_every = 0
  if test_every is None:
    test_every = print_every

  for step in range(num_steps):
    batch = next(train_iter)
    if preprocess_train_batch is not None:
      batch = preprocess_train_batch(batch, rng)
    optimizer, train_metrics = train_fn(model, optimizer, batch)
    if (print_every > 0) & (step % print_every == 0):
       print('train step: {:d}, loss: {:0.4f}, accuracy: {:0.2f}'.format(
              step, train_metrics['loss'], 
                 train_metrics['accuracy']))
       
    if (test_every > 0) & (step % test_every == 0):
      batch = next(test_iter)
      if preprocess_test_batch is not None:
        batch = preprocess_test_batch(batch, rng)
      test_metrics = test_fn(model, optimizer.target, batch)
      history  = append_history(history, train_metrics, test_metrics)
      
  params = optimizer.target
  return params, history

############ Testing

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

def l2norm_sq(x):
  leaves, _ = tree_util.tree_flatten(x)
  return sum([np.sum(leaf ** 2) for leaf in leaves])

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

  # test apply
  logprobs = model.apply({'params': params}, batch['X'])
  assert logprobs.shape==(N,C)

  # test loss
  labels = batch['y']
  loss = softmax_cross_entropy(logprobs, labels)
  assert loss.shape==()

  # test test_fn
  metrics = eval_classifier(model, params, batch)
  assert np.allclose(loss, metrics['loss'])

  # test train_fn
  make_optimizer = optim.Momentum(learning_rate=0.1, beta=0.9)
  optimizer = make_optimizer.create(params)
  optimizer, metrics = update_classifier(model, optimizer, batch)

  # test fit_model
  num_steps = 2
  train_iter = make_iterator_from_batch(batch);
  test_iter = make_iterator_from_batch(batch);
  params_init = params
  params_new, history =  fit_model(
    model, rng, num_steps, train_iter, test_iter)

  diff = tree_util.tree_multimap(lambda x,y: x-y, params_init, params_new)
  print(diff)
  norm = l2norm_sq(diff)
  assert norm > 0 # check that parameters have changed :)

  print(history)
  print('test passed')
  

def main():
    test()

if __name__ == "__main__":
    main()
    
