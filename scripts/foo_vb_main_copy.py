from jax.config import config
config.update("jax_enable_x64", True)

from jax.config import config
config.update("jax_debug_nans", True)
from random import randint
from jax import random

import flax.linen as nn
from typing import Sequence, Callable

import foo_vb_utils_copy as utils
import foo_vb_datasets as ds
import foo_vb_lib_copy as  foo_vb_lib

import ml_collections


'''from jax.config import config
config.update("jax_enable_x64", True)'''

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.batch_size = 128
  config.test_batch_size = 1000

  config.epochs = 20
  config.seed = 1
  config.train_mc_iters = 3

  config.s_init = 0.27
  config.eta = 1.
  config.alpha = 0.5

  config.tasks = 10
  config.results_dir = "."

  config.dataset = "continuous_permuted_mnist"
  config.iterations_per_virtual_epc = 468

  config.diagonal=True
  
  return config

class Net(nn.Module):
  features : Sequence[int]
  activation_fn : Callable = nn.activation.relu
  
  @nn.compact
  def __call__(self, x):
    for feature in self.features[:-1]:
      x = self.activation_fn(nn.Dense(feature)(x))
    return nn.log_softmax(nn.Dense(self.features[-1])(x))

Net100 = Net([100, 100, 10])
Net200 = Net([200, 200, 10])

if __name__ == '__main__':
  config = get_config()
  config.alpha = 0.6
  model = Net200
  key = random.PRNGKey(0)
  perm_key, key = random.split(key)
  
  image_size = 784
  n_permutations = 10

  perm_lst = utils.create_random_perm(perm_key, image_size, n_permutations)
  perm_lst = perm_lst[1:11]
  train_loaders, test_loaders = ds.ds_padded_cont_permuted_mnist(num_epochs=int(config.epochs*config.tasks), iterations_per_virtual_epc=config.iterations_per_virtual_epc,
                                                                  contpermuted_beta=4, permutations=perm_lst,
                                                                  batch_size=config.batch_size)

                                                      
  ava_test = foo_vb_lib.train_continuous_mnist(key, model, train_loaders,
                            test_loaders, image_size, n_permutations, config)
  print(ava_test)