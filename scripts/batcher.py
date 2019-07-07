import sklearn
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import itertools
import time
from functools import partial

import os
figdir = "../figures" # set this to '' if you don't want to save figures
def save_fig(fname):
    if figdir:
        plt.savefig(os.path.join(figdir, fname))

import numpy as onp
onp.set_printoptions(precision=3)
import jax
import jax.numpy as np
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.experimental import optimizers
print("jax version {}".format(jax.__version__))
from jax.lib import xla_bridge
print("jax backend {}".format(xla_bridge.get_backend().platform))

import torch
import torchvision
print("torch version {}".format(torch.__version__))
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print("current device {}".format(torch.cuda.current_device()))
else:
    print("Torch cannot find GPU")
    
import tensorflow as tf
from tensorflow import keras
print("tf version {}".format(tf.__version__))
if tf.test.is_gpu_available():
    print(tf.test.gpu_device_name())
else:
    print("TF cannot find GPU")



# First we create a dataset.

import sklearn.datasets
from sklearn.model_selection import train_test_split

iris = sklearn.datasets.load_iris()
X = iris["data"][:,:3] # Just take first 3 features to make problem harder
y = (iris["target"] == 2).astype(onp.int)  # 1 if Iris-Virginica, else 0'
N, D = X.shape # 150, 4


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)


# To implement SGD, we need a way to work with minibatches of data.
# Here is a first attempt.

batch_size = 2
num_train = X_train.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

def data_stream():
    rng = onp.random.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield X_train[batch_idx], y_train[batch_idx]
batch_it = data_stream()

nbatches = 3
for i in range(nbatches):
    batch = next(batch_it)
    x, y = batch
    print(x)
     
dataset = tf.data.Dataset.from_tensor_slices(
    {"X": X_train, "y": y_train})
batches = dataset.repeat().batch(batch_size)
batch_it = tf.data.Dataset.make_one_shot_iterator(batches)

i = 0
for batch in batches:
    if i >= nbatches:
        break
    x, y = batch["X"].numpy(), batch["y"]
    print(x)
    print(y)
    i = i + 1
    
#batch_it = tf.data.Dataset.make_one_shot_iterator(batches)

#https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
    
import torch.utils.data as utils

    
my_x = [onp.array([[1.0,2],[3,4]]),onp.array([[5.,6],[7,8]])] # a list of numpy arrays
my_y = [onp.array([4.]), onp.array([2.])] # another list of numpy arrays (targets)

tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = utils.DataLoader(my_dataset) # create your dataloader

for x, y in my_dataloader:
    print(x)
    print(y)
    
    import itertools
import time

def epoch_callback(params, step, epoch, train_loss):
    if True: #epoch % 500 == 0:
            print('Epoch {}, train NLL {}'.format(epoch, train_loss))
    return True

def sgd_v1(params, loss_fn, batcher, max_epochs, lr, callback=None):
    itercount = itertools.count()
    loss_history = []
    for epoch in range(max_epochs):
        start_time = time.time()
        for step in range(batcher.num_batches):
            batch = next(batcher.batch_stream)
            batch_loss = loss_fn(params, batch)
            batch_grad = grad(loss_fn)(params, batch)
            params = params - lr*batch_grad
        epoch_time = time.time() - start_time
        train_loss = loss_fn(params, (batcher.X, batcher.y))
        loss_history.append(train_loss)
        stop = epoch_callback(params, epoch, train_loss)
        if stop:
            break
    return params, loss_history
                 

onp.random.seed(43)
D = X_train.shape[1]
w_init = onp.random.randn(D)

batcher = MyBatcher(X_train, y_train, batch_size=10, seed=0)
w_mle_sgd1, history = sgd_v1(w_init, NLL, batcher, num_epochs=3,
                             lr=0.1, callback=epoch_callback)
print(w_mle_sgd1)
print(history)


# Finally we get to SGD.

from jax.experimental import optimizers
import itertools
import time

schedule = optimizers.constant(step_size=0.1)
#schedule = optimizers.exponential_decay(step_size=0.1, decay_steps=10, decay_rate=0.9)
#schedule = optimizers.piecewise_constant([50, 100], [0.1, 0.05, 0.01])

opt_init, opt_update, get_params = optimizers.momentum(step_size=schedule, mass=0.9)
#opt_init, opt_update, get_params = optimizers.adam(step_size=schedule)

    
@jit
def update(i, opt_state, batch):
  params = get_params(opt_state)
  g = grad(NLL)(params, batch)
  return opt_update(i, g, opt_state) # update internal state using gradient and iteration number

# Make sure everything is reproducible!
onp.random.seed(43)
w_init = onp.random.randn(D)
opt_state = opt_init(w_init)
num_epochs = 2000
loss_history = []

itercount = itertools.count()
for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time
    params = get_params(opt_state)
    train_loss = NLL(params, (X_train, y_train))
    loss_history.append(train_loss)
    if epoch % 500 == 0:
        #print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print('Epoch {}, train NLL {}'.format(epoch, train_loss))
        
w_mle_jax = get_params(opt_state)

# Rather than comparing parameters, we compare predictions
prob_scipy = predict_prob(w_mle_scipy, X_test)
prob_jax = predict_prob(w_mle_jax, X_test)
print(np.round(prob_scipy, 3))
print(np.round(prob_jax, 3))
assert np.allclose(prob_jax, prob_sklearn, atol=1e-1) # This is only true for some random seeds!

