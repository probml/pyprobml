
# Sometimes (eg when using JAX) we want to have an infinite stream
# of numpy minibatches. Here are some handy functions for this.

import numpy as np

USE_TORCH = True
USE_TF = True

if USE_TORCH:
    import torch
    print("torch version {}".format(torch.__version__))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print("current device {}".format(torch.cuda.current_device()))
    else:
        print("Torch cannot find GPU")
    
if USE_TF:
    import tensorflow as tf
    print("tf version {}".format(tf.__version__))
    if tf.test.is_gpu_available():
        print(tf.test.gpu_device_name())
    else:
        print("TF cannot find GPU")

class NumpyBatcher():
    def __init__(self, X, y, batch_size, shuffle=False):
        self.num_data = X.shape[0]
        num_complete_batches, leftover = divmod(self.num_data, batch_size)
        self.num_batches = num_complete_batches + bool(leftover)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = self._make_data_stream()
                
    def _make_data_stream(self):
        while True:
            if self.shuffle:
                perm = np.random.permutation(self.num_data)
            else:
                perm = range(self.num_data)
            for i in range(self.num_batches):
                batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                yield self.X[batch_idx], self.y[batch_idx]

# Test the Batcher

N_train = 5
D = 4            
np.random.seed(0)    
X_train = np.random.randn(N_train, D)
y_train = np.random.randn(N_train, 1)
batch_size = 2


# If we know how much of the stream we want
train_iterator = NumpyBatcher(X_train, y_train, batch_size).generator
num_minibatches = 4
print("read fixed number")
for step in range(num_minibatches):
    batch = next(train_iterator)
    x, y = batch
    print(y)

# If we want to keep reading the stream until we meet a stopping criterion   
train_iterator = NumpyBatcher(X_train, y_train, batch_size).generator    
step = 0
print("read till had enough")
for batch in train_iterator:
    x, y = batch
    print(y)
    step = step + 1
    if step >= num_minibatches:
        break
    


    
##########
# Pytorch version
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


from torch.utils.data import DataLoader, TensorDataset

np.random.seed(0)
train_set = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False) 

print("One epoch")
for step, (x,y) in enumerate(train_loader):
    print(y)

# If we try to read past the end of the dataset, the loop just terminates.
step = 0
for batch in train_loader:
    x, y = batch
    print(y)
    step = step + 1
    if step >= num_minibatches:
        break

# DataLoader is not an iterator, and does not support next()
try:       
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)     
    for step in range(num_minibatches):
        batch = next(train_loader)
        x, y = batch
        print(y)
except Exception as e:
    print(e)

# It can converted to an iterator. But if we try to read past ened of
# the dataset, we get a stopIteration error.
try:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)  
    train_iterator = iter(train_loader)
    for step in range(num_minibatches):
        batch = next(train_iterator)
        x, y = batch
        print(y)
except Exception as e:
    print(e)

############    
# Use pre-canned dataset
    
class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=np.float32))

import torchvision.datasets as datasets
mnist_dataset = datasets.MNIST('/tmp/mnist/',  transform=FlattenAndCast())

mnist_dataset = datasets.MNIST('/tmp/mnist/')
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

print("MNIST labels")
step = 0
for batch in training_generator:
    if step >= num_minibatches:
        break
    x, y = batch
    y = y.numpy()
    print(y)
    step = step + 1
    
training_iterator = iter(training_generator)
print("MNIST labels")
for step in range(num_minibatches):
    batch = next(training_iterator)
    x, y = batch
    print(y)
  
##############    
# Lets make an Infinite stream
#https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
    
train_loader_infinite = InfiniteDataLoader(train_set, batch_size=batch_size, shuffle=False)
step = 0
print("read till done")
for batch in train_loader_infinite:
    x, y = batch
    print(y)
    step = step + 1
    if step >= num_minibatches:
        break

print("read fixed number")
train_loader_infinite = InfiniteDataLoader(train_set, batch_size=batch_size, shuffle=False)
for step in range(num_minibatches):
    batch = next(train_loader_infinite)
    x, y = batch
    print(y)
    

    
############
# Make Torch DataLoader return numpy arrays instead of Tensors.
# https://github.com/google/jax/blob/master/notebooks/neural_network_and_data_loading.ipynb

def numpy_collate(batch):
    print('collate')
    print(batch)
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

N = 5; D = 2;
X = np.random.randn(N,D)
y = np.random.rand(N)
data_set = TensorDataset(torch.Tensor(X), torch.Tensor(y))
data_loader = DataLoader(data_set, batch_size=2, shuffle=False) 
for step, (x,y) in enumerate(data_loader):
    print(y)
    
numpy_loader = NumpyLoader(data_set, batch_size=2, shuffle=False) 
for step, (x,y) in enumerate(numpy_loader):
    print(y)
    
train_loader_numpy = NumpyLoader(train_set, batch_size=batch_size, shuffle=False) 
print("One epoch")
for step, (x,y) in enumerate(train_loader_numpy):
    print(y)
    

#################
# TF gives us an infinite stream of minibatches, all with the same size.
    
dataset = tf.data.Dataset.from_tensor_slices({"X": X_train, "y": y_train})
batches = dataset.repeat().batch(batch_size)

step = 0
for batch in batches:
    if step >= num_minibatches:
        break
    x, y = batch["X"].numpy(), batch["y"].numpy()
    print(y)
    step = step + 1

# Pre-canned datasets.
import tensorflow_datasets as tfds
dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)
    
