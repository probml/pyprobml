
# Sometimes (eg when using JAX) we want to have an infinite stream
# of numpy minibatches. Here are some handy functions for this.

import numpy as np

import torch
print("torch version {}".format(torch.__version__))
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print("current device {}".format(torch.cuda.current_device()))
else:
    print("Torch cannot find GPU")
    
if False:
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
        self.iterator = self._make_data_stream()
                
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

np.random.seed(0)
train_iterator = NumpyBatcher(X_train, y_train, batch_size).iterator
num_minibatches = 4
print("Prefix of infinite stream")
for i in range(num_minibatches):
    batch = next(train_iterator)
    x, y = batch
    print(y)


##########
# Pytorch version
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958

from torch.utils.data import DataLoader, TensorDataset

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
    
np.random.seed(0)
train_set = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader_finite = DataLoader(train_set, batch_size=batch_size, shuffle=False) 
train_loader_infinite = InfiniteDataLoader(train_set, batch_size=batch_size, shuffle=False)
train_iterator = iter(train_loader_infinite)

print("One epoch")
for step, (x,y) in enumerate(train_loader_finite):
    print(y)
    
print("Prefix of infinite stream")
for step in range(num_minibatches):
    batch = next(train_iterator)
    x, y = batch
    print(y)
    

############
# Make Torch DataLoader return numpy arrays instead of Tensors.
# https://github.com/google/jax/blob/master/notebooks/neural_network_and_data_loading.ipynb

def numpy_collate(batch):
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

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=np.float32))

from torchvision.datasets import MNIST
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=128, num_workers=0)

train_loader_finite = NumpyLoader(train_set, batch_size=batch_size, shuffle=False) 

print("One epoch")
for step, (x,y) in enumerate(train_loader_finite):
    print(y)
    


'''     
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
'''


    
