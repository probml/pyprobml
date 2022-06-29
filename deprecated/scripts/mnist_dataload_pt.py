# Tutorial on PyTorch dataloader
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
    
# Convert to stream of numpy arrays
# https://github.com/google/jax/blob/master/notebooks/neural_network_and_data_loading.ipynb


import superimport

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

#from pdb import set_trace as bp

np.random.seed(42)


class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=np.float32))

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    bundle = zip(*batch) # [(X1,y1), (X2,y2)] -> [(X1,X2), (y1,y2)]
    return [numpy_collate(samples) for samples in bundle] # [stack(X1,X2), stack(y1,y2)] 
  else:
    return np.array(batch)

#transform = transforms.ToTensor()
#transform = FlattenAndCast()
transform = lambda x: np.array(x, dtype=np.float32)

mnist_dataset = datasets.MNIST('/tmp/mnist/',  download=True, train=True, transform=transform)
train_loader = DataLoader(mnist_dataset, batch_size=3, shuffle=False)
                                                        

for batch_id, (data, label) in enumerate(train_loader):
    print(type(data)) # <class 'torch.Tensor'>
    print(data.shape) # torch.Size([3, 28, 28])
    print(label) # tensor([5, 0, 4])
    if batch_id > 1:
        break
    
train_loader = DataLoader(mnist_dataset, batch_size=3, shuffle=False, collate_fn=numpy_collate)

for batch_id, (data, label) in enumerate(train_loader):
    print(type(data)) # <class 'numpy.ndarray'>
    print(data.shape) # (3, 28, 28)
    print(label) # [5 0 4]
    if batch_id > 1:
        break
    

'''
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


mnist_dataset = datasets.MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
datastream  = NumpyLoader(mnist_dataset, batch_size=3, num_workers=0)

for i_batch, sample_batched in enumerate(datastream):
    image, label = sample_batched
    print(image.shape)  ## (3, 784)
    print(label)
    if i_batch == 3:
        break
'''