# check gpu

print("\n\n\n\nTest Torch\n\n")
import superimport

import torch
print("torch version {}".format(torch.__version__))
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print("current device {}".format(torch.cuda.current_device()))
else:
    print("Torch cannot find GPU")
    x = torch.Tensor(3,3)

print("\n\n\n\nTest JAX\n\n")  
import jax
import jax.numpy as np
print("jax version {}".format(jax.__version__))
from jax.lib import xla_bridge
print("jax backend {}".format(xla_bridge.get_backend().platform))
from jax import random
key = random.PRNGKey(0)
x = random.normal(key, (5,5))
    
print("\n\n\n\nTest TF\n\n")
import tensorflow as tf
print("tf version {}".format(tf.__version__))
if tf.test.is_gpu_available():
    print(tf.test.gpu_device_name())
else:
    print("TF cannot find GPU")
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(c)

