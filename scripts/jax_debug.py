
import numpy as onp

import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/home/murphyk/miniconda3/lib"
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit"
os.environ["CUDA_HOME"]="/usr"


import jax
import jax.numpy as np
print("jax version {}".format(jax.__version__))
from jax.lib import xla_bridge
print("jax backend {}".format(xla_bridge.get_backend().platform))


from jax import random
key = random.PRNGKey(0)
x = random.normal(key, (5,5))
print(x)

