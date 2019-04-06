
import os
os.environ["XLA_FLAGS"]="--xla_cpu_enable_fast_math=false"

import numpy as onp # original numpy
import jax.numpy as np

print(np.isnan(np.nan)) #F
print(onp.isnan(np.nan)) #T
print(np.isnan(onp.nan)) #F
print(onp.isnan(onp.nan)) #T
