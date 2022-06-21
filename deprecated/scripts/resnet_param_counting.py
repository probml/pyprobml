# https://github.com/google/flax/tree/main/examples/imagenet
# resnet 18

params = state.params


from jax.flatten_util import ravel_pytree
params_flat, flat_to_pytree_fn = ravel_pytree(params)
print(params_flat.shape)

def count_blocks(names):
  n = 0
  for name in names:
    parts = name.split('_')
    #print(parts)
    if parts[0] == 'ResNetBlock':
      n = n + 1
      #print(parts)
  return n



def count_conv(names):
  n = 0
  for name in names:
    parts = name.split('_')
    #print(parts)
    if parts[0] == 'Conv':
      n = n + 1
      #print(parts)
  return n


def print_params(layer, name):
  if 'bias' in layer.keys():
    bias = layer['bias'].shape
  else:
    bias = 0
  if 'kernel' in layer.keys():
     kernel = layer['kernel'].shape
  else:
    kernel = 0
  if 'scale' in layer.keys():
     scale = layer['scale'].shape
  else:
    scale = 0
  n = np.prod(bias) + np.prod(kernel) + np.prod(scale)
  print(name, 'bias ', bias, 'kernel ', kernel, 'scale ', scale, 'n ', n)
  return n

n = 0
nn = print_params(params['conv_init'], 'conv_init')
n += nn
nn = print_params(params['bn_init'], 'bn_init')
n += nn
nblocks = count_blocks(params.keys())
for b in range(nblocks):
  block = params[f'ResNetBlock_{b}']
  for k in block.keys():
    name = f'{b}/{k}'
    nn = print_params(block[k], name)
    n += nn
nn = print_params(params['Dense_0'], 'Dense_0')
n += nn

print(n)
print(params_flat.shape)

'''
conv_init bias  0 kernel  (8, 7, 7, 3, 64) scale  0 n  75264
bn_init bias  (8, 64) kernel  0 scale  (8, 64) n  1024
0/BatchNorm_0 bias  (8, 64) kernel  0 scale  (8, 64) n  1024
0/BatchNorm_1 bias  (8, 64) kernel  0 scale  (8, 64) n  1024
0/Conv_0 bias  0 kernel  (8, 3, 3, 64, 64) scale  0 n  294912
0/Conv_1 bias  0 kernel  (8, 3, 3, 64, 64) scale  0 n  294912
1/BatchNorm_0 bias  (8, 64) kernel  0 scale  (8, 64) n  1024
1/BatchNorm_1 bias  (8, 64) kernel  0 scale  (8, 64) n  1024
1/Conv_0 bias  0 kernel  (8, 3, 3, 64, 64) scale  0 n  294912
1/Conv_1 bias  0 kernel  (8, 3, 3, 64, 64) scale  0 n  294912
2/BatchNorm_0 bias  (8, 128) kernel  0 scale  (8, 128) n  2048
2/BatchNorm_1 bias  (8, 128) kernel  0 scale  (8, 128) n  2048
2/Conv_0 bias  0 kernel  (8, 3, 3, 64, 128) scale  0 n  589824
2/Conv_1 bias  0 kernel  (8, 3, 3, 128, 128) scale  0 n  1179648
2/conv_proj bias  0 kernel  (8, 1, 1, 64, 128) scale  0 n  65536
2/norm_proj bias  (8, 128) kernel  0 scale  (8, 128) n  2048
3/BatchNorm_0 bias  (8, 128) kernel  0 scale  (8, 128) n  2048
3/BatchNorm_1 bias  (8, 128) kernel  0 scale  (8, 128) n  2048
3/Conv_0 bias  0 kernel  (8, 3, 3, 128, 128) scale  0 n  1179648
3/Conv_1 bias  0 kernel  (8, 3, 3, 128, 128) scale  0 n  1179648
4/BatchNorm_0 bias  (8, 256) kernel  0 scale  (8, 256) n  4096
4/BatchNorm_1 bias  (8, 256) kernel  0 scale  (8, 256) n  4096
4/Conv_0 bias  0 kernel  (8, 3, 3, 128, 256) scale  0 n  2359296
4/Conv_1 bias  0 kernel  (8, 3, 3, 256, 256) scale  0 n  4718592
4/conv_proj bias  0 kernel  (8, 1, 1, 128, 256) scale  0 n  262144
4/norm_proj bias  (8, 256) kernel  0 scale  (8, 256) n  4096
5/BatchNorm_0 bias  (8, 256) kernel  0 scale  (8, 256) n  4096
5/BatchNorm_1 bias  (8, 256) kernel  0 scale  (8, 256) n  4096
5/Conv_0 bias  0 kernel  (8, 3, 3, 256, 256) scale  0 n  4718592
5/Conv_1 bias  0 kernel  (8, 3, 3, 256, 256) scale  0 n  4718592
6/BatchNorm_0 bias  (8, 512) kernel  0 scale  (8, 512) n  8192
6/BatchNorm_1 bias  (8, 512) kernel  0 scale  (8, 512) n  8192
6/Conv_0 bias  0 kernel  (8, 3, 3, 256, 512) scale  0 n  9437184
6/Conv_1 bias  0 kernel  (8, 3, 3, 512, 512) scale  0 n  18874368
6/conv_proj bias  0 kernel  (8, 1, 1, 256, 512) scale  0 n  1048576
6/norm_proj bias  (8, 512) kernel  0 scale  (8, 512) n  8192
7/BatchNorm_0 bias  (8, 512) kernel  0 scale  (8, 512) n  8192
7/BatchNorm_1 bias  (8, 512) kernel  0 scale  (8, 512) n  8192
7/Conv_0 bias  0 kernel  (8, 3, 3, 512, 512) scale  0 n  18874368
7/Conv_1 bias  0 kernel  (8, 3, 3, 512, 512) scale  0 n  18874368
Dense_0 bias  (8, 1000) kernel  (8, 512, 1000) scale  0 n  4104000
93516096
(93516096,)

'''


# for mnist

from jax.flatten_util import ravel_pytree
params = models[name]
params_flat, flat_to_pytree_fn = ravel_pytree(params)
print(params_flat.shape)

params = models[name]
nparams = 0
for k in params.keys():
  print(k)
  print(params[k]['bias'].shape)
  print(params[k]['kernel'].shape)
  nparams += np.prod(params[k]['bias'].shape)
  nparams += np.prod(params[k]['kernel'].shape)
print(nparams)
