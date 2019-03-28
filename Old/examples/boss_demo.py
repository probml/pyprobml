import numpy as np
import matplotlib.pyplot as plt
from time import time

import boss_problems, boss_models

np.random.seed(0)


#problem = 'motif'
#problem = 'tfbind'
problem = 'tfbind-small'

if problem == 'motif':
  seq_len = 8
  noise = 0.1
  oracle, oracle_batch, Xall, yall, Xtrain, ytrain, train_ndx = boss_problems.motif_problem(
      seq_len, noise)
  hparams = {'epochs': 10, 'nlayers': 2, 'nhidden': 10, 'embed_dim': 5, 'seq_len': seq_len}
  
if problem == 'tfbind':
  oracle, oracle_batch, Xall, yall, Xtrain, ytrain, train_ndx = boss_problems.tfbind_problem(
      lower_bin=50, upper_bin=99)
  seq_len = np.shape(Xall)[1]
  hparams = {'epochs': 30, 'nlayers': 4, 'nhidden': 100, 'embed_dim': 64, 'seq_len': seq_len}

if problem == 'tfbind-small':
  max_nseq=20000
  oracle, oracle_batch, Xall, yall, Xtrain, ytrain, train_ndx = boss_problems.tfbind_problem(
      lower_bin=50, upper_bin=99, max_nseq=max_nseq)
  seq_len = np.shape(Xall)[1]
  hparams = {'epochs': 10, 'nlayers': 3, 'nhidden': 50, 'embed_dim': 32, 'seq_len': seq_len}



nseq = np.shape(Xall)[0]
 
plt.figure()
plt.plot(range(nseq), yall, 'b')
plt.plot(train_ndx, ytrain, 'r')
m = np.max(yall)
s = np.std(yall)
maxima = np.where(yall >= m-1*s)[0]
plt.title('{} seq, {} maxima of value {:0.3f}'.format(len(yall), len(maxima), m))
plt.show()

time_start = time()
predictor = boss_models.learn_supervised_model(Xtrain, ytrain, hparams)
print('time spent training {:0.3f}\n'.format(time() - time_start))

ypred = predictor.predict(Xall)
plt.figure()
plt.scatter(yall, ypred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

embedder = boss_models.convert_supervised_to_embedder(predictor, hparams)

Z = embedder.predict(Xtrain)
plt.figure()
plt.scatter(Z[:,0], Z[:,1], c=ytrain)
plt.title('embeddings of training set')
plt.colorbar()
plt.show()


from sklearn.metrics.pairwise import pairwise_distances
sources = np.arange(4)
dist_matrix = pairwise_distances(Z[sources], Z)
nearest = np.argsort(dist_matrix, axis=1)
knn = min(nseq, 100)
fig, ax = plt.subplots(2,2,figsize=(10,10))
for i, source in enumerate(sources):
  ysource = oracle(Xall[source])
  nbrs = nearest[source, 0:knn];
  dst = dist_matrix[source, nbrs];
  ytargets = oracle_batch(Xall[nbrs])
  #plt.figure()
  r = i // 2
  c = i % 2
  ax[r,c].plot(dst, ytargets-ysource, 'o')
  ax[r,c].set_title('s={}'.format(source))
plt.xlabel('distance(s,t) in embedding space vs t')
plt.ylabel('f(s)-f(t)')
plt.show()

########


import boss_problems, boss_models
#from bayes_opt_utils import BayesianOptimizer
from bayes_opt_utils import BayesianOptimizerEmbedEnum
#from bayes_opt_utils import EnumerativeDiscreteOptimizer
from bayes_opt_utils import RandomDiscreteOptimizer
from bayes_opt_utils import expected_improvement

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
#from bayes_opt_utils import EmbedKernel

  
noise = 1e-5 # np.std(yinit)
# We tell EI the objective is noise free so that it trusts
# the current best, rather than re-evaluating
def EI(X, X_sample, Y_sample, surrogate):
  X = np.atleast_2d(X)
  return expected_improvement(X, X_sample, Y_sample, surrogate,
                              improvement_thresh=0.01, trust_incumbent=True)
  
def oracle_embed_fn(X):
  return np.reshape(oracle_batch(X), (-1,1))


def predictor_embed_fn(X):
  return predictor.predict(X, batch_size=1000)

def super_embed_fn(X):
  return embedder.predict(X, batch_size=1000)
  

# One-hot encode integers before passing to kernel
from sklearn.preprocessing import OneHotEncoder
alpha_size = 4
cat = np.array(range(alpha_size)); 
cats = [cat]*seq_len
enc =  OneHotEncoder(sparse=False, categories=cats)
enc.fit(Xall)

def onehot_embed_fn(X):
  return enc.transform(X)
#Xhot = enc.transform(X)
#Xcold = enc.inverse_transform(Xhot)
#assert (Xcold==X).all()

  
n_bo_iter = 50
n_bo_init = 10 # # Before starting BO, we perform N random queries
nseq = np.shape(Xall)[0] 

  
def do_expt(seed):
  np.random.seed(seed)
  perm = np.random.permutation(nseq)
  perm = perm[:n_bo_init]
  Xinit = Xall[perm]
  yinit = yall[perm]


    
  rnd_solver = RandomDiscreteOptimizer(Xall, n_iter=n_bo_iter+n_bo_init)
  
  """
  # Embed sequence then pass to kernel.
  # We use Matern kernel 1.5 since this only assumes first-orer differentiability.
  kernel = ConstantKernel(1.0) * EmbedKernel(length_scale=1.0, nu=1.5,
                         embed_fn=lambda x: embedder.predict(x))
  gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
  acq_fn = expected_improvement
  n_seq = np.shape(Xall)[0]
  acq_solver =  EnumerativeDiscreteOptimizer(Xall, n_iter=n_seq)
  bo_embed_solver_slow = BayesianOptimizer(
      Xinit, yinit, gpr, acq_fn, acq_solver, n_iter=n_bo_iter)
  """
  
  
  
  kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
  gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
  acq_fn = EI
  bo_oracle_embed_solver = BayesianOptimizerEmbedEnum(
      Xall, oracle_embed_fn, Xinit, yinit, gpr, acq_fn, n_iter=n_bo_iter) 
  
  
  kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
  gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
  acq_fn = EI
  bo_predictor_embed_solver = BayesianOptimizerEmbedEnum(
      Xall, predictor_embed_fn, Xinit, yinit, gpr, acq_fn, n_iter=n_bo_iter) 
  
  
  kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
  gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
  acq_fn = EI
  bo_super_embed_solver = BayesianOptimizerEmbedEnum(
      Xall, super_embed_fn, Xinit, yinit, gpr, acq_fn, n_iter=n_bo_iter) 
  
  kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
  gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
  acq_fn = EI
  bo_onehot_embed_solver = BayesianOptimizerEmbedEnum(
      Xall, onehot_embed_fn, Xinit, yinit, gpr, acq_fn, n_iter=n_bo_iter)
  
  """
  # Pass integers to kernel.
  kernel = ConstantKernel(1.0) * EmbedKernel(length_scale=1.0, nu=1.5,
                         embed_fn=lambda x: x)
  gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
  acq_fn = expected_improvement
  n_seq = 4**seq_len
  acq_solver =  EnumerativeStringOptimizer(seq_len, n_iter=n_seq)
  bo_int_solver = BayesianOptimizer(Xinit, yinit, gpr, acq_fn, acq_solver, n_iter=n_bo_iter)
  """
  
      
  methods = []
  methods.append((bo_oracle_embed_solver, 'BO-oracle-embed-enum'))
  methods.append((bo_predictor_embed_solver, 'BO-predictor_embed-enum'))
  methods.append((bo_super_embed_solver, 'BO-super-embed-enum'))
  methods.append((bo_onehot_embed_solver, 'BO-onehot-enum'))
  #methods.append((bo_int_solver, 'BO-int-enum'))
  methods.append((rnd_solver, 'RndSolver')) # Always do random last
  
  ytrace = dict()
  for solver, name in methods:
    print("Running {}".format(name))
    time_start = time()
    solver.maximize(oracle)   
    print('time spent by {} = {:0.3f}\n'.format(name, time() - time_start))
    ytrace[name] = np.maximum.accumulate(solver.val_history)
    
  plt.figure()
  styles = ['k-o', 'r:o', 'b--o', 'g-o', 'c:o', 'm--o', 'y-o']
  for i, tuple in enumerate(methods):
    style = styles[i]
    name = tuple[1]
    plt.plot(ytrace[name], style, label=name)
  plt.axvline(n_bo_init)
  plt.legend()
  plt.title("seed = {}".format(seed))
  plt.show()
    
seeds = [0, 1, 2]
for seed in seeds:
  do_expt(seed)