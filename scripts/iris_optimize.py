# Try optimizing  binary logistic reg on iris dataset using various solvers

import numpy as np
np.set_printoptions(precision=3)
import sklearn
import scipy
from scipy.misc import logsumexp
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

# We make some wrappers around random number generation
# so it works even if we switch from numpy to JAX
import numpy as onp # original numpy

def set_seed(seed):
    onp.random.seed(seed)
    
def randn(*args):
    return onp.random.randn(*args)
        
def randperm(args):
    return onp.random.permutation(args)

USE_JAX = False
USE_TORCH = True
USE_TF = False

if USE_TORCH:
    import torch
    import torchvision
    print("torch version {}".format(torch.__version__))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print("current device {}".format(torch.cuda.current_device()))
    else:
        print("Torch cannot find GPU")
    
    def set_seed(seed):
        onp.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
            
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #torch.backends.cudnn.benchmark = True
   
if USE_JAX:        
    import jax
    import jax.numpy as np
    from jax.scipy.special import logsumexp
    from jax import grad, hessian, jacfwd, jacrev, jit, vmap
    from jax.experimental import optimizers
    print("jax version {}".format(jax.__version__))
    from jax.lib import xla_bridge
    print("jax backend {}".format(xla_bridge.get_backend().platform))
    import os
    os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/home/murphyk/miniconda3/lib"
    

if USE_TF:
    import tensorflow as tf
    from tensorflow import keras
    print("tf version {}".format(tf.__version__))
    if tf.test.is_gpu_available():
        print(tf.test.gpu_device_name())
    else:
        print("TF cannot find GPU")


##
    
# First we create a dataset.

import sklearn.datasets
from sklearn.model_selection import train_test_split

if True:
    iris = sklearn.datasets.load_iris()
    X = iris["data"][:,:] 
    y = (iris["target"] == 2).astype(onp.int)  # 1 if Iris-Virginica, else 0
else:
    X, y = sklearn.datasets.make_classification(
            n_samples=1000, n_features=10, n_informative=5, random_state=42)

N, D = X.shape 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
N_train = X_train.shape[0]
N_test = X_test.shape[0]



#####
# Define model and  objective

def sigmoid(x): return 0.5 * (np.tanh(x / 2.) + 1)

def predict_logit(weights, inputs):
    return np.dot(inputs, weights) # Already vectorized, no bias term

def predict_prob(weights, inputs):
    return sigmoid(predict_logit(weights, inputs))


def NLL(weights, batch):
    # Use log-sum-exp trick
    inputs, targets = batch
    # p1 = 1/(1+exp(-logit)), p0 = 1/(1+exp(+logit))
    logits = predict_logit(weights, inputs).reshape((-1,1))
    N = logits.shape[0]
    logits_plus = np.hstack([np.zeros((N,1)), logits]) # e^0=1
    logits_minus = np.hstack([np.zeros((N,1)), -logits])
    logp1 = -logsumexp(logits_minus, axis=1)
    logp0 = -logsumexp(logits_plus, axis=1)
    logprobs = logp1 * targets + logp0 * (1-targets)
    return -np.sum(logprobs)/N

def NLL_grad(weights, batch):
    X, y = batch
    N = X.shape[0]
    mu = predict_prob(weights, X)
    g = np.sum(np.dot(np.diag(mu - y), X), axis=0)/N
    return g

###########
# Define a test function for comparing solvers

def evaluate_preds(w_opt, w_est, X):
    p_opt = predict_prob(w_opt, X)
    p_est = predict_prob(w_est, X)
    delta = np.max(np.abs(p_opt - p_est))
    print("predictions max delta: {}".format(delta))
    return delta

def evaluate(w_opt, w_est, name):
    print("evaluating {}".format(name))
    delta = np.max(np.abs(w_opt - w_est))
    print("parameters max delta: {}".format(delta))
    train_delta = evaluate_preds(w_opt, w_est, X_train)
    test_delta = evaluate_preds(w_opt, w_est, X_test)
    train_delta = NLL(w_est, (X_train, y_train))
    test_delta = NLL(w_est, (X_test, y_test)) 
    return train_delta, test_delta

###
# Fit with sklearn. We will use this as the "gold standard"

from sklearn.linear_model import LogisticRegression

# We set C to a large number to turn off regularization.
# We don't fit the bias term to simplify the comparison below.
log_reg = LogisticRegression(solver="lbfgs", C=1e5, fit_intercept=False)
log_reg.fit(X_train, y_train)
w_mle_sklearn = np.ravel(log_reg.coef_)

#### Use scipy-BFGS

import scipy.optimize

def training_loss(w):
    return NLL(w, (X_train, y_train))

def training_grad(w):
    return NLL_grad(w, (X_train, y_train))

set_seed(0)
w_init = randn(D)
w_mle_scipy = scipy.optimize.minimize(training_loss, w_init, jac=training_grad, method='BFGS').x   
evaluate(w_mle_sklearn, w_mle_scipy, "scipy-bfgs")


#### Use scipy-BFGS + JAX

if USE_JAX:

    @jit
    def training_loss(w):
        return NLL(w, (X_train, y_train))
    
    @jit
    def training_grad(w):
        return grad(training_loss)(w) 
    
    set_seed(0)
    w_init = randn(D)
    w_mle_scipy = scipy.optimize.minimize(training_loss, w_init, jac=training_grad, method='BFGS').x   
    evaluate(w_mle_sklearn, w_mle_scipy, "scipy-bfgs-jax")
    
###################
# pytorch 

#https://github.com/yangzhangalmo/pytorch-iris/blob/master/main.py
#https://m-alcu.github.io/blog/2018/02/10/logit-pytorch/


import torch
from torch.utils.data import DataLoader, TensorDataset

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(D, 1, bias=False) 
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
x_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
data_set = TensorDataset(x_train_tensor, y_train_tensor)
criterion = torch.nn.BCELoss(reduction='mean')

expts = []
#expts.append({'lr':0.1, 'bs':N_train, 'epochs':1000})
ep = 100
#expts.append({'lr':1, 'bs':2, 'epochs':ep})
expts.append({'lr':0.1, 'bs':2, 'epochs':ep})
expts.append({'lr':0.01, 'bs':2, 'epochs':ep})
expts.append({'lr':'armijo', 'bs':2, 'epochs':ep})
expts.append({'lr':'armijo', 'bs':10, 'epochs':ep})
expts.append({'lr':'armijo', 'bs':N_train, 'epochs':ep})

#  pytorch using SGD with armijo line search
# https://github.com/IssamLaradji/stochastic_line_search/blob/master/main.py
from armijo_sgd import SGD_Armijo, ArmijoModel
   
for expt in expts:
    lr = expt['lr']
    bs = expt['bs']
    max_epochs = expt['epochs']
    seed = 0
    set_seed(seed)
    model = Model()
    model.train() # set to training mode
    data_loader = DataLoader(data_set, batch_size=bs, shuffle=True)
    n_batches = len(data_loader)
    loss_history = []
    print_every = max(1, int(0.25*max_epochs))
    if lr == 'armijo':
        name = 'sgd-armijo-bs{}'.format(bs)
        opt_model = ArmijoModel(model, criterion)
        optimizer = SGD_Armijo(opt_model, batch_size=bs, dataset_size=N_train)   
        opt_model.opt = optimizer
        armijo = True
    else:
        name = 'sgd-lr{:0.3f}-bs{}'.format(lr, bs)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        armijo = False
    
    print('starting {}'.format(name))
    for epoch in range(max_epochs):
        loss_sum = 0.0
        for step, (x_batch, y_batch) in enumerate(data_loader):
            if armijo:     
                loss = opt_model.step((x_batch, y_batch))
                loss_sum += loss
            else:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().numpy()
        train_loss = loss_sum / n_batches
        loss_history.append(train_loss)
        if epoch % print_every == 0:
            print("epoch {}, loss {}".format(epoch, train_loss)) 
    print("Final epoch {}, loss {}".format(epoch, train_loss))   
        
    params_torch = list(model.parameters())
    w_torch = params_torch[0][0].detach().numpy() #(D,) vector
    #offset = params_torch[1].detach().numpy() # scalar
    train_delta, test_delta = evaluate(w_mle_sklearn, w_torch, name)
    plt.plot(loss_history)
    plt.title('{}, train {:0.3f}, test {:0.3f}'.format(name, train_delta, test_delta))
    plt.show()
    
    



# Bare bones SGD


def sgd_v1(params, loss_fn, batcher, max_epochs, lr):
    loss_history = []
    total_steps = 0
    print_every = max(1, int(0.1*max_epochs))
    for epoch in range(max_epochs):
        start_time = time.time()
        for step in range(batcher.num_batches):
            total_steps = total_steps + 1
            batch = next(batcher.batch_stream)
            batch_loss = loss_fn(params, batch)
            batch_grad = grad(loss_fn)(params, batch)
            params = params - lr*batch_grad
        epoch_time = time.time() - start_time
        train_loss = onp.float(loss_fn(params, (batcher.X, batcher.y)))
        loss_history.append(train_loss)
        if epoch % print_every == 0:
            print('Epoch {}, train NLL {}'.format(epoch, train_loss))
    return params, loss_history
    

