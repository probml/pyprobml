# Desmonstrate automatic differentiaiton on binary logistic regression
# using JAX, Torch and TF

import numpy as np
from scipy.misc import logsumexp

USE_JAX = True
USE_TORCH = True
USE_TF = True

# We make some wrappers around random number generation
# so it works even if we switch from numpy to JAX
import numpy as onp # original numpy

def set_seed(seed):
    onp.random.seed(seed)
    
def randn(args):
    return onp.random.randn(args)
        
def randperm(args):
    return onp.random.permutation(args)


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
    import numpy as onp
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

def sigmoid(x): return 0.5 * (np.tanh(x / 2.) + 1)

def predict_logit(weights, inputs):
    return np.dot(inputs, weights) # Already vectorized

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


import sklearn.datasets
from sklearn.model_selection import train_test_split

iris = sklearn.datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 2).astype(onp.int)  # 1 if Iris-Virginica, else 0'
N, D = X.shape # 150, 4


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

# We set C to a large number to turn off regularization.
# We don't fit the bias term to simplify the comparison below.
log_reg = LogisticRegression(solver="lbfgs", C=1e5, fit_intercept=False)
log_reg.fit(X_train, y_train)
w_mle_sklearn = np.ravel(log_reg.coef_)

set_seed(0)
w = w_mle_sklearn

y_pred = predict_prob(w, X_test)
loss = NLL(w, (X_test, y_test))
grad_np = NLL_grad(w, (X_test, y_test))
print("params {}".format(w))
print("pred {}".format(y_pred))
print("loss {}".format(loss))
print("grad {}".format(grad_np))

if USE_JAX:
    print("Starting JAX demo")
    grad_jax = grad(NLL)(w, (X_test, y_test))
    print("grad {}".format(grad_jax))
    
if USE_TORCH:
    import torch
    
    print("Starting torch demo")
    w1 = torch.Tensor(np.reshape(w, [D, 1])).to(device)
    w_torch = w1
    w_torch.requires_grad_() 
    x_test_tensor = torch.Tensor(X_test).to(device)
    y_test_tensor = torch.Tensor(y_test).to(device)
    y_pred = torch.sigmoid(torch.matmul(x_test_tensor, w_torch))[:,0]
    criterion = torch.nn.BCELoss(reduction='mean')
    loss_torch = criterion(y_pred, y_test_tensor)
    loss_torch.backward()
    grad_torch = w_torch.grad
    print("params {}".format(w_torch))
    #print("pred {}".format(y_pred))
    print("loss {}".format(loss_torch))
    print("grad {}".format(grad_torch))
 
if USE_TORCH:
    print("Starting torch demo: nn.Module version")
    
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(D, 1, bias=False) 
            
        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred
    
    model = Model()
    print(model.state_dict())
    from collections import OrderedDict
    w1 = torch.Tensor(np.reshape(w, [1, D])).to(device) # row vector
    new_state_dict = OrderedDict({'linear.weight': w1})
    model.load_state_dict(new_state_dict, strict=False)
    #print(model.state_dict())
    model.to(device) # make sure new params are on same device as data
    
    criterion = torch.nn.BCELoss(reduction='mean')
    y_pred = model(x_test_tensor)[:,0]
    loss_torch = criterion(y_pred, y_test_tensor)
    loss_torch.backward()
    params_torch = list(model.parameters())
    grad_torch = params_torch[0].grad
    
    print("params {}".format(w1))
    #print("pred {}".format(y_pred))
    print("loss {}".format(loss_torch))
    print("grad {}".format(grad_torch))
    
if USE_TF:
    print("Starting TF demo")
    w_tf = tf.Variable(np.reshape(w, (D,1)))  
    x_test_tf = tf.convert_to_tensor(X_test, dtype=np.float64) 
    y_test_tf = tf.convert_to_tensor(np.reshape(y_test, (-1,1)), dtype=np.float64)
    with tf.GradientTape() as tape:
        logits = tf.linalg.matmul(x_test_tf, w_tf)
        y_pred = tf.math.sigmoid(logits)
        loss_batch = tf.nn.sigmoid_cross_entropy_with_logits(y_test_tf, logits)
        loss_tf = tf.reduce_mean(loss_batch, axis=0)
    grad_tf = tape.gradient(loss_tf, [w_tf])
    print("params {}".format(w_tf))
    print("pred {}".format(y_pred))
    print("loss {}".format(loss_tf))
    print("grad {}".format(grad_tf))