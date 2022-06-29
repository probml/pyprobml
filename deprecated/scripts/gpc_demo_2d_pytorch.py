# Gaussian Process Classifier demo
# Author: Drishtii@
# Based on
# https://github.com/probml/pmtk3/blob/master/demos/gpcDemo2d.m

#!pip install gpytorch

import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls.variational_elbo import VariationalELBO

# make synthetic data
np.random.seed(9)
n1=80
n2=40
S1 = np.eye(2)
S2 = np.array([[1, 0.95], [0.95, 1]])
m1 = np.array([0.75, 0]).reshape(-1, 1)
m2 = np.array([-0.75, 0])
xx = np.repeat(m1, n1).reshape(2, n1)
yy = np.repeat(m2, n2).reshape(2, n2)
x1 = np.linalg.cholesky(S1).T @ np.random.randn(2,n1) + xx
x2 = np.linalg.cholesky(S2).T @ np.random.randn(2,n2) + yy
x = np.concatenate([x1.T, x2.T])
y1 = -np.ones(n1).reshape(-1, 1)
y2 = np.ones(n2).reshape(-1, 1)
y = np.concatenate([y1, y2])
q = np.linspace(-4, 4, 81)
r = np.linspace(-4, 4, 81)
t1, t2 = np.meshgrid(q, r)
tgrid = np.hstack([t1.reshape(-1, 1), t2.reshape(-1, 1)])

class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

def train(model, init_lengthscale, init_sigmaf_2):
  model.covar_module.base_kernel.lengthscale = init_lengthscale
  model.covar_module.outputscale = init_sigmaf_2
  likelihood = gpytorch.likelihoods.BernoulliLikelihood()
  # Training:
  model.train()
  likelihood.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
  mll = VariationalELBO(likelihood, model, train_y.numel())
  training_iter = 200
  for i in range(training_iter):
      optimizer.zero_grad()
      model = model.double()
      output = model(train_x.double()) 
      train_y2 = train_y.view(120)
      loss = -mll(output, train_y2)
      loss.backward()
      optimizer.step()

def plot(model, likelihood):  
    model.eval()
    likelihood.eval()
    with torch.no_grad(): 
      model = model.double()    
      observed_pred = likelihood(model(test_x))
      # Plot:
      f, ax = plt.subplots(1, 1, figsize=(6, 5))
      pred_labels = observed_pred.mean.view(81, 81) 
      ax.scatter(x1_torch[0, :], x1_torch[1, :], marker='o')
      ax.scatter(x2_torch[0, :], x2_torch[1, :], marker='+')
      ax.contour(test_x1, test_x2, pred_labels.numpy()) 
      ax.contour(test_x1, test_x2, pred_labels.numpy(), [0.5], colors=['red'])

# Converting numpy data to torch:
train_x = torch.from_numpy(x)
train_x = train_x.double()
train_y = torch.from_numpy(y)
train_y = train_y.double()
test_x = torch.from_numpy(tgrid) 
test_x1 = torch.from_numpy(t1)
test_x2 = torch.from_numpy(t2)
x1_torch = torch.from_numpy(x1)
x2_torch = torch.from_numpy(x2)

# Manual parameters: [No training]
model = GPClassificationModel(train_x)
init_lengthscale = 0.5
init_sigmaf_sq = 10.0
model.covar_module.base_kernel.lengthscale = init_lengthscale
model.covar_module.outputscale = init_sigmaf_sq
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
# Plotting initial model
plot(model, likelihood)
pml.savefig('gpc2d_manual_params.pdf')


# Learned parameters:
model2 = GPClassificationModel(train_x)
init_lengthscale_2 = 1.0
init_sigmaf_sq_2 = 1.0
train(model2, init_lengthscale_2, init_sigmaf_sq_2)
# Plotting fitted model:
plot(model2, likelihood)
pml.savefig('gpc2d_learned_params.pdf')