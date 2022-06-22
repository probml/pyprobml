#https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb
#https://pytorch.org/tutorials/beginner/nn_tutorial.html

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml


import torch
from torch import nn
import torch.nn.functional as F

np.random.seed(42)

n = 100
x = torch.ones(n, 2, requires_grad=False) 
x[:,0].uniform_(-1.,1)


def mse(y_hat, y): return ((y_hat-y)**2).mean()
#def mse(y, y_pred): return (y_pred - y).pow(2).sum()

a = torch.as_tensor(np.array([3.0,2.0])).float()
y = x@a + torch.rand(n)

plt.scatter(x[:,0],y)


# must cast parameters to float to match type of x
#a = torch.as_tensor(np.array([-1.,1])).float()
#a = nn.Parameter(a);
a = torch.randn(2, requires_grad=True)
print(a)

# must prevent backprop passing through y to a
#y = x@a.detach() + torch.rand(n)

lr = 1e-1

def update():
    y_hat = x@a
    loss = mse(y, y_hat)
    if t % 10 == 0: print(loss)
    loss.backward() 
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()

for t in range(100): update()

plt.scatter(x[:,0],y)
plt.scatter(x[:,0],x@a.detach())
plt.show()
pml.savefig('linreg_sgd.pdf')
