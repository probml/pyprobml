# https://course.fast.ai/videos/?lesson=2
#https://pytorch.org/tutorials/beginner/nn_tutorial.html

import numpy as np
import matplotlib.pyplot as plt
import os
#figdir = os.path.join(os.environ["PYPROBML"], "figures")
#def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


import torch
from torch import nn
import torch.nn.functional as F


np.random.seed(42)

n = 100
x = torch.ones(n,2) 
x[:,0].uniform_(-1.,1)


def mse(y_hat, y): return ((y_hat-y)**2).mean()
#def mse(y, y_pred): return (y_pred - y).pow(2).sum()

# must cast parameters to float to match type of x
#a = torch.as_tensor(np.array([-1.,1])).float()
#a = nn.Parameter(a);
a = torch.randn(2, requires_grad=True)
print(a)

y = x@a + torch.rand(n)
y_hat = x@a
lr = 1e-1

def update():
    y_hat = x@a
    #loss = mse(y, y_hat)
    loss = ((y_hat-y)**2).mean()
    if t % 10 == 0: print(loss)
    loss.backward() #retain_graph=True)
    #Trying to backward through the graph a second time, but the buffers have already been freed
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()


for t in range(100): update()

plt.scatter(x[:,0],y)
plt.scatter(x[:,0],x@a);

