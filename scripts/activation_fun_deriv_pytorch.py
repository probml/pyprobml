# Plot some neural net activation functions and their derivatives
# Based on sec 4.1 of
# http://d2l.ai/chapter_multilayer-perceptrons/mlp.html

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

np.random.seed(seed=1)
import torch


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
fns = [torch.relu, torch.sigmoid]
names = ['relu', 'sigmoid']

for i in range(len(fns)):
    fn = fns[i]
    name = names[i]
    y = fn(x)

    plt.figure()
    plt.plot(x.detach(), y.detach())
    plt.title(name)
    plt.show()
    
    plt.figure()
    plt.plot(x.detach(), x.grad)
    plt.title(f'gradient of {name}')
    plt.show()
    
    