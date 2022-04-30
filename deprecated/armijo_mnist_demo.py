# We compare armijo line search to fixed learning rate SGD 
# when used to fit a CNN / MLP to MNIST

# Linesearch code is from
# https://github.com/IssamLaradji/stochastic_line_search/blob/master/main.py
import superimport

from armijo_sgd import SGD_Armijo, ArmijoModel

# Neural net code is based on various tutorials
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
#https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb


import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import pyprobml_utils as pml
import warnings
warnings.filterwarnings('ignore')



import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('Using PyTorch version:', torch.__version__, ' Device:', device)


figdir = "../figures"
import os

############
# Get data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


batch_size = 32
train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

test_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)


for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

bs, ncolors, height, width = X_train.shape
nclasses = 10
N_train = train_dataset.data.shape[0]

#####
# Define model 

import torch.nn as nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss(reduction='mean')
# https://pytorch.org/docs/stable/nn.html#crossentropyloss
# This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single clas
# Therefore we don't need the LogSoftmax on the final layer
# But we do need it if we use NLLLoss

# The Armijo method assumes gradient noise goes to zero,
# so it is important that we don't have dropout layers.

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(ncolors, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        # input is 28x28x1
        # conv1(kernel=5, filters=10) 28x28x10 -> 24x24x10
        # max_pool(kernel=2) 24x24x10 -> 12x12x10
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # conv2(kernel=5, filters=20) 12x12x20 -> 8x8x20
        # max_pool(kernel=2) 8x8x20 -> 4x4x20
        #x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # flatten 4x4x20 = 320
        x = x.view(-1, 320)
        
        # 320 -> 50
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        
        # 50 -> 10
        x = self.fc2(x)
        
        return x
        #return F.log_softmax(x)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(ncolors*height*width, 50)
        #self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        #self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = x.view(-1, ncolors*height*width)
        x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        #x = self.fc2_drop(x)
        x = self.fc3(x)
        #return F.log_softmax(x, dim=1)
        return x

class Logreg(nn.Module):
    def __init__(self):
        super(Logreg, self).__init__()
        self.fc1 = nn.Linear(ncolors*height*width, nclasses)

    def forward(self, x):
        x = x.view(-1, ncolors*height*width)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=1)
        return x
    
def make_model(name, seed=0):
    np.random.seed(seed)
    if name == 'CNN':
        net = CNN()
    elif name == 'MLP':
        net = MLP()
    else:
        net = Logreg()
    net = net.to(device)
    return net

###############

# Define each expermental configuration
expts = []
ep = 4
#model = 'Logreg'
model = 'MLP'
#model = 'CNN'
bs = 10
expts.append({'lr':'armijo', 'bs':bs, 'epochs':ep, 'model': model})
expts.append({'lr':0.01, 'bs':bs, 'epochs':ep, 'model': model})
expts.append({'lr':0.1, 'bs':bs, 'epochs':ep, 'model': model})
#expts.append({'lr':0.5, 'bs':bs, 'epochs':ep, 'model': model})

@torch.no_grad()
def eval_loss(model, loader):    
    avg_loss = 0.0
    model.eval()
    for step, (x_batch, y_batch) in enumerate(loader):
        # Copy data to GPU if needed
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        avg_loss += loss.item()
    # Compute average loss per example
    # Note that the criterion already averages within each batch.
    n_batches = len(loader)
    avg_loss /= n_batches 
    return avg_loss       
    
def fit_epoch(model, optimizer, train_loader, loss_history):    
    epoch_loss = 0.0
    model.train()
    for step, (x_batch, y_batch) in enumerate(train_loader):
        # Copy data to GPU if needed
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # Function to (re)evaluate loss and its gradient for this step.
        def closure():
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        batch_loss = loss.item()
        epoch_loss += batch_loss
        loss_history.append(batch_loss)
    # Compute average loss per example for this epoch.
    # Note that the criterion already averages within each batch.
    n_batches = len(train_loader)
    epoch_loss /= n_batches 
    return epoch_loss 

def fit_epoch_armijo(model, optimizer, train_loader, loss_history, step_size_history):    
    epoch_loss = 0.0
    for step, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_loss, step_size = model.step((x_batch, y_batch))
        epoch_loss += batch_loss
        loss_history.append(batch_loss)
        step_size_history.append(step_size)
    n_batches = len(train_loader)
    epoch_loss /= n_batches
    return epoch_loss

    
results_dict = {}
for expt in expts:
    lr = expt['lr']
    bs = expt['bs']
    max_epochs = expt['epochs']
    model_name = expt['model']
    model = make_model(model_name)
    model.train() # set to training mode
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
                                          shuffle=True, num_workers=2)
    n_batches = len(train_loader)
    batch_loss_history = []
    epoch_loss_history = []
    step_size_history = []
    print_every = max(1, int(0.1*max_epochs))
    if lr == 'armijo':
        name = '{}-armijo-bs{}'.format(model_name, bs)
        model = ArmijoModel(model, criterion)
        optimizer = SGD_Armijo(model, batch_size=bs, dataset_size=N_train)  
        model.opt = optimizer
        armijo = True
    else:
        name = '{}-lr{:0.3f}-bs{}'.format(model_name, lr, bs)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        armijo = False
    
    print('starting {}'.format(name))
    for epoch in range(max_epochs):
        if armijo:
           avg_batch_loss = fit_epoch_armijo(model, optimizer, train_loader, batch_loss_history, step_size_history)
        else:
            avg_batch_loss = fit_epoch(model, optimizer, train_loader, batch_loss_history)
        epoch_loss = eval_loss(model, train_loader)
        epoch_loss_history.append(epoch_loss)
        if epoch % print_every == 0:
            print("epoch {}, loss {}".format(epoch, epoch_loss)) 
            
    label = '{}-final-loss{:0.3f}'.format(name, epoch_loss)
    results = {'label': label, 'batch_loss_history': batch_loss_history,
               'epoch_loss_history': epoch_loss_history, 'step_size_history': step_size_history}
    results_dict[name] = results


plt.figure()
name = 'MLP-armijo-bs10'
results = results_dict[name]
plt.plot(results['step_size_history'])
plt.ylabel('stepsize')
pml.savefig('armijo-mnist-stepsize.pdf')
plt.show()

plt.figure()
for name, results in results_dict.items():
    label = results['label']
    y = results['epoch_loss_history']
    plt.plot(y, label=label)
    plt.legend()
pml.savefig('armijo-mnist-epoch-loss.pdf')
plt.show()

# Add smoothed version of batch loss history to results dict    
import pandas as pd
for name, results in results_dict.items():
    loss_history = results['batch_loss_history']  
    df = pd.Series(loss_history)
    nsteps = len(loss_history)
    smoothed = pd.Series.ewm(df, span=0.1*nsteps).mean()
    results['batch_loss_history_smoothed'] = smoothed
    
# Plot curves on one figure
plt.figure() 
for name, results in results_dict.items():
    label = results['label']
    y = results['batch_loss_history_smoothed']
    nsteps = len(y)
    x = np.arange(nsteps)
    ndx = np.arange(int(0.2*nsteps), nsteps) # skip first 20%
    #plt.figure()
    plt.plot(x[ndx], y[ndx], label=label)
plt.legend()
pml.savefig('armijo-mnist-batch-loss.pdf')
plt.show()

