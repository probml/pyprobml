


# From
# https://cloud.coiled.io/examples/notebooks

import superimport

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class HiddenLayerNet(nn.Module):
    def __init__(self, n_features=10, n_outputs=1, n_hidden=100, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)
        self.activation = getattr(F, activation)

    def forward(self, x, **kwargs):
        return self.fc2(self.activation(self.fc1(x)))