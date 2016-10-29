#http://pymc-devs.github.io/pymc3/getting_started/#generalized-linear-models
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import *
from pymc3.glm import glm
import pandas as pd
import scipy

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

df = pd.DataFrame({'x1': X1, 'x2': X2, 'y': Y})

with Model() as model_glm:
    glm('y ~ x1 + x2', df)
    trace = sample(5000)

traceplot(trace);
