
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import os
#%matplotlib inline

# Set random seeds
np.random.seed(0)

# Load data
iris = load_iris()
# Original Iris : (150,4)
X = iris.data 
y = iris.target

from sklearn import logistic_regression
