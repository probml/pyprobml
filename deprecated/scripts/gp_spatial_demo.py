# Gaussian Process prior with poisson liklihood 
# Author: Drishtii@ 
# Based on 
# https://github.com/probml/pmtk3/blob/master/demos/gpSpatialDemoLaplace.m

#!pip install GPy

import superimport

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import GPy
import pandas as pd
import pyprobml_utils as pml
import requests
import re

url="https://github.com/probml/probml-data/blob/main/data/spatial_data_Finland.txt"
s = requests.get(url)

def clean_data(text):
    clean1 = re.compile('<.*?>')
    clean = re.sub(clean1, '', text)
    clean = re.sub("   ", " ", clean)
    clean = re.sub("   ", " ", clean)
    clean = clean.strip()
    lines = clean.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    string_without_empty_lines = ""
    for line in non_empty_lines:
        if (len(line.split())==4):
          string_without_empty_lines += line + "\n" 
    out = re.sub(r"< .*? >", "", string_without_empty_lines) 
    new = ""
    for i in range(13, len(string_without_empty_lines.split('\n'))-3):
      new = new + string_without_empty_lines.split('\n')[i] + "\n"

    i = 0
    data = np.zeros((911, 4))
    for line in new.split('\n'):
      j = 0
      for item in line.split():
        data[i][j] = float(item)
        j = j + 1
      i = i + 1

    return data

data = clean_data(s.text)

X1 = data[:, 0].reshape(-1, 1) 
X2 = data[:, 1].reshape(-1, 1)
X = np.concatenate((X1, X2), axis=1)
YE = data[:, 2].reshape(-1, 1)
YY = data[:, 3].reshape(-1, 1)
x1 = data[:, 0].astype(int)
x2 = data[:, 1].astype(int)

kernel = GPy.kern.RBF(1, variance=0.3, lengthscale=5.0)

poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

model = GPy.core.GP(X=X, Y=YY, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)

model.optimize()

f_mean, f_var = model._raw_predict(YE)

def plot(f):
    G = np.empty((60, 35))
    G[:] = np.nan 
    i = 0
    for k in range(len(f)):
      G[x2[i], x1[i]] = f[k]
      i = i + 1
    X1, X2= np.meshgrid(np.linspace(1, 35, 35), np.linspace(1, 60, 60))
    G = G[:-1, :-1]
    G_min, G_max = np.min(f), np.max(f)
  
    fig, ax = plt.subplots(figsize=(5, 6))
    c = plt.pcolor(X1, X2, G, cmap='RdBu', vmin=G_min, vmax=G_max)
    fig.colorbar(c, ax=ax)

plot(f_mean)
plt.title('Mean')
pml.savefig('gp_spatial_mean.pdf')
plt.show()

plot(f_var)
plt.title('Variance')
pml.savefig('gp_spatial_variance.pdf')
plt.show()
