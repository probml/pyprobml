'''
Finds the stationary distributions for the graph consisting of 6 nodes and one for Harvard500. Then, compares
the results found by matrix inversion and power method. Plots the bar plots.
Author: Cleve Moler, Aleyna Kara
This file is converted to Python from https://github.com/probml/pmtk3/blob/master/demos/pagerankDemoPmtk.m
'''

import superimport

import numpy as np
from pagerank_power_method_sparse import pagerank_power_method_sparse
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from scipy.io import loadmat

import requests
from io import BytesIO

url = 'https://github.com/probml/probml-data/blob/main/data/harvard500.mat?raw=true'
response = requests.get(url)
rawdata = BytesIO(response.content)
mat = loadmat(rawdata)
G_harvard = mat['G']

#plt.figure(figsize=(6, 6))
plt.spy(G_harvard, c='blue', marker='.', markersize=1)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
pml.savefig('harvard500-spy')
plt.show()

p = 0.85
pi_sparse_harvard = pagerank_power_method_sparse(G_harvard, p)[0]

fig, ax = plt.subplots()
plt.bar(np.arange(0, pi_sparse_harvard.shape[0]), pi_sparse_harvard, width=1.0, color='darkblue')
ax.set_ylim([0, 0.02])
pml.savefig('harvard500-pagerank')
plt.show()

