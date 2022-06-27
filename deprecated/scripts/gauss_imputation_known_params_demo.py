# Authors: shivaditya-meduri@, murphyk@
# Illustration of data imputation using an MVN.
# Based on https://github.com/probml/pmtk3/blob/master/demos/gaussImputationDemo.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
import gauss_utils as gauss
import pyprobml_utils as pml


np.random.seed(12)
data_dim = 8
n_data = 10
threshold_missing = 0.5
mu = np.random.randn(data_dim, 1)
sigma = make_spd_matrix(n_dim=data_dim)  # Generate a random positive semi-definite matrix
# test if the matrix is positive definite
# print(is_pos_def(sigma))
x_full = gauss.gauss_sample(mu, sigma, n_data)
missing = np.random.rand(n_data, data_dim) < threshold_missing
x_miss = np.copy(x_full)
x_miss[missing] = np.nan
x_imputed = gauss.gauss_impute(mu, sigma, x_miss)
#Create a matrix from x_miss by replacing the NaNs with 0s to display the hinton_diagram
xmiss0 = np.copy(x_miss)
for g in np.argwhere(np.isnan(x_miss)):
    xmiss0[g[0], g[1]] = 0

plot_1 = plt.figure(1)
pml.hinton_diagram(xmiss0, ax=plot_1.gca())
plot_1.suptitle('Observed')
pml.savefig("gauss_impute_observed.pdf", dpi=300)

plot_2 = plt.figure(2)
pml.hinton_diagram(x_full, ax=plot_2.gca())
plot_2.suptitle('Hidden truth')
pml.savefig("gauss_impute_truth.pdf", dpi=300)

plot_3 = plt.figure(3)
pml.hinton_diagram(x_imputed, ax=plot_3.gca())
plot_3.suptitle('Imputation with true params')
pml.savefig("gauss_impute_pred.pdf", dpi=300)
plt.show()

