# Kernel PCA toy example for k(x,y)=exp(-||x-y||^2/rbf_var)
# Author: Animesh Gupta

import superimport

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.spatial.distance import pdist, cdist, squareform

rbf_var = 0.1
xnum = 4
ynum = 2
max_ev = xnum*ynum
x_test_num = 15
y_test_num = 15
cluster_pos = np.array([[-0.5, -0.2], [0, 0.6], [0.5, 0]])
cluster_size = 30

# generate a toy data set
###########################################################################
num_clusters = np.shape(cluster_pos)[0]
train_num = num_clusters*cluster_size
patterns = np.zeros((train_num, 2))
rnge = 1
np.random.seed(1337)
for i in range(num_clusters):
    patterns[(i)*cluster_size:(i+1)*cluster_size,0] = cluster_pos[i,0]+0.1*np.random.randn(cluster_size)
    patterns[(i)*cluster_size:(i+1)*cluster_size,1] = cluster_pos[i,1]+0.1*np.random.randn(cluster_size)

test_num = x_test_num*y_test_num
x_range_gap = (2*rnge /(x_test_num - 1))
x_range = np.arange(-rnge,rnge+x_range_gap,x_range_gap)
y_offset = 0.5
y_range_gap = (2*rnge/(y_test_num - 1))
y_range = np.arange(-rnge + y_offset,rnge + y_offset + y_range_gap, y_range_gap)  
xs, ys = np.meshgrid(x_range, y_range)
test_patterns = np.zeros((225,2))
test_patterns[:, 0] = np.ravel(xs)
test_patterns[:, 1] = np.ravel(ys)
cov_size = train_num  # use all patterns to compute the covariance matrix

# carry out Kernel PCA
#############################################################################
K = np.zeros((cov_size,cov_size))

K = np.exp(-squareform(pdist(patterns)**2)/rbf_var)
K = K.T

unit = np.ones((cov_size, cov_size))/cov_size
# centering in feature space!
K_n = K - unit @ K - K @ unit + unit @ K @ unit

[evals, evecs] = la.eig(K_n)
evals = np.real((evals))
evals = -np.sort(-evals)

for i in range(cov_size):
  evecs[:,i] = evecs[:,i]/(np.sqrt(evals[i]))

unit_test = np.ones((test_num,cov_size))/cov_size
K_test = np.zeros((test_num,cov_size))
K_test = np.exp(-cdist(test_patterns, patterns)**2/rbf_var)
  
K_test_n = K_test - unit_test @ K - K_test @ unit + unit_test @ K @ unit
test_features = np.zeros((test_num, max_ev))                                                             
test_features = K_test_n @ evecs[:,0:max_ev]

# plot it
###############################################################################

fig, ax = plt.subplots(2,4,figsize=(20,10))
for i, axi in enumerate(ax.flat):
    imag = np.reshape(test_features[:,i], [y_test_num, x_test_num])
    axi.set_xticks(np.arange(-1,2,1))
    axi.set_yticks(np.arange(-0.5,2,0.5))
    axi.contour(x_range, y_range, imag, 9, colors= 'b')
    axi.plot(patterns[:,0], patterns[:,1], 'r.')
    axi.text(-0.5,1.60,f'Eigenvalue={evals[i]:.3f}')
fig.savefig('../figures/kpcaScholkopf.pdf', dpi=300)
plt.show()
      


    