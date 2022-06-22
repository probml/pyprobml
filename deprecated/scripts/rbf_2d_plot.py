
import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
from pyprobml_utils import save_fig

from sklearn.cluster import KMeans
import matplotlib.colors


from scipy.stats import multivariate_normal
    
import scipy.io
from mpl_toolkits.mplot3d import Axes3D


mu = np.array([[-1.5,1.5], [-0.75,-1.5], [0.75,0]]) # K*2
s = 0.08
Sigma = [ s*np.eye(2),  s*np.eye(2), np.array([[s,0], [0, 3]])]
weights = np.array([2,2,10])
mixdist = dict()
K = np.shape(mu)[0]
for k in range(K):
    pdist = multivariate_normal(mean=mu[k], cov=Sigma[k])
    mixdist[k] = pdist
    
    
def oracle(X):
    K = np.shape(mu)[0]
    N = np.shape(X)[0]
    lik = np.zeros((N))
    for k in range(K):
        lik = lik + weights[k] * mixdist[k].pdf(X)
    return lik

def noisy_oracle(X):
    f = oracle(X)
    n = len(f)
    return f + np.random.randn(n) * 0.1

n = 50
xrange = np.linspace(-2, 2, n)
yrange = np.linspace(-2, 2, n)
xx, yy = np.meshgrid(xrange, yrange)
flatxx = xx.reshape((n**2, 1))
flatyy = yy.reshape((n**2, 1))
X = np.column_stack((flatxx, flatyy))
f = noisy_oracle(X)  

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xx, yy, f.reshape(n, n),
              rstride=1, cstride=1, cmap='jet')

save_fig('rbf-boss-surface.png')
plt.show()

N  = X.shape[0]
perm = np.random.permutation(N)
Ntrain = 500
ndx = perm[:Ntrain]
XX = X[ndx,:]
ff = f[ndx]
fmax = np.max(ff)
thresh = fmax*0.6
thresh = 1.0
yy = ff >= thresh

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(XX[yy==1, 0], XX[yy==1, 1], "ro")
plt.plot(XX[yy==0, 0], XX[yy==0, 1], "bx")
save_fig('rbf-boss-binary.png')
plt.show()



def quantize_data(XX, centroids):
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.cluster_centers_ = centroids
    zz = kmeans.predict(XX)
    return zz

def plot_quantized_data(XX, centroids, zz, offset, fname):
    K = np.max(zz)+1
    cmap = plt.cm.rainbow
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=K-1)
    #https://stackoverflow.com/questions/43009724/how-can-i-convert-numbers-to-a-color-scale-in-matplotlib
    plt.figure()
    for k in range(K):
        ndx = np.where(zz==k)
        color = cmap(cmap_norm(k))
        plt.plot(XX[ndx, 0], XX[ndx, 1], 'o', color=color)
        plt.plot(centroids[k,0], centroids[k,1], 'kx')
        plt.text(centroids[k,0], centroids[k,1], '{}'.format(k+offset), fontsize=14)
    save_fig(fname)
    plt.show()


B = 5
xrange = np.linspace(-1.5, 1.5, B)
yrange = np.linspace(-1.5, 1.5,  B)
K = B**2
centroids2d = np.zeros((K,2))
k = 0
for i in range(B):
    for j in range(B):
        centroids2d[k] = np.array([xrange[i], yrange[j]])
        k += 1
        
centroidsv = np.zeros((B,2))
for b in range(B):
    centroidsv[b] = np.array([0, yrange[b]])
   
centroidsh = np.zeros((B,2))
for b in range(B):
    centroidsh[b] = np.array([xrange[b], 0])
      
        

zv = quantize_data(XX, centroidsv)
plot_quantized_data(XX, centroidsv, zv, 0, 'rbf-boss-vqv.png')

zh = quantize_data(XX, centroidsh)
plot_quantized_data(XX, centroidsh, zh, B, 'rbf-boss-vqh.png')

z2d = quantize_data(XX, centroids2d)
plot_quantized_data(XX, centroids2d, z2d, 2*B, 'rbf-boss-vq2d.png')


if 0:
    #centroids2 = np.array([[0,1], [0,-1]]) # split vertically (x0=*)
    #centroids3 = np.array([[-1,0], [1,0]]) # split horizontally (x1=*)
    #ZZ2 = quantize_data(XX, centroids2)
    #plot_quantized_data(XX, centroids2, ZZ2)
    ZZ3 = quantize_data(XX, centroids3)
    plot_quantized_data(XX, centroids3, ZZ3)
