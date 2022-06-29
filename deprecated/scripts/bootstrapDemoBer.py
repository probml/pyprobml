# Bootstrap demo for the MLE for a Bernoulli
# Author: Animesh Gupta

import superimport

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

fs = 12
theta = 0.7
Ns = [10, 100]

for i in range(len(Ns)):
    N = Ns[i]
    B = 10000
    
    X = np.random.randn(1,N) < theta
    bmle = np.mean(X)
    mleBoot = np.zeros((1,B))
    
    for j in range(B):
        Xb = np.random.randn(1,N) < bmle
        mleBoot[:,j] = np.mean(Xb)
        ndx = np.random.randint(0, N, size = N)
        Xnonparam = X[:,ndx]
        mleBootNP = np.zeros((1,B))
        mleBootNP[:,j] = np.mean(Xnonparam)
        
    counts, nbinslocation = np.histogram(mleBoot, bins=10)
    plt.figure(figsize=(8,8))
    plt.title(f'Boot: true = {theta:.2f}, n={N}, mle = {bmle}, se = {(np.std(mleBoot)/np.sqrt(B)):.3f}')
    plt.bar(nbinslocation[:-1], counts, width=nbinslocation[1]-nbinslocation[0], color='tab:blue',align="center")
    plt.xlim(0, 1)
    plt.savefig(f'../figures/bootstrapDemo_{i}.pdf', dpi=300)
    
    N1 = len(np.argwhere(X==1))
    N0 = len(np.argwhere(X==0))
    alpha1 = 1
    alpha0 = 1
    model_a = N1+alpha1
    model_b = N0+alpha0
    Xpost = np.random.beta(model_a,model_b, [1, B])
    
    counts, nbinslocation = np.histogram(Xpost, bins=10)
    plt.figure(figsize=(8,8))
    plt.title(f'Bayes: true = {theta:.2f}, n={N}, post mean = {np.mean(Xpost):.2f}, se = {(np.std(Xpost)/np.sqrt(B)):.3f}')
    plt.bar(nbinslocation[:-1], counts, width=nbinslocation[1]-nbinslocation[0], color='tab:blue',align="center")
    plt.xlim(0, 1)
    plt.show()
    plt.savefig(f'../figures/bayesDemo_{i}.pdf', dpi=300)