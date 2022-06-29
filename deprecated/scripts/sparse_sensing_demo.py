import superimport

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import Lasso

def mse(w, f):
  return np.mean((w-f)**2)

np.random.seed(0)
n, k, sigma = 2**12, 2**10, 1e-2
n_spikes = 160
f = np.zeros((n,1))
perm = np.random.permutation(n)
f[perm[:n_spikes]] = np.sign(np.random.randn(n_spikes,1))

R = np.linalg.qr( np.random.randn(n, k))[0].T
y = R @ f + sigma*np.random.randn(k, 1)
l_max = 0.1 * np.linalg.norm(R.T  @ y, np.inf)

clf = Lasso(alpha=l_max/k, tol=1e-2)
clf.fit(R, y)

w = clf.coef_
ndx = np.where(np.abs(w) > 0.01 * np.max(np.abs(w)))[0]
w_debiased = np.zeros((n,1))
w_debiased[ndx,:] = np.linalg.pinv(R[:,ndx]) @ y
w_ls = R.T @ y

titles= [f'Original (D = {n}, number of nonzeros = {n_spikes})', 'L1 Reconstruction (K0 = {}, lambda = {:.4f}, MSE = {:.4f})'.format(k, l_max, mse(w,f)),
         'Debiased (MSE = {:.4E})'.format(mse(w_debiased,f)), 'Minimum Norm Solution (MSE = {:.4f})'.format(mse(w_ls, f))]

fig, axes = plt.subplots(nrows=4, ncols=1)
fig.set_figheight(8)
fig.set_figwidth(6)

weights = [f, w, w_debiased, w_ls]

for i, ax in enumerate(axes.flat, start=1):
    ax.plot(weights[i-1], linewidth=1.1)
    ax.set_title(titles[i-1])
    ax.set_xlim(0, n)
    ax.set_ylim(-1, 1)

fig.tight_layout()
