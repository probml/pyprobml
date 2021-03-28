import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

CB_color = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']


def reconstruct_FA(model, muPost):
    Wpsi = model.components_ / model.noise_variance_
    Ih = np.eye(len(model.components_))
    inv_cov_z = Ih + np.dot(Wpsi, model.components_.T)
    tmp = np.dot(muPost, inv_cov_z)
    X_recon = np.dot(tmp, np.linalg.pinv(Wpsi.T)) + model.mean_
    return X_recon


np.random.seed(10)
D = 16
K = 3
proto = np.random.rand(D, K) < 0.5
M = 50
source = np.ravel([0 * np.ones(M), 1 * np.ones(M), 2 * np.ones(M)])
N = len(source)
dataClean = np.zeros((N, D))
for n in range(N):
    src = int(source[n])
    dataClean[n, :] = np.ravel(proto[:, src])

noiseLevel = 0.05
flipMask = np.random.rand(N, D) < noiseLevel
dataNoisy = dataClean.copy()
dataNoisy[flipMask] = 1 - dataClean[flipMask]
dataMissing = dataClean.copy()
dataMissing[flipMask] = np.nan

fig, ax1 = plt.subplots(1, 2, figsize=(10, 5))
ax1[0].imshow(dataNoisy, interpolation='nearest', aspect='auto', cmap='gray')
ax1[0].set_title('Noisy Data')
ax1[1].imshow(dataClean, interpolation='nearest', aspect='auto', cmap='gray')
ax1[1].set_title('Clean Data')
fig.savefig('../figures/binaryPCAinput.png')

model = FactorAnalysis(n_components=2, max_iter=10)
muPost = model.fit_transform(dataNoisy.copy())

fig, ax2 = plt.subplots(1, 1, figsize=(5, 4))
for k in range(K):
    ndx = source == k
    if k == 0:
        ax2.scatter(muPost[ndx, 0], muPost[ndx, 1], color=CB_color[0], marker='o')
    elif k == 1:
        ax2.scatter(muPost[ndx, 0], muPost[ndx, 1], color=CB_color[1], marker='s')
    else:
        ax2.scatter(muPost[ndx, 0], muPost[ndx, 1], color=CB_color[2], marker='*')
ax2.set_title('latent embedding')
fig.savefig('../figures/binaryPCAembedding.png')

X_recon = reconstruct_FA(model, muPost)
yhat = X_recon > 0.5

fig, ax3 = plt.subplots(1, 2, figsize=(10, 5))
ax3[0].imshow(X_recon, interpolation='nearest', aspect='auto', cmap='gray')
ax3[0].set_title('Posterior predictive')
ax3[1].imshow(yhat, interpolation='nearest', aspect='auto', cmap='gray')
ax3[1].set_title('Reconstruction')
fig.savefig('../figures/binaryPCArecon.png')
