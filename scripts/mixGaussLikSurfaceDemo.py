import numpy as np
import matplotlib.pyplot as plt
import os


if os.path.isdir('scripts'):
    os.chdir('scripts')
fs = 12
np.random.seed(0)
true_mu1 = -10
true_mu2 = 10
true_pi = 0.5

sigmas = np.array([5])
obs = None
for sigmai in sigmas:
    true_sigma = sigmai
    
    n_obs = 100
    obs = ([true_mu1 + true_sigma*np.random.randn(1, n_obs), true_mu2 + true_sigma*np.random.randn(1, n_obs)])
    obs = np.reshape(obs, [1, 200])
    obs = np.transpose(obs)

    histogram = plt.hist(obs)
    plt.savefig(r'../figures/gmmLikSurfaceHistSigma'+str(true_sigma))
    plt.show()
    #print(obs.shape)

