import numpy as np
import matplotlib.pyplot as plt
import os

if os.path.isdir('scripts'):
    os.chdir('scripts')

  
'''def gaussProb(X, mu, Sigma):
    d = np.array([Sigma, 2]).shape[0]
    X  = X.reshape(100, d)
    # X = bsxfun(@minus, X, np.full(mu, mu))
    for i in range(0, 100):
        X[i][0] = X[i][0] - mu
        X[i][1] = X[i][1] - mu
    logp = -0.5*np.sum((X/(Sigma))*X, 2) 
    logZ = (d/2)*np.log(2*np.pi) + 0.5*np.linalg.slogdet(Sigma)
    logp = logp - logZ;
    p = np.exp(logp);    
    return p'''

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

    dmu = .5;
    mu1_bins = np.arange(-20, 20+dmu, dmu)
    mu2_bins = np.arange(-20, 20+dmu, dmu)

    n_mu1_bins = mu1_bins.size
    n_mu2_bins = mu2_bins.size

    lik_bins = np.zeros((n_mu1_bins, n_mu2_bins))

    for b1 in range(0, n_mu1_bins):
        for b2 in range(0, n_mu2_bins):
            p1 = true_pi * np.random.multivariate_normal(obs)
            p2 = (1-true_pi) * np.random.multivariate_normal(obs)
            lik_bins[b1, b2] = sum(np.log(p1 + p2));

    
    
    plt.contour(lik_bins)
