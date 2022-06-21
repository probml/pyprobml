# Based on https://github.com/probml/pmtk3/blob/master/toolbox/BasicModels/sub/shrinkcov.m
# Converted by John Fearns - jdf22@infradead.org
# Wrapped Ledoit-Wolf optimal shrinkage estimator for cov(X)

import superimport

from sklearn.covariance import LedoitWolf

# Returns C, shrinkage, S
# C = shrinkage*T + (1-shrinkage)*S
# Using the diagonal variance "target" T=diag(S) with the
# unbiased sample cov S as the unconstrained estimate
def shrinkcov(X):
    # The LedoitWolf estimator by default centres the provided data before
    # estimating covariance.
    estimator = LedoitWolf().fit(X)
    C = estimator.covariance_
    shrinkage = estimator.shrinkage_
    
    # Unbiased estimate, for which we need to centre the data ourselves.
    X_centred = X - np.mean(X, 0)
    n = X.shape[0]
    S = np.matmul(X_centred.transpose(), X_centred) / (n-1)
    
    return C, shrinkage, S