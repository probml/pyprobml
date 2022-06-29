# This file implements Relevance Vector Machine Classifier.
# Author Srikar Reddy Jilugu(@always-newbie161)

import superimport

import numpy as np

# This is a python implementation of Relevance Vector Machine Classifier, 
# it's based on github.com/ctgk/PRML/blob/master/prml/kernel/relevance_vector_classifier.py
class RVC:
    def sigmoid(self,a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    # Kernel matrix using rbf kernel with gamma = 0.3.
    def kernel_mat(self,X, Y):
        (x, y) = (np.tile(X, (len(Y), 1, 1)).transpose(1, 0, 2),
                  np.tile(Y, (len(X), 1, 1)))
        d = np.repeat(1 / (0.3 * 0.3), X.shape[-1]) * (x - y) ** 2
        return np.exp(-0.5 * np.sum(d, axis=-1))
    def __init__(self, alpha=1.):
        self.threshold_alpha = 1e8
        self.alpha = alpha
        self.iter_max = 100
        self.relevance_vectors_ = []

    # estimates for singulat matrices.
    def ps_inv(self, m):
        # assuming it is a square matrix.
        a = m.shape[0]
        i = np.eye(a, a)
        return np.linalg.lstsq(m, i, rcond=None)[0]

    '''
    For the current fixed values of alpha, the most probable
    weights are found by maximizing w over p(w/t,alpha) 
    using the Laplace approximation of finding an hessian.
    (E step)
    w = mean of p(w/t,alpha)
    cov = negative hessian of p(w/t,alpha)
    
    '''
    def _map_estimate(self, X, t, w, n_iter=10):
        for _ in range(n_iter):
            y = self.sigmoid(X @ w)
            g = X.T @ (y - t) + self.alpha * w
            H = (X.T * y * (1 - y)) @ X + np.diag(self.alpha)  # negated Hessian of p(w/t,alpha)
            w -= np.linalg.lstsq(H, g, rcond=None)[0]  # works even if for singular matrices.
        return w, self.ps_inv(H)        # inverse of H is the covariance of the gaussian approximation.

    '''
    Fitting of input-target pairs works by
    iteratively finding the most probable weights(done by _map_estimate method)
    and optimizing the hyperparameters(alpha) until there is no
    siginificant change in alpha.
    
    (M step)
    Optimizing alpha:
        For the given targets and current variance(sigma^2) alpha is optimized over p(t/alpha,variance)
        It is done by Mackay approach(ARD).
        alpha(new) = gamma/mean^2
        where gamma = 1 - alpha(old)*covariance.
    
    After finding the hyperparameters(alpha),
    the samples which have alpha less than the threshold(hence weight >> 0)
    are choosen as relevant vectors.
    
    Now predicted y = sign(phi(X) @ mean) ( mean contains the optimal weights)
    '''
    def fit(self, X, y):
        Phi = self.kernel_mat(X, X)
        N = len(y)
        self.alpha = np.zeros(N) + self.alpha
        mean = np.zeros(N)
        for i in range(self.iter_max):
            param = np.copy(self.alpha)
            mean, cov = self._map_estimate(Phi, y, mean, 10)
            gamma = 1 - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 1e10, out=self.alpha)
            if np.allclose(param, self.alpha):
                break

        ret_alpha = self.alpha < self.threshold_alpha
        self.relevance_vectors_ = X[ret_alpha]
        self.y = y[ret_alpha]
        self.alpha = self.alpha[ret_alpha]
        Phi = self.kernel_mat(self.relevance_vectors_, self.relevance_vectors_)
        mean = mean[ret_alpha]
        self.mean, self.covariance = self._map_estimate(Phi, self.y, mean, 100)


    # gives probability for target to be class 0.
    def predict_proba(self, X):
        phi = self.kernel_mat(X, self.relevance_vectors_)
        mu_a = phi @ self.mean
        var_a = np.sum(phi @ self.covariance * phi, axis=1)
        return 1 - self.sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))

    def predict(self, X):
        phi = self.kernel_mat(X, self.relevance_vectors_)
        return (phi @ self.mean > 0).astype(np.int)
