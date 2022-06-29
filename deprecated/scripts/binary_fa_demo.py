import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from jax import vmap

class BinaryFA:
 
  def __init__(self, input_dim, latent, max_iter, conv_tol=1e-4, compute_ll=True):
    self.W = 0.1 * np.random.randn(latent, input_dim) # 2x16
    self.b = 0.01 * np.random.randn(input_dim, 1) # 16x1
    self.mu_prior = np.zeros((latent,1)) # 2x1
    self.sigma_prior = np.eye(latent) # 2x2
    self.input_dim = input_dim
    self.latent = latent
    self.max_iter = max_iter
    self.compute_ll = compute_ll
    if compute_ll :
      self.ll_hist = np.zeros((max_iter + 1, 1))  # 51x1
 
 
  def variational_em(self, data):
    ll_hist = np.zeros((self.max_iter + 1, 1))
    i = 0
    while i < 3:
      S1, S2, ll = self.estep(data)
      ll_hist[i,0] = ll
      self.mstep(S1, S2)
      if i!=0:
        delta_fval = abs(ll_hist[i] - ll_hist[i-1])
        avg_fval = (abs(ll_hist[i]) + abs(ll_hist[i-1]) + np.finfo(float).eps)/2
        if (delta_fval / avg_fval) < conv_tol:
          break
      i += 1
    return ll_hist[:i]
  
  def estep(self, data):
    S1 = np.zeros((self.latent + 1, self.input_dim)) # 3x16 
    S2 = np.zeros((self.latent + 1, self.latent + 1, self.input_dim)) # 3x3x16
    W, b, mu_prior = self.W , self.b, self.mu_prior
    ll = 0
    for i in range(data.T.shape[1]):
      mu_post, sigma_post, logZ, lambd = self.compute_latent_posterior_statistics(data.T[:,i], max_iter=3)
      ll += logZ
      EZZ = np.zeros((self.latent+1, self.latent+1))
      EZZ[:self.latent,:self.latent] = sigma_post + np.outer(mu_post, mu_post)
      EZZ[self.latent,:self.latent] = mu_post.T
      EZZ[:self.latent,self.latent] = np.squeeze(np.asarray(mu_post))
      EZZ[self.latent,self.latent] = 1
      EZ = np.append(mu_post,np.ones((1,1)))
      for j in range(self.input_dim):
        S1[:,j] = S1[:,j] + (data.T[j,i] - 0.5) * EZ
        S2[:,:,j] = S2[:,:,j] - 2* lambd[j] * EZZ
    return S1, S2, ll
 
  def mstep(self, S1, S2):
    for i in range(self.input_dim):
      what = np.linalg.lstsq(S2[:,:,i] , S1[:,i])[0]
      self.W[:,i] = what[:self.latent]
      self.b[i] = what[self.latent]
 
  def compute_latent_posterior_statistics(self, y, output=[0,0,0,0], max_iter=3):
    W, b = np.copy(self.W), np.copy(self.b)
    y = y.reshape((-1,1))
    # variational parameters
    mu_prior = self.mu_prior
    xi = (2 * y -1) * (W.T @ mu_prior + b)
    xi[xi==0] = 0.01 * np.random.rand(np.count_nonzero(xi==0)) # 16x1
    sigma_inv, iter = np.linalg.inv(self.sigma_prior), 0
    for iter in range(max_iter):
      lambd = (0.5 - sigmoid(xi)) / (2*xi)
      tmp = W @ np.diagflat(lambd) @ W.T # 2x2
      sigma_post = np.linalg.inv(sigma_inv - (2 * tmp))
      tmp = y -0.5 + 2* lambd * b
      tmp2 = np.sum(W @ np.diagflat(tmp), axis=1).reshape((2,1)) 
      mu_post = sigma_post @ (sigma_inv @ mu_prior + tmp2)

      tmp = np.diag(W.T @ (sigma_post + mu_post @ mu_post.T) @ W)
      tmp = tmp.reshape((tmp.shape[0],1))
      tmp2 = 2*(W @ np.diagflat(b)).T @ mu_post
      xi = np.sqrt(tmp + tmp2 + b**2)
      logZ = 0
      if self.compute_ll:
        lam = -lambd
        A = np.diagflat(2*lam)
        invA = np.diagflat(1/(2*lam))
        bb = -0.5 * np.ones((y.shape[0],1))
        c = -lam * xi**2 - 0.5 * xi + np.log(1+ np.exp(xi))
        ytilde = invA @ (bb + y)
        B = W.T 
        logconst1 = -0.5* np.sum(np.log(lam/np.pi))
        logconst2 = 0.5 * ytilde.T @ A @ ytilde - np.sum(c)
        gauss = multivariate_normal.logpdf(np.squeeze(np.asarray(ytilde)), mean=np.squeeze(np.asarray(B @ mu_prior + b)), cov=(invA + B @ sigma_post @ B.T))
        logZ = logconst1 + logconst2 + gauss
        output = [mu_post, sigma_post, logZ,lambd]
    return output
 
  def predict_missing(self, y):
    N, T = y.shape # 150 x 16 
    prob_on = np.zeros(y.shape) # 150 x 16
    post_pred = np.zeros((N,T,2)) # 150 x 16 x 2
    L,p = self.W.shape # 16 x 3
    B = np.c_[np.copy(self.b),self.W.T] # 16 x 3
    for n in range(N):
      mu_post, sigma_post, logZ, lambd = self.compute_latent_posterior_statistics(y[n,:].T, False)
      mu1 = np.r_[np.ones((1,1)), mu_post]
      sigma1 = np.zeros((L+1,L+1))
      sigma1[1:,1:] = sigma_post
      prob_on[n,:] = sigmoid_times_gauss(B, mu1, sigma1)

    return prob_on
  
  def infer_latent(self, y):
    N, T = y.shape
    W, b, mu_prior = self.W, self.b, self.mu_prior
    K, T2 = self.W.shape
    mu_post, loglik  = np.zeros((K,N)),np.zeros((1,N))
    sigma_post = np.zeros((K,K,N))
    for n in range(N):
      mu_p , sigma_p, loglik[0,n] , _ = self.compute_latent_posterior_statistics(y[n,:].T)
      mu_post[:,n] = np.squeeze(np.asarray(mu_p))
      sigma_post[:,:,n] = np.squeeze(np.asarray(sigma_p))
    return mu_post, sigma_post, loglik

def sigmoid_times_gauss(X, wMAP, C):
  vv = lambda x, y: jnp.vdot(x, y)  
  mv = vmap(vv, (None, 0), 0)
  mm = vmap(mv, (0, None), 0) 
  vm = vmap(vv, (0, 0), 0)
  
  mu = X @ wMAP;
  n = X.shape[1]
  if n < 1000:
    sigma2 = np.diag(X @ C @ X.T)
  else:
    sigma2 = vm(X , mm(C,X))
  kappa = 1 / np.sqrt(1 + np.pi * sigma2 /8);
  p = sigmoid(kappa * mu.reshape(kappa.shape))
  return p

np.random.seed(1)

max_iter, conv_tol = 50, 1e-4
sigmoid = lambda x : 1/(1 + np.exp(-1 * x))
d, k, m = 16, 3, 50
noise_level = 0.5
 
proto = np.random.rand(d, k) < noise_level
src = np.concatenate((np.tile(proto[:,0], (1,m)), np.tile(proto[:,1],(1,m)), np.tile(proto[:,2],(1,m))),axis=1)
clean_data = np.concatenate((np.tile(proto[:,0], (m,1)), np.tile(proto[:,1],(m,1)), np.tile(proto[:,2],(m,1))), axis=0)
n = clean_data.shape[0]
 
 
mask, noisy_data, missing_data, = np.random.rand(n,d) < 0.05, np.copy(clean_data), np.copy(clean_data)
 
noisy_data[mask] = 1 - noisy_data[mask]
missing_data[mask] = np.nan

plt.figure()
ax = plt.gca()
plt.imshow(noisy_data, aspect='auto', interpolation='none',
           origin='lower', cmap="gray")
plt.title('Noisy Binary Data')
plt.show()

binaryFA = BinaryFA(d, 2, 50, 1e-4, True)
binaryFA.variational_em(noisy_data)
 
mu_post, sigma_post, loglik  = binaryFA.infer_latent(noisy_data)

symbols = ['ro', 'gs', 'k*']
plt.figure()
plt.plot(mu_post[0,:m], mu_post[1,0:m], symbols[0])
plt.plot(mu_post[0,m:2*m], mu_post[1,m:2*m], symbols[1])
plt.plot(mu_post[0,2*m:], mu_post[1,2*m:], symbols[2])
plt.title('Latent Embedding')
plt.show()
 
prob_on = binaryFA.predict_missing(noisy_data)
plt.figure()
plt.imshow(prob_on, aspect='auto', interpolation='none',
           origin='lower', cmap="gray")
plt.title('Posterior Predictive')
plt.show()

plt.figure()
plt.imshow(prob_on>0.5, aspect='auto', interpolation='none',
           origin='lower', cmap="gray")
plt.title('Reconstruction')
plt.show()
