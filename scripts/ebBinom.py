# -*- coding: utf-8 -*-
"""
Author: Ang Ming Liang

Based on https://github.com/probml/pmtk3/blob/master/demos/ebBinom.m
"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from scipy.special import digamma
import pyprobml_utils as pml

y = np.array([
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  
              1, 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  5,  
              2, 5,  3,  2,  7,  7,  3,  3,  2,  9, 10,  4,  4,  4,  4,  4,  4,  
              4, 10,  4,  4,  4,  5, 11, 12,  5,  5,  6,  5,  6,  6,  6,  6, 
              16, 15, 15,  9,  4])

n = np.array([20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 
               20, 20, 20, 20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 
               20, 20, 20, 10, 49, 19, 46, 27, 17, 49, 47, 20, 20, 13, 
               48, 50, 20, 20, 20, 20, 20, 20, 20, 48, 19, 19, 19, 22, 
               46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46, 47, 24, 14])

X = np.array([y, n-y]).T

def dirichlet_moment_match(data):
  a = np.mean(data, axis=0)
  m2 = np.mean(data*data, axis=0)
  ok = a>0
  s = (a[ok] - m2[ok]) / (m2[ok] - a[ok]**2)
  s = np.median(s)
  if s == 0:
    s = 1
  return a*s

def polya_moment_match(data):
  sdata = np.expand_dims(np.sum(X, axis=1) ,axis=1)
  p = data / sdata
  a = dirichlet_moment_match(p)
  return a 

def polya_fit_simple(data):
  a = polya_moment_match(data)
  N,K = data.shape
  for _ in range(100):
    sa = np.sum(a)
    g = np.sum(digamma(data + a), axis=0) - N*digamma(a)
    h = sum(digamma(np.sum(X, axis=1) + np.sum(a))) - N*digamma(np.sum(a))
    a = a * g / h
  return a

alphas = polya_fit_simple(X);
a, b = alphas

popMean = a/(a+b)
aPost = a + y
bPost = b + n - y
meantheta = aPost/(aPost + bPost)
quartiles = np.array([[beta.ppf(0.25, a,b), beta.ppf(0.75, a, b),beta.ppf(0.50, a, b)] 
             for (a,b) in zip(aPost, bPost)])
CItheta, mediantheta = quartiles[:, :2], quartiles[:, 2]

thetaMLE = y/n
thetaPooledMLE = np.sum(y)/np.sum(n)
x = np.arange(0, len(y))

# Plot

plt.figure(figsize=(10, 3))
plt.title('num. positives')
plt.bar(x, y)
plt.xlim(0, 70)
plt.ylim(0, 20)
pml.savefig('num_positives.pdf')
plt.show()

plt.figure(figsize=(10, 3))
plt.title('pop size')
plt.bar(x, n)
plt.xlim(0, 70)
plt.ylim(0, 50)
pml.savefig('pop_size.pdf')
plt.show()

plt.figure(figsize=(10, 3))
plt.title("MLE (red line = pooled MLE)")
plt.bar(x, thetaMLE)
plt.plot([0, len(thetaMLE)], [thetaPooledMLE, thetaPooledMLE], color="red")
plt.xlim(0, 70)
plt.ylim(0, 0.5)
pml.savefig('mle.pdf')
plt.show()

plt.figure(figsize=(10, 3))
plt.title("posterior mean (red line=population mean)")
plt.bar(x, meantheta)
plt.plot([0, len(meantheta)], [popMean, popMean], color="red")
plt.xlim(0, 70)
plt.ylim(0, 0.5)
pml.savefig('post_mean.pdf')
plt.show()

plt.figure(figsize=(15, 10))
plt.title("95% confidence interval")
for (height, q, median) in zip(range(len(n)-1, 1, -1), CItheta, mediantheta):
    plt.plot([q[0], q[1]], [height, height], 'b', alpha=0.5)
    plt.plot(median, height, 'b*')
plt.yticks(x)
plt.show()
pml.savefig('CI.pdf')
plt.show()