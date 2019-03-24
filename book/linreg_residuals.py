
# Plot resisudal from 1d linear regression
# Based on https://github.com/probml/pmtk3/blob/master/demos/linregResiduals.m


import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


def make_1dregression_data(n=21):
    np.random.seed(0)
    # Example from Romaine Thibaux
    xtrain = np.linspace(0.0, 20, n)
    xtest = np.arange(0.0, 20, 0.1)
    sigma2 = 4
    w = np.array([-1.5, 1/9.])
    fun = lambda x: w[0]*x + w[1]*np.square(x)
    # Apply function to make data
    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2

N = 21
X,y,_,_,_,_ = make_1dregression_data(N)

X = np.concatenate((np.ones((N,1)), X.reshape(N,1)), axis=1)  
w = np.linalg.lstsq(X, y)[0]
print(w)
y_estim = np.dot(X,w)

plt.plot(X[:,1], y, 'o')
plt.plot(X[:,1], y_estim, '-')
save_fig('linregResidualsNoBars.pdf')
plt.show()

for x0, y0, y_hat in zip(X[:,1], y, y_estim):
  plt.plot([x0, x0],[y0, y_hat],'k-')
plt.plot(X[:,1], y, 'o')
plt.plot(X[:,1], y_estim, '-')
save_fig('linregResidualsBars.pdf')
plt.show()
