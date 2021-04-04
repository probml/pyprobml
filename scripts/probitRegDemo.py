import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from cycler import cycler
import jax.numpy as jnp
import jax.scipy.stats.norm as jnorm
from jax import grad
import pyprobml_utils as pml
from statsmodels.discrete.discrete_model import Probit

CB_color = ['#377eb8', '#ff7f00']

cb_cycler = (cycler(linestyle=['-', '--', '-.']) * cycler(color=CB_color))
plt.rc('axes', prop_cycle=cb_cycler)

np.random.seed(0)


class ProbitReg:

    def __init__(self):
        self.loglikhist = []
        self.max_iter = 100
        self.tolerance = 1e-4
        self.w = []

    def Probitloss(self, X, y, w):  # NLL

        # (1-y)*log(1-cdf(X.w)) = (1-y)*log(cdf(-(X.w))

        return -jnp.sum(y * jnorm.logcdf(jnp.dot(X, w))) - jnp.sum((1 - y) * jnorm.logcdf(-jnp.dot(X, w)))

    def objfn(self, X, y, w, lambd): # penalized likelihood.
        return jnp.sum(lambd * jnp.square(w[1:])) - self.Probitloss(X, y, w)

    def probRegFit_EM(self, X, y, lambd):

        self.w = np.linalg.lstsq(X + np.random.rand(X.shape[0], X.shape[1]), y, rcond=None)[0].reshape(-1, 1)

        def estep(w):
            u = X @ w
            z = u + norm.pdf(u) / ((y == 1) - norm.cdf(-u))
            loglik = self.objfn(X, y, w, lambd)
            return z, loglik

        # mstep function is the ridge regression

        i = 1
        stop = False
        while not stop:
            z, loglik = estep(self.w)
            self.loglikhist.append(loglik)
            self.w = ridgeReg(X, z, lambd)  # mstep
            if i >= self.max_iter:
                stop = True
            elif i > 1:
                # if slope becomes less than tolerance.
                stop = np.abs((self.loglikhist[i - 1] - self.loglikhist[i - 2]) / (
                        self.loglikhist[i - 1] + self.loglikhist[i - 2])) <= self.tolerance / 2

            i += 1

        self.loglikhist = self.loglikhist[0:i - 1]

        return self.w, np.array(self.loglikhist)

    def probitRegFit_gradient(self, X, y, lambd):
        winit = jnp.linalg.lstsq(X + np.random.rand(X.shape[0], X.shape[1]), y, rcond=None)[0].reshape(-1, 1)

        self.loglikhist = []

        self.loglikhist.append((-self.objfn(X,y,winit,lambd)))

        def obj(w):
            w = w.reshape(-1, 1)
            return self.Probitloss(X, y, w) + jnp.sum(lambd * jnp.square(w[1:]))  # PNLL


        def grad_obj(w):
            return grad(obj)(w)

        def callback(w):
            loglik = obj(w)  # LL

            self.loglikhist.append(loglik)

        res = minimize(obj, x0=winit, jac=grad_obj, callback=callback,method='BFGS')
        return res['x'], np.array(self.loglikhist[0:-1])

    def predict(self, X, w):
        p = jnorm.cdf(jnp.dot(X,w))
        y = np.array((p > 0.5), dtype='int32')
        return y, p


# using matrix inversion for ridge regression
def ridgeReg(X, y, lambd):  # returns weight vectors.
    D = X.shape[1]
    w = np.linalg.inv(X.T @ X + lambd * np.eye(D, D)) @ X.T @ y

    return w


def flipBits(y, p):
    x = np.random.rand(y.shape[0], 1) < p
    y[x < p] = 1 - y[x < p]
    return y


N, D = 100, 2
X = np.random.randn(N, D)
w = np.random.randn(D, 1)
y = flipBits((X @ w > 0), 0)

lambd = 1e-1

# statsmodel.Probit
res = Probit(exog=X,endog=y).fit_regularized(disp=0)
smProbit_prob = res.predict(exog=X)

# Our Implementation:
proreg = ProbitReg()

# EM:
em_w, objTraceEM = proreg.probRegFit_EM(X, y, lambd)
em_yhat, em_prob = proreg.predict(X, em_w)

# gradient:
gradient_w, objTraceGradient = proreg.probitRegFit_gradient(X, y, lambd)
gradient_yhat, gradient_prob = proreg.predict(X, gradient_w)

plt.figure()
plt.plot(smProbit_prob,em_prob,'o')
plt.xlabel('statsmodel.probit')
plt.ylabel('em')

plt.figure()
plt.plot(gradient_prob, em_prob, 'o')
plt.xlabel('bfgs')
plt.ylabel('em')
plt.title('probit regression with L2 regularizer of {0:.3f}'.format(lambd))
plt.show()

plt.figure()
plt.plot(-objTraceEM.flatten(), '-o', linewidth=2)
plt.plot(objTraceGradient.flatten(), ':s', linewidth=1)
plt.legend(['em','bfgs'])
plt.title('probit regression with L2 regularizer of {0:.3f}'.format(lambd))
plt.ylabel('logpost')
plt.xlabel('iter')
pml.save_fig('../figures/probitRegDemoNLL.pdf')
plt.show()
