import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, fmin_bfgs
from matplotlib import pyplot as plt
from cycler import cycler
import pyprobml_utils as pml

CB_color = ['#377eb8', '#ff7f00']

cb_cycler = (cycler(linestyle=['-', '--', '-.']) * cycler(color=CB_color))
plt.rc('axes', prop_cycle=cb_cycler)

np.random.seed(0)

value = 0

class ProbitReg:

    def __init__(self):
        self.loglikhist = []
        self.max_iter = 100
        self.tolerance = 1e-4
        self.w = []

    def Probitloss(self, X, y, w):

        value = 1 - norm.cdf(X @ w)
        return -np.sum(y * norm.logcdf(X @ w)) - np.sum((1 - y) * np.ma.log(1 - (X@w)).filled(-(1e+20)))

    def objfn(self, X, y, w, lambd):
        return np.sum(lambd * np.square(w[1:])) - self.Probitloss(X, y, w)

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
                stop = np.abs(self.loglikhist[i - 1] - self.loglikhist[i - 2]) / (
                        self.loglikhist[i - 1] + self.loglikhist[i - 2]) <= self.tolerance / 2

            i += 1

        self.loglikhist = self.loglikhist[0:i - 1]

        return self.w, np.array(self.loglikhist)

    def probitRegFit_MinFunc(self, X, y, lambd):
        winit = np.zeros(X.shape[1])
        self.loglikhist = []

        def obj(w):
            return self.Probitloss(X, y, w) + np.sum(lambd * np.square(w[1:])) #NLL

        def callback(w):
      
            loglik = -obj(w) #LL

            self.loglikhist.append(loglik)

        res = minimize(obj, x0=winit, callback=callback)
        return res['x'], np.array(self.loglikhist)

    def predict(self, X, w):
        p = norm.cdf(X @ w)
        y = np.array((p > 0.5), dtype='int32')
        return y, p


# using matrix inversion for ridge regression 
def ridgeReg(X, y, lambd):  # returns weight vectors.
    D = X.shape[1]
    w = np.linalg.inv(X.T@X + (lambd)*np.eye(D,D)) @ (X.T) @ y

    return w


def flipBits(y, p):
    x = np.random.rand(y.shape[0], 1) < p
    y[x < p] = 1 - y[x < p]
    return y


N, D = 100, 2
X = np.random.randn(N, D)
w = np.random.randn(D, 1)
y01 = flipBits((X @ w > 0), 0)
ypm1 = np.sign(y01 - 0.5)

lambd = 1e-1

proreg = ProbitReg()

# EM:
em_w, objTraceEM = proreg.probRegFit_EM(X, ypm1, lambd)
em_yhat, em_prob = proreg.predict(X, em_w)

# Minfunc:
minfunc_w, objTraceMinfunc = proreg.probitRegFit_MinFunc(X, ypm1, lambd)
minfunc_yhat, minfunc_prob = proreg.predict(X, minfunc_w)


plt.figure()
plt.plot(minfunc_prob, em_prob, 'o')
plt.xlabel('minfunc')
plt.ylabel('em')
plt.title('probit regression with L2 regularizer of {}'.format(lambd))
plt.show()

plt.figure()
plt.plot(objTraceEM.flatten(), '-o', linewidth=2)
plt.plot(objTraceMinfunc.flatten(), ':s', linewidth=1)
plt.legend(['em', 'minfunc'])
plt.title('probit regression with L2 regularizer of {0:.3f}'.format(lambd))
plt.ylabel('logpost')
plt.xlabel('iter')
plt.show()
pml.save_fig('../figures/probitRegDemoNLL.pdf')
