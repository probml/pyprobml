import superimport

import numpy as np
import math
from cycler import cycler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import digamma
import pyprobml_utils as pml

np.random.seed(0)

# Using Colorblind friendly colors

cb_color = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

cb_cycler = cycler(linestyle=["-", "--", "-."]) * cycler(color=cb_color)
plt.rc("axes", prop_cycle=cb_cycler)


def func(x, w):
    return w[0] * x + w[1] * np.square(x)


# 'Data as mentioned in the matlab code'
def polydatemake():
    n = 21
    sigma2 = 4
    xtrain = np.linspace(0, 20, n)
    xtest = np.arange(0, 20.1, 0.1)
    w = np.array([-1.5, 1 / 9])
    ytrain = func(xtrain, w).reshape(-1, 1) + math.sqrt(sigma2) * np.random.randn(
        xtrain.shape[0], 1
    )
    ytest_noisefree = func(xtest, w).reshape(-1, 1)
    ytest_noisy = ytest_noisefree + math.sqrt(sigma2) * np.random.randn(xtest.shape[0], 1)

    return xtrain, ytrain, xtest, ytest_noisefree, ytest_noisy, sigma2


def rescale(X):
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(X.reshape(-1, 1))


def poly_features(X, deg):
    X_deg = np.tile(X, deg)
    n_deg = np.arange(14) + 1
    degs = np.tile(np.repeat(n_deg, X.shape[1]), X.shape[0]).reshape(X.shape[0], -1)
    X_poly = np.power(X_deg, degs)

    return X_poly


# Not used, wrote it,as to use if needed.
def addones(X):
    # x is of shape (s,)
    return np.insert(X[:, np.newaxis], 0, [[1]], axis=1)


[xtrain, ytrain, xtest, ytest_noisefree, ytest, sigma2] = polydatemake()

deg = 14

poly_train = poly_features(rescale(xtrain), deg)
poly_test = poly_features(rescale(xtest), deg)

ytrain = ytrain - ytrain.mean()
ytest = ytest - ytest.mean()


class BayesianLinearRegression:
    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
        (n, d) = X.shape
        sigma2 = 1 / beta
        sigma = np.sqrt(sigma2)
        winit = np.zeros((d, 1))
        lam0 = self.alpha * np.eye(d)  # positive definite.
        v0 = (1 / self.alpha) * np.eye(d)

        self.posterior_sol(X, y, lam0, winit, sigma)
        mu = X @ winit
        Sigma = sigma2 * np.eye(n) + X @ v0 @ X.T
        self.logev = multivariate_normal.logpdf(
            x=y.T, mean=mu.reshape(-1, ), cov=Sigma, allow_singular=True
        )
        return self

    def posterior_sol(self, X, y, lam0, winit, sigma):
        lam0root = np.linalg.cholesky(lam0)
        X2 = np.vstack((X / sigma, lam0root))
        y2 = np.vstack((y / sigma, lam0root @ winit))
        q, r = np.linalg.qr(X2)
        self.w = np.linalg.lstsq(r, q.T @ y2, rcond=None)[0].reshape(-1, 1)

    def predict(self, X):
        ypred = X @ self.w
        return ypred


# Algorithm Ref from Andreas C.Kapourani (kapouranis.andreas@gmail.com)
# https://rpubs.com/cakapourani/variational-bayes-lr


class VBLinearRegression:
    def __init__(self):
        self.maxiter = 400
        self.a_init = 1e-3
        self.b_init = 1e-3
        self.max_tol = 1e-5
        self.lam = 0.5

    def fit(self, X, y):

        """
        The optimal precision factor follows Gamma distribition with
            alpha = alpha_int + no.of.features/2
            beta =  beta_init + Exp[w.T * w]/2

        The optimal parameters follow Gaussian Distribution with
            Var = inv(Exp[precision_factor] + lambda*(X.T*X))
            mean = lambda*Var*X.T*y

        These values are updated iteratively until log.marginal.likelihood converges.
        """

        (n, d) = X.shape

        a = self.a_init + d / 2
        self.expec_a = self.a_init / self.b_init  # Expectation of precision_factor

        self.logev = []  # log.marginal.likelihood

        for i in range(self.maxiter):
            self.S = np.linalg.pinv(self.expec_a * np.eye(d) + self.lam * (X.T @ X))
            self.mu = self.lam * (self.S @ (X.T @ y))
            E_ww = self.mu.T @ self.mu + np.trace(self.S)
            b = self.b_init + 0.5 * E_ww
            self.expec_a = a / b

            lb_py = (
                    0.5 * n * math.log(self.lam / (2 * math.pi))
                    - 0.5 * self.lam * (y.T @ y)
                    + self.lam * (self.mu.T @ (X.T @ y))
                    - 0.5 * self.lam * np.trace((X.T @ X) * (self.mu @ self.mu.T + self.S))
            )
            lb_pw = (
                    -0.5 * d * (math.log(2 * math.pi))
                    + 0.5 * d * (digamma(a) - math.log(b))
                    - 0.5 * self.expec_a @ E_ww
            )

            lb_pa = (
                    self.a_init * math.log(self.b_init)
                    + (self.a_init - 1) * (digamma(a) - math.log(b))
                    - self.b_init * self.expec_a
            )

            lb_qw = -0.5 * math.log(np.linalg.det(self.S)) - 0.5 * d * (
                    1 + math.log(2 * math.pi)
            )

            lb_qa = -math.lgamma(a) + (a - 1) * digamma(a) + math.log(b) - a

            self.logev.append(lb_py + lb_pw + lb_pa - lb_qw - lb_qa)

            if i >= 1:
                if (self.logev[i] - self.logev[i - 1]) < self.max_tol:
                    break

        return self, self.logev

    def predict(self, X):
        ypred = X @ self.mu
        return ypred


# Bayes
# -------------------------------------------
# linear Bayes with gaussian prior
lambdas = np.logspace(-10, 2.5, 15)
lambdas = lambdas
beta = 1 / sigma2
alphas = beta * lambdas
train_mse_bayes, test_mse_bayes = [], []
logev = []
for alpha in alphas:
    reg = BayesianLinearRegression(alpha=alpha, beta=beta)
    reg.fit(poly_train, ytrain)
    logev.append(reg.logev)
    ypred_train = reg.predict(poly_train)
    ypred_test = reg.predict(poly_test)
    train_mse_bayes.append(((ypred_train - ytrain) ** 2).mean())
    test_mse_bayes.append(((ypred_test - ytest) ** 2).mean())

plt.figure()
plt.semilogx(alphas, train_mse_bayes, "-s")
plt.semilogx(alphas, test_mse_bayes, "-x")
plt.legend(["train_mse", "test_mse"])
plt.xlabel("log alpha")
plt.ylabel("mean squared error")

plt.figure(0)
plt.plot(np.log(alphas), logev, "-o")
plt.xlabel("log alpha")
plt.ylabel("log evidence")

# -------------------------------------------
# Infering alpha using VB

plt.figure(0)
reg = VBLinearRegression()
(reg, logev_vb) = reg.fit(poly_train, ytrain)
alpha_vb = reg.expec_a
plt.axvline(math.log(alpha_vb), ls="--")
plt.legend(["log evidence", "alpha by VB"])
pml.save_fig("../figures/polyfitRidgeModelSelVB.pdf")

plt.figure()
logev_err = -np.array(logev)
logev_err = logev_err / np.max(logev_err)
plt.plot(alphas, logev_err, "o-")
plt.xlabel("log lambda")
plt.xscale("log")
plt.legend(["negative log marg. likelihood"])
pml.save_fig("../figures/polyfitRidgeModelSelEB.pdf")
