import numpy as np
import math
from cycler import cycler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, make_scorer
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import digamma
import pyprobml_utils as pml

np.random.seed(0)

# Using Colorblind friendly colors

CB_color = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']

cb_cycler = (cycler(linestyle=['-', '--', '-.']) * cycler(color=CB_color))
plt.rc('axes', prop_cycle=cb_cycler)


def fun(x, w):
    return w[0] * x + w[1] * np.square(x)


# 'Data as mentioned in the matlab code'
def polydatemake():
    n = 21
    sigma2 = 4
    xtrain = np.linspace(0, 20, n)
    xtest = np.arange(0, 20.1, 0.1)
    w = np.array([-1.5, 1 / 9])
    ytrain = fun(xtrain, w).reshape(-1, 1) + math.sqrt(sigma2) * np.random.randn(xtrain.shape[0], 1)
    ytestNoisefree = fun(xtest, w).reshape(-1, 1)
    ytestNoisy = ytestNoisefree + math.sqrt(sigma2) * np.random.randn(xtest.shape[0], 1)

    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2


def rescale(X):
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(X.reshape(-1,1))


def poly_features(X, deg):
    X_deg = np.tile(X, deg)
    n_deg = np.arange(14) + 1
    degs = np.tile(np.repeat(n_deg, X.shape[1]), X.shape[0]).reshape(X.shape[0], -1)
    X_poly = np.power(X_deg, degs)

    return X_poly

# Not used, wrote it,as to use if needed.
def addones(x):
    # x is of shape (s,)
    return np.insert(x[:, np.newaxis], 0, [[1]], axis=1)


[xtrain, ytrain, xtest, ytestNoisefree, ytest, sigma2] = polydatemake()

deg = 14

poly_train = poly_features(rescale(xtrain), deg)
poly_test = poly_features(rescale(xtest), deg)

ytrain = ytrain- ytrain.mean()
ytest = ytest - ytest.mean()

# -------------------------------------------
# error vs lambda
lambdas = np.logspace(-10, 2.5, 15)
lambdas = lambdas
train_mse, test_mse = [], []

for lam in lambdas:
    reg = linear_model.Ridge(alpha=lam).fit(poly_train, ytrain)
    ypred_train = reg.predict(poly_train)
    ypred_test = reg.predict(poly_test)
    train_mse.append(((ypred_train - ytrain) ** 2).mean())
    test_mse.append(((ypred_test - ytest) ** 2).mean())

plt.figure()
plt.semilogx(lambdas, train_mse, '-s')
plt.semilogx(lambdas, test_mse, '-x')
plt.legend(['train_mse', 'test_mse'])
plt.xlabel('log lambda')
plt.ylabel('mean sqaured error')
pml.save_fig('../figures/polyfitRidgeModelSelUcurve.pdf')


# -------------------------------------------
# cv vs lambda
cv_means = []
cv_stand_errors = []
n_folds = 5
scorer = make_scorer(mean_squared_error, greater_is_better=False)
for lam in lambdas:
    cross_validations = -np.array(
        cross_val_score(linear_model.Ridge(alpha=lam), poly_train, ytrain, cv=n_folds, scoring=scorer))
    cv_means.append(cross_validations.mean())
    cv_stand_errors.append(cross_validations.std() / np.sqrt(n_folds))

plt.figure()
plt.errorbar(lambdas, np.log(cv_means), yerr=np.log(np.array(cv_stand_errors)) / 2, fmt='-o')
plt.title('{}-fold cross validation, ntrain = {}'.format(n_folds, poly_train.shape[0]))
plt.axvline(lambdas[np.argmin(cv_means)], ls='--')  # lambda corresponding to minimum cv_mean.
plt.xscale('log')
pml.save_fig('../figures/polyfitRidgeModelSelCV.pdf')


class BayesianLinearRegression:
    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
        (N, d) = X.shape
        sigma2 = 1 / beta
        sigma = np.sqrt(sigma2)
        winit = np.zeros((d, 1))
        Lam0 = self.alpha * np.eye(d)  # positive definite.
        V0 = (1 / self.alpha) * np.eye(d)

        self.posteriorSol(X, y, Lam0, winit, sigma)
        mu = X @ winit
        Sigma = sigma2 * np.eye(N) + X @ V0 @ X.T
        self.logev_ = multivariate_normal.logpdf(x=y.T, mean=mu.reshape(-1, ), cov=Sigma,
                                                 allow_singular=True)
        return self

    def posteriorSol(self, X, y, Lam0, winit, sigma):
        Lam0root = np.linalg.cholesky(Lam0)
        X2 = np.vstack((X / sigma, Lam0root))
        y2 = np.vstack((y / sigma, Lam0root @ winit))
        q, r = np.linalg.qr(X2)
        self.w_ = np.linalg.lstsq(r, q.T @ y2, rcond=None)[0].reshape(-1, 1)

    def predict(self, X):
        ypred = X @ self.w_
        return ypred


# Algorithm Ref from Andreas C.Kapourani (kapouranis.andreas@gmail.com)
class VBLinearRegression:
    def __init__(self):
        self.maxiter = 400
        self.a_init = 1e-3
        self.b_init = 1e-3
        self.max_tol = 1e-5
        self.lam = 0.5

    def fit(self, X, y):

        '''
        The optimal precision factor follows Gamma distribition with
            alpha = alpha_int + no.of.features/2
            beta =  beta_init + Exp[w.T * w]/2

        The optimal parameters follow Gaussian Distribution with
            Var = inv(Exp[precision_factor] + lambda*(X.T*X))
            mean = lambda*Var*X.T*y

        These values are updated iteratively until log.marginal.likelihood converges.
        '''

        (N, d) = X.shape

        a = self.a_init + d / 2
        self.E_a = self.a_init / self.b_init  # Expectation of precision_factor

        self.logev = []  # log.marginal.likelihood

        for i in range(self.maxiter):
            self.S = np.linalg.pinv(self.E_a * np.eye(d) + self.lam * (X.T @ X))
            self.mu = self.lam * (self.S @ (X.T @ y))
            E_ww = self.mu.T @ self.mu + np.trace(self.S)
            b = self.b_init + 0.5 * E_ww
            self.E_a = a / b

            lb_py = 0.5 * N * math.log(self.lam / (2 * math.pi)) - 0.5 * self.lam * (y.T @ y) + self.lam * (
                    self.mu.T @ (X.T @ y)) - 0.5 * self.lam * np.trace((X.T @ X) * (self.mu @ self.mu.T + self.S))
            lb_pw = -0.5 * d * (math.log(2 * math.pi)) + 0.5 * d * (digamma(a) - math.log(b)) - 0.5 * self.E_a @ E_ww

            lb_pa = self.a_init * math.log(self.b_init) + (self.a_init - 1) * (
                    digamma(a) - math.log(b)) - self.b_init * self.E_a

            lb_qw = -0.5 * math.log(np.linalg.det(self.S)) - 0.5 * d * (1 + math.log(2 * math.pi))

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
beta = 1 / sigma2
alphas = beta * lambdas
train_mseB, test_mseB = [], []
logev = []
for alpha in alphas:
    reg = BayesianLinearRegression(alpha=alpha, beta=beta)
    reg.fit(poly_train, ytrain)
    logev.append(reg.logev_)
    ypred_train = reg.predict(poly_train)
    ypred_test = reg.predict(poly_test)
    train_mseB.append(((ypred_train - ytrain) ** 2).mean())
    test_mseB.append(((ypred_test - ytest) ** 2).mean())



plt.figure()
plt.semilogx(alphas, train_mseB, '-s')
plt.semilogx(alphas, test_mseB, '-x')
plt.legend(['train_mse', 'test_mse'])
plt.xlabel('log alpha')
plt.ylabel('mean sqaured error')

plt.figure(0)
plt.plot(np.log(alphas), logev, '-o')
plt.xlabel('log alpha')
plt.ylabel('log evidence')

# -------------------------------------------
# Infering alpha using VB

plt.figure(0)
reg = VBLinearRegression()
(reg, logevVB) = reg.fit(poly_train, ytrain)
alphaVB = reg.E_a
plt.axvline(math.log(alphaVB), ls='--')
plt.legend(['log evidence', 'alpha by VB'])
pml.save_fig('../figures/polyfitRidgeModelSelVB.pdf')

plt.figure()
logevErr = -np.array(logev)
logevErr = logevErr / np.max(logevErr)
plt.plot(alphas, logevErr, 'o-')
cvErr = np.log(cv_means) / np.max(np.log(cv_means))
cv_se = np.log(cv_stand_errors) / np.max(np.log(cv_stand_errors))
plt.errorbar(alphas, cvErr, yerr=cv_se / 2, fmt='-x')
plt.xlabel('log lambda')
plt.xscale('log')
plt.legend(['negative log marg. likelihood', 'CV estimate of MSE'])
pml.save_fig('../figures/polyfitRidgeModelSelCVEB.pdf')

plt.show()
