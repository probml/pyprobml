# Plot polynomial regression on 1d problem
# Based on https://github.com/probml/pmtk3/blob/master/demos/linregPolyVsDegree.m

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from probml_tools import savefig


def fun(w: np.array, x: np.array) -> np.array:
    return w[0] * x + w[1] * np.square(x)


def make_1dregression_data(n: int = 21) -> Tuple[np.array, np.array, np.array, np.array]:
    np.random.seed(0)
    xtrain = np.linspace(0.0, 20, n)
    xtest = np.arange(0.0, 20, 0.1)
    sigma2 = 4
    w = np.array([-1.5, 1 / 9.])
    ytrain = fun(w, xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytest = fun(w, xtest) + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytest


xtrain, ytrain, xtest, ytest = make_1dregression_data(n=21)

# Rescaling data
scaler = MinMaxScaler(feature_range=(-1, 1))
Xtrain = scaler.fit_transform(xtrain.reshape(-1, 1))
Xtest = scaler.transform(xtest.reshape(-1, 1))

degs = np.arange(1, 21, 1)
ndegs = np.max(degs)
mse_train = np.empty(ndegs)
mse_test = np.empty(ndegs)
ytest_pred_stored = np.empty(ndegs, dtype=np.ndarray)
ytrain_pred_stored = np.empty(ndegs, dtype=np.ndarray)
for deg in degs:
    model = LinearRegression()
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    Xtrain_poly = poly_features.fit_transform(Xtrain)
    model.fit(Xtrain_poly, ytrain)
    ytrain_pred = model.predict(Xtrain_poly)
    ytrain_pred_stored[deg - 1] = ytrain_pred
    Xtest_poly = poly_features.transform(Xtest)
    ytest_pred = model.predict(Xtest_poly)
    mse_train[deg - 1] = mse(ytrain_pred, ytrain)
    mse_test[deg - 1] = mse(ytest_pred, ytest)
    ytest_pred_stored[deg - 1] = ytest_pred

# Plot MSE vs degree
fig, ax = plt.subplots()
mask = degs <= 15
ax.plot(degs[mask], mse_test[mask], color='r', marker='x', label='test')
ax.plot(degs[mask], mse_train[mask], color='b', marker='s', label='train')
ax.legend(loc='upper right', shadow=True)
plt.xlabel('degree')
plt.ylabel('mse')
savefig('polyfitVsDegree.pdf')
plt.show()

# Plot fitted functions
chosen_degs = [1, 2, 3, 14, 20]
for deg in chosen_degs:
    fig, ax = plt.subplots()
    ax.scatter(xtrain, ytrain)
    ax.plot(xtest, ytest_pred_stored[deg - 1])
    ax.set_ylim((-10, 15))
    plt.title(f'degree {deg}')
    savefig(f'polyfitDegree{deg}.pdf')
    plt.show()

# Plot residuals
for deg in chosen_degs:
    fig, ax = plt.subplots()
    ypred = ytrain_pred_stored[deg - 1]
    residuals = ytrain - ypred
    # ax.plot(ypred, residuals, 'o')
    # ax.set_xlabel('predicted y')
    ax.plot(xtrain, residuals, 'o')
    ax.set_xlabel('x')
    ax.set_ylabel('residual')
    ax.set_ylim(-6, 6)
    plt.title(f'degree {deg}. Predictions on the training set')
    savefig(f'polyfitDegree{deg}Residuals.pdf')
    plt.show()

# Plot fit vs actual
for deg in chosen_degs:
    for train in [True, False]:
        if train:
            ytrue = ytrain
            ypred = ytrain_pred_stored[deg - 1]
            dataset = 'Train'
        else:
            ytrue = ytest
            ypred = ytest_pred_stored[deg - 1]
            dataset = 'Test'
        fig, ax = plt.subplots()
        ax.scatter(ytrue, ypred)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.set_xlabel('true y')
        ax.set_ylabel('predicted y')
        r2 = sklearn.metrics.r2_score(ytrue, ypred)
        plt.title(f'degree {deg}. R2 on {dataset} = {r2:0.3f}')
        savefig(f'polyfitDegree{deg}FitVsActual{dataset}.pdf')
        plt.show()
