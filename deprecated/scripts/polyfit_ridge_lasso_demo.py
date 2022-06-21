# Ridge and lasso regression: 
# visualize effect of changing lambda on degree 14 polynomial
# https://github.com/probml/pmtk3/blob/master/demos/polyfitRidgeLasso.m
# Duane Rich

"""
Ridge and lasso regression:
Visualize effect of changing lambda on degree 14 polynomial.
This is a simplified version of linregPolyVsRegDemo.m
These are the steps:
    - Generate the data
    - Create a preprocessor pipeline that applies a degree 14 polynomial
    and rescales values to be within [-1, 1] (no hypers to CV)
    - Create a pipeline with the preprocessor and a ridge estimator
    - Create a pipeline with the preprocessor and a lasso estimator
    - Create the grid where we show coefficients decrease as regularizers
    increase (for both ridge and lasso)
    - Plot fitted values vs y values for ridge and lasso (with standard errors)
    - For increasing log values of lambda, plot the training and test error
    for ridge regression.
"""

import superimport

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from polyDataMake import polyDataMake
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from pyprobml_utils import save_fig

deg = 14

# Generate data and split into in and out of sample
#xtrain, ytrain, xtest, ytestNoisefree, ytest, sigma2 = polyDataMake(sampling='thibaux')

def make_1dregression_data(n=21):
    np.random.seed(0)
    xtrain = np.linspace(0.0, 20, n)
    xtest = np.arange(0.0, 20, 0.1)
    sigma2 = 4
    w = np.array([-1.5, 1/9.])
    fun = lambda x: w[0]*x + w[1]*np.square(x)
    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytest= fun(xtest) + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytest

xtrain, ytrain, xtest, ytest = make_1dregression_data(n=21)
def shp(x):
        return np.asarray(x).reshape(-1,1)
xtrain = shp(xtrain)
xtest = shp(xtest)


preprocess = Pipeline([('Poly', PolynomialFeatures(degree=deg, include_bias=False)),
                       ('MinMax', MinMaxScaler((-1, 1)))])

ridge_pipe = Pipeline([('PreProcess', preprocess),
                       ('Estimator', Ridge())])

lasso_pipe = Pipeline([('PreProcess', preprocess),
                       ('Estimator', Lasso())])

ridge_lambdas = [0.00001, 0.1]
lasso_lambdas = [0.00001, 0.1]

coefs_by_lambda = {}

def extract_coefs(pipe, lambdas):

    coefs_by_lambda = {}

    for lamb in lambdas:
        pipe.set_params(Estimator__alpha=lamb)
        pipe.fit(xtrain, ytrain)
        coefs_by_lambda[lamb] = pipe.named_steps['Estimator'].coef_.reshape(-1)

    return pd.DataFrame(coefs_by_lambda, index=range(1, deg+1))

lasso_coefs_by_lambda = extract_coefs(lasso_pipe, lasso_lambdas)
lasso_coefs_by_lambda.columns = ['lamb1='+str(lm) for lm in lasso_lambdas]
ridge_coefs_by_lambda = extract_coefs(ridge_pipe, ridge_lambdas)
ridge_coefs_by_lambda.columns = ['lamb2='+str(lm) for lm in ridge_lambdas]

coefs = lasso_coefs_by_lambda.join(ridge_coefs_by_lambda)

print(coefs)

def make_plot_fit(pipe, lamb, num):

    fig, ax = plt.subplots()

    pipe.set_params(Estimator__alpha=lamb)
    pipe.fit(xtrain, ytrain)
    ypred = pipe.predict(xtest)

    ax.plot(xtest, ypred, linewidth=3)
    ax.scatter(xtrain, ytrain)
    std_err = np.std(ypred - ytest)
    ax.plot(xtest, ypred - std_err, linewidth=1,
                  linestyle='dotted', color='blue')
    ax.plot(xtest, ypred + std_err, linewidth=1,
                  linestyle='dotted', color='blue')
    ax.set_title('L{0} lambda = {1}'.format(str(num), str(lamb)[:6]))
    ax.set_xlim(0, 20)
    ax.set_ylim(-10, 20)
    return fig, ax

for i, lamb in enumerate(ridge_lambdas):
    fig_ridge, ax_ridge = make_plot_fit(ridge_pipe, lamb, 2)
    save_fig('polyfitRidgeK' + str(i+1) + '.pdf')

for i, lamb in enumerate(lasso_lambdas):
    fig_lasso, ax_lasso = make_plot_fit(lasso_pipe, lamb, 1)
    save_fig('polyfitRidgeLassoK' + str(i+1) + '.pdf')

def mse(ypred, ytest):
    return np.mean((ypred - ytest)**2)

def make_train_test_mse(pipe, log_lambdas):

    train_mse = []
    test_mse = []

    for i, llamb in enumerate(log_lambdas):
        pipe.set_params(Estimator__alpha=np.exp(llamb))
        pipe.fit(xtrain, ytrain)
        ypred_test = pipe.predict(xtest)
        ypred_train = pipe.predict(xtrain)
        train_mse.append(mse(ypred_train, ytrain))
        test_mse.append(mse(ypred_test, ytest))

    fig, ax = plt.subplots()
    ax.plot(log_lambdas, train_mse, label='train mse', color='blue', marker='s', markersize=10)
    ax.plot(log_lambdas, test_mse, label='test mse', color='red', marker='x', markersize=10)
    ax.set_title('Mean Squared Error')
    ax.set_xlabel('log lambda')
    ax.set_xlim(-25, 5)
    ax.legend(loc='upper left')

    return fig, ax

fig, ax = make_train_test_mse(ridge_pipe, np.linspace(-24, 4, 10))
save_fig('polyfitRidgeUcurve.pdf')

plt.show()