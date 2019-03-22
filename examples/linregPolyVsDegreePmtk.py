
#       Author:    J. Benjamin Cook
#       E-mail:    jbenjamincook@gmail.com
#


# Based on https://github.com/probml/pmtk3/blob/master/demos/linregPolyVsDegree.m

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.util import preprocessor_create
#from utils.util import poly_data_make
from examples.linregPmtkEmulator import linreg_fit
from examples.linregPmtkEmulator import linreg_predict
#from examples.linearRegression import linreg_fit_bayes

def make_1dregression_data(n=21):
    np.random.seed(0)
    # Example from Romaine Thibaux
    xtrain = np.linspace(0, 20, n)
    xtest = np.arange(0, 20, 0.1)
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
#xtrain, ytrain, xtest, _, ytest, _ = poly_data_make(sampling='thibaux', n=N)
xtrain, ytrain, xtest, _, ytest, _ = make_1dregression_data(n=N)


degs = np.arange(1, 22)
Nm = len(degs)

# Plot error vs degree
mseTrain = np.zeros(Nm)
mseTest = np.zeros(Nm)
for m in range(len(degs)):
    deg = degs[m]
    pp = preprocessor_create(rescale_X=True, poly=deg, add_ones=True)
    model = linreg_fit(xtrain, ytrain, preproc=pp)
    ypredTrain = linreg_predict(model, xtrain)
    ypredTest = linreg_predict(model, xtest)
    mseTrain[m] = np.mean(np.square(ytrain - ypredTrain))
    mseTest[m] = np.mean(np.square(ytest - ypredTest))

ndx = degs <= 16
fig = plt.figure()
plt.plot(degs[ndx], mseTrain[ndx], 'bs-', lw=3)
plt.plot(degs[ndx], mseTest[ndx], 'r*-', lw=3)
plt.xlabel('degree')
plt.ylabel('mse')
leg = plt.legend(('train', 'test'), loc='upper left')
leg.draw_frame(False)
plt.savefig(os.path.join('figures','linregPolyVsDegreeUcurve.pdf'))
plt.show()

degs = [1, 2, 14, 20]
mseTrain = np.zeros(len(degs))
mseTest = np.zeros(len(degs))

for m, deg in enumerate(degs):
    pp = preprocessor_create(rescale_X=True, poly=deg, add_ones=True)
    model = linreg_fit(xtrain, ytrain, preproc=pp)
    ypredTrain = linreg_predict(model, xtrain)
    ypredTest = linreg_predict(model, xtest)
    mseTrain[m] = np.mean(np.square(ytrain - ypredTrain))
    mseTest[m] = np.mean(np.square(ytest - ypredTest))

    plt.figure()
    plt.plot(xtrain, ytrain, 'o')
    plt.plot(xtest, ypredTest, 'k', lw=3)
    plt.title("degree %d" % deg)
    plt.xlim([-1, 21])
    #plt.xlim([min(xtest), max(xtest)])
    plt.ylim([-10, 15])
    plt.savefig(os.path.join('figures','polyfitDemo%d.pdf' % deg))
    plt.show()

