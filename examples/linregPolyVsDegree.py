#!/usr/bin/env python
#
#       Author:    J. Benjamin Cook
#       E-mail:    jbenjamincook@gmail.com
#
#       File Name: linregPolyVsDegree.py
#       Description:
#           Linear Regression with Polynomial Basis of different degrees
#           based on code code by Romain Thibaux
#           (Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)

import numpy as np
import matplotlib.pyplot as pl
from utils.util import preprocessor_create
from utils.util import poly_data_make
from SupervisedModels.linearRegression import linreg_fit
from SupervisedModels.linearRegression import linreg_fit_bayes
from SupervisedModels.linearRegression import linreg_predict

N = 21
xtrain, ytrain, xtest, _, ytest, _ = poly_data_make(sampling='thibaux', n=N)

degs = np.arange(1, 22)
Nm = len(degs)

# Plot error vs degree
mseTrain = np.zeros(Nm)
mseTest = np.zeros(Nm)
for m in xrange(len(degs)):
    deg = degs[m]
    pp = preprocessor_create(rescale_X=True, poly=deg, add_ones=True)
    model = linreg_fit(xtrain, ytrain, preproc=pp)
    ypredTrain = linreg_predict(model, xtrain)
    ypredTest = linreg_predict(model, xtest)
    mseTrain[m] = np.mean(np.square(ytrain - ypredTrain))
    mseTest[m] = np.mean(np.square(ytest - ypredTest))

ndx = degs <= 16
fig = pl.figure()
pl.plot(degs[ndx], mseTrain[ndx], lw=3)
pl.plot(degs[ndx], mseTest[ndx], lw=3)
pl.xlabel('degree')
pl.ylabel('mse')
leg = pl.legend(('train', 'test'), loc='upper left')
leg.draw_frame(False)
pl.savefig('linregPolyVsDegreeUcurve.png')
pl.show()


degs = [1, 2, 10, 14, 20]
mseTrain = np.zeros(len(degs))
mseTest = np.zeros(len(degs))

for m, deg in enumerate(degs):
    pp = preprocessor_create(rescale_X=True, poly=deg, add_ones=True)
    model = linreg_fit(xtrain, ytrain, preproc=pp)
    ypredTrain = linreg_predict(model, xtrain)
    ypredTest = linreg_predict(model, xtest)
    mseTrain[m] = np.mean(np.square(ytrain - ypredTrain))
    mseTest[m] = np.mean(np.square(ytest - ypredTest))

    pl.figure(m)
    pl.plot(xtrain, ytrain, 'o')
    pl.plot(xtest, ypredTest, lw=3)
    pl.title("degree %d" % deg)
    pl.savefig('polyfitDemo%d.png' % deg)
    pl.xlim([-1, 21])
    pl.ylim([-10, 15])
    pl.show()
