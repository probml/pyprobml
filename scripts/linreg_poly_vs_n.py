# Linear regression as a function of training set size
# Based on https://github.com/probml/pmtk3/blob/master/demos/linregPolyVsN.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml



from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler 

TrueDeg = 2 #The true degree of the model
degrees = [1, 2, 10, 20] #The degrees of our design matrices
    
def ExpandtoDeg(x,deg):
    return np.array([x**i for i in range(deg+1)]).transpose().reshape(-1,deg+1)

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

    
for ModDeg in degrees:
    ns = [int(n) for n in np.round(np.linspace(10, 210, 20))]
    err = []
    errtrain = []
    for n in ns:
        xtrain, ytrain, xtest, ytest = make_1dregression_data(n=n)

        #Rescaling data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        xtrain = scaler.fit_transform(xtrain.reshape(-1, 1))
        xtest = scaler.transform(xtest.reshape(-1, 1))

        #Fitting ridge regression. Small differences in alpha near zero make a visual difference in the plot when n is close to 0.
        regr = Ridge(alpha=0, fit_intercept=False) #Using ridge instead of ordinary least squares for numerical stability
        XDesignTrain = ExpandtoDeg(xtrain, ModDeg)
        XDesignTest = ExpandtoDeg(xtest, ModDeg)
        regr.fit(XDesignTrain,ytrain)   
        ypred = regr.predict(XDesignTest)
        err.append(np.mean((ytest-ypred)**2))
        errtrain.append(np.mean((ytrain-regr.predict(XDesignTrain))**2))
    
    #Plotting
    fig, ax = plt.subplots()
    ax.plot(ns, err, color = 'r', marker = 's',label='test')
    ax.plot(ns, errtrain, marker = 'x', label='train')
    ax.legend(loc='upper right', shadow=True)
    ax.set_xlim([0,200])
    ax.set_ylim([0,22])
    plt.axhline(y=4, color='k', linewidth=2)
    plt.xlabel('size of training set')
    plt.ylabel('mse')
    plt.title('truth = degree {}, model = degree {}'.format(TrueDeg, ModDeg))
    pml.savefig('polyfitN{}.pdf'.format(ModDeg))
    plt.show()

plt.show()
