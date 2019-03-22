
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import numpy as np
import os

TrueDeg = 2 #The tree degree of the model
degrees = [1, 2, 10, 50] #The degrees of our design matrices
    
#Function to expand from x to design matrix of degree deg
def ExpandtoDeg(x,deg):
    return np.array([x**i for i in range(deg+1)]).transpose().reshape(-1,deg+1)


def poly_data_make(sampling="sparse", deg=3, n=21):
    """
    Create an artificial dataset
    """
    np.random.seed(0)

    if sampling == "irregular":
        xtrain = np.concatenate(
            (np.arange(-1, -0.5, 0.1), np.arange(3, 3.5, 0.1)))
    elif sampling == "sparse":
        xtrain = np.array([-3, -2, 0, 2, 3])
    elif sampling == "dense":
        xtrain = np.arange(-5, 5, 0.6)
    elif sampling == "thibaux":
        xtrain = np.linspace(0, 20, n)
        xtest = np.arange(0, 20, 0.1)
        sigma2 = 4
        w = np.array([-1.5, 1/9.])
        fun = lambda x: w[0]*x + w[1]*np.square(x)

    if sampling != "thibaux":
        assert deg < 4, "bad degree, dude %d" % deg
        xtest = np.arange(-7, 7, 0.1)
        if deg == 2:
            fun = lambda x: (10 + x + np.square(x))
        else:
            fun = lambda x: (10 + x + np.power(x, 3))
        sigma2 = np.square(5)

    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)

    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2

for ModDeg in degrees:
    
    ns = np.round(np.linspace(10, 200, 10))
    
    err = []
    errtrain = []
    for n in ns:
        #Forming data
        xtrain, ytrain, xtest, _, ytest, _ = poly_data_make(n=n, sampling='thibaux',deg=TrueDeg)
        
        #Rescaling data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

        #Fitting ridge regression. Small differences in alpha near zero make a visual difference in the plot when n is close to 0.
        regr = Ridge(alpha=0.00001, fit_intercept=False) #Using ridge instead of ordinary least squares for numerical stability
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
    plt.title('truth = degree 2, model = degree '+ str(ModDeg))
    plt.savefig(os.path.join('figures', 'polyfitN{}.pdf'.format(ModDeg)),orientation='landscape')
    plt.draw()

plt.show()
