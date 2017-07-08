import os
os.chdir("..")
from utils.util import poly_data_make
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import numpy as np

TrueDeg = 2 #The true degree of the model
degrees = [1, 2, 14, 20] #The degrees of our design matrices
    
#Function to expand from x to design matrix of degree deg
def ExpandtoDeg(x,deg):
    return np.array([x**i for i in range(deg+1)]).transpose().reshape(-1,deg+1)

for ModDeg in degrees:
    
    ns = np.round(np.linspace(10, 210, 20))
    
    err = []
    errtrain = []
    for n in ns:
        #Forming data
        xtrain, ytrain, xtest, _, ytest, _ = poly_data_make(sampling='thibaux', deg=TrueDeg, n=n)
        
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
    plt.title('truth = degree 2, model = degree '+ str(ModDeg))
    plt.savefig(os.path.join('figures', 'polyfitN{}.png'.format(ModDeg)),orientation='landscape')
    plt.draw()

plt.show()
