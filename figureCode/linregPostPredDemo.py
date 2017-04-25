import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.stats import multivariate_normal
from polyDataMake import polyDataMake
from sklearn.linear_model import Ridge

polydeg = 2 #Degree of design matrix
alph = 0.001 #Alpha of ridge regression
NFuncSamples = 10 #Number of sample coefficients to draw and use for prediction
vis = 1 #Transparency of plotted lines - in case we wish to plot a bunch.

np.random.seed(seed=1)

#Generate data
xtrain, ytrain, xtest, ytestNoisefree, ytest, sigma2 =  polyDataMake(sampling = 'sparse', deg = 2)

def polyBasis(x, deg):
    #Expands a vector to a polynomial design matrix: from a constant to the deg-power
    return(np.column_stack([x**deg for deg in range(0, deg+1)]))

def MakePlot(ypreds, SaveN, Title, lowerb = None, upperb = None):
    #Function for creating and saving plots
    fig, ax = plt.subplots()
    ax.scatter(xtrain, ytrain, s=140, facecolors='none', edgecolors='r', label='training data')
    plt.ylim([-10,80])    
    plt.xlim([-8,8])
    Errlogi = lowerb is not None or upperb is not None #Determines where we will be plotting error bars as well
    if Errlogi: 
        errspacing = [int(round(s)) for s in np.linspace(0,xtest.shape[0]-1,30)]
        ax.errorbar(xtest[errspacing], ypreds[errspacing,0], yerr=[lowerb[errspacing],upperb[errspacing]])
    for j in range(ypreds.shape[1]):
        ax.plot(xtest,ypreds[:,j],color='k', linewidth=2.0, label='prediction', alpha = vis)
    if Errlogi:
        plt.legend(loc=2)
    plt.title(Title)
    plt.savefig(os.path.join('figures', SaveN +'.pdf'))

xtrainp = polyBasis(xtrain,polydeg)
xtestp = polyBasis(xtest,polydeg)

#Declare and fit linear regression model
LinR = Ridge(alpha=alph,fit_intercept=False)
LinR.fit(xtrainp,ytrain)

#Determine coefficient distribution
wmle = LinR.coef_.reshape(-1,) #Mean of coefficients
wcov = sigma2 * np.linalg.inv(np.diag([alph]*(polydeg+1)) + xtrainp.T.dot(xtrainp)) #Covariance of coefficients
CoefPostDist = multivariate_normal(mean = wmle, cov = wcov)
Samples = CoefPostDist.rvs(NFuncSamples)
 
#Sample predictions according to samples of coefficients
SamplePreds = xtestp.dot(Samples.T)

ypredmle = LinR.predict(xtestp) #MLE prediction
noisemle = np.var(ytrain - LinR.predict(xtrainp),ddof=(polydeg + 1)) #MLE noise estimation

#plot a
noisevec = np.array([np.sqrt(noisemle)] * ypredmle.shape[0])
MakePlot(ypredmle, 'linregPostPredPlugin', 'plugin approximation (MLE)', noisevec, noisevec)

#plot b
postnoise = np.array([np.sqrt(sigma2 + xtestp[i,:].T.dot(wcov.dot(xtestp[i,:]))) for i in range(xtestp.shape[0])])
MakePlot(ypredmle, 'linregPostPredBayes', 'Posterior predictive (known variance)', postnoise, postnoise)

#plot c
MakePlot(ypredmle, 'linregPostPredSamplesPlugin', 'functions sampled from plugin approximation to posterior')

#plot d
MakePlot(SamplePreds, 'linregPostPredSamples', 'functions sampled from posterior')

plt.show()
