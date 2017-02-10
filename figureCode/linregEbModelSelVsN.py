import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Bayesian model selection demo for polynomial regression
# This illustartes that if we have more data, Bayes picks a more complex model.

# Based on a demo by Zoubin Ghahramani

random.seed(0)

Ns = [5, 30] #Number of points
degs = [1, 2, 3] #Degrees of linear regression models.

def polyBasis(x, deg):
    #Takes a vector and returns a polynomial basis matrix up to degree deg (not including ones)
    return(np.column_stack([x**deg for deg in range(1, deg+1)]))

def linregFitBayes(X, ytrain, **kwargs):
    #Bayesian inference for a linear regression model.

    #The model is p(y | x) = N(y | w*[1 x], (1/beta)) so beta is the precision of the measurement noise

    # OUTPUTS model contains the parameters of the posterior, suitable for input to linregPredictBayes.
    # logev is the log marginal likelihood

    #This function is structured so it can be expanded with the full set of options provided in
    #linregFitBayes.m in pmtk3.
    if kwargs['prior'] == 'EB':
        args = {v for k, v in kwargs.iteritems() if k not in 'preproc'}
        [model, logev, LHist] = linregFitEB(X, ytrain, kwargs['preproc'], maxIter=kwargs['maxIter'])
    else:
        raise ValueError('Unrecognized Prior type given')
    model['modelType'] = 'linregBayes'
    model['prior'] = kwargs['prior']
    return model, logev

def preprocessorApplyToTrain(preproc, X):
    if('addOnes' in preproc.keys() and preproc['addOnes']):
        X = np.column_stack((np.ones(X.shape[0]), X))
    return preproc, X

def linregFitEB(X, y, preproc, **kwargs):
    #This closely follows the code of linregFitEbChen giving in pmtk3/toolbox

    [preproc, X] = preprocessorApplyToTrain(preproc, X)

    [N, M] = X.shape

    XX = np.dot(np.transpose(X), X)
    XX2 = np.dot(X, np.transpose(X))
    Xy = np.dot(np.transpose(X), y)

    #This method can get stuck in local minima, so we should do multiple restarts.

    alpha = 0.01 #initially don't trust the prior
    beta = 1 #initially trust the data

    L_old = - float('inf')
    Lhist = np.empty((kwargs['maxIter'], 1))

    for i in range(kwargs['maxIter']):
        if(N > M):
            T = alpha*np.identity(M) + XX*beta
            cholT = np.transpose(np.linalg.cholesky(T))
            Ui = np.linalg.inv(cholT)
            Sn = np.dot(Ui, np.transpose(Ui))
            logdetS = - 2 * sum(np.log(np.diag(cholT)))
        else:
            T = np.identity(N) / beta + XX2 / alpha
            cholT = np.transpose(np.linalg.cholesky(T))
            print(T)
            Ui = np.linalg.inv(cholT)
            XU = np.dot(np.transpose(X), Ui)
            Sn = np.identity(M) / alpha - np.dot(XU, np.transpose(XU))/alpha/alpha
            logdetS = sum(np.log(np.diag(cholT))) * 2 + M * np.log(alpha) + N * np.log(beta)
            logdetS = - logdetS

        mn = beta*np.dot(Sn, Xy)

        t1 = sum((y - np.dot(X, mn))*(y - np.dot(X, mn)))
        t2 = np.dot(np.transpose(mn), mn)

        M = float(M)
        N = float(N)
        gamma = M - alpha * np.trace(Sn)
        beta = (N - gamma) / t1

        L = M * np.log(alpha) - N * np.log(2 * np.pi) + N * np.log(beta) - beta * t1 - alpha * t2 + logdetS
        L = L / 2
        Lhist[i] = L

        if abs(L - L_old) < 1e-2:
            break
        else:
            L_old = L
            alpha = gamma/t2

    model = {'wN': mn, 'VN': Sn, 'beta': beta, 'alpha': alpha, 'gamma': gamma, 'preproc': preproc}

    return model, L, Lhist

def linregPredictBayes(model, X):
    #This accepts a model of the form produced by linregFitBayes and an array of X to form posterior predictions
    [_, X] = preprocessorApplyToTrain(model['preproc'], X)
    yhat = np.dot(X, model['wN'])
    sigma2Hat = (1.0/model['beta']) + np.diag(np.dot(np.dot(X, model['VN']), np.transpose(X)))
    return yhat, sigma2Hat

#We loop over each setting for the number of data points
for n in Ns:
    x1d = np.random.uniform(0, 10, n) #input points
    e = np.random.normal(0, 1, n) #noise
    ytrain = (x1d - 4.0)**2 + 5.0*e #observed y
    plotvals1d = np.arange(-2.0, 12.1, .1) #grid for plotting/testing
    trueOutput = (plotvals1d - 4.0) ** 2 #true function
    logevs = []
    #We loop over the number of degree in our regression.
    for deg in degs:
        X = polyBasis(x1d, deg) #Polynomial basis
        pp = {'addOnes': True} #Setting for feature preprocessing
        [mod, logev] = linregFitBayes(X, ytrain, prior='EB', preproc=pp, maxIter=20) #Fit the model
        logevs.append(logev)
        Xtest = polyBasis(plotvals1d, deg) #Grid to test our prediction on
        [mu, sig2] = linregPredictBayes(mod, Xtest)
        sig2 = np.sqrt(sig2)
        #Form line graph
        fig, ax = plt.subplots()
        plt.scatter(x1d, ytrain, s=140, facecolors='none', edgecolors='k')
        lower = mu - sig2
        upper = mu + sig2
        plt.plot(plotvals1d, trueOutput, 'g', plotvals1d, mu, 'r--', linewidth=2)
        plt.plot(plotvals1d, lower, 'b-' , plotvals1d, upper, 'b-', linewidth=0.5)
        plt.title('d='+str(deg)+', logev='+str(np.round(logev, 2))+', EB')
        plt.savefig(os.path.join('figures', 'linregEbModelSelVsN%dD%dEB'%(n, deg) + '.pdf'))
        plt.draw()

    #Form bar graph showing the posterior probabilities for each model
    PP = np.exp(logevs)
    PP = PP/sum(PP)
    fig, ax = plt.subplots()
    ax.bar(list(range(len(PP))), PP, align='center')
    plt.xticks(list(range(len(PP))))
    plt.ylim([0, 1])
    ax.set_ylabel('P(M|D)')
    plt.title('N='+str(n)+', Method=EB')
    plt.savefig(os.path.join('figures', 'linregEbModelSelVsN' + str(n) + 'PostEB.pdf'))
    plt.draw()

plt.show()


