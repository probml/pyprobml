from polyDataMake import polyDataMake
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import os

#Generate data and split into in and out of sample
xtrain, ytrain, xtest, ytest, _, _ = polyDataMake(sampling='thibaux')

#Determine the width and number of RBFs to use
taus = [.5, 50, 2000]
K = 10 #Set this to 20 to get a perfect interpolation of the data
centers = np.asarray(np.linspace(min(xtrain), max(xtrain), K)).reshape(-1,1)

fig, axs = plt.subplots(3,3)
axs = axs.ravel()

for i in range(len(taus)):
    gamma = 1.0/(taus[i]**2) #Using Sklearn RBF function, which requires a reparameteriation of tau

    XDesignTrain = rbf_kernel(xtrain,centers,gamma)
    
    regr = linear_model.Ridge(alpha=0.00001) #Using ridge instead of ordinary least squares for numerical stability
    regr.fit(XDesignTrain,ytrain)   
    XDesignTest = rbf_kernel(xtest,centers,gamma)
    ypred = regr.predict(XDesignTest)
    
    #Form row i of graph
    axs[i*3].scatter(xtrain,ytrain,c='blue',edgecolor='none')
    axs[i*3].plot(xtest,ypred,c='black')
    axs[i*3].set_xlim([min(xtest),max(xtest)])
    if i==1:
        xtestG = np.array(np.linspace(-20,40,201)).reshape(-1,1)
        XDesignTestG = rbf_kernel(xtestG,centers,gamma)
    elif i==2:
        xtestG = np.array(np.linspace(-1000,1000,201)).reshape(-1,1)
        XDesignTestG = rbf_kernel(xtestG,centers,gamma)
    else:
        xtestG = xtest
        XDesignTestG = XDesignTest
    for col in range(XDesignTest.shape[1]):
        axs[i*3+1].plot(xtestG,XDesignTestG[:,col],c='blue')
    axs[i*3+1].set_ylim([0,1])  
    axs[i*3+2].pcolor(-XDesignTestG, cmap='Greys')    
    ylabs = xtestG[::int(round(len(xtestG)/4))] #In general, should avoid this kind of manual labeling..
    ylabs = [str(int(yl[0])) for yl in ylabs]
    axs[i*3+2].set_yticklabels(ylabs)
    axs[i*3+2].set_ylim([0,XDesignTestG.shape[0]])
    
plt.draw()
plt.show()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'rbfDemoAll.pdf'),orientation='landscape')
