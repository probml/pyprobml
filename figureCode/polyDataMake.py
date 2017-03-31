import numpy as np
from random import seed, getstate, setstate, normalvariate

def polyDataMake(n=21,deg=3,sampling='sparse'):
    old_state = getstate()
    seed(0)

    if sampling == 'irregular':
        xtrain = np.array([np.linspace(-1,-.5,6),np.linspace(3,3.5,6)]).reshape(-1,1)
    elif sampling == 'sparse':
        xtrain = np.array([-3, -2, 0, 2, 3])
    elif sampling == 'dense':
        xtrain = np.array(np.arange(-5,5,.6))
    elif sampling == 'thibaux':
        xtrain = np.linspace(0,20,n)
    else:
        raise ValueError('Unrecognized sampling provided.')
        
    if sampling == 'thibaux':
        seed(654321)
        xtest = np.linspace(0,20,201)
        sigma2 = 4
        w = np.array([-1.5,1.0/9.0]).T
        def fun(x):
            return w[0]*x + w[1]*(x**2)
    else:
        xtest = np.linspace(-7,7,141)
        if deg == 2:
            def fun(x):
                return 10 + x + x**2
        elif deg == 3 :
            def fun(x):
                return 10 + x + x**3
        else:
            raise ValueError('Unrecognized degree.')
        sigma2 = 25
        
    ytrain = fun(xtrain) + [normalvariate(0,np.sqrt(sigma2)) for i in range(xtrain.shape[0])]
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + [normalvariate(0,np.sqrt(sigma2)) for i in range(xtest.shape[0])]
    
    def shp(x):
        return np.asarray(x).reshape(-1,1)
    
    setstate(old_state)
    return shp(xtrain), shp(ytrain), shp(xtest), shp(ytestNoisefree), shp(ytestNoisy), sigma2
