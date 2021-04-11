import pyprobml_utils as pml

#pip install GPy

#pip install GPyOpt

import numpy as np
import matplotlib.pyplot as plt 
import GPy
from scipy import linalg
from scipy import special
import scipy.spatial.distance as spdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def covNoise(hyp, x, z=None, der=None):
  
    tol = 1.e-9                 # Tolerance for declaring two vectors "equal"
    if hyp.all() == None:             # report number of parameters
        return [1]
    
    s2 = np.exp(2.*hyp[0])      # noise variance
    n,D = x.shape

    if z == 'diag':
        A = np.ones((n,1))
    elif z == None:
        A = np.eye(n)
    else:                       # compute covariance between data sets x and z
        M = spdist.cdist(x, z, 'sqeuclidean')
        A = np.zeros_like(M,dtype=np.float)
        A[M < tol] = 1.

    if der == None:
        A = s2*A
    else:                       # compute derivative matrix
        if der == 0:
            A = 2.*s2*A
        else:
            raise Exception("Wrong derivative index in covNoise")

    return A

def covSEiso(hyp, x, z=None, der=None):
    
    if hyp.all() == None:               # report number of parameters
        return [2]

    ell = np.exp(hyp[0])          # characteristic length scale
    sf2 = np.exp(2.*hyp[1])       # signal variance
    n,D = x.shape

    if z == 'diag':
        A = np.zeros((n,1))
    elif z == None:
        A = spdist.cdist(x/ell, x/ell, 'sqeuclidean')
    else:                                # compute covariance between data sets x and z
        A = spdist.cdist(x/ell, z/ell, 'sqeuclidean') # self covariances

    if der == None:                      # compute covariance matix for dataset x
        A = sf2 * np.exp(-0.5*A)
    else:
        if der == 0:                    # compute derivative matrix wrt 1st parameter
            A = sf2 * np.exp(-0.5*A) * A

        elif der == 1:                  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * np.exp(-0.5*A)
        else:
            raise Exception("Calling for a derivative in covSEiso that does not exist")
    
    return A

# Dataset according to the matlab code:

n = 20
x = 15*(np.random.uniform(low=0, high=1, size=20) - 0.5).reshape((-1,1))
q = np.array([1.0, 1.0, 0.1])
log_hyper = np.log(q)

j = covSEiso(hyp=log_hyper, x=x)
k = covNoise(hyp=log_hyper, x=x)
A = j + k                               #covSum(covfunc = covfunc, hyp=log_hyper, x=x)
B = linalg.cholesky(A)
B = B.conjugate()
y = B.dot(np.random.randn(n, 1))        

xstar = np.linspace(-7.5, 7.5, 201)
xstar = xstar.reshape(-1, 1)

l = np.array([1.0, 0.3, 3.0])
sigma_f = np.array([1, 1.08, 1.16])
sigma_y = np.array([0.1, 0.00005, 0.89])

def generate_plots(sigma_f, l, sigma_y):
    kernel = GPy.kern.RBF(1, sigma_f, l) #+ GPy.kern.White(1)
    model = GPy.models.GPRegression(x , y, kernel) 
    model.Gaussian_noise.variance = sigma_y**2
    model.Gaussian_noise.variance.fix()
    mean, variance = model.predict(xstar)
    model.plot()
    plt.title("Hyperparameters (l, sigma_f, sigma_y) are {}, {}, {}".format(l, sigma_f, sigma_y))
    pml.save_fig('Plot' + '-' + str(i) + '.png')
    plt.savefig('Plot' + '-' + str(i) + '.png')
    plt.show()

for i in range(len(l)):
    generate_plots(sigma_f[i], l[i], sigma_y[i])

