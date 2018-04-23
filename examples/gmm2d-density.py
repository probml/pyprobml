#density plot  of 2d GMM
import matplotlib.pyplot as plt
import numpy as np
import os

# create data
N = 50000
#x = np.random.normal(size=N)
#y = x * 3 + np.random.normal(size=N)

if 0:
    mu1 = [-2, -2]
    Sigma1 = 1* np.asarray([[1, 0], [0, 1]])
    x1, y1 = np.random.multivariate_normal(mu1, Sigma1, N).T
    
#x1 = np.random.uniform(-5.0, 5.0, N)
#y1 = np.random.uniform(-5, 5, N)

x1, y1 = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))


if 0:
    mu2 = [2, 2]
    Sigma2 = 1* np.asarray([[1, 0], [0, 1]])
    x2, y2 = np.random.multivariate_normal(mu2, Sigma2, N).T
    
    x = np.append(x1,  x2)
    y = np.append(y1, y2)
else:
    x = x1
    y = y1
    
# Big bins
nbins = 10
plt.hist2d(x, y, bins=(nbins, nbins), cmap=plt.cm.jet)

plt.show()
 
folder = "/Users/kpmurphy/github/pyprobml/figures"
fname = "gmm2d-density"
plt.savefig(os.path.join(folder, "{}.png".format(fname)))
