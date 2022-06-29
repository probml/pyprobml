# Figure 11.12 (b)
# Plot the full L1 regularization path for the prostate data set

import superimport

from scipy.io import loadmat
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import requests
from io import BytesIO
import pyprobml_utils as pml

# Load prostate cancer data 
url = 'https://github.com/probml/probml-data/blob/main/data/prostate/prostateStnd.mat?raw=true'
response = requests.get(url)
rawdata = BytesIO(response.content)
data = loadmat(rawdata)

names = list(map(lambda x: x[0], data['names'][0]))
X, y = data['X'], data['y']

# Finding coefficients for lasso path

_,_,coefs = linear_model.lars_path(X, y.flatten(), method='lasso')

tau = np.sum(np.abs(coefs.T), axis=1)

# Figure 11.12 (b) 
#Profile of lasso coeficients for prostate cancer example

fig, ax = plt.subplots()
xs = tau
ys = coefs.T
plt.xlabel(r'$\tau$')
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.plot(xs,ys,marker='o')
plt.legend(names)
pml.savefig('L1.pdf')
plt.show()

