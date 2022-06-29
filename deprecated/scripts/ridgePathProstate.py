

# Plot the full L2 regularization path for the prostate data set

import superimport

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pyprobml_utils as pml
import requests
from io import BytesIO
from scipy.io import loadmat

# Load prostate cancer data 
#!wget https://github.com/probml/probml-data/blob/main/data/prostateStnd.mat?raw=true -O prostateStnd.mat

# matlab data is created by this
# https://github.com/probml/pmtk3/blob/master/data/prostate/prostateDataMake.m

#data = loadmat('prostateStnd.mat')

url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/prostateStnd.mat'
response = requests.get(url)
#rawdata = response.text
rawdata = BytesIO(response.content)
data = loadmat(rawdata)


names = list(map(lambda x: x[0], data['names'][0]))
X, y = data['X'], data['y']

# Ridge regression

n_alpha = 30
alphas = np.logspace(5,0,n_alpha)

coefs = map(lambda a: linear_model.Ridge(a).fit(X,y).coef_.flatten(), alphas)
coefs = np.array(list(coefs))

# Ridge regression with cross validation

best_model = linear_model.RidgeCV(alphas)
b = best_model.fit(X,y)

# Figure 11.12 (a) 
# Profile of ridge coeficients for prostate cancer example
# Vertical line is values chosen by cross validation

fig, ax = plt.subplots()
plt.plot(coefs,marker='o')
plt.axvline(x=np.where(alphas==best_model.alpha_), c="r")
plt.legend(names)
pml.savefig('prostate_data.pdf')
plt.show()
