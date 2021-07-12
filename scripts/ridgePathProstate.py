
# Figure 11.12 (a)
# Plot the full L2 regularization path for the prostate data set

from scipy.io import loadmat
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load prostate cancer data 

data = loadmat('../data/prostate/prostateStnd')
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
plt.show()