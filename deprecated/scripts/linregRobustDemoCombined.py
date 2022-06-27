import superimport

import pyprobml_utils as pml
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
from statsmodels.miscmodels.tmodel import TLinearModel
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
import statsmodels.formula.api as smf

#!pip install statsmodels

"""Dataset according to the matlab code and book"""

np.random.seed(1) #1
x = np.random.uniform(low=0.3, high=1.1, size=(10,)) 
x = x.flatten()
x = np.sort(x)

y = (np.random.rand(len(x))-0.5 + 1 + 2*x)
y = y.flatten()
y = np.sort(y)

x = np.append(x, 0.1)
x = np.append(x, 0.5)
x = np.append(x, 0.9)
x = x.reshape(-1, 1)

y = np.append(y, -5)
y = np.append(y, -5)
y = np.append(y, -5)
y = y.reshape(-1, 1)

x_test = np.arange(0, 1.1, 0.1)
x_test = x_test.reshape((len(x_test), 1))

xmax = 1 #np.max(x)
xmin = 0 #np.min(x)
ymax = 4 #np.max(y) 
ymin = -6 #np.min(y)

"""L2"""

reg1 = LinearRegression()

model1 = reg1.fit(x, y)
y_pred1 = model1.predict(x_test)

"""Huber"""

reg2 = HuberRegressor(epsilon = 1)

model2 = reg2.fit(x, y)
y_pred2 = model2.predict(x_test)

"""L1"""

dfx = pd.DataFrame(x, columns = ['x'])
dfy = pd.DataFrame(y, columns = ['y'])
exog = sm.add_constant(dfx['x'])
endog = dfy['y']
dft = pd.DataFrame(x_test, columns = ['test'])

qrmodel = QuantReg(endog, exog)
result = qrmodel.fit(q=0.5)

ypred_qr = np.dot(dft, result.params[1]) + result.params[0] #results.predict(dft)

"""Student-t"""

tmodel = TLinearModel(endog, exog)
results = tmodel.fit(df=0.6)

ypred_t = np.dot(dft, results.params[1]) + results.params[0] #results.predict(dft)

"""Plot"""

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.yticks(np.arange(ymin, ymax, 1.0))
plt.scatter(x, y, color="none", edgecolor="black")
plt.plot(x_test, y_pred1, '-.', color='black') #Least squares
plt.plot(x_test, y_pred2, '--', color='green') #Huber
plt.plot(x_test, ypred_t, color='red')         #student
plt.plot(x_test, ypred_qr, '--', color='blue') 
plt.legend(["Least squares", "Huber, \u0394 =1", "Student-t, \u03BD =0.6", "Laplace"])
pml.savefig('Robust.pdf')
plt.show()
