# Boston housing demo


import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

import pandas as pd
import sklearn.datasets
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

# Prevent numpy from printing too many digits
np.set_printoptions(precision=3)


# Load data (creates numpy arrays)
boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target

# Convert to Pandas format
df = pd.DataFrame(X)
df.columns = boston.feature_names
df['MEDV'] = y.tolist()

df.describe()


# plot marginal histograms of each column (13 features, 1 response)
df.hist()
plt.show()



# scatter plot of response vs each feature 
nrows = 3; ncols = 4;
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=[15, 10])
plt.tight_layout()
plt.clf()
for i in range(0,12):
    plt.subplot(nrows, ncols, i+1)
    plt.scatter(X[:,i], y)
    plt.xlabel(boston.feature_names[i])
    plt.ylabel("house price")
    plt.grid()
save_fig("boston-housing-scatter.pdf")
plt.show()


# Rescale input data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(X_train)
Xscaled = scaler.transform(X_train)
# equivalent to Xscaled = scaler.fit_transform(X_train)

# Fit model
linreg = lm.LinearRegression()
linreg.fit(Xscaled, y_train)

# Extract parameters
coef = np.append(linreg.coef_, linreg.intercept_)
names = np.append(boston.feature_names, 'intercept')
print([name + ':' + str(round(w,1)) for name, w in zip(names, coef)])
"""
['CRIM:-1.0', 'ZN:0.9', 'INDUS:0.4', 'CHAS:0.9', 'NOX:-1.9', 'RM:2.8', 'AGE:-0.4', 
'DIS:-3.0', 'RAD:2.0', 'TAX:-1.4', 'PTRATIO:-2.1', 'B:1.0', 'LSTAT:-3.9', 'intercept:23.0']
"""

# Assess fit on test set
Xscaled = scaler.transform(X_test)
ypred = linreg.predict(Xscaled) 

plt.figure()
plt.scatter(y_test, ypred)
plt.xlabel("true price")
plt.ylabel("predicted price")
mse = sklearn.metrics.mean_squared_error(y_test, ypred)
plt.title("Boston housing, rmse {:.2f}".format(np.sqrt(mse)))
xs = np.linspace(min(y), max(y), 100)
plt.plot(xs, xs, '-')
save_fig("boston-housing-predict.pdf")
plt.show()

