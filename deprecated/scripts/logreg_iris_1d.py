
# Logistic regression on the iris flower dataset.

import superimport

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


iris = datasets.load_iris()
X = iris.data 
y = iris.target

# vsetosa = class 0, versicolor = class 1
# use sepal length feature
df_iris = pd.DataFrame(data=iris.data, 
                        columns=['sepal_length', 'sepal_width', 
                                 'petal_length', 'petal_width'])
df_iris['species'] = pd.Series(iris.target_names[y], dtype='category')
    
df = df_iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length' 
x_0 = df[x_n].values
#xmean = np.mean(x_0)    
#x_c = x_0 - xmean
#X = x_c.reshape(-1,1)
X = x_0.reshape(-1,1)
y = y_0

#log_reg = LogisticRegression(solver="lbfgs", penalty='none')
# Penalty='none' introduced in sklearn 0.21.
# For older versions, use this method:
log_reg = LogisticRegression(solver="lbfgs", C=1000)
log_reg.fit(X, y)

X_new = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
p1 = y_proba[:,1]
decision_boundary = X_new[p1 >= 0.5][0]
plt.figure()
plt.plot(X_new, p1,  '-', linewidth=2, label="Versicolor")
plt.vlines(decision_boundary, 0, 1)
plt.xlabel(x_n)
plt.ylabel('p(y=1)')
# plot jittered data
plt.scatter(x_0, np.random.normal(y_0, 0.02),
            marker='.', color=[f'C{label}' for label in y_0])
plt.tight_layout()
plt.savefig('../figures/logreg_iris_1d.pdf', dpi=300) 
plt.show()
