# Based on 
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import utils
   
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
#X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# L2-regularizer lambda=1/C
logreg = LogisticRegression(C=1e3, solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train, y_train)

ypred = logreg.predict(X_test)
errs = (ypred != y_test)
nerrs = np.sum(errs)
print("Made {} errors out of {}, on instances {}".format(nerrs, len(ypred), np.where(errs)))
# Made 10 errors out of 50, on instances (array([ 4, 15, 21, 32, 35, 36, 40, 41, 42, 48]),)


fig, ax = utils.plot_decision_regions(X, y, logreg, iris.target_names)
ax.set(xlabel = 'Sepal length')
ax.set(ylabel = 'Sepal width')
utils.save_fig("iris-logistic")
plt.show()

  
