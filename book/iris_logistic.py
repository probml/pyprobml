# Based on 
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import utils
   
iris = datasets.load_iris()
ndims = 2 #4
X = iris.data[:, :ndims]  # we only take the first two features.
#X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# L2-regularizer lambda=1/C, set to np.inf to get MLE
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
errs = (y_pred != y_test)
nerrs = np.sum(errs)
print("Made {} errors out of {}, on instances {}".format(nerrs, len(y_pred), np.where(errs)))
# With ndims=2: Made 10 errors out of 50, on instances
#  (array([ 4, 15, 21, 32, 35, 36, 40, 41, 42, 48]),)


from sklearn.metrics import zero_one_loss
err_rate_test = zero_one_loss(y_test, y_pred)
assert np.isclose(err_rate_test, nerrs / len(y_pred))
err_rate_train =  zero_one_loss(y_train, logreg.predict(X_train))
print("Error rates on train {:0.3f} and test {:0.3f}".format(
    err_rate_train, err_rate_test))
#Error rates on train 0.180 and test 0.200

if ndims==2:
  fig, ax = utils.plot_decision_regions(X, y, logreg, iris.target_names)
  ax.set(xlabel = 'Sepal length')
  ax.set(ylabel = 'Sepal width')
  utils.save_fig("iris-logistic")
  plt.show()
  
  # Get predictive distribution for some ambiguous test points
  X = [[5.7, 3.5]] # (1,2) array
  y_probs = logreg.predict_proba(X)
  print(np.round(y_probs, 2))