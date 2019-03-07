# Based on 
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import save_fig
import seaborn as sns
from matplotlib.colors import ListedColormap
   
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


  
# Based on # https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch05/ch05.py#L308
def plot_decision_regions(X, y, classifier, class_names = None):
    sns.set(style="ticks", color_codes=True)
    fig, ax = plt.subplots()
    markers = ('s', 'x', 'o', '^', 'v')
    #colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    #cmap = ListedColormap(colors[:len(np.unique(y))])
    cmap = ListedColormap(sns.color_palette())

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    npoints = 1000
    X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, npoints),
                           np.linspace(x2_min, x2_max, npoints))
    Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape) # NxN array of ints, 0..C-1
    class_ids = np.unique(y)
    nclasses = len(class_ids)
    colors = sns.color_palette()[0:nclasses]
    levels = np.arange(0, nclasses+1)-0.1 # fills in regions z1 < Z <= z2
    ax.contourf(X1, X2, Z, levels=levels, colors=colors, alpha=0.4)
    ax.set(xlim = (X1.min(), X1.max()))
    ax.set(ylim = (X2.min(), X2.max()))

    # plot raw data
    handles = []
    for idx, cl in enumerate(class_ids):
      color = np.atleast_2d(cmap(idx))
      id = ax.scatter(x=X[y == cl, 0], 
                  y=X[y == cl, 1],
                  alpha=0.6, 
                  c=color,
                  edgecolor='black',
                  marker=markers[idx], 
                  label=cl)
      handles.append(id)
    
    if class_names is not None: 
      ax.legend(handles, class_names, scatterpoints=1)
    return fig, ax


if ndims==2:
  fig, ax = plot_decision_regions(X, y, logreg, iris.target_names)
  ax.set(xlabel = 'Sepal length')
  ax.set(ylabel = 'Sepal width')
  save_fig("iris-logistic")
  plt.show()
  
  # Get predictive distribution for some ambiguous test points
  X = [[5.7, 3.5]] # (1,2) array
  y_probs = logreg.predict_proba(X)
  print(np.round(y_probs, 2))