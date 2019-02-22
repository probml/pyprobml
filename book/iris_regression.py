
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import os

# Where to save the figures
IMAGES_PATH = "../figures"
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="pdf", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# import some data to play with
iris = datasets.load_iris()
xidx = 2

ys = [1, 3]
for yidx in ys:
  X = iris.data[:, xidx:xidx+1]  # we only take the first  feature
  Y = iris.data[:, yidx:yidx+1]
  
  linreg = LinearRegression()
  linreg.fit(X, Y)
  
  xs = np.arange(np.min(X), np.max(X), 0.1).reshape(-1,1)
  yhat = linreg.predict(xs)
  plt.plot(xs, yhat)
  sns.scatterplot(x=X[:,0], y=Y[:,0])
  plt.xlabel(iris.feature_names[xidx])
  plt.ylabel(iris.feature_names[yidx])
  plt.xlim(np.min(X), np.max(X))
  plt.ylim(np.min(Y), np.max(Y))
  
  fname = "iris-linreg{}".format(yidx)
  save_fig(fname)
  plt.show()