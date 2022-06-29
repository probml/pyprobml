import superimport

# feature importance using random forests  on ,mnist
# Based on https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)
X= mnist["data"]
y = mnist["target"]

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.figure()
    plt.imshow(image, cmap = mpl.cm.hot,
               interpolation="nearest")
    plt.axis("off")
    

    
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
ndxA = np.where(y==0)[0]
ndxB = np.where(y==8)[0]
ndx = np.concatenate((ndxA, ndxB))
Xc = X[ndx,:]
yc = y[ndx]
rnd_clf.fit(Xc, yc)

plot_digit(rnd_clf.feature_importances_)
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])
plt.savefig("../figures/rf_feature_importance_mnist.pdf", dpi=300)
#plt.savefig("../figures/rf_feature_importance_mnist_class{}.pdf".format(c), dpi=300)
plt.show()