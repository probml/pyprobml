import numpy as np
from sklearn.svm import SVC 
from sklearn_rvm import EMRVC 
import h5py  
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.kernel_approximation import RBFSampler

# scipy.io loadmat doesn't seem to be accept the version of this data file

data = {}
with h5py.File('/pyprobml/data/bishop2class.mat', 'r') as f:
  for l,d in f.items():
    data[l] = np.array(d)

X = data['X'].transpose()
Y = data['Y']
y = Y.flatten() 

# Feature Mapping X to rbf_features to simulate non-linear logreg using linear ones.
rbf_feature = RBFSampler(gamma=0.3, random_state=1) 
X_rbf = rbf_feature.fit_transform(X)

# Using CV to find SVM regularization parameter.
C = np.power(2,np.linspace(-5,5,10))
mean_scores = [cross_val_score(SVC(kernel = 'rbf',  gamma=0.3, C=c), X, y, cv=5).mean() for c in C]
c = C[np.argmax(mean_scores)]


classifiers = {
    'logregL2,': LogisticRegression(C=0.2, penalty='l2',
                                                    solver='saga',
                                                    multi_class='ovr',
                                                    max_iter=10000),
    'logregL1,': LogisticRegression(C=1, penalty='l1',
                                      solver='saga',
                                      multi_class='ovr',
                                      max_iter=10000),
    'RVM': EMRVC(kernel="rbf",gamma=0.3),
    'SVM': SVC(kernel = 'rbf',  gamma=0.3, C=c,probability=True)
}

h = 0.05  # step size in the mesh

# Mesh to use in the boundary plotting.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))


for (i,(name,clf)) in enumerate(classifiers.items()):
  if i == 0 or i == 1:
    clf.fit(X_rbf,y)
  else: 
    clf.fit(X, y)

  # decision boundary plot
  # point in the mesh [x_min, m_max]x[y_min, y_max].
  
  if i == 0 or i == 1:  # for logregs
    Z = clf.predict_proba(rbf_feature.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
  else:   # for VMs
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

  # Putting the result into a color plot
  Z = Z[:,0]
  Z = Z.reshape(xx.shape)
  plt.figure(i)
  
  if i == 0 or i == 1: # for logregs
    plt.title(name+", nerr= {}".format(np.sum(y!=clf.predict(X_rbf))))

  else: # for VMs
    plt.title(name+", nerr= {}".format(np.sum(y!=clf.predict(X))))

  plt.contour(xx, yy, Z, colors=['w','w','w','black']) # taking some levels of contours.
  
  for class_value in range(1,3):  
      # get row indexes for samples with this class
      row_ix = np.where(y == class_value)
      # creating scatter of these samples
      plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired',marker='X',s=30)

  # for support_vectors
  if i > 0:
      if i == 1:
        conf_scores = np.abs(clf.decision_function(X_rbf))
        SV = X[(conf_scores > conf_scores.mean())]
      elif i == 3:
        SV = clf.support_vectors_
      elif i== 2:
        SV = clf.relevance_vectors_
      plt.scatter(SV[:, 0], SV[:, 1], s=100, facecolor="none", edgecolor="black")
  plt.savefig("/pyprobml/figures/kernelBinaryClassifDemo{}.pdf".format(name),  dpi=300)
