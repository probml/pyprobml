# Based on 
#https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb

import numpy as np
from time import time

np.random.seed(42)

import matplotlib.pyplot as plt
from scipy.special import logit

import tensorflow as tf
from tensorflow import keras

from sklearn.linear_model import LogisticRegression

from utils import load_mnist_data_keras, save_fig

x_train, x_test, y_train, y_test = load_mnist_data_keras(flatten=True)
# Take tiny train subset for speed
N = 10000
Xtrain = x_train[:N, :]
Xtest = x_test
# Convert to a binary problem
target_class = 5
ytrain = (y_train[:N] == target_class)
ytest = (y_test == target_class)
#ytrain_onehot = keras.utils.to_categorical(ytrain)
#ytest_onehot = keras.utils.to_categorical(ytest)


from sklearn.base import BaseEstimator
class BaselineClassifier(BaseEstimator):
    def fit(self, X, y):
        n1 = np.sum(y==1)
        n0 = np.sum(y==0)
        self.prob1 = n1/(n1+n0)
    def predict_proba(self, X):
      N = len(X)
      p1 = np.repeat(self.prob1, N)
      p0 = np.repeat(1-self.prob1, N)
      return np.c_[p0, p1]
    def predict(self, X):
      probs = self.predict_proba(X)
      return probs[:,1] > 0.5 
    

def make_model(nhidden):
  num_classes = 2
  if nhidden == 0:
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])
  else:
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(nhidden, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
  ])
  lr = 0.001
  opt = tf.train.AdamOptimizer(lr)
  model.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

nepochs = 5
batch_size = 32



model_baseline = BaselineClassifier()

model_logreg_skl = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

model_logreg_keras = keras.wrappers.scikit_learn.KerasClassifier(
        make_model, verbose=0, nhidden=0, epochs=nepochs, batch_size=batch_size)
 
model_mlp = keras.wrappers.scikit_learn.KerasClassifier(
        make_model, verbose=0, nhidden=100, epochs=nepochs, batch_size=batch_size)


models = []  # (name, color, linestyle, model)
models.append(('baseline', 'b', '-',  model_baseline))
models.append(('LR-skl', 'b', ':', model_logreg_skl))
models.append(('LR-keras', 'r', '-', model_logreg_keras))
models.append(('MLP', 'k', '--', model_mlp))

for name, color, ls, model in models:
  print(name)
  time_start = time()
  model.fit(Xtrain, ytrain)
  print('time spent training {:0.3f}'.format(time() - time_start))
  
"""
2 epochs:
baseline
time spent training 0.000
LR-keras
time spent training 1.314
LR-skl
time spent training 1.849
MLP
time spent training 2.345
"""


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def eval_classifier(ytrue, ypred, name):
  print("evaluating {}".format(name))
  A = accuracy_score(ytrue, ypred)  
  P = precision_score(ytrue, ypred)
  R = recall_score(ytrue, ypred)
  C = confusion_matrix(ytrue, ypred)
  F1 = f1_score(ytrue, ypred)
  N = len(ytrue)
  TN = C[0,0]; FP=C[0,1]; FN=C[1,0]; TP=C[1,1];
  eps = 1e-10
  assert np.isclose(A, (TN+TP)/N)
  assert np.isclose(P, TP/(TP+FP+eps))
  assert np.isclose(R, TP/(TP+FN+eps))
  assert np.isclose(F1, 2*(P*R)/(P+R+eps))
  print("acc {:0.3f}, prec {:0.3f}, recall {:0.3f}, F1 {:0.3f}".format(A, P, R, F1))
  print(C)
  
  
for name, color, ls, model in models:
  print(name)
  ypred = model.predict(Xtest)
  eval_classifier(ytest, ypred, name)


"""

5 epochs
evaluating baseline
acc 0.911, prec 0.000, recall 0.000, F1 0.000
[[9108    0]
 [ 892    0]]

evaluating LR-keras
acc 0.971, prec 0.918, recall 0.744, F1 0.822
[[9049   59]
 [ 228  664]]

evaluating LR-skl
acc 0.965, prec 0.802, recall 0.806, F1 0.804
[[8930  178]
 [ 173  719]]

evaluating MLP
acc 0.988, prec 0.981, recall 0.883, F1 0.930
[[9093   15]
 [ 104  788]]

"""

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
  

fig, ax = plt.subplots()
for name, color, ls, model in models:
  print(name)
  yprobs = model.predict_proba(Xtest)
  yscores = yprobs[:,1]
  ytrue = ytest
  fpr, tpr, thresholds = roc_curve(ytrue, yscores)
  auc = roc_auc_score(ytrue, yscores)
  label = "{}, auc={:0.2f}".format(name, auc)
  ax.plot(fpr, tpr, ls=ls, color=color, label=label, linewidth=2)
  ax.set(xlabel = "False positive rate", xlim=[0, 1])
  ax.set(ylabel = "True positive rate (recall)", ylim=[0, 1])
  ax.grid(True)
plt.legend(loc="best")
save_fig("mnist-ROC-curves.pdf")
plt.show()

fig, ax = plt.subplots()
for name, color, ls, model in models:
  print(name)
  yprobs = model.predict_proba(Xtest)
  yscores = yprobs[:,1]
  ytrue = ytest
  precisions, recalls, thresholds = precision_recall_curve(ytrue, yscores)
  ap = average_precision_score(ytrue, yscores)
  label = "{}, ap={:0.2f}".format(name, ap)
  ax.plot(recalls, precisions, ls=ls, color=color, label=label, linewidth=2)
  ax.set(xlabel = "Recall", xlim=[0, 1])
  ax.set(ylabel = "Precision", ylim=[0, 1])
  ax.grid(True)
plt.legend(loc="best")
save_fig("mnist-PR-curves.pdf")
plt.show()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)        
    plt.grid(True)                            

    
model  = model_mlp
yprobs = model.predict_proba(Xtest)
ytrue = ytest

precisions, recalls, thresholds = precision_recall_curve(ytrue, yprobs[:,1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.title("Probs")
save_fig("mnist-PR-thresh-probs.pdf")
plt.show()

precisions, recalls, thresholds = precision_recall_curve(ytrue, logit(yprobs[:,1]))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.title("Logits")
save_fig("mnist-PR-thresh-logits.pdf")
plt.show()


