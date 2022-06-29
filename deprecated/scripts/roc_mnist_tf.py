# ROC curve for different classifiers on mnist data
# We use tensorflow/keras to fit MLPs
# Based on 
#https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb


import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


from time import time

np.random.seed(42)

from scipy.special import logit

import tensorflow as tf
from tensorflow import keras

from sklearn.linear_model import LogisticRegression

def load_mnist_data_keras(flatten=False):
   # Returns X_train: (60000, 28, 28), X_test: (10000, 28, 28), scaled [0..1] 
  # y_train: (60000,) 0..9 ints, y_test: (10000,)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if flatten: 
      Ntrain, D1, D2 = np.shape(x_train)
      D = D1*D2
      assert D == 784
      Ntest = np.shape(x_test)[0]
      x_train = np.reshape(x_train, (Ntrain, D))
      x_test = np.reshape(x_test, (Ntest, D))
    return x_train, x_test, y_train, y_test
  

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


from sklearn.metrics import confusion_matrix, log_loss, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


def eval_classifier(ytrue, yprobs1, name):
  ypred = yprobs1>0.5
  A = accuracy_score(ytrue, ypred)  
  L = log_loss(ytrue, yprobs1)
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
  print("logloss {:0.3f}, acc {:0.3f}, prec {:0.3f}, recall {:0.3f}, F1 {:0.3f}".format(L, A, P, R, F1))
  print(C)
  
  
for name, color, ls, model in models:
  print("\nEvaluating {} on train set".format(name))
  ypred = model.predict(Xtrain)
  yprobs = model.predict_proba(Xtrain)
  ytrue = ytrain
  assert np.isclose(ypred, np.argmax(yprobs, axis=1)).all()
  yprobs1 = yprobs[:,1]
  assert np.isclose(ypred, yprobs1>0.5).all()
  eval_classifier(ytrue, yprobs1, name)

#ytrue=ytrain; yprobs1=np.repeat(0.9, len(ytrain)); eval_classifier(ytrue, yprobs1, 'baseline1')
#ytrue=ytrain; yprobs1=np.repeat(0.1, len(ytrain)); eval_classifier(ytrue, yprobs1, 'baseline0')

"""
Evaluating baseline on train set
evaluating baseline
logloss 0.294, acc 0.914, prec 0.000, recall 0.000, F1 0.000
[[9137    0]
 [ 863    0]]

Evaluating LR-skl on train set
evaluating LR-skl
logloss 0.041, acc 0.988, prec 0.947, recall 0.906, F1 0.926
[[9093   44]
 [  81  782]]

Evaluating LR-keras on train set
evaluating LR-keras
logloss 0.080, acc 0.975, prec 0.936, recall 0.761, F1 0.840
[[9092   45]
 [ 206  657]]

Evaluating MLP on train set
evaluating MLP
logloss 0.009, acc 0.998, prec 0.994, recall 0.986, F1 0.990
[[9132    5]
 [  12  851]]
"""

for name, color, ls, model in models:
  print("\nEvaluating {} on test set".format(name))
  ypred = model.predict(Xtest)
  yprobs = model.predict_proba(Xtest)
  ytrue = ytest
  assert np.isclose(ypred, np.argmax(yprobs, axis=1)).all()
  yprobs1 = yprobs[:,1]
  assert np.isclose(ypred, yprobs1>0.5).all()
  eval_classifier(ytrue, yprobs1, name)


"""
Test set
5 epochs
baseline
evaluating baseline
logloss 0.301, acc 0.911, prec 0.000, recall 0.000, F1 0.000
[[9108    0]
 [ 892    0]]

LR-skl
evaluating LR-skl
logloss 0.154, acc 0.965, prec 0.802, recall 0.806, F1 0.804
[[8930  178]
 [ 173  719]]

LR-keras
evaluating LR-keras
logloss 0.095, acc 0.971, prec 0.924, recall 0.737, F1 0.820
[[9054   54]
 [ 235  657]]

MLP
evaluating MLP
logloss 0.036, acc 0.989, prec 0.966, recall 0.910, F1 0.937
[[9079   29]
 [  80  812]]

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


