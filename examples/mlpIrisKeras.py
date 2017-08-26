
# MLP  on 3 class Iris data


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets, metrics

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

# import the data 
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

ndim = X.shape[1]
nclasses = len(np.unique(Y))
one_hot_labels = keras.utils.to_categorical(Y, num_classes=nclasses)


# logreg as baseline
logreg = linear_model.LogisticRegression(C=1e5, multi_class='multinomial', solver='lbfgs')
logreg.fit(X, Y)
predicted = logreg.predict(X)
accuracy_logreg = metrics.accuracy_score(Y, predicted)
print(accuracy_logreg) # 0.833

# MLP with 0 hidden layers - should match logreg
model = Sequential([
    Dense(nclasses, input_shape=(ndim,)),
    Activation('softmax'),
])

# Fit

import tensorflow as tf
import scipy

#import custom_optimizers
import imp
# we assume we're executing from /Users/kpmurphy/github/pyprobml
imp.load_source('custom_opt', 'examples/custom_optimizers.py')
import custom_opt

opt = keras.optimizers.Adam()
#opt = custom_opt.ScipyOpt(model=model, x=X, y=Y, nb_epoch=10)

lossfn = keras.losses.categorical_crossentropy

# 
# opt_bfgs_scipy = scipy.optimize.fmin_l_bfgs_b
# lossfn_train = lambda ypred: lossfn(Y, ypred)
# #tfopt = tf.contrib.opt.ScipyOptimizerInterface(lossfn_train, options={'maxiter': 100})
# opt_bfgs_tf = opt_bfgs_scipy
# opt_bfgs = keras.optimizers.TFOptimizer(opt_bfgs_tf)

model.compile(loss=lossfn,
              optimizer=opt,
              metrics=['accuracy'])
history = model.fit(X, one_hot_labels, epochs=500, batch_size=20, verbose=1)

# Plot training speed
loss_trace = history.history['loss']
acc_trace = history.history['acc']
plt.figure()
plt.subplot(1,2,1)
plt.plot(loss_trace)
plt.title('loss')
plt.subplot(1,2,2)
plt.plot(acc_trace)
plt.title('accuracy')
plt.show()

