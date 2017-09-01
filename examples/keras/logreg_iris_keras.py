# MLP  on 3 class Iris data


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets, metrics

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import scipy

np.random.seed(123) # try to enforce reproduacability

# import the data 
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

ncases = X.shape[0]
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


lossfn = keras.losses.categorical_crossentropy
#lossfn_train = lambda ypred: lossfn(Y, ypred)

# Use a keras optimizer - works
opt = keras.optimizers.Adam()
#https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L385

# Use a TF optimizer - works
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
#opt_tf = tf.train.AdamOptimizer()
#opt = keras.optimizers.TFOptimizer(opt_tf)

# Use a scipy optimizer - FAILS
#import custom_optimizers
#import imp
# we assume we're executing from /Users/kpmurphy/github/pyprobml
#imp.load_source('custom_opt', 'examples/custom_optimizers.py')
#import custom_opt
##opt = custom_opt.ScipyOpt(model=model, x=X, y=Y, nb_epoch=10)

# opt_bfgs_scipy = scipy.optimize.fmin_l_bfgs_b
# #tfopt = tf.contrib.opt.ScipyOptimizerInterface(lossfn_train, options={'maxiter': 100})

batch_size = ncases # full batch
model.compile(loss=lossfn,
              optimizer=opt,
              metrics=['accuracy'])
history = model.fit(X, one_hot_labels, epochs=500, batch_size=batch_size, verbose=0)
final_acc = history.history['acc'][-1]
print('final accuracy of model with 0 hidden layers {0:.2f}'.format(final_acc))

# Plot training speed - gets close to performance of batch logreg
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


############
# MLP with 1 hidden layers


nhidden = 10
# model = Sequential([
#     Dense(nhidden, input_shape=(ndim,)),
#     Activation('relu'),
#     Dense(nclasses),
#     Activation('softmax'),
# ])

model = Sequential()
model.add(Dense(nhidden, activation='relu', input_dim=ndim))
model.add(Dense(nclasses, activation='softmax'))

optimizers = [keras.optimizers.Adam(), 
              keras.optimizers.TFOptimizer(tf.train.AdamOptimizer())];
optimizer_names = ['AdamKeras', 'AdamTF'];
opt_acc = {}
for i, opt in enumerate(optimizers):
    opt_name = optimizer_names[i]
    model.compile(loss=lossfn,
                optimizer=opt,
                metrics=['accuracy'])
    history = model.fit(X, one_hot_labels, epochs=50, batch_size=batch_size, verbose=0)
    final_acc = history.history['acc'][-1]
    opt_acc[opt_name] = final_acc
    print('final accuracy of model with 1 hidden layers {0:.2f}'.format(final_acc))
print(opt_acc)

