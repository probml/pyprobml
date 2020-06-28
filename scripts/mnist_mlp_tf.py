from sklearn.metrics import zero_one_loss
from tensorflow import keras
import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from time import time
import os
figdir = "../figures"
#figdir = os.path.join(os.environ["PYPROBML"], "figures")


def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


# print(tf.__version__)
np.random.seed(0)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# print(np.shape(train_images))
# print(np.shape(test_images))
#(60000, 28, 28)
#(10000, 28, 28)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
save_fig("mnist-data.pdf")
plt.show()


def load_mnist_data_keras(flatten=False):
   # Returns X_train: (60000, 28, 28), X_test: (10000, 28, 28), scaled [0..1]
  # y_train: (60000,) 0..9 ints, y_test: (10000,)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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


x_train, x_test, y_train, y_test = load_mnist_data_keras()

# convert class vectors to binary class matrices
num_classes = 10
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# If we use integers as labels, we should use sparse_categorical_crossentropy.
# If we use onehots, we should use categorical_crossentropy.


def make_model(nhidden):
    if nhidden == 0:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(nhidden, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
        ])
    lr = 0.001
    opt = tf.optimizers.Adam(lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


nepochs = 2
batch_size = 32

for nhidden in [0, 100]:
    print("using {} hidden units".format(nhidden))
    model = make_model(nhidden)
    time_start = time()
    model.fit(x_train, y_train, epochs=nepochs, batch_size=batch_size)
    print('time spent training {:0.3f}'.format(time() - time_start))
    y_pred_probs = model.predict(x_test)  # (10000, 10)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = 1-zero_one_loss(y_test, y_pred)
    print("test accuracy {:0.3f}".format(acc))  # 0.915
    metric_names = model.metrics_names
    metric_values = model.evaluate(x_test, y_test)
    for metric, value in zip(metric_names, metric_values):
        print(metric, ': ', value)


"""
1 layer MLP with 512 hidden units
Epoch 1/5
60000/60000 [==============================] - 13s 217us/sample - loss: 0.2178 - acc: 0.9357
Epoch 2/5
60000/60000 [==============================] - 19s 309us/sample - loss: 0.0981 - acc: 0.9698
Epoch 3/5
60000/60000 [==============================] - 15s 253us/sample - loss: 0.0689 - acc: 0.9784
Epoch 4/5
60000/60000 [==============================] - 16s 265us/sample - loss: 0.0536 - acc: 0.9830
Epoch 5/5
60000/60000 [==============================] - 17s 276us/sample - loss: 0.0436 - acc: 0.9854
10000/10000 [==============================] - 1s 87us/sample - loss: 0.0726 - acc: 0.9798
time spent training 80.55410885810852
loss: 0.07
acc: 0.98


logistic regression
loss: 0.275
acc: 0.923
"""


print("\nsklearn version")
# SKlearn interface
# https://github.com/keras-team/keras/blob/master/examples/mnist_sklearn_wrapper.py

#from keras.wrappers.scikit_learn import KerasClassifier
for nhidden in [0, 100]:
    print("using {} hidden units".format(nhidden))
    model_sk = keras.wrappers.scikit_learn.KerasClassifier(
        make_model, nhidden=nhidden, epochs=nepochs, batch_size=batch_size)
    time_start = time()
    history = model_sk.fit(x_train, y_train)
    print('time spent training {}'.format(time() - time_start))
    ypred_probs = model_sk.predict_proba(x_test)  # (1000, 10)
    y_pred = model_sk.predict(x_test)  # (10000,)
    acc = 1-zero_one_loss(y_test, y_pred)
    print("test accuracy {:0.3f}".format(acc))  # 0.915


"""
# Reformat to 3d tensor for CNN
img_rows, img_cols = 28, 28

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
"""
