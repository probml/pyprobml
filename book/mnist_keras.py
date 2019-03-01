import tensorflow as tf
from time import time
import numpy as np

from utils import load_mnist_data_keras
from tensorflow import keras

from sklearn.metrics import zero_one_loss

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
  opt = tf.train.AdamOptimizer(lr)
  model.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

nepochs = 2
batch_size = 32

for nhidden in [0, 100]:
  print("using {} hidden units".format(nhidden))
  model= make_model(nhidden)
  time_start = time()
  model.fit(x_train, y_train, epochs=nepochs, batch_size=batch_size)
  print('time spent training {:0.3f}'.format(time() - time_start))
  y_pred_probs = model.predict(x_test) # (10000, 10)
  y_pred = np.argmax(y_pred_probs, axis=1)
  acc = 1-zero_one_loss(y_test, y_pred)
  print("test accuracy {:0.3f}".format(acc)) # 0.915
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
        make_model, nhidden=0, epochs=nepochs, batch_size=batch_size)
  time_start = time()
  history = model_sk.fit(x_train, y_train)
  print('time spent training {}'.format(time() - time_start))
  ypred_probs = model_sk.predict_proba(x_test) # (1000, 10)
  y_pred = model_sk.predict(x_test) # (10000,)
  acc = 1-zero_one_loss(y_test, y_pred)
  print("test accuracy {:0.3f}".format(acc)) # 0.915


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

