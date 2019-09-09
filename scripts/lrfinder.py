# Illustrate the learning rate finder and 1cycle heuristic from Leslie Smith
# It is described in this WACV'17 paper (https://arxiv.org/abs/1506.01186)
# and this  blog post:
# https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
#
# The code below is modified from 
# https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb
# It trains an MLP on FashionMNIST



import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import os


    
np.random.seed(42)



import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import pandas as pd

figdir = "../figures"
def save_fig(fname):
    if figdir: plt.savefig(os.path.join(figdir, fname))
    
K = keras.backend

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(10, activation="softmax")
])
'''
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
'''
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"]) 

batch_size = 128
rates, losses = find_learning_rate(model, X_train_scaled, y_train, epochs=1, batch_size=batch_size)
plt.figure()
plot_lr_vs_loss(rates, losses)
save_fig("lrfinder-raw.pdf")
plt.show()

# https://sites.google.com/site/hardwaremonkey/blog/ewmafilterexmpleusingpandasandpython
#x = np.linspace(0, 2 * np.pi, 100)
#y = 2 * np.sin(x) + 0.1 * np.random.normal(x)
x = rates
y = np.array(losses, dtype=np.float64)
df = pd.Series(y)
filtered = pd.Series.ewm(df, span=10).mean()
plt.figure() #figsize=(10,6))
plt.plot(x, y)
#plt.plot(x, filtered)
plt.gca().set_xscale('log')
plt.xlabel("Learning rate")
plt.ylabel("Loss")
save_fig("lrfinder-unfiltered.pdf")
plt.show()

plt.figure() #figsize=(10,6))
#plt.plot(x, y)
plt.plot(x, filtered)
plt.gca().set_xscale('log')
plt.xlabel("Learning rate")
plt.ylabel("Loss")
save_fig("lrfinder-filtered.pdf")
plt.show()

plt.figure() #figsize=(10,6))
plt.plot(x, y)
plt.plot(x, filtered)
plt.gca().set_xscale('log')
plt.xlabel("Learning rate")
plt.ylabel("Loss")
save_fig("lrfinder-filtered-both.pdf")
plt.show()


class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        #self.half_iteration = (iterations - self.last_iterations) // 2
        self.half_iteration = self.last_iterations // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
        self.rate_hist = []
    def _interpolate_broken(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (iter2 - self.iteration)
                / (iter2 - iter1) + rate1)
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)
        self.rate_hist.append(K.get_value(self.model.optimizer.lr))


#https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009
n_epochs = 5
n_steps_per_epoch = len(X_train) // batch_size
onecycle = OneCycleScheduler(n_steps_per_epoch * n_epochs, max_rate=0.05)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[onecycle])
#lr_hist = history.history["lr"] # only stored by LRScheduler
lr_hist = onecycle.rate_hist

plt.figure()
plt.plot(lr_hist, "o-")
plt.xlabel("Step")
plt.ylabel("Learning rate")
plt.title('onecycle', fontsize=14)
plt.grid(True)
save_fig('lrschedule-onecycle.pdf')
plt.show()