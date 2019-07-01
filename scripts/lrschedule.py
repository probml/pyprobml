# Illustrate various learning rate schedules
# Based on 
# https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

figdir = "../figures"
def save_fig(fname):
    if figdir: plt.savefig(os.path.join(figdir, fname))
    
K = keras.backend

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
# There are 60k training examples. For speed, we use 10k for training
# and 10k for validation.
n_train = 1000
n_valid = 1000
X_valid, X_train = X_train_full[:n_valid], X_train_full[n_valid:n_valid+n_train]
y_valid, y_train = y_train_full[:n_valid], y_train_full[n_valid:n_valid+n_train]

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds


n_epochs = 20
lr0 = 0.01
batch_size = 32
n_steps_per_epoch = len(X_train) // batch_size
epochs = np.arange(n_epochs)


def make_model(lr0=0.01, momentum=0.9):
    optimizer = keras.optimizers.SGD(lr=lr0, momentum=momentum)
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# Power scheduling
# lr = lr0 / (1+ steps/s)**c
# Keras.optimizer.SGD uses power scheduling with c=1 and s=1/decay

def power_decay(lr0, s, c=1):
    def power_decay_fn(epoch):
        return lr0 / (1 + epoch/s)**c
    return power_decay_fn

power_schedule = keras.callbacks.LearningRateScheduler(
                     power_decay(lr0=lr0, s=20))

# Exponential scheduling
# lr = lr0 * 0.1**(epoch / s)
 
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_schedule = keras.callbacks.LearningRateScheduler(
        exponential_decay(lr0=lr0, s=20))

# Piecewise constant

def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)
    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]
    return piecewise_constant_fn

piecewise_schedule = keras.callbacks.LearningRateScheduler(
        piecewise_constant([5, 15], [0.01, 0.005, 0.001]))

# Performance scheduling
perf_schedule = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)


# Make plots
schedules = {'power': power_schedule,
             'exp': exponential_schedule,
             'piecewise': piecewise_schedule}

schedules = {'perf': perf_schedule}

for name, lr_scheduler in schedules.items():
    tf.random.set_seed(42)
    np.random.seed(42)
    model = make_model()
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        callbacks=[lr_scheduler])
    plt.figure()
    plt.plot(history.epoch, history.history["lr"], "o-")
    #plt.axis([0, n_epochs - 1, 0, 0.011])
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title(name, fontsize=14)
    plt.grid(True)
    
    if name == 'perf':
        ax2 = plt.gca().twinx()
        ax2.plot(history.epoch, history.history["val_loss"], "r^-")
        ax2.set_ylabel('Validation Loss', color='r')
        ax2.tick_params('y', colors='r')
        
    fname = 'lrschedule-{}.pdf'.format(name)
    save_fig(fname)
    plt.show()


