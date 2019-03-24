'''Trains a simple convnet on the MNIST dataset.

Modified from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
by Kevin Murphy.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import os

batch_size = 20 # 128
num_classes = 10
epochs = 2 # 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


if False:
    # Version from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    # This model has 1,199,882 params and takes 120 seconds per epoch
    # on my laptop CPU.
    # After 2 epochs
    #Test loss: 0.0666620718201
    #Test accuracy: 0.9809
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
if True:
    # Version from Keras book listing 5.1
    # This model has 93,322 params. This takes 60 seconds per epoch on
    # my laptop's CPU.
    # After 2 epochs:
    #Test loss: 0.0315938582575
    #Test accuracy: 0.9906
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    

model.summary()

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])


start = timer()
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
end = timer()
print('Training took {:f} seconds'.format(end - start))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot loss over time
if True:
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.figure()
    plt.plot(epochs, loss_values, 'bo-')
    plt.plot(epochs, val_loss_values, 'k+-')
    plt.legend({'train', 'test'})
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Cross entropy loss')
    plt.show()
    plt.savefig(os.path.join('figures','mnist-cnn-keras-loss.png'))


    plt.figure()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc_values, 'bo-')
    plt.plot(epochs, val_acc_values, 'k+-')
    plt.legend({'train', 'test'})
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Classification accuracy')
    plt.show()
    plt.savefig(os.path.join('figures','mnist-cnn-keras-acc.png'))



    