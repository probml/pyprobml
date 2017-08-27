'''Trains an MLP on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

Modified from https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
by Kevin Murphy.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

np.random.seed(123) # try to enforce reproduacability

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data() # downloads from AWS
# data is 60000 x 28 x 28

ntrain = x_train.shape[0] #60k
ntest = x_test.shape[0] # 10k
num_classes = len(np.unique(y_train)) # 10
ndims = x_train.shape[1] * x_train.shape[2] # 28*28=784

# Preprocess data
x_train = x_train.reshape(ntrain, ndims)
x_test = x_test.reshape(ntest, ndims)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

'''
# Specify architecture
#layer_sizes = [ndims, 512, 512, num_classes]
layer_sizes = [ndims, 512, num_classes]
#layer_sizes = [ndims, num_classes]

model = Sequential()
nlayers = len(layer_sizes)-1
for l, layer_size in enumerate(layer_sizes):
    if l > 0:
        if l == nlayers:
            activation = 'softmax'
        else:
            activation = 'relu'
        print('layer {:d}, size {:d}, input {:d}, act {:s}'.format(
                    l, layer_size, layer_sizes[l-1], activation))
        model.add(Dense(layer_size, activation=activation,
                    input_shape=(layer_sizes[l-1],)))
'''
  
# Two hidden layers
if False:
    model = Sequential()        
    model.add(Dense(512, activation='relu', input_shape=(ndims,)))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
# One hidden layer
if True:
    model = Sequential()        
    model.add(Dense(512, activation='relu', input_shape=(ndims,)))
    model.add(Dense(num_classes, activation='softmax'))

# Logistic regression model
if False:
    model = Sequential()        
    model.add(Dense(num_classes, input_shape=(ndims,), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
              

# Train
batch_size = 128
epochs =  5# 20
start = timer()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
end = timer()
print('Training took {:f} seconds'.format(end - start))

# Plot loss over time
if True:
    loss_trace = history.history['loss']
    acc_trace = history.history['acc']
    plt.figure()
    plt.plot(loss_trace, 'o-')
    plt.title('log loss on training set')
    plt.show()
    plt.figure()
    plt.plot(acc_trace, 'o-')
    plt.title('classif. accuracy on testset')
    plt.show()
    
# Test
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


    

'''   
Layer (type)                 Output Shape              Param #   
=================================================================
dense_229 (Dense)            (None, 512)               401920    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_230 (Dense)            (None, 512)               262656    
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_231 (Dense)            (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0

Sizes [512, 512]
Dropout [0.2, 0.2]
Train  for 5 epochs on a macbook CPU tooks 47 seconds:
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 9s - loss: 0.0753 - acc: 0.9768 - val_loss: 0.0826 - val_acc: 0.9774
Epoch 2/5
60000/60000 [==============================] - 8s - loss: 0.0613 - acc: 0.9818 - val_loss: 0.0797 - val_acc: 0.9785
Epoch 3/5
60000/60000 [==============================] - 9s - loss: 0.0502 - acc: 0.9857 - val_loss: 0.0820 - val_acc: 0.9805
Epoch 4/5
60000/60000 [==============================] - 9s - loss: 0.0428 - acc: 0.9879 - val_loss: 0.0728 - val_acc: 0.9827
Epoch 5/5
60000/60000 [==============================] - 9s - loss: 0.0360 - acc: 0.9892 - val_loss: 0.0778 - val_acc: 0.9808
Training took 47.130551 seconds
Test loss: 0.0778279642943
Test accuracy: 0.9808

Without dropout:
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 10s - loss: 0.2200 - acc: 0.9325 - val_loss: 0.0891 - val_acc: 0.9729
Epoch 2/5
60000/60000 [==============================] - 8s - loss: 0.0820 - acc: 0.9750 - val_loss: 0.0972 - val_acc: 0.9715
Epoch 3/5
60000/60000 [==============================] - 7s - loss: 0.0524 - acc: 0.9840 - val_loss: 0.1004 - val_acc: 0.9695
Epoch 4/5
60000/60000 [==============================] - 8s - loss: 0.0394 - acc: 0.9877 - val_loss: 0.0780 - val_acc: 0.9796
Epoch 5/5
60000/60000 [==============================] - 8s - loss: 0.0285 - acc: 0.9911 - val_loss: 0.0703 - val_acc: 0.9798
Training took 43.370110 seconds
Test loss: 0.0703338779202
Test accuracy: 0.9798


1 hidden layer
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_236 (Dense)            (None, 512)               401920    
_________________________________________________________________
dense_237 (Dense)            (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0

_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 7s - loss: 0.2589 - acc: 0.9252 - val_loss: 0.1232 - val_acc: 0.9646
Epoch 2/5
60000/60000 [==============================] - 6s - loss: 0.1043 - acc: 0.9692 - val_loss: 0.0980 - val_acc: 0.9710
Epoch 3/5
60000/60000 [==============================] - 5s - loss: 0.0685 - acc: 0.9793 - val_loss: 0.0796 - val_acc: 0.9756
Epoch 4/5
60000/60000 [==============================] - 6s - loss: 0.0499 - acc: 0.9846 - val_loss: 0.0648 - val_acc: 0.9798
Epoch 5/5
60000/60000 [==============================] - 6s - loss: 0.0368 - acc: 0.9888 - val_loss: 0.0654 - val_acc: 0.9798
Training took 32.637105 seconds
Test loss: 0.065425345597
Test accuracy: 0.9798


No hidden layers (logistic regression): 11 seconds
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_235 (Dense)            (None, 10)                7850      
=================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 3s - loss: 0.6030 - acc: 0.8468 - val_loss: 0.3419 - val_acc: 0.9095
Epoch 2/5
60000/60000 [==============================] - 1s - loss: 0.3306 - acc: 0.9085 - val_loss: 0.2998 - val_acc: 0.9160
Epoch 3/5
60000/60000 [==============================] - 1s - loss: 0.3013 - acc: 0.9157 - val_loss: 0.2878 - val_acc: 0.9198
Epoch 4/5
60000/60000 [==============================] - 1s - loss: 0.2883 - acc: 0.9201 - val_loss: 0.2776 - val_acc: 0.9232
Epoch 5/5
60000/60000 [==============================] - 1s - loss: 0.2801 - acc: 0.9222 - val_loss: 0.2740 - val_acc: 0.9236
Training took 11.059814 seconds
Test loss: 0.273956710047
Test accuracy: 0.9236

'''