
# Movie review classifier using keras. Based on 
# https://www.tensorflow.org/tutorials/keras/basic_text_classification


import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras


print(tf.__version__)
np.random.seed(0)

imdb = keras.datasets.imdb

vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
print(np.shape(train_data)) # (25000)
print(len(train_data[0]))
print(len(train_data[1]))
print(train_data[0])
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941...]

word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[0])
"""
<START> this film was just brilliant casting location scenery story direction everyone's really suited the part 
they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director <UNK> 
father came from the same scottish island as myself ...
"""

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

print(train_data[0])
"""
[   1   14   22   16   43  530  973 1622 1385   65  458 4468   66 3941
    4  173   36  256    5   25  100   43  838  112   50  670    2    9
   35  480  284    5  150    4  172  112  167    2  336  385   39    4
  172 4536 1111   17  546   38   13  447    4  192   50   16    6  147
 2025   19   14   22    4 1920 4613  469    4   22   71   87   12   16
   43  530   38   76   15   13 1247    4   22   17  515   17   12   16
  626   18    2    5   62  386   12    8  316    8  106    5    4 2223
 5244   16  480   66 3785   33    4  130   12   16   38  619    5   25
  124   51   36  135   48   25 1415   33    6   22   12  215   28   77
   52    5   14  407   16   82    2    8    4  107  117 5952   15  256
    4    2    7 3766    5  723   36   71   43  530  476   26  400  317
   46    7    4    2 1029   13  104   88    4  381   15  297   98   32
 2071   56   26  141    6  194 7486   18    4  226   22   21  134  476
   26  480    5  144   30 5535   18   51   36   28  224   92   25  104
    4  226   65   16   38 1334   88   12   16  283    5   16 4472  113
  103   32   15   16 5345   19  178   32    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0]
"""

embed_size = 16
def make_model(embed_size):
  model = keras.Sequential()
  model.add(keras.layers.Embedding(vocab_size, embed_size))
  model.add(keras.layers.GlobalAveragePooling1D())
  model.add(keras.layers.Dense(16, activation=tf.nn.relu))
  model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
  model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
  return model

model = make_model(embed_size)
model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 16)          160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 16)                272       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________
"""


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
print(history_dict.keys())


acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
fig, ax = plt.subplots()
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'r-', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
save_fig("imdb-loss.pdf")
plt.show()


fig, ax = plt.subplots()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
save_fig("imdb-acc.pdf")
plt.show()

# Now turn on early stopping
# https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
callbacks = [PrintDot(),
             keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
             keras.callbacks.ModelCheckpoint(filepath='imdb_keras_best_model.ckpt',
                                             monitor='val_loss', save_best_only=True)]

# Reset parameters to a new random state
model = make_model(embed_size)
history = model.fit(
    partial_x_train, partial_y_train, epochs=40, batch_size=512, 
    validation_data=(x_val, y_val), verbose=0, callbacks=callbacks)
print(history)

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
fig, ax = plt.subplots()
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'r-', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
save_fig("imdb-loss-early-stop.pdf")
plt.show()