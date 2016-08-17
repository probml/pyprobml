#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util


data = util.load_mat('mnistAll')
mnist = data['mnist']
train_images = mnist['train_images'][0][0]  # 28*28*60000
train_labels = mnist['train_labels'][0][0]  # 60000*1
test_images = mnist['test_images'][0][0]  # 28*28*10000
test_labels = mnist['test_labels'][0][0]  # 10000*1

fig1 = pl.figure(1)
fig2 = pl.figure(2)
np.random.seed(seed=10)
for i in range(10):
    img = test_images[:, :, i]
    ax1 = fig1.add_subplot(3, 3, i)
    ax1.imshow(img)
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title('true class = %s' % test_labels[i])

    img_shuffled = img.copy()
    # np.shuffle only along the first index, ravel it first
    np.random.shuffle(img_shuffled.ravel())
    img_shuffled = img_shuffled.reshape(img.shape)
    ax2 = fig2.add_subplot(3, 3, i)
    ax2.imshow(img_shuffled)
    ax2.set_xticks(())
    ax2.set_yticks(())
    ax2.set_title('true class = %s' % test_labels[i])
fig1_name = 'shuffledDigitsDemo_unshuffled.png'
fig2_name = 'shuffledDigitsDemo_shuffled.png'
fig1.savefig(fig1_name)
fig2.savefig(fig2_name)
pl.show()
