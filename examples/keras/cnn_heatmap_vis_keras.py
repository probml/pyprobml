# -*- coding: utf-8 -*-
# Keras  book sec 5.4.3 Visualizing heatmaps of class activation
# Implements this paper
# R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra,
# “Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization,” 
# arXiv [cs.CV], 07-Oct-2016 [Online]. Available: http://arxiv.org/abs/1610.02391

import os
#import cv2
import scipy
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import urllib

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
model = VGG16(weights='imagenet')

#from keras.applications import ResNet50
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#model = ResNet50(weights='imagenet')


img_path = 'figures/Elephant_mating_ritual_3.jpg'
#img_path = 'figures/dog-cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))

#url = "https://en.wikipedia.org/wiki/African_elephant#/media/File:Elephant_mating_ritual_3.jpg"
#urllib.request.urlretrieve(url, "/tmp/img.png") 
#img = plt.imread('/tmp/img.png')

x = image.img_to_array(img)
im_batch = np.expand_dims(x, axis=0)
im_batch = preprocess_input(im_batch)
im_batch.shape # 1, 224, 224, 3


preds = model.predict(im_batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
ndx = np.argmax(preds[0]) 

output = model.output[:, ndx]

last_conv_layer = model.get_layer('block5_conv3') # VGG16, 14x14x512
#last_conv_layer = model.get_layer('activation_196') # Resnet, 7x7x2048
nfeatures = 512 # D=nfeatures. 2048 for Resnet50, 512 for VGG16

# grads(i,j,k) = d y^c / dA(i,j,k)
# where c is max class, i and j are pixels, k is feature map.
# Note that this is a deferred evaluation, not an instantiated tensor.
grads = K.gradients(output, last_conv_layer.output)[0]

# pooled_grads(k) = mean_{ij} grads(i,j,k)
# This is alpha_k^c (eqn 1) in the GradCAM paper.
pooled_grads = K.mean(grads, axis=(0, 1, 2))


# This function extracts the relevant tensors from the model
# by evaluating the above "symbolic" expressions.
# In the Keras book, this function is called "iterate".  

# To handle resnet, which has  different behavior in training and testing phase 
# (since it uses Dropout and BatchNormalization only in training), 
# we need to pass the 'learning phase' flag to your function
# https://medium.com/towards-data-science/https-medium-com-manishchablani-useful-keras-features-4bac0724734c
get_grads_and_vals = K.function([model.input, K.learning_phase()],
                               [pooled_grads, last_conv_layer.output[0]])
                               
# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
test_mode = 1
pooled_grads_value, conv_layer_output_value = get_grads_and_vals([im_batch, test_mode])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the max class
# conv_layer_output[i,j,k] *= pooled_grads[k], 14x14x512
# Eqn 2 of GradCAM paper
for k in range(nfeatures):
    #conv_layer_output_value[:, :, k] *= pooled_grads_value[k]
    conv_layer_output_value[:, :, k] *= np.maximum(0.0, pooled_grads_value[k])
    
    
# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
# heatmap[i,j] = mean_k conv_layer_output[i,j,k]
heatmap = np.mean(conv_layer_output_value, axis=-1) #14x14

# Rescale to 0..1
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

plt.figure()
plt.imshow(heatmap)
plt.show()
plt.savefig(os.path.join('figures','cnn-heatmap-elephant.png'))

# Plot heatmap on top of image
# https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image
   
if True:
    # Rescale image to 0..1
    x = image.img_to_array(img)
    x = np.maximum(x, 0)
    x /= np.max(x)
    
    plt.figure()
    plt.imshow(x, origin='upper')
    #plt.axis('off')
    plt.savefig(os.path.join('figures','cnn-elephant-original.png'))
    plt.show()
    
    # Resize heatmap
    heatmap_big = scipy.misc.imresize(heatmap, x.shape[0:2])
    #heatmap_big = np.ones((x.shape[0], x.shape[1])) # select all pixels
    #heatmap_big = np.uint8(255 * heatmap_big)
    
    plt.figure()
    plt.imshow(x, origin='upper')
    plt.imshow(heatmap_big, alpha=.5, origin='upper')
    #plt.axis('off')
    plt.savefig(os.path.join('figures','cnn-heatmap-overlayed-elephant.png'))
    plt.show()

if False:
    # We use cv2 to load the original image
    img = cv2.imread(img_path)
    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img
    # Save the image to disk
    cv2.imwrite('../figures/elephant_cam.jpg', superimposed_img)


