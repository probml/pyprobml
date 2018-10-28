# -*- coding: utf-8 -*-
# Visualize which parts of an image 'cause' the class label
# Uses the GradCAM algorithm
# R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra,
# “Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization,” 
# arXiv [cs.CV], 07-Oct-2016 [Online]. Available: http://arxiv.org/abs/1610.02391
# Uses the keras-vis package https://github.com/raghakot/keras-vis
# You first need to run 'pip install keras-vis'
# This script is based on https://github.com/raghakot/keras-vis/blob/master/examples/vggnet/attention.ipynb

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from vis.utils import utils
from vis.visualization import visualize_cam, overlay


# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)

img_path = 'figures/cat_dog.jpg' # From https://github.com/ramprs/grad-cam/blob/master/images/cat_dog.jpg
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
im_batch = np.expand_dims(x, axis=0)
im_batch = preprocess_input(im_batch)
im_batch.shape # 1, 224, 224, 3

preds = model.predict(im_batch) # (1, 1000)
topK = 20
decoded = decode_predictions(preds, top=topK)[0]
print('Predicted:', decoded)

topK_synsets = [triple[0] for triple in decoded]
topK_names = [triple[1] for triple in decoded]
topK_scores = [triple[2] for triple in decoded]

class_names = ['boxer', 'tiger_cat']
topK_ndx = []
imagenet_ndx = [] # indexes into the softmax entries of final layer
for i, name in enumerate(class_names):
    ndx = topK_names.index(name)
    topK_ndx.append(ndx)
    imagenet_ndx.append(np.argwhere(preds[0] == topK_scores[ndx])[0][0])
# 282 = Tiger cat, 242 = Boxer (0 indexed)

img = utils.load_img(img_path, target_size=(224, 224))
N = len(class_names)
fig, ax = plt.subplots(1, N+1)
ax[0].imshow(img)
ax[0].axis('off')

# Lets overlay the heatmap for each desired class onto original image.    
# Guided backprop implemented here
# https://github.com/raghakot/keras-vis/blob/master/vis/backend/tensorflow_backend.py#L23

for i in range(N):
    ndx = imagenet_ndx[i]
    layer_idx = utils.find_layer_idx(model, 'predictions') # final layer
    grads = visualize_cam(model, layer_idx, filter_indices=ndx, 
                        seed_input=img, backprop_modifier='guided') 
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    ax[i+1].imshow(overlay(jet_heatmap, img))
    ax[i+1].axis('off')
    ax[i+1].set_title(class_names[i])
plt.show()

plt.savefig(os.path.join('figures','grad-cam-keras.pdf'))
