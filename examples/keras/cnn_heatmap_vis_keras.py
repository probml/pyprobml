# -*- coding: utf-8 -*-
# Keras  book sec 5.4.3 Visualizing heatmaps of class activation
# Implements this paper
# R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra,
# “Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization,” 
# arXiv [cs.CV], 07-Oct-2016 [Online]. Available: http://arxiv.org/abs/1610.02391
# https://github.com/ramprs/grad-cam (torch/lua code)

#import cv2
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from skimage import transform, filters

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)

img_path = 'figures/cat_dog.jpg' # From https://github.com/ramprs/grad-cam/blob/master/images/cat_dog.jpg   
img_pil = image.load_img(img_path, target_size=(224, 224))

def normalize(a):
    # Normalize between 0 and 1
    return (a - a.min())/(a.max() - a.min())
    
def get_heatmap(model, im_batch, ndx):
    output = model.output[:, ndx]
    
    last_conv_layer = model.get_layer('block5_conv3') # VGG16, 14x14x512
    #last_conv_layer = model.get_layer('activation_196') # Resnet, 7x7x2048
    nfeatures = 512 # D=nfeatures. 2048 for Resnet50, 512 for VGG16
    
    # Eqn 1 of GradCAM paper
    # grads(i,j,k) = d y^c / dA(i,j,k)
    # where c is max class, i and j are pixels, k is feature map.
    # Note that this is a deferred evaluation, not an instantiated tensor.
    grads = K.gradients(output, last_conv_layer.output)[0]
    # pooled_grads(k) = mean_{ij} grads(i,j,k) = alpha_k^c
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
    # given our sample image:
    test_mode = 1
    pooled_grads_value, conv_layer_output_value = get_grads_and_vals([im_batch, test_mode])

    # Eqn 2 of GradCAM paper
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the max class
    # conv_layer_output[i,j,k] *= pooled_grads[k], 14x14x512
    for k in range(nfeatures):
        conv_layer_output_value[:, :, k] *= pooled_grads_value[k]
    # heatmap[i,j] = relu(sum_k conv_layer_output[i,j,k])
    heatmap = np.sum(conv_layer_output_value, axis=-1) #14x14
    heatmap = np.maximum(heatmap, 0.0) # Relu
    heatmap = normalize(heatmap)
    return heatmap


def make_att_map(img_np, attMap, blur = True, overlap = True):
    # From Ramprasaath Ramasamy Selvaraju 
    # based on https://github.com/jimmie33/Caffe-ExcitationBP/blob/master/excitationBP/util.py#L24
    # Rescale image to 0..1
    img = normalize(img_np)
    attMap = normalize(attMap)
    #attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
    attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

def show_att_map(img, attMap):
    att_map = make_att_map(img, attMap)
    plt.figure()
    plt.imshow(att_map)
    
def show_att_map_scipy(img_np, heatmap):
    # https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image 
    # Rescale image to 0..1
    x = np.maximum(img_np, 0)
    x /= np.max(x)
    # Resize heatmap  
    heatmap_big = scipy.misc.imresize(heatmap, x.shape[0:2])
    plt.figure()
    plt.imshow(x, origin='upper')
    plt.imshow(heatmap_big, alpha=.5, origin='upper')
    #plt.axis('off')

def show_att_map_opencv(img_path, heatmap, fname):
    # Code from Keras book needs OpenCV
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
    cv2.imwrite(fname, superimposed_img)
    

img_np = image.img_to_array(img_pil)
im_batch = np.expand_dims(img_np, axis=0)
im_batch = preprocess_input(im_batch)
im_batch.shape # 1, 224, 224, 3

preds = model.predict(im_batch)
decoded = decode_predictions(preds, top=10)[0]
print('Predicted:', decoded)

'''
Predicted: [
('n02108422', 'bull_mastiff', 0.40943894), 
('n02108089', 'boxer', 0.3950904), 
('n02109047', 'Great_Dane', 0.039510112), 
('n02109525', 'Saint_Bernard', 0.031701218), '
('n02129604', 'tiger', 0.019169593), 
('n02093754', 'Border_terrier', 0.018684039),
('n02110958', 'pug', 0.014893572), 
('n02123159', 'tiger_cat', 0.014403002),
('n02105162', 'malinois', 0.010533252), 
('n03803284', 'muzzle', 0.005662783)]
'''

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

for i, name in enumerate(class_names):
    ndx = imagenet_ndx[i]
    heatmap = get_heatmap(model, im_batch, ndx)
    plt.figure()
    plt.imshow(heatmap)
    plt.title(name)
    plt.show()
    fname = 'cnn-heatmap-{}.png'.format(name)
    plt.savefig(os.path.join('figures',fname))
    
    plt.figure()
    #show_att_map_scipy(img_np, heatmap)
    show_att_map(img_np, heatmap)
    plt.title(name)
    plt.show()
    fname = 'cnn-heatmap-overlayed-{}.png'.format(name)
    plt.savefig(os.path.join('figures',fname))





