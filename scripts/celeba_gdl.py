# Illustrate latent space embedding and arithmetic for  VAE on CelebA faces images
# Code is based on 
# https://nbviewer.jupyter.org/github/davidADSP/GDL_code/blob/master/03_06_vae_faces_analysis.ipynb
# Full dataset can be downloaded from https://www.kaggle.com/jessicali9530/celeba-dataset

# In this code, we use a vacuous vae model that does nothing,
# just to check the plumbing. You should redefine vae_encode and
# vae_decode for a real model.

import numpy as np
import matplotlib.pyplot as plt

import os
figdir = "../figures"
def save_fig(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, fname))

#import tensorflow as tf
#from tensorflow import keras
#import tensorflow_datasets as tfds


import pandas as pd
#from scipy.stats import norm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array
#from keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array


vae = []
latent_dim = 50
INPUT_DIM = (128,128,3)

# Make some dummy functions
def vae_encode(model, x):
  #mean, logvar = model.inference_net(x)
  #return mean
  N = x.shape[0]
  return np.zeros((N, latent_dim))

def vae_decode(model, z_points):
  #return model.decode(z_points)
  N = z_points.shape[0]
  return np.zeros((N,128,128,3))

DATA_FOLDER = '/home/murphyk/Data/CelebA/'
IMAGE_FOLDER = '/home/murphyk/Data/CelebA/img_align_celeba/'



att = pd.read_csv(os.path.join(DATA_FOLDER, 'list_attr_celeba.csv'))
att.head()


class ImageLabelLoader():
    def __init__(self, image_folder, target_size):
        self.image_folder = image_folder
        self.target_size = target_size

    def build(self, att, batch_size, label = None):

        data_gen = ImageDataGenerator(rescale=1./255)
        if label:
            data_flow = data_gen.flow_from_dataframe(
                att
                , self.image_folder
                , x_col='image_id'
                , y_col=label
                , target_size=self.target_size 
                , class_mode='other'
                , batch_size=batch_size
                , shuffle=True
            )
        else:
            data_flow = data_gen.flow_from_dataframe(
                att
                , self.image_folder
                , x_col='image_id'
                , target_size=self.target_size 
                , class_mode='input'
                , batch_size=batch_size
                , shuffle=True
            )

        return data_flow


imageLoader = ImageLabelLoader(IMAGE_FOLDER, INPUT_DIM[:2])


#######
# Reconstructing images

n_to_show = 10

data_flow_generic = imageLoader.build(att, n_to_show)

example_batch = next(data_flow_generic)
example_images = example_batch[0]

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(n_to_show):
    img = example_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i+1)
    sub.axis('off')        
    sub.imshow(img)

z_points = vae_encode(vae, example_images)
reconst_images = vae_decode(vae, z_points)

for i in range(n_to_show):
    img = reconst_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    sub.axis('off')
    sub.imshow(img)
    
    
###############
# Latent space distribution
# We skip this since it requires predict_generator
    

###############
# Newly generated faces
    
n_to_show = 30

znew = np.random.normal(size = (n_to_show, latent_dim))

#reconst = vae.decoder.predict(np.array(znew))
reconst = vae_decode(vae, np.array(znew))

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(n_to_show):
    ax = fig.add_subplot(3, 10, i+1)
    ax.imshow(reconst[i, :,:,:])
    ax.axis('off')

plt.show()


def get_vector_from_label(label, batch_size):

    data_flow_label = imageLoader.build(att, batch_size, label = label)

    #origin = np.zeros(shape = latent_dim, dtype = 'float32')
    current_sum_POS = np.zeros(shape = latent_dim, dtype = 'float32')
    current_n_POS = 0
    current_mean_POS = np.zeros(shape = latent_dim, dtype = 'float32')

    current_sum_NEG = np.zeros(shape = latent_dim, dtype = 'float32')
    current_n_NEG = 0
    current_mean_NEG = np.zeros(shape = latent_dim, dtype = 'float32')

    current_vector = np.zeros(shape = latent_dim, dtype = 'float32')
    current_dist = 0

    print('label: ' + label)
    print('images : POS move : NEG move :distance : 𝛥 distance')
    while(current_n_POS < 10000):

        batch = next(data_flow_label)
        im = batch[0]
        attribute = batch[1]

        #z = vae.encoder.predict(np.array(im))
        z = vae_encode(vae, np.array(im))

        z_POS = z[attribute==1]
        z_NEG = z[attribute==-1]

        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis = 0)
            current_n_POS += len(z_POS)
            new_mean_POS = current_sum_POS / current_n_POS
            movement_POS = np.linalg.norm(new_mean_POS-current_mean_POS)

        if len(z_NEG) > 0: 
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis = 0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_NEG
            movement_NEG = np.linalg.norm(new_mean_NEG-current_mean_NEG)

        current_vector = new_mean_POS-new_mean_NEG
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist


        print(str(current_n_POS)
              + '    : ' + str(np.round(movement_POS,3))
              + '    : ' + str(np.round(movement_NEG,3))
              + '    : ' + str(np.round(new_dist,3))
              + '    : ' + str(np.round(dist_change,3))
             )

        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        if np.sum([movement_POS, movement_NEG]) < 0.08:
            current_vector = current_vector / current_dist
            print('Found the ' + label + ' vector')
            break

    return current_vector   


def add_vector_to_images(feature_vec):

    n_to_show = 5
    factors = [-4,-3,-2,-1,0,1,2,3,4]

    example_batch = next(data_flow_generic)
    example_images = example_batch[0]
    #example_labels = example_batch[1]

    #z_points = vae.encoder.predict(example_images)
    z_points = vae_encode(vae, example_images)

    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):

        img = example_images[i].squeeze()
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis('off')        
        sub.imshow(img)

        counter += 1

        for factor in factors:

            changed_z_point = z_points[i] + feature_vec * factor
            #changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]
            changed_image = vae_decode(vae, np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis('off')
            sub.imshow(img)

            counter += 1

    plt.show()

BATCH_SIZE = 500
attractive_vec = get_vector_from_label('Attractive', BATCH_SIZE)
mouth_open_vec = get_vector_from_label('Mouth_Slightly_Open', BATCH_SIZE)
smiling_vec = get_vector_from_label('Smiling', BATCH_SIZE)
lipstick_vec = get_vector_from_label('Wearing_Lipstick', BATCH_SIZE)
young_vec = get_vector_from_label('High_Cheekbones', BATCH_SIZE)
male_vec = get_vector_from_label('Male', BATCH_SIZE)
eyeglasses_vec = get_vector_from_label('Eyeglasses', BATCH_SIZE)
blonde_vec = get_vector_from_label('Blond_Hair', BATCH_SIZE)
    
print('Eyeglasses Vector')
add_vector_to_images(eyeglasses_vec)

##########
# Face morphs

def morph_faces(start_image_file, end_image_file):

    factors = np.arange(0,1,0.1)

    att_specific = att[att['image_id'].isin([start_image_file, end_image_file])]
    att_specific = att_specific.reset_index()
    data_flow_label = imageLoader.build(att_specific, 2)

    example_batch = next(data_flow_label)
    example_images = example_batch[0]
    #example_labels = example_batch[1]

    #z_points = vae.encoder.predict(example_images)
    z_points = vae_encode(vae, example_images)


    fig = plt.figure(figsize=(18, 8))

    counter = 1

    img = example_images[0].squeeze()
    sub = fig.add_subplot(1, len(factors)+2, counter)
    sub.axis('off')        
    sub.imshow(img)

    counter+=1


    for factor in factors:

        changed_z_point = z_points[0] * (1-factor) + z_points[1]  * factor
        #changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]
        changed_image = vae_decode(vae, np.array([changed_z_point]))[0]

        img = changed_image.squeeze()
        sub = fig.add_subplot(1, len(factors)+2, counter)
        sub.axis('off')
        sub.imshow(img)

        counter += 1

    img = example_images[1].squeeze()
    sub = fig.add_subplot(1, len(factors)+2, counter)
    sub.axis('off')        
    sub.imshow(img)


    plt.show()

start_image_file = '000238.jpg' 
end_image_file = '000193.jpg' #glasses
morph_faces(start_image_file, end_image_file)

start_image_file = '000112.jpg'
end_image_file = '000258.jpg'
morph_faces(start_image_file, end_image_file)

start_image_file = '000230.jpg'
end_image_file = '000712.jpg'
morph_faces(start_image_file, end_image_file)
