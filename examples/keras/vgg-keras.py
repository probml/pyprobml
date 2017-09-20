# Example of applying VGG16 classifier
# Based on Keras  book sec 5.4.3.

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
#import matplotlib.pyplot as plt
import numpy as np
#import urllib

#from keras.applications import ResNet50
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#model = ResNet50(weights='imagenet')


model = VGG16(weights='imagenet')
model.summary() # see vgg16-summary.txt for details

# Load image from file
img_path = 'figures/cat_dog.jpg' # From https://github.com/ramprs/grad-cam/blob/master/images/cat_dog.jpg 
#img_path = 'figures/Elephant_mating_ritual_3.jpg' #  https://en.wikipedia.org/wiki/African_elephant#/media/File:Elephant_mating_ritual_3.jpg
#img_path = 'figures/Serengeti_Elefantenherde2.jpg' #https://en.wikipedia.org/wiki/African_elephant#/media/File:Serengeti_Elefantenherde2.jpg
#img_path = 'figures/dog-cat-openimages.jpg'
#img_path = 'figures/dog-ball-openimages.jpg' # https://www.flickr.com/photos/akarmy/5423588107
#img_path = 'figures/dog-car-backpack-openimages.jpg' # https://www.flickr.com/photos/mountrainiernps/14485323038

# Retrieve image from web
#url = "https://en.wikipedia.org/wiki/African_elephant#/media/File:Elephant_mating_ritual_3.jpg"
#urllib.request.urlretrieve(url, "/tmp/img.png") 
#img = plt.imread('/tmp/img.png')

# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))
# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)
# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=10)[0])

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

# For img_path = 'data/Elephant_mating_ritual_3.jpg'
#Predicted: [('n02504458', 'African_elephant', 0.93163019), 
#('n01871265', 'tusker', 0.053829707), ('n02504013', 'Indian_elephant', 0.014539864)]

# For img_path = 'data/Serengeti_Elefantenherde2.jpg'
#Predicted: [('n01871265', 'tusker', 0.61881274),
#('n02504458', 'African_elephant', 0.25420085), ('n02504013', 'Indian_elephant', 0.11940476)]

# Animal > Chordate > Vertebrate > Mammal > Tusker
#http://image-net.org/synset?wnid=n01871265
#Any mammal with prominent tusks (especially an elephant or wild boar)

# Animal > Chordate > Vertebrate > Mammal > Placental Mammal ...
# > Proboscidian > Elephant > African Elephant
# http://image-net.org/synset?wnid=n02504458
# An elephant native to Africa having enormous flapping ears and ivory tusks




