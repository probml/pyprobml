# Example of applying VGG16 classifier
# Based on Keras  book sec 5.4.3 but uses a slightly different image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.preprocessing import image
import numpy as np

model = VGG16(weights='imagenet')
model.summary() # see vgg16-summary.txt for details

# Elephant image source https://en.wikipedia.org/wiki/African_elephant#/media/File:Elephant_mating_ritual_3.jpg
#By Charlesjsharp - Own work, CC BY 3.0, https://commons.wikimedia.org/w/index.php?curid=8916787
img_path = 'figures/Elephant_mating_ritual_3.jpg'


#https://en.wikipedia.org/wiki/African_elephant#/media/File:Serengeti_Elefantenherde2.jpg
#By Ikiwaner - Own work, GFDL 1.2, https://commons.wikimedia.org/w/index.php?curid=11232893
#img_path = 'figures/Serengeti_Elefantenherde2.jpg'

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
print('Predicted:', decode_predictions(preds, top=3)[0])

# For img_path = 'data/Elephant_mating_ritual_3.jpg'
#Predicted: [('n02504458', 'African_elephant', 0.93163019), 
#('n01871265', 'tusker', 0.053829707), ('n02504013', 'Indian_elephant', 0.014539864)]

# For img_path = 'data/Serengeti_Elefantenherde2.jpg'
#Predicted: [('n01871265', 'tusker', 0.61881274),
#('n02504458', 'African_elephant', 0.25420085), ('n02504013', 'Indian_elephant', 0.11940476)]

ndx = np.argmax(preds[0]) # 386 African Elephant

# Animal > Chordate > Vertebrate > Mammal > Tusker
#http://image-net.org/synset?wnid=n01871265
#Any mammal with prominent tusks (especially an elephant or wild boar)


# Animal > Chordate > Vertebrate > Mammal > Placental Mammal ...
# > Proboscidian > Elephant > African Elephant
# http://image-net.org/synset?wnid=n02504458
# An elephant native to Africa having enormous flapping ears and ivory tusks




