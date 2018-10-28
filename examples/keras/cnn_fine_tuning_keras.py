# Use pretrained molde as feature extractor
# Keras book Listing 5.24 for VGG
# https://keras.io/applications/

from keras.applications import VGG16, ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


if True:
    # 553MB file, this can take several minutes to download!            
    vgg = VGG16(weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3))
    vgg.summary() # see vgg16-summary.txt for details

#Total params: 138,357,544
#Trainable params: 138,357,544
#Non-trainable params: 0    

if False:
    # 58MB file
    conv_base = VGG16(weights='imagenet',
                    include_top=False, # omit FC layers
                    input_shape=(150, 150, 3))
    conv_base.summary() # see vgg16-no-top-summary.txt for details
    
# ends at block5_pool (MaxPooling2D)
#Total params: 14,714,688
#Trainable params: 14,714,688
#Non-trainable params: 0

if False:
    resnet = ResNet50(weights='imagenet')
    resnet.summary() # see resnet50-summary.txt for details

#Total params: 25,636,712
#Trainable params: 25,583,592
#Non-trainable params: 53,120

model = vgg


img_path = 'figures/Elephant_mating_ritual_3.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
#Predicted: [('n02504458', 'African_elephant', 0.98539239),
#('n01871265', 'tusker', 0.0081480648), ('n02504013', 'Indian_elephant', 0.0064583868)]
