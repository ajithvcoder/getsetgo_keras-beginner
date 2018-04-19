# -*- coding: utf-8 -*-
"""Copy of VGG19_extractinglayer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qDJt-jH8htt_98_q5IXXPb4Q7ui_2bOK
"""

##line1:VGG19 model's fc1 layer (last before layer) of the architecture is extracted and used for prediction
#http://image-net.org/about-stats
#https://arxiv.org/abs/1512.03385
#below code has been tested and executed only in colab.research.google.com .Please make sure you have enabled GPU from notebook settings before execution

#importing preprocess_input and decode_predictions from vgg19 model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input,decode_predictions
from keras.models import Model
import numpy as np

#For visualising picture
from matplotlib import pyplot as plt
# %matplotlib inline

#extracting 'fc1'(fully connected layer 1) layer from base_model and storing it in model
base_model=VGG19(weights='imagenet',include_top=True)
model=Model(inputs=base_model.input,outputs=base_model.get_layer('fc1').output)

#image is loaded from url
#you can use imread to load loacally 
from urllib.request import urlopen

#url of the image is stored in url_link1
urllink=urlopen("https://secure.img1-fg.wfcdn.com/im/60243122/resize-h800%5Ecompr-r85/4037/40372281/Corona+Extendable+Dining+Table.jpg")

#(224,224) is the target size of resnet50 model
img=image.load_img(urllink,target_size=(224,224))

#visuvalising input image
plt.imshow(img)

#preprocessing input image
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

from keras.models import Sequential

from keras.layers import Dense, Activation

#its not possible to give predictions from fc1 layer immediately only after passing it through softmax layer (assigning probabilities) we can use predict function to classify 
#in order to add layer you need to take a sequential initialized model and then add fc1 model along with it

#sequential model is initialised with model2
model2=Sequential()
#fc1 layer is added in a sequential manner to model2
model2.add(model)
#adding softmax layer for prediction
model2.add(Dense(1000,activation='softmax'))

fc1_layer=model2.predict(x)

#you can also take fc2 layer but make sure that its (1,1000) shape when decoding
fc1_layer.shape

#decode_predictions decodes the values of pred1 and provides the output
print("predict:",decode_predictions(fc1_layer))