#final


#part layer VGG 
from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import SGD,adadelta
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from sklearn.metrics import log_loss


#train_img=x_train.astype(float)
#test_img=x_val.astype(float)

import cv2 
import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle as pkl
import numpy
from __future__ import print_function
import argparse
import tensorflow as tf 
import torch 

from keras.layers import Dense
import keras


def largest_digit(binary_image):  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image,8,cv2.CV_32S)
    stats=stats[1:] #add
    largest_component = np.array([])
    largest_bounding_box = 0
    largest_width = 0
    largest_height = 0
    
    for i in range(num_labels-1): #excludes the background
            CC_STAT_WIDTH = stats[i][2]
            CC_STAT_HEIGHT = stats[i][3]
            #largest bounding rectangule produces error 
            #area=length*height
            bounding_box = max(CC_STAT_WIDTH,CC_STAT_HEIGHT) * max(CC_STAT_WIDTH,CC_STAT_HEIGHT)
            if(bounding_box > largest_bounding_box):
                largest_bounding_box = bounding_box
                
                #parameters for the largest bounding box
                CC_STAT_LEFT_Largest = stats[i][0]
                CC_STAT_TOP_Largest = stats[i][1]
                CC_STAT_WIDTH_Largest=stats[i][2]
                CC_STAT_HEIGHT_Largest=stats[i][3]

 
    out = np.zeros((64, 64),dtype=np.uint8)
    MNIST_digit = binary_image[CC_STAT_TOP_Largest:CC_STAT_TOP_Largest+CC_STAT_HEIGHT_Largest, CC_STAT_LEFT_Largest:CC_STAT_LEFT_Largest+CC_STAT_WIDTH_Largest]
    
    positioning=5
    #positioning could be manipulated to center the image, however imperically centered numbers take longer to train with no noticable advantage 
    out[positioning:positioning+CC_STAT_HEIGHT_Largest,positioning:positioning+CC_STAT_WIDTH_Largest] = MNIST_digit
    
    
    
    
    #uncommenting the code bellow produces a centered image
    #centering was found to severly cripple learning speed 
    
    #######
    #bounding box coordinates
    #coords = cv2.findNonZero(out)
    #x, y, w, h = cv2.boundingRect(coords)
    #out = out[y:y+h, x:x+w]
    #######
    #######
    #bounding box border
    #border=2 #margin
    #out = cv2.resize(out,(64-border*2,64-border*2))
    #######
    #######
    #removes a margin on the sides 
    #b = np.zeros((64-border*2,border))
    #result=np.append(result, b, axis=1)
    #result=np.append(b, result, axis=1)
    #b=np.zeros((border,64))
    #result=np.append(result, b, axis=0)
    #result=np.append(b, result, axis=0)
    #######
    return out


def largest_digit_set (train):
    threshold=250
    binary_map = (train > threshold).astype(np.uint8) #thresholding 
    for i in range (len(train)):
        img=binary_map[i]
        img=largest_digit(img)
        binary_map[i]=img
        print(i)
    return binary_map


def split(train_images,y_label):
    
    
  images=train_images.copy()
  y=y_label.copy()
  X_train=img[0:int(0.8*len(img))]
  y_train=y[0:int(0.8*len(img))]
  X_test=img[int(0.8*len(img)):]
  y_test=y[int(0.8*len(img)):]
  return X_train, y_train,X_test,y_test




#import data 
train_images = pd.read_pickle('input/train_images.pkl')
train_labels = pd.read_csv('input/train_labels.csv')
test_images = pd.read_pickle('input/test_images.pkl')
y=np.array(train_labels['Category'])
y=np.int64(y)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

img=train_images.copy()


largest= largest_digit_set(img)

X_train,y_train,X_test,y_test=split(largest,y)





























SIZE=64

dim = (SIZE, SIZE)
X_train=X_train.astype(float)/255

#Y_train=train_y
X_valid=X_test.astype(float)/255
#Y_valid=valid_y

#convert 28x28 grayscale to 48x48 rgb channels
def to_rgb(img):
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    #img_rgb=np.swapaxes(img_rgb, 1, 2)
    #img_rgb=np.swapaxes(img_rgb, 0, 1)
    return img_rgb

rgb_list = []
rgbval_list=[]
#convert X_train data to 48x48 rgb values
for i in range(len(X_train)):
    rgb = to_rgb(X_train[i])
    #rgbval=to_rgb(X_valid[i])
    rgb_list.append(rgb)
    #rgbval_list.append(rgbval)
    #print(rgb.shape)
    
for i in range(len(X_valid)):
    #rgb = to_rgb(X_train[i])
    rgbval=to_rgb(X_valid[i])
    #rgb_list.append(rgb)
    rgbval_list.append(rgbval)
    #print(rgb.shape)
    

    
rgb_arr = np.stack([rgb_list],axis=4)
rgb_arr_to_3d = np.squeeze(rgb_arr, axis=4)



rgb_arrval=np.stack([rgbval_list],axis=4)
rgb_arr_to_3dval = np.squeeze(rgb_arrval, axis=4)


#X_train=rgb_arr_to_3d
#X_valid=rgb_arr_to_3dval

print(rgb_arr_to_3d.shape)

























from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

def vgg16_model( num_classes=None):
    IMAGE_SIZE=[64,64]
    
    num_classes=10
    vgg_model = VGG16(input_shape = IMAGE_SIZE +[3],weights='imagenet', include_top=False,pooling='avg')
    
    
    #used for removing layers 
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block1_conv1'].output
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    
# Stacking a new simple convolutional network on top of it    
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    custom_model = Model(input=vgg_model.input, output=x)
    #for layer in custom_model.layers[:7]:
        #layer.trainable = False
    
    
    
    custom_model.compile(loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])
    return custom_model 


num_classes=10
model = vgg16_model( num_classes)



model.summary()


#####one hot encoding 

train_y=y_train
valid_y=y_test

le = LabelEncoder()

train_y = le.fit_transform(train_y)

train_y=to_categorical(train_y)

train_y=np.array(train_y)


le = LabelEncoder()

valid_y = le.fit_transform(valid_y)

valid_y=to_categorical(valid_y)

valid_y=np.array(valid_y)








img_rows, img_cols = 64, 64 # Resolution of inputs
channel = 1
num_classes = 10 
batch_size = 16 
nb_epoch = 10







model = vgg16_model(num_classes)



model.summary()




from keras.applications.vgg16 import preprocess_input

#X_train = np.expand_dims(X_train, axis=3)
#X_valid= np.expand_dims(X_valid, axis=3)


#X_train=preprocess_input(X_train)


model.fit(rgb_arr_to_3d, train_y,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(rgb_arr_to_3dval, valid_y))



# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

# Cross-entropy loss score
#score = log_loss(Y_valid, predictions_valid)



