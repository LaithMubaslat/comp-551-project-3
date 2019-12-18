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
#in colab
#from tensorflow import keras 
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



#####################model

#Ensemble CNN
#CNN with data augmentation Code 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers.normalization import BatchNormalization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler


#We import classes and function then load and prepare the data the same as in the previous CNN example.
# Larger CNN for the MNIST Dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data again



# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64).astype('float32')

# normalize inputs from 0-255 to 0-1


X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# BUILD CONVOLUTIONAL NEURAL NETWORKS
nets = 15
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (1, 64, 64)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(MaxPool2D(pool_size=(2,2)))
    model[j].add(Dropout(0.4))    
    
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets
epochs = 10

####################################################
#Running model
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                        height_shift_range=0.08, zoom_range=0.08)
test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, y_train, batch_size=64)
test_generator = test_gen.flow(X_test, y_test, batch_size=64)
for j in range(nets):
    history[j] = model[j].fit_generator(train_generator, steps_per_epoch=60000//64, epochs=epochs, 
                    validation_data=test_generator, validation_steps=10000//64)    
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
###################################################    
#for j in range(nets):
 #   X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, y_train, test_size = 0.1)
  #  history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
   #     epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
    #    validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    #print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
     #   j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
    
    


