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



####random sample code 1/2
XX=[]
YY=[]

import random 
for j in range (15):
    idx = np.random.choice(np.arange(len(X_train)), len(X_train), replace=True)
    X_train_sample=X_train[idx]
    y_train_sample=y_train[idx]
    
    #X_train_sample=numpy.array(X_train_sample)
    X_train_sample = X_train_sample.reshape(X_train.shape[0], 1, 64, 64).astype('float32')
    X_train_sample=X_train_sample/ 257
    XX.append(X_train_sample)  
    y_train_sample = np_utils.to_categorical(y_train_sample)
    YY.append(y_train_sample)
####random sample code 1/2



# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64).astype('float32')

# normalize inputs from 0-255 to 0-1


X_train = X_train / 257
X_test = X_test / 257
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
#train_generator = gen.flow(X_train, y_train, batch_size=64)
test_generator = test_gen.flow(X_test, y_test, batch_size=64)
for j in range(nets):
  ####random sample code 2/2
    train_generator = gen.flow(XX[j], YY[j], batch_size=64)
    ####random sample code 2/2
    history[j] = model[j].fit_generator(train_generator, steps_per_epoch=60000//64, epochs=epochs, 
                    validation_data=test_generator, validation_steps=10000//64)    
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
    
    
    
    
    
    
    #save model after each iteration 
    #model=model[j]
    #name=j
    #name='RSmodel'+str(j)+'.h5'
    
    #model.save(name)
    #model_file = drive.CreateFile({'title' : name})
    #model_file.SetContentFile(name)
    #model_file.Upload()
###################################################    
#for j in range(nets):
 #   X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, y_train, test_size = 0.1)
  #  history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
   #     epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
    #    validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    #print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
     #   j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))