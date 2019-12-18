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
from tensorflow import keras
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
train_images=pd.read_pickle('../root/COMP 551-Project 3-Data/train_images.pkl')
test_images = pd.read_pickle('../root/COMP 551-Project 3-Data/test_images.pkl')
train_labels=pd.read_csv('../root/COMP 551-Project 3-Data/train_labels.csv')
y=np.array(train_labels['Category'])
y=np.int64(y)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

img=train_images.copy()


largest= largest_digit_set(img)

X_train,y_train,X_test,y_test=split(largest,y)




