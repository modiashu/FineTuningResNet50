'''
# File: Test.py
# Description: 
#			 This is the testing script used to check performance of learned weights for resnet model
#            Transfer learning has been used to fine tune ResNet50 model pretuned on Imagenet.
#
# Date: 3 March 2018
# Author: Ashutosh Modi
'''

##### Import required packages and libraries #####
import numpy as np
np.random.seed(2016)
import scipy
import os
import glob
import math
import pickle
import datetime

from keras.applications import vgg16 as keras_vgg16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense ,Input, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.layers.core import  ActivityRegularization

import json, sys

import keras
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.preprocessing import image

import matplotlib.pyplot as plt


##### Initialize environment #####

w_img, h_img = 224, 224

testing_path = 'test'

classes = 7;
BATCH_SIZE = 16;
EPOCHS_NB = 2;

input_shape = (224,224,3)

num_test_samples = sum([len(files) for r, d, files in os.walk(testing_path)])

print(num_test_samples)


##### Data preprocessing #####


test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(testing_path, target_size=(w_img,h_img), batch_size=4)


##### Load pre trained model from keras #####

inputLayer = Input(shape=(224, 224,3),
              name='image_input')
base_modelResNet = keras.applications.resnet50.ResNet50(include_top=False,  weights='imagenet')

x = base_modelResNet(inputLayer)

base_modelResNet.summary()


##### Modify model #####


x = Conv2D(512, (2, 2), name='convNew', padding='same')(x)
x = BatchNormalization(axis=-1, name='banNew')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', name='fc-2')(x)
x = ActivityRegularization(l2=0.01, l1=0.01)(x)
x = Dropout(0.2)(x)
output = Dense(classes, activation='softmax', name='outputLayer')(x)

customModel = Model(inputs=inputLayer, outputs=output)

customModel.summary()


##### Learn only new layers #####


#for layer in customModel.layers[:-7]:
#        layer.trainable=False

for layer in base_modelResNet.layers:
    layer.trainable = False
        
#customModel.layers[-1].trainable

##### Load weights learned from transfer learning #####

customModel.load_weights("weights-improvement-01-0.26.hdf5")

##### Set optimizer and parameters #####
rmsProp =RMSprop( lr=0.0001)
#adm = Adam(lr=0.0001)

customModel.compile(optimizer=rmsProp, loss='categorical_crossentropy', metrics=['accuracy'])
#customModel.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])


##### Evaluate performance by checking prediction accuracy on test data #####

score = customModel.evaluate_generator(test_batches, num_test_samples)
print("Accuracy = ", score[1])

