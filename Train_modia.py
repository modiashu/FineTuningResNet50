'''
# File: Train.py
# Description: 
#			 This is the training script used to learn weights for resnet model
# 		     Transfer learning has been used to fine tune ResNet50 model pre tuned on Imagenet.
# Hyper parameters to train: Epochs,
#							 Batch Size,
#							 Convolution Kernel size,
#							 Number of convolution Kernels,
# 							 Regularization for L1 and L2,
# 							 Optimizer learning rate,
#    						 New layers involved in transfer learning
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

training_path = 'train'
validation_path = 'validation'
testing_path = 'test'

classes = 7;
BATCH_SIZE = 16;
EPOCHS_NB = 2;

input_shape = (224,224,3)

num_train_samples = sum([len(files) for r, d, files in os.walk(training_path)])
num_valid_samples = sum([len(files) for r, d, files in os.walk(validation_path)])
num_test_samples = sum([len(files) for r, d, files in os.walk(testing_path)])

print(num_train_samples)
print(num_valid_samples)
print(num_test_samples)

num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

print(num_train_steps)
print(num_valid_steps)


##### Data augmentation and preprocessing #####


train_batches = ImageDataGenerator(rescale=1./255,
        rotation_range=10,
        width_shift_range=0.2, 
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.1, 
        horizontal_flip=True,
        vertical_flip=True).flow_from_directory(training_path, target_size=(w_img,h_img), batch_size=BATCH_SIZE)



"""
train_batches = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(training_path, target_size=(w_img,h_img), batch_size=BATCH_SIZE)
"""
valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path, target_size=(w_img,h_img), batch_size=BATCH_SIZE)
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(testing_path, target_size=(w_img,h_img), batch_size=4)


##### Load pre trained model from keras #####


inputLayer = Input(shape=(224, 224,3),
              name='image_input')
base_modelResNet = keras.applications.resnet50.ResNet50(include_top=False,  weights='imagenet')
#base_modelResNet.layers.pop()

### To avoid dangling pointer issue
#base_modelResNet.outputs = [base_modelResNet.layers[-1].output]
#base_modelResNet.layers[-1].outbound_nodes = []

x = base_modelResNet(inputLayer)


##### Check model summary #####


#x= Pop()(x)
base_modelResNet.summary()


##### Modify model #####


#lastLayer = x.output
x = Conv2D(512, (2, 2), name='convNew', padding='same')(x)
x = BatchNormalization(axis=-1, name='banNew')(x)
x = GlobalAveragePooling2D()(x)
#x= MaxPooling2D(pool_size=(2, 2))(x)
#x = Flatten(name='flat')(x)
#x = Dense(2048, activation='elu', name='fc-1')(x)
#x = BatchNormalization(axis=-1, name='banNew')(x) #gives low accuracy
#x = Dropout(0.5)(x)
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


##### Set optimizer and parameters #####


rmsProp =RMSprop( lr=0.0001)
#adm = Adam(lr=0.0001)

customModel.compile(optimizer=rmsProp, loss='categorical_crossentropy', metrics=['accuracy'])
#customModel.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])


##### Model Checkpoint to have weights giving best performance #####
fileName="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkPoint = ModelCheckpoint(fileName, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkPoint]


##### Train model for custom dataset #####


customModel.fit_generator(
        train_batches,
        steps_per_epoch=num_train_steps,
        epochs=EPOCHS_NB,
        validation_data=valid_batches,
        validation_steps=num_valid_steps,
        callbacks=callbacks_list)


##### Evaluate performance by checking prediction accuracy on test data #####

score = customModel.evaluate_generator(test_batches, num_test_samples)
print("Accuracy = ", score[1])

##### Save model and weights #####
customModel.save_weights('Weights_v1.h5',overwrite=True)
customModel.save('ResNetModel.h5')


##### References #####

#https://www.youtube.com/watch?v=m5RjXjvAAhQ
#https://www.youtube.com/watch?v=14syUbL16k4
#https://www.youtube.com/watch?v=kqj6ltmD3aA&index=46&list=PLBAGcD3siRDguyYYzhVwZ3tLvOyyG5k6K
#https://www.youtube.com/watch?v=OoUX-nOEjG0&index=2&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
#https://keras.io/
#https://github.com/deeplizard2/Keras_Jupyter_Notebooks/blob/master/CNN.ipynb
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#https://en.wikipedia.org/wiki/Keras
#http://image-net.org/
#https://github.com/keras-team/keras/issues/5179
#https://github.com/keras-team/keras/issues/2790
