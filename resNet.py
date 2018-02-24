import numpy as np
np.random.seed(2016)
import scipy
import os
import glob
import math
import pickle
import datetime
#import pandas as pd

from keras.applications import vgg16 as keras_vgg16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense ,Input, BatchNormalization
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
#########################################################################
w_img, h_img = 224, 224

training_path = 'train'
validation_path = 'validation'
testing_path = 'test'

classes = 7;
BATCH_SIZE = 16;
EPOCHS_NB = 500;

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

#########################################################################
train_batches = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(training_path, target_size=(w_img,h_img), batch_size=BATCH_SIZE)
valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path, target_size=(w_img,h_img), batch_size=BATCH_SIZE)
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(testing_path, target_size=(w_img,h_img), batch_size=4)

#########################################################################


# Create a Model
X_input = Input(input_shape)

#X = ZeroPadding2D((3, 3))(X_input)

modelResNet = keras.applications.resnet50.ResNet50(include_top=False,  weights='imagenet', )

#########################################################################

#Freeze layers in resnet and do not update weights while trainng model.
for layer in modelResNet.layers:
	layer.trainable=False

#########################################################################

X = modelResNet(X_input)


X = Conv2D(1024, (3, 3), name='convNew', padding='same')(X)
X = BatchNormalization(axis=1, name='bnNew')(X)
X = Activation('relu')(X)

# MAXPOOL
#X= MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
#X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxPoolNew')(X)

## FLATTEN X (means convert it to a vector) + FULLYCONNECTED
X = Flatten()(X)
output = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

#########################################################################
#Model to train 
newModelResNet = Model(input=X_input, output=output)

rmsProp =RMSprop( lr=0.0001)
#adm = Adam(lr=0.0001)

newModelResNet.compile(optimizer=rmsProp, loss='categorical_crossentropy', metrics=['accuracy'])

#########################################################################
# Model Checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkPoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkPoint]
# Data augmentation

#########################################################################

newModelResNet.fit_generator(
        train_batches,
        steps_per_epoch=num_train_steps,
        epochs=EPOCHS_NB,
        validation_data=valid_batches,
        validation_steps=num_valid_steps,
        callbacks=callbacks_list)
		
#########################################################################
score = newModelResNet.evaluate_generator(test_batches, num_test_samples)
print("Accuracy = ", score[1])

newModelResNet.save_weights('Weights_v1.h5')

