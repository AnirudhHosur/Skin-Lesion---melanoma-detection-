# Using thano for a change lol

# Import Libraries
import numpy as np 
import pandas as pd 
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Define the Path
train_path = 'D:/ML_Projects/Melanoma Classification/skin-lesions/train'
valid_path = 'D:/ML_Projects/Melanoma Classification/skin-lesions/valid'
test_path = 'D:/ML_Projects/Melanoma Classification/skin-lesions/test'

# Create Batches
train_batches = ImageDataGenerator().flow_from_directory(
    train_path, target_size=(128,128), color_mode='rgb', 
    classes=['melanoma','nevus','seborrheic_keratosis'], batch_size=32)

valid_batches = ImageDataGenerator().flow_from_directory(
    valid_path, target_size=(128,128), color_mode='rgb', 
    classes=['melanoma','nevus','seborrheic_keratosis'], batch_size=16)

test_batches = ImageDataGenerator().flow_from_directory(
    test_path, target_size=(128,128), color_mode='rgb', 
    classes=['melanoma','nevus','seborrheic_keratosis'], batch_size=32)

# Lets just plot our train images
imgs, labels = next(train_batches)
plots(imgs, titles=labels)

# Build and Train CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3),
           Flatten(),
           Dense(3, activation='softmax'))])

# Compile
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit_generator(train_batches, steps_per_epoch=64, validation_data=valid_batches,
                    validation_steps=32, epochs=30, verbose=2)



























