# Import Libraries
import numpy as np # linear algebra
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Path specification
import os
import cv2
print(os.listdir("D:/ML_Projects/Melanoma Classification/skin-lesions/"))

# Reading the Trainig data
def load_data():
    img_dir = "D:/ML_Projects/Melanoma Classification/skin-lesions/train/"
    y_map = {"Melanoma":0, "Nevus": 1, "seborrheic_keratosis": 2}
    
    x_data = []
    y_data = []
    
    for dir_n in y_map:
        label = int(y_map[dir_n])
        for ip in tqdm(os.listdir(os.path.join(img_dir, dir_n))):
            try:
                x_data.append(cv2.resize(cv2.imread(os.path.join(img_dir, dir_n, ip)), (64, 64)))
                y_data.append(label)
            except:
                continue
                
    return np.array(x_data), np.array(y_data)

# Load the training data
x_data, y_data = load_data()
y_data_cat = tf.keras.utils.to_categorical(y_data, num_classes=3)

# Reading the Validation data
def load_data():
    img_dir = "D:/ML_Projects/Melanoma Classification/skin-lesions/valid/"
    y_map = {"Melanoma":0, "Nevus": 1, "seborrheic_keratosis": 2}
    
    x_data = []
    y_data = []
    
    for dir_n in y_map:
        label = int(y_map[dir_n])
        for ip in tqdm(os.listdir(os.path.join(img_dir, dir_n))):
            try:
                x_data.append(cv2.resize(cv2.imread(os.path.join(img_dir, dir_n, ip)), (64, 64)))
                y_data.append(label)
            except:
                continue
                
    return np.array(x_data), np.array(y_data)

# Load the validation data
x_data_v, y_data_v = load_data()
y_data_cat_v = tf.keras.utils.to_categorical(y_data_v, num_classes=3)

# Reading the testing data
def load_data():
    img_dir = "D:/ML_Projects/Melanoma Classification/skin-lesions/test/"
    y_map = {"Melanoma":0, "Nevus": 1, "seborrheic_keratosis": 2}
    
    x_data = []
    y_data = []
    
    for dir_n in y_map:
        label = int(y_map[dir_n])
        for ip in tqdm(os.listdir(os.path.join(img_dir, dir_n))):
            try:
                x_data.append(cv2.resize(cv2.imread(os.path.join(img_dir, dir_n, ip)), (64, 64)))
                y_data.append(label)
            except:
                continue
                
    return np.array(x_data), np.array(y_data)

# Load the testing data
x_data_t, y_data_t = load_data()
y_data_cat_t = tf.keras.utils.to_categorical(y_data_t, num_classes=3)

# Shuffle the dataset
from sklearn.utils import shuffle
x_data, y_data_cat = shuffle(x_data, y_data_cat)
x_data_v, y_data_cat_v = shuffle(x_data_v, y_data_cat_v)
x_data_t, y_data_cat_t = shuffle(x_data_t, y_data_cat_t)

"""#Splitting the train dataset into training and validation 
from sklearn.model_selection import train_test_split
x_data, y_data_cat, x_valid , y_valid = train_test_split(
    x_data, y_data_cat, test_size = 0.25 , random_state = 0)"""


# Import libraries for CNN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# CNN Model
###2 conv and pool layers. with some normalization and drops in between.

INPUT_SHAPE = (64, 64, 3)   #change to (SIZE, SIZE, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(64, kernel_size=(3, 3), 
                               activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(rate=0.1)(norm1)
conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(rate=0.1)(norm2)

flat = keras.layers.Flatten()(drop2)  #Flatten the matrix to get it ready for dense.

hidden1 = keras.layers.Dense(256, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.1)(norm3)

out = keras.layers.Dense(3, activation='sigmoid')(drop3)   #units=1 gives error

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='categorical_crossentropy',   #Check between binary_crossentropy and categorical_crossentropy
                metrics=['accuracy'])
print(model.summary())
    


# Fit the model
history = model.fit(
    x_data, y_data_cat, batch_size=64, epochs=20, validation_split=0.3)

# Visualize rise in accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Visualise drop in loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# ## Accuracy calculation
# 
# I'll now calculate the accuracy on the test data.
prediction = model.predict(x_data_t)
