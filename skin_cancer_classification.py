# Import libraries for pre-processing
import numpy as np
np.random.seed(1000)
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras


# Set the path & size of images
image_directory = 'D:/ML_Projects/Melanoma Classification/skin-lesions/train/'
SIZE = 128
dataset_train = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label_train = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

""" We shall load the images from melanoma, nevus and keratosis path 
    into a single variable called dataset. The label of each shall also be loaded into label variabe.
    0 - Melanoma
    1 - Nevus
    2 - Keratosis """
    
Mel_images = os.listdir(image_directory + 'melanoma/')

for i, image_name in enumerate(Mel_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'melanoma/' + image_name)
        image = Image.fromarray(image, '1')
        image = image.resize((SIZE, SIZE))
        dataset_train.append(np.array(image))
        label_train.append(0)
        
Nev_images = os.listdir(image_directory + 'nevus/')

for i, image_name in enumerate(Nev_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'nevus/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset_train.append(np.array(image))
        label_train.append(1)
        
Ker_images = os.listdir(image_directory + 'seborrheic_keratosis/')

for i, image_name in enumerate(Ker_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'seborrheic_keratosis/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset_train.append(np.array(image))
        label_train.append(2)


""" Now we shall import the validation data in a similar manner """
image_directory_valid = 'D:/ML_Projects/Melanoma Classification/skin-lesions/valid/'
SIZE = 128
dataset_valid = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label_valid = []

Mel_images_valid = os.listdir(image_directory_valid + 'melanoma/')

for i, image_name in enumerate(Mel_images_valid):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory_valid + 'melanoma/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset_valid.append(np.array(image))
        label_valid.append(0)
        
Nev_images_valid = os.listdir(image_directory_valid + 'nevus/')

for i, image_name in enumerate(Nev_images_valid):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory_valid + 'nevus/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset_valid.append(np.array(image))
        label_valid.append(1)
        
Ker_images_valid = os.listdir(image_directory_valid + 'seborrheic_keratosis/')

for i, image_name in enumerate(Ker_images_valid):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory_valid + 'seborrheic_keratosis/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset_valid.append(np.array(image))
        label_valid.append(2)
        
""" TRAIN = 
    dataset_train - X_train
    label_train - y_train 
    
    VALIDATION = 
    dataset_valid - X_test
    label_valid - y_test """
    
# Lets import test dataset also
image_directory_test = 'D:/ML_Projects/Melanoma Classification/skin-lesions/test/'
SIZE = 128
dataset_test = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label_test = []

Mel_images_test = os.listdir(image_directory_test + 'melanoma/')

for i, image_name in enumerate(Mel_images_test):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory_test + 'melanoma/' + image_name)
        image = Image.fromarray(image, 'BW')
        image = image.resize((SIZE, SIZE))
        dataset_test.append(np.array(image))
        label_test.append(0)
        
Nev_images_test = os.listdir(image_directory_test + 'nevus/')

for i, image_name in enumerate(Nev_images_test):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory_test + 'nevus/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset_test.append(np.array(image))
        label_test.append(1)
        
Ker_images_test = os.listdir(image_directory_test + 'seborrheic_keratosis/')

for i, image_name in enumerate(Ker_images_test):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory_test + 'seborrheic_keratosis/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset_test.append(np.array(image))
        label_test.append(2)
      
# Shuffle the dataset
from sklearn.utils import shuffle
dataset_train, label_train = shuffle(dataset_train, label_train)
dataset_valid, label_valid = shuffle(dataset_valid, label_valid)
dataset_test, label_test = shuffle(dataset_test, label_test) 

       
# Noe lets add the train and valid dataset
dataset_train = dataset_train + dataset_valid
label_train = label_train + label_valid


# Import libraries for CNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
 

#Apply CNN
# ### Build the model

#############################################################
###2 conv and pool layers. with some normalization and drops in between.

INPUT_SHAPE = (SIZE, SIZE, 1)   
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
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
hidden2 = keras.layers.Dense(128, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.1)(norm4)

out = keras.layers.Dense(4, activation='sigmoid')(drop4)   #units=1 gives error

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='categorical_crossentropy',   #Check between binary_crossentropy and categorical_crossentropy
                metrics=['accuracy'])
print(model.summary())


# One hot encoding our categorical variables
from keras.utils import to_categorical
label_train = to_categorical(label_train)
label_valid = to_categorical(label_valid)

history = model.fit(np.array(dataset_train), label_train, epochs=15, validation_split=0.1)


### Accuracy calculation
# 
# I'll now calculate the accuracy on the test data.
label_test = to_categorical(label_test)
print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(dataset_test), label_test)[1]*100))


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")