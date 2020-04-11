# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:41:05 2020

@author: arunc
"""

# Imports
import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, pooling, Dropout
import matplotlib.pyplot as plt
# Get path to the current working directory.
cwd_path = os.getcwd()
folder_path = cwd_path + "/data/IMG" + "/"

## Function for converting BGR to RGB image.
def preprocess(img):
    return np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Read image paths from driving_log.csv file
with open("data/driving_log.csv") as csvfile:
    images = []
    measurements = []
    reader = csv.reader(csvfile)
    for line in reader:        
        ## Get image name from first column
        img_name_center = line[0].split('/')[-1]
        img_name_left = line[1].split('/')[-1]
        img_name_right = line[2].split('/')[-1] 
        ## Get the image paths from csv files
        img_path_1 = folder_path + img_name_center
        img_path_2 = folder_path + img_name_left
        img_path_3 = folder_path + img_name_right
       
        ## Read images 
        image_1 = cv2.imread(img_path_1)        
        image_2 = cv2.imread(img_path_2)        
        image_3 = cv2.imread(img_path_3)           
        
        ## Convert images from BGR to RGB color space
        image_center = preprocess(image_1)        
        image_left = preprocess(image_2)        
        image_right = preprocess(image_3) 
        
        
        image_center_flipped = np.fliplr(image_center)
        image_left_flipped = np.fliplr(image_left)
        image_right_flipped = np.fliplr(image_right)
        
        ## Get streering angle from csv file
        measurement_center = float(line[3])
        measurement_left = measurement_center + 0.2
        measurement_right = measurement_center - 0.2
        
        m_center_flipped = -measurement_center        
        m_left_flipped = -measurement_left        
        m_right_flipped = -measurement_right
        
        ## Extend image and measurements        
        images.extend([image_center, image_center_flipped, 
                       image_left, image_left_flipped,
                       image_right, image_right_flipped]) 
        
        measurements.extend([measurement_center, m_center_flipped, 
                             measurement_left, m_left_flipped, 
                             measurement_right, m_right_flipped])

# Convert to arrays        
X_train = np.array(images)
y_train = np.array(measurements)
print(len(X_train))
print(len(y_train))

from keras.models import Sequential
from keras.layers import MaxPooling2D, Dense, Flatten, Convolution2D, Lambda

model = Sequential()
## Normlization using lambda helps parallelization 
model.add(Lambda(lambda x: (x/255) - 0.5))
## set up cropping2D layer
model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

## Compile and Fit
model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 4)

## Save the model
model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()