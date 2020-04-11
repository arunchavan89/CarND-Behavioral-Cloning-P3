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
from keras.layers import MaxPooling2D, Dense, Flatten, Convolution2D, Lambda, Cropping2D

# Get path to the current working directory.
cwd_path = os.getcwd()
folder_path = cwd_path + "\data\IMG" + "\\"

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
        img_name_center = line[0].split('\\')[-1]
        img_name_left = line[1].split('\\')[-1]
        img_name_right = line[2].split('\\')[-1]   

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

model = Sequential()
## Normlization using lambda helps parallelization 
model.add(Lambda(lambda x: (x/255) - 0.5))

###
# Set up cropping2D layer.     
# 70 rows pixels from the top of the image
# 20 rows pixels from the bottom of the image
# 0 columns of pixels from the rignt and left of the image
###
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
# Layer 1: 24 x 5 x 5
model.add(Convolution2D(24, 5, 5, activation='relu'))
# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer 2: 36 x 5 x 5
model.add(Convolution2D(36, 5, 5, activation='relu'))
# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer 3: 48 x 5 x 5
model.add(Convolution2D(48, 5, 5, activation='relu'))
# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten
model.add(Flatten())
## Dense 100
model.add(Dense(100))
## Dense 50
model.add(Dense(50))
## Dense 10
model.add(Dense(10))
# Final output layer
model.add(Dense(1))

## Compile and Fit
model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 4)

## Save the model
model.save('model.h5')
