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
import matplotlib.pyplot as plt 
import matplotlib.image as img 
from keras.models import Sequential
from keras.layers import Flatten, Dense
# Get path to the current working directory.
cwd_path = os.getcwd()

# Read image paths from driving_log.csv file
with open("data1/driving_log.csv") as csvfile:
    images = []
    measurements = []
    reader = csv.reader(csvfile)
    for line in reader:
        # Get image name from first column
        img_name = line[0].split('/')[-1]
        #print(img_name)
        img_path = cwd_path + "/data1/IMG" + "/" + img_name
        image = img.imread(img_path)
        images.append(image)
        # Get streering angle from csv file
        measurement = line[3]
        measurements.append(measurement)

# Convert to arrays        
X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Flatten(input_shape = (160, 320, 3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 7)

# Save the model
model.save('model.h5')
