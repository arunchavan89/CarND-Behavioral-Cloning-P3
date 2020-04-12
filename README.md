
[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/model_cnn.png "model Image"
[image9]: ./examples/Figure_1.png "graph Image"

# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains files for the Behavioral Cloning Project.

In this project, deep neural networks and convolutional neural networks are implemented to clone driving behavior. The network model is  trained, validated and tested using Keras. The model outputs a steering angle to an autonomous vehicle.

A simulator [self-driving-car-sim](https://github.com/udacity/self-driving-car-sim.git) is provided where a car can be steered around a track for data collection. The collected image data and steering angles are used to train a neural network and then use this model to drive the car autonomously around the track. Download the simulator using the link: [Windows](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip) [Linux](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip) [Mac](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip)

The basic modules are as the following: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* video.py (to create a video)
* video_output/video.mp4 (a video recording of the vehicle driving autonomously around the track for at least one full lap)

### `model.py`
* Implement a neural network model in this file and run the following command. The model will be saved in `model.h5`
```sh
python model.py
```
### `drive.py`

* Run the saved model `model.h5`using the following ccommand.
```sh
python drive.py model.h5
```
* The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

* Saving a video of the autonomous agent using the following command.
```sh
python drive.py model.h5 run1
```
* The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

### `video.py`

* Creates a video based on images found in the `run1` directory using the following command.
```sh
python video.py run1
```
* Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```
---

# **Behavioral Cloning Project**

The goals of this project are the following:
* Use the [self-driving-car-simulator](https://github.com/udacity/self-driving-car-sim.git) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md file summarizing the results
* video_output/run1.mp4 the output-video 

#### 2. Submission includes functional code
* Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### 3. Submission code is usable and readable

* The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
* The model designed by NVIDIA corporation in the paper [End to End Learning for Self-Driving Cars](https://www.researchgate.net/publication/301648615_End_to_End_Learning_for_Self-Driving_Cars) is implemetned in this project. 

#### 2. Attempts to reduce overfitting in the model

* The model contains MaxPooling2D layers in order to reduce overfitting (model.py lines 81-96). 

* The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

* The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

#### 4. Appropriate training data

* Training data was chosen to keep the vehicle driving on the road.
* Recorded set 1 : One lap clockwise driving.
* Recorded set 2 : One lap anti-clockwise driving.
* Flipped images of set 1 and set 2

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

* The NVIDIA [model´s](https://www.researchgate.net/publication/301648615_End_to_End_Learning_for_Self-Driving_Cars) end-to-end approach
proved surprisingly powerful. The paper claims that "With minimum training data from humans the system learns to drive in traffic on local roads with or without lane markings and on highways. It also operates in areas with unclear visual guidance such as in parking lots and on unpaved roads".

* Therefore, the above model was starting point for this project. In order to gauge how well the model was working, the image and steering angle data are splitted into a training (80%) and validation set (20%). 

#### 2. Final Model Architecture

The following diagram shows the network layers.
![alt text][image8]

#### 3. Creation of the Training Set & Training Process

* As mentioned earlier almost 40k images and streering data is collected by doing the following:
* Recorded set 1 : One lap clockwise driving.
* Recorded set 2 : One lap anti-clockwise driving.
* Flipped images of set 1 and set 2
* The data set is randomly shuffled and put 20% of the data into a validation set. 
* The graph shown below descibes traning loss vs validation loss achieved.
![alt text][image9]

# **Output**
Here is the final result.!
![](video_output/behavioral_cloning.gif)
