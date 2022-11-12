# A Simple Neural Network Machine Learning Project
# language: Python
# Framework: Tensorflow 
# Timestamp: 3:32pm 11/12/2022 
# Author: AkashP (@quibdev)

## imports
import cv2 
import os 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## loading training dataset
mnist = tf.keras.datasets.mnists
(x_train, y_train), (x_test, y_test) = mnist.load_data()


## preprocessing
# normalize the pixel data
# makes it easier for nueral network to work on the data
x_train = tf.keras.utils.normalize(x_train, axis=1)


## model
model = tf.keras.models.Sequential()

# flatten layer turns a 28*28 grid to a strain line 28*28 long
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# dense layer - where each neuron of other layer is connected to each other
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))

# each 10 neuron have a value btw 0-10 of how likely the image
# is that digit [confidence]
model.add(tf.keras.layers.Dense(10,activation='softmax'))

# optimizer 
model.compile(optimizer='adam', lose='sparse_categorical_crossentropy',metrics=['accuracy'])

## training
model.fit(x_train, y_train, epoch=3)
# epoch is iteration

 

model.save('handwritten.model')



