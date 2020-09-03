import os
import sys
import numpy as np
import pandas as pd
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm 

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

'''
The following dummy code for demonstration.
'''

def preprocess_input(trainfile):

    data = pd.read_csv(trainfile)

    uniqueValues = data['emotion'].unique()

    emotion_arr = []
    image_list = []

    rows = len(data.index)

    for j in tqdm (range (rows), desc="Loading..."): 
        image_arr = []
        for i in range(48*48):
            pixels = data['pixel_'+str(i)]
            pixel = pixels[j]
            image_arr.append(pixel)

        image_list.append(np.array(image_arr).reshape((48,48)))

        emotion = data['emotion'][j]
        
        if emotion == uniqueValues[0]:
            index = 0
        elif emotion == uniqueValues[1]:
            index = 1
        elif emotion == uniqueValues[2]:
            index = 2
        else:
            print("Error Occured")
            break
        
        emotion_arr.append(index)

    return np.array(image_list), np.array(emotion_arr), uniqueValues



def train_a_model(trainfile):
    '''
    :param trainfile:
    :return:
    '''

    image, label, class_names = preprocess_input(trainfile)

    train_images, test_images, train_labels, test_labels = train_test_split(image, label, test_size=0.2, random_state=42)

    print(np.array(train_images).shape,np.array(train_labels).shape)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(48, 48)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=300)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    return model


def test_the_model(testfile):
    '''

    :param testfile:
    :return:  a list of predicted values in same order of
    '''

    return "Not Implemented Yet"
