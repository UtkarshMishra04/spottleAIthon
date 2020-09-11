import os
import sys
import numpy as np
import pandas as pd
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm 

# TensorFlow and keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras import optimizers

'''
The following dummy code for demonstration.
'''

def data(trainfile):

    test=pd.read_csv(trainfile)

    train_val_X=test.iloc[:,1:].values
    train_val_X=np.reshape(train_val_X, (len(train_val_X),48,48,1))
    train_val_y=test.iloc[:,0].values

    label_encoder = preprocessing.LabelEncoder()
    train_val_y=label_encoder.fit_transform(train_val_y)

    train_val_X = train_val_X / 255.0
    X_train, X_val, y_train, y_val =train_test_split(train_val_X, train_val_y, test_size=0.2,stratify=train_val_y, random_state=42)    #keep this fixed at 0.2
    
    return X_train, y_train, X_val, y_val

def build_net():

    net = models.Sequential(name='DCNN')
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            input_shape=(48, 48, 1),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
    net.add(BatchNormalization(name='batchnorm_2'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
    net.add(Dropout(0.4, name='dropout_1'))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
    net.add(BatchNormalization(name='batchnorm_4'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
    net.add(Dropout(0.3, name='dropout_2'))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
    net.add(BatchNormalization(name='batchnorm_6'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
    net.add(Dropout(0.6, name='dropout_3'))
    net.add(Flatten(name='flatten'))   
    net.add(
        Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_7')) 
    net.add(Dropout(0.5, name='dropout_4'))
    net.add(
        Dense(
            3,
            activation='softmax',
            name='out_layer'
        )
    )
    net.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    net.summary()
    return net

def train_a_model(trainfile):
    '''
    :param trainfile:
    :return:
    '''

    X_train, y_train, X_val, y_val = data(trainfile)
    model=build_net()

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True)   

    model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                  epochs=100,
                  verbose=2,
                  steps_per_epoch=len(X_train)/32,
                  validation_data=(X_val, y_val))

    score, acc = model.evaluate(X_val, y_val, verbose=0)

    print('Val accuracy:', acc)

    model.save('./trained_model') 

    return model


def test_the_model(testfile, model):
    '''

    :param testfile:
    :return:  a list of predicted values in same order of
    '''
    test=pd.read_csv(testfile)

    X_test=test.iloc[:,1:].values
    X_test=np.reshape(X_test, (len(X_test),48,48,1))
    y_test=test.iloc[:,0].values

    y_test=label_encoder.transform(y_test)

    X_test = X_test / 255.0

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print('test accuracy:', test_acc)

    return test_acc
