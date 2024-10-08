# -*- coding: utf-8 -*-
"""Digit Classification.ipynb
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

(X_train,Y_train),(X_test,Y_test) = keras.datasets.mnist.load_data()

plt.matshow(X_train[0])

X_train_flatten = X_train.reshape(-1, 28, 28, 1)
X_test_flatten = X_test.reshape(-1, 28, 28, 1)
X_train_flatten = X_train_flatten / 255.0
X_test_flatten = X_test_flatten / 255.0

def CNN():
    model = keras.Sequential()
    # CONV > CONV > BN > RELU > MAXPOOLING > DROPOUT
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='valid', input_shape=(28, 28, 1), name='conv2d_1_1'))
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv2d_1_2'))
    model.add(layers.BatchNormalization(name='bn_1'))
    model.add(layers.Activation('relu', name='relu_1'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='mp2d_1'))
    model.add(layers.Dropout(0.2, name='drop_1'))

    # CONV > CONV > BN > RELU > MAXPOOLING > DROPOUT
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', name='conv2d_2_1'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv2d_2_2'))
    model.add(layers.BatchNormalization(name='bn_2'))
    model.add(layers.Activation('relu', name='relu_2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='mp2d_2'))
    model.add(layers.Dropout(0.2, name='drop_2'))

    # FLATTEN > DENSE > CLASSIFICATION
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

model = CNN()

model.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=5, verbose=1)

Y_pred=model.predict( X_test_flatten)

Y_pred_label=[np.argmax(i) for i in Y_pred]
Y_test_label=[np.argmax(i) for i in Y_test]

cm=tf.math.confusion_matrix(Y_test_label, Y_pred_label)
cm

import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.matshow(X_test[0])
print("the value is:", Y_test_label[0])

