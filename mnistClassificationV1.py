# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:41:35 2022

@author: USER
"""

import tensorflow as tf
from tensorflow.keras import datasets,layers,models
mnist = datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0 , x_test/255.0
num_classes = 10
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
