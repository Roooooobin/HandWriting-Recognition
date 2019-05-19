from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import layers
import numpy as np

def model_convolution():
    # 定义卷积神经网络模型
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    return model

def model_baseline():
    # 多层感知机的baseline模型
    model = Sequential()
    model.add(Dense(input_dim=28*28, units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model
