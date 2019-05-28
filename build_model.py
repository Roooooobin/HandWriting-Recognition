from keras.layers import Dense
from keras.models import Sequential
from keras import layers

def model_convolution():
    # 定义卷积神经网络模型（数字）
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
    # 多层感知机的baseline模型（数字）
    model = Sequential()
    model.add(Dense(input_dim=28*28, units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

def model_baseline_letter():
    # 多层感知机的baseline模型（字母）
    model = Sequential()
    model.add(Dense(input_dim=28*28, units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    # 测试集的标签1表示a/A，所以softmax为27维
    model.add(Dense(units=27, activation='softmax'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

def model_convolution_letter():
    # 定义卷积神经网络模型（字母）
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # 测试集的标签1表示a/A，所以softmax为27维
    model.add(layers.Dense(27, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    return model

def model_baseline_combined():
    # 多层感知机的baseline模型（字母）
    model = Sequential()
    model.add(Dense(input_dim=28*28, units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    # 测试集的标签1表示a/A，所以softmax为27维
    model.add(Dense(units=37, activation='softmax'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

def model_convolution_combined():
    # 定义卷积神经网络模型（字母）
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # 测试集的标签1表示a/A，所以softmax为27维
    model.add(layers.Dense(37, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    return model