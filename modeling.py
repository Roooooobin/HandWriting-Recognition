from H
from keras import models
from keras import layers
import numpy as np

def model_conv():
    model = models.Sequential()
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


x_train, x_test, y_train, y_test = get_data()
# 定义模型
model = model_conv()
model.summary()
his = model.fit(x_train, y_train, epochs=5, batch_size=256, validation_split=0.1)

# 看下测试集的损失值和准确率
loss, acc = model.evaluate(x_test, y_test)
print('loss {}, acc {}'.format(loss, acc))
model.save("my_mnist_model.h5")
'''
    output:
            (60000, 28, 28, 1) (60000, 10)
            loss 0.02437469101352144, acc 0.9927
    测试集结果是99.27%，非常不错的模型
'''
