from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from load_data_letter import *


def load_data_convolution_number():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    # 数据预处理
    print(train_data.shape)
    x_train = train_data.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    x_test = test_data.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    return (x_train, y_train), (x_test, y_test)

def load_data_baseline_number():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 60000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)


def load_data_baseline_letter():
    train_data = load_train_images()
    # print(train_images[0].shape)
    train_labels = load_train_labels()
    test_data = load_test_images()
    test_labels = load_test_labels()

    data_len = train_data.shape[0]
    x_train = train_data[0:data_len]
    y_train = train_labels[0:data_len]
    x_train = x_train.reshape(data_len, 28 * 28)
    x_test = test_data.reshape(test_data.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 27)
    y_test = to_categorical(test_labels, 27)
    x_train = x_train
    x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)

def load_data_convolution_letter():
    train_data = load_train_images()
    # print(train_images[0].shape)
    train_labels = load_train_labels()
    test_data = load_test_images()
    test_labels = load_test_labels()
    x_train = train_data.reshape((train_data.shape[0], 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    x_test = test_data.reshape((test_data.shape[0], 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    return (x_train, y_train), (x_test, y_test)
