from keras.datasets import mnist
from keras.utils.np_utils import to_categorical


def get_data():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    # 数据预处理
    x_train = train_data.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    x_test = test_data.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    return x_train, x_test, y_train, y_test