import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件
from keras import Sequential, layers
from keras.datasets import mnist
from keras.utils import to_categorical

from build_model import model_baseline_letter

train_images_idx3_ubyte_file = './Letter DataSet/emnist-letters-train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = './Letter DataSet/emnist-letters-train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = './Letter DataSet/emnist-letters-test-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = './Letter DataSet/emnist-letters-test-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_data_baseline_letter():
    train_data = load_train_images()
    # print(train_images[0].shape)
    train_labels = load_train_labels()
    test_data = load_test_images()
    test_labels = load_test_labels()

    # reshape to regular shape
    train_data = reshape_data_letter(train_data)
    test_data = reshape_data_letter(test_data)

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

    # reshape to regular shape
    train_data = reshape_data_letter(train_data)
    test_data = reshape_data_letter(test_data)

    x_train = train_data.reshape((train_data.shape[0], 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    x_test = test_data.reshape((test_data.shape[0], 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    return (x_train, y_train), (x_test, y_test)

# 哈皮数据集，图片格式居然经过了镜像+旋转，重新转回来
def reshape_data_letter(x_train):
    x_train = x_train.reshape(len(x_train), 28, 28)
    n_data = len(x_train)
    for i in range(n_data):
        x_train[i] = np.rot90(x_train[i], -1)
    for i in range(n_data):
        for m in x_train[i]:
            for j in range(14):
                m[j], m[28 - 1 - j] = m[28 - 1 - j], m[j]
    return x_train

def model_convolution_letter():
    # 定义卷积神经网络模型
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(27, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    return model

def showDatainPicture(_x, _y):
    _x = _x.reshape(len(_x), 28, 28)
    print(np.argmax(_y[10]), np.argmax(_y[11]),
          np.argmax(_y[12]), np.argmax(_y[13]))
    print(np.argmax(_y[14]), np.argmax(_y[15]))
    plt.subplot(331)
    plt.imshow(_x[10], cmap=plt.get_cmap('gray'))
    plt.subplot(332)
    plt.imshow(_x[11], cmap=plt.get_cmap('gray'))
    plt.subplot(333)
    plt.imshow(_x[12], cmap=plt.get_cmap('gray'))
    plt.subplot(334)
    plt.imshow(_x[13], cmap=plt.get_cmap('gray'))
    plt.subplot(335)
    plt.imshow(_x[14], cmap=plt.get_cmap('gray'))
    plt.subplot(336)
    plt.imshow(_x[15], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()

def run():
    # x_train = train_data.reshape((60000, 28, 28, 1))
    # x_train = x_train.astype('float32') / 255
    # x_test = test_data.reshape((10000, 28, 28, 1))
    # x_test = x_test.astype('float32') / 255
    # y_train = to_categorical(train_labels)
    # y_test = to_categorical(test_labels)

    (x_train, y_train), (x_test, y_test) = load_data_baseline_letter()
    showDatainPicture(x_train, y_train)
    model = model_baseline_letter()
    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=256)

    # # model = load_model(""model_baseline.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss {}, acc {}'.format(loss, accuracy))
    model.save("model_baseline_letter_test1.h5")

    # (x_train, y_train), (x_test, y_test) = load_data_convolution_letter()
    # showDatainPicture(x_train, y_train)
    # model = model_convolution_letter()
    # model.fit(x_train, y_train, epochs=1, batch_size=512, validation_split=0.1)
    # # model = load_model(""model_baseline.h5")
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print('loss {}, acc {}'.format(loss, accuracy))
    # model.save("model_convolution_letter_test1.h5")


if __name__ == '__main__':
    run()