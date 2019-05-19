import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件
from keras.utils import to_categorical

from build_model import model_baseline_letter

train_images_idx3_ubyte_file = r'./Letter DataSet/emnist-letters-train-images-idx3-ubyte'
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
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
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
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
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


def run():
    train_data = load_train_images()
    # print(train_images[0].shape)
    train_labels = load_train_labels()
    test_data = load_test_images()
    test_labels = load_test_labels()

    # # 查看前十个数据及其标签以读取是否正确
    # x_train = train_data.reshape((60000, 28, 28, 1))
    # x_train = x_train.astype('float32') / 255
    # x_test = test_data.reshape((10000, 28, 28, 1))
    # x_test = x_test.astype('float32') / 255
    # y_train = to_categorical(train_labels)
    # y_test = to_categorical(test_labels)

    number = train_data.shape[0]
    x_train = train_data[0:number]
    y_train = train_labels[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = test_data.reshape(test_data.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(y_train.shape)
    y_train = to_categorical(y_train, 27)
    y_test = to_categorical(test_labels, 27)
    x_train = x_train
    x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
    model = model_baseline_letter()
    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=256)

    # model = load_model(""model_baseline.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss {}, acc {}'.format(loss, accuracy))
    model.save("model_baseline_letter.h5")


if __name__ == '__main__':
    run()