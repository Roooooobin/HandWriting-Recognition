import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件
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

# dd数据集，图片格式居然经过了镜像+旋转，重新转回来
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