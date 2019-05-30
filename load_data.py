from keras.datasets import mnist
from keras.utils import to_categorical
from process_data_letter import *

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

def load_data_convolution_number():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    # 数据预处理
    x_train = train_data.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    x_test = test_data.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    return (x_train, y_train), (x_test, y_test)

def load_data_baseline_letter():
    train_data = load_train_images()
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

def load_data_baseline_combined():

    train_data_letter = load_train_images()
    train_labels_letter = load_train_labels()
    test_data_letter = load_test_images()
    test_labels_letter = load_test_labels()

    # reshape to regular shape
    train_data_letter = reshape_data_letter(train_data_letter)
    test_data_letter = reshape_data_letter(test_data_letter)

    data_len = train_data_letter.shape[0]
    x_train_letter = train_data_letter[0:data_len]
    y_train_letter = train_labels_letter[0:data_len]
    x_train_letter = x_train_letter.reshape(data_len, 28 * 28)
    x_test_letter = test_data_letter.reshape(test_data_letter.shape[0], 28 * 28)
    x_train_letter = x_train_letter.astype('float32')
    x_test_letter = x_test_letter.astype('float32')
    y_train_letter = to_categorical(y_train_letter, 27)
    y_test_letter = to_categorical(test_labels_letter, 27)
    print(y_test_letter[0])
    x_train_letter = x_train_letter
    x_test_letter = x_test_letter
    x_train_letter = x_train_letter / 255
    x_test_letter = x_test_letter / 255

    (x_train_number, y_train_number), (x_test_number, y_test_number) = mnist.load_data()
    number = 60000
    x_train_number = x_train_number[0:number]
    y_train_number = y_train_number[0:number]
    x_train_number = x_train_number.reshape(number, 28 * 28)
    x_test_number = x_test_number.reshape(x_test_number.shape[0], 28 * 28)
    x_train_number = x_train_number.astype('float32')
    x_test_number = x_test_number.astype('float32')
    y_train_number = to_categorical(y_train_number, 10)
    y_test_number = to_categorical(y_test_number, 10)
    x_train_number = x_train_number
    x_test_number = x_test_number
    x_train_number = x_train_number / 255
    x_test_number = x_test_number / 255

    y_train = label_extended2(y_train_number, y_train_letter)
    y_test = label_extended2(y_test_number, y_test_letter)
    x_train = data_combined2(x_train_number, x_train_letter)
    x_test = data_combined2(x_test_number, x_test_letter)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return (x_train, y_train), (x_test, y_test)

def load_data_convolution_combined():
    (train_data_number, train_labels_number), (test_data_number, test_labels_number) = mnist.load_data()
    # 数据预处理
    x_train_number = train_data_number.reshape((60000, 28, 28, 1))
    x_train_number = x_train_number.astype('float32') / 255
    x_test_number = test_data_number.reshape((10000, 28, 28, 1))
    x_test_number = x_test_number.astype('float32') / 255
    y_train_number = to_categorical(train_labels_number)
    y_test_number = to_categorical(test_labels_number)

    train_data_letter = load_train_images()
    train_labels_letter = load_train_labels()
    test_data_letter = load_test_images()
    test_labels_letter = load_test_labels()

    # reshape to regular shape
    train_data_letter = reshape_data_letter(train_data_letter)
    test_data_letter = reshape_data_letter(test_data_letter)

    x_train_letter = train_data_letter.reshape((train_data_letter.shape[0], 28, 28, 1))
    x_train_letter = x_train_letter.astype('float32') / 255
    x_test_letter = test_data_letter.reshape((test_data_letter.shape[0], 28, 28, 1))
    x_test_letter = x_test_letter.astype('float32') / 255
    y_train_letter = to_categorical(train_labels_letter)
    y_test_letter = to_categorical(test_labels_letter)

    y_train = label_extended(y_train_number, y_train_letter)
    y_test = label_extended(y_test_number, y_test_letter)
    x_train = data_combined(x_train_number, x_train_letter)
    x_test = data_combined(x_test_number, x_test_letter)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return (x_train, y_train), (x_test, y_test)

# 将标签扩展为10+27维
def label_extended(label_number, label_letter):
    label_new = []
    for i in range(len(label_number)):
        # 在number的label后添加27个0
        label_new.append(np.append(label_number[i], np.array([0]*27)))
    zeros = [0] * 10
    for i in range(len(label_letter)):
        label_new.append(np.append(zeros, label_letter[i]))
    return np.array(label_new)

# 将标签扩展为10+27维
def label_extended2(label_number, label_letter):
    label_new = []
    zeros = [0] * 10
    for i in range(len(label_letter) // 2):
        label_new.append(np.append(zeros, label_letter[i]))
    for i in range(len(label_number)):
        # 在number的label后添加27个0
        label_new.append(np.append(label_number[i], np.array([0]*27)))
    return np.array(label_new)

def data_combined(data_number, data_letter):
    data_new = list(data_number)
    for i in range(len(data_letter)):
        data_new.append(list(data_letter[i]))
    return np.array(data_new)

def data_combined2(data_number, data_letter):
    data_new = list(data_letter[:len(data_letter) // 2])
    for i in range(len(data_number)):
        data_new.append(list(data_number[i]))
    return np.array(data_new)

