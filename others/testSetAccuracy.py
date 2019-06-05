from sklearn.externals import joblib

from load_data import *
from keras.models import load_model
import random as rd
from datetime import datetime as dt

from utils import reshape_label_classifier

f = open("accuracy.txt", "a")

def acc_convolution_number():
    (_, _), (x_test, y_test) = load_data_convolution_number()
    model = load_model("models\model_convolution_number.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy of CNN for recognizing number: %.2f%%" % (accuracy*100), file=f)

def acc_convolution_letter():
    (_, _), (x_test, y_test) = load_data_convolution_letter()
    model = load_model("models\model_convolution_letter.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy of CNN for recognizing letter: %.2f%%" % (accuracy * 100), file=f)

def acc_baseline_number():
    (_, _), (x_test, y_test) = load_data_baseline_number()
    model = load_model("models\model_baseline_number.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy of NN for recognizing number: %.2f%%" % (accuracy*100), file=f)

def acc_baseline_letter():
    (_, _), (x_test, y_test) = load_data_baseline_letter()
    model = load_model("models\model_baseline_letter.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy of NN for recognizing letter: %.2f%%" % (accuracy * 100), file=f)

def acc_SVM_number():
    (_, _), (x_test, y_test) = load_data_baseline_number()
    y_test = reshape_label_classifier(y_test)
    clf_SVM = joblib.load(r"C:\Users\robin\Desktop\Courses\models\SVM(C=1.0)_number.m")
    classifierResult = clf_SVM.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1
    print("Accuracy of SVM for recognizing number: %.2f%%" % ((len(y_test) - errorCount) / len(y_test) * 100), file=f)


def acc_SVM_letter():
    (_, _), (x_test, y_test) = load_data_baseline_letter()
    # 测试一下运行大概需要的时间
    # sequence = list(range(0, len(x_test)))
    # rd.shuffle(sequence)
    # y_test = reshape_label_classifier(y_test)
    # x_test_new = []
    # y_test_new = []
    # for i in sequence:
    #     x_test_new.append(x_test[i])
    #     y_test_new.append(y_test[i])
    # x_test_new = np.array(x_test_new)
    # y_test_new = np.array(y_test_new)
    clf_SVM = joblib.load(r"C:\Users\robin\Desktop\Courses\models\SVM(C=1.0)_letter.m")
    classifierResult = clf_SVM.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1
    print("Accuracy of SVM for recognizing letter: %.2f%%" % ((len(y_test) - errorCount) / len(y_test) * 100), file=f)

def acc_KNN_number():
    (_, _), (x_test, y_test) = load_data_baseline_number()
    y_test = reshape_label_classifier(y_test)
    clf_KNN = joblib.load(r"C:\Users\robin\Desktop\Courses\models\KNN(n=3)_number.m")
    classifierResult = clf_KNN.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1
    print("Accuracy of KNN for recognizing number: %.2f%%" % ((len(y_test) - errorCount) / len(y_test) * 100), file=f)

def acc_KNN_letter():
    print(dt.now())
    (_, _), (x_test, y_test) = load_data_baseline_letter()
    # 测试一下运行大概需要的时间
    sequence = list(range(0, len(x_test)))
    rd.shuffle(sequence)
    y_test = reshape_label_classifier(y_test)
    x_test_new = []
    y_test_new = []
    for i in sequence[:]:
        x_test_new.append(x_test[i])
        y_test_new.append(y_test[i])
    x_test_new = np.array(x_test_new)
    y_test_new = np.array(y_test_new)
    clf_KNN = joblib.load(r"C:\Users\robin\Desktop\Courses\models\KNN(n=3)_letter.m")
    classifierResult = clf_KNN.predict(x_test_new)
    print(dt.now())
    errorCount = 0
    for i in range(len(y_test_new)):
        if classifierResult[i] != y_test_new[i]:
            errorCount += 1
    print("Accuracy of KNN for recognizing letter: %.2f%%" % ((len(y_test_new) - errorCount) / len(y_test_new) * 100), file=f)


if __name__ == "__main__":
    # acc_convolution_number()
    # acc_convolution_letter()
    # acc_baseline_number()
    # acc_baseline_letter()
    # acc_SVM_number()
    # acc_SVM_letter()
    # acc_KNN_number()
    acc_KNN_letter()