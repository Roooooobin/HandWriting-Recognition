from sklearn.externals import joblib

from load_data import *
from keras.models import load_model

from utils import reshape_label_classifier

f = open("accuracy.txt", "w+")

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
    clf_svm = joblib.load(r"C:\Users\robin\Desktop\Courses\models\SVM(C=1.0)_number.m")
    classifierResult = clf_svm.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1
    print("Accuracy of SVM for recognizing number: %.2f%%" % ((len(y_test) - errorCount) / len(y_test) * 100), file=f)

def acc_SVM_letter():
    (_, _), (x_test, y_test) = load_data_baseline_letter()
    y_test = reshape_label_classifier(y_test)
    clf_svm = joblib.load(r"C:\Users\robin\Desktop\Courses\models\SVM(C=1.0)_letter.m")
    classifierResult = clf_svm.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1
    print("Accuracy of SVM for recognizing letter: %.2f%%" % ((len(y_test) - errorCount) / len(y_test) * 100), file=f)

def acc_KNN_number():
    (_, _), (x_test, y_test) = load_data_baseline_number()
    y_test = reshape_label_classifier(y_test)
    clf_svm = joblib.load(r"C:\Users\robin\Desktop\Courses\models\KNN(n=3)_number.m")
    classifierResult = clf_svm.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1
    print("Accuracy of KNN for recognizing number: %.2f%%" % ((len(y_test) - errorCount) / len(y_test) * 100), file=f)

def acc_KNN_letter():
    (_, _), (x_test, y_test) = load_data_baseline_letter()
    y_test = reshape_label_classifier(y_test)
    clf_svm = joblib.load(r"C:\Users\robin\Desktop\Courses\models\KNN(n=3)_letter.m")
    classifierResult = clf_svm.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1
    print("Accuracy of KNN for recognizing letter: %.2f%%" % ((len(y_test) - errorCount) / len(y_test) * 100), file=f)


if __name__ == "__main__":
    # acc_convolution_number()
    # acc_convolution_letter()
    # acc_baseline_number()
    # acc_baseline_letter()
    # acc_SVM_number()
    acc_SVM_letter()
    # acc_KNN_number()
    acc_KNN_letter()