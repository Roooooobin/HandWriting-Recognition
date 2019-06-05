from sklearn import svm
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from load_data import *
from build_model import *

from utils import reshape_label_classifier


def fit_number(method):
    if method == "convolution":
        (x_train, y_train), (x_test, y_test) = load_data_convolution_number()
        model = model_convolution_number()
        model.summary()
        model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.1)

        # model = load_model("model_convolution_number.h5")
        # 看下测试集的损失值和准确率
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_convolution_number.h5")
    # baseline仅仅是一般神经网络的代称（“对照组”/“基准”）
    elif method == "baseline":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_number()
        model = model_baseline_number()
        model.summary()  # 显示模型信息摘要
        model.fit(x_train, y_train, epochs=10, batch_size=256)

        # model = load_model(""model_baseline_number.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_baseline_number.h5")
    else:
        pass


def fit_letter(method):
    # 根据method来选择训练的模型种类
    if method == "baseline":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_letter()
        print(x_test.shape)
        model = model_baseline_letter()
        model.summary()
        model.fit(x_train, y_train, epochs=10, batch_size=256)

        # model = load_model(""model_baseline_letter.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_baseline_letter.h5")  # 保存模型
    elif method == "convolution":
        (x_train, y_train), (x_test, y_test) = load_data_convolution_letter()
        model = model_convolution_letter()
        model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.1)
        # model = load_model(""model_convolution_letter.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_convolution_letter.h5")  # 保存模型


def fit_combined(method):
    if method == "convolution":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_combined()
        model = model_convolution_combined()
        model.summary()
        model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.1)

        # model = load_model("model_convolution_combined.h5")
        # 看下测试集的损失值和准确率
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_convolution_combined.h5")

    elif method == "baseline":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_combined()
        model = model_baseline_combined()
        model.summary()
        model.fit(x_train, y_train, epochs=10, batch_size=256)

        # model = load_model(""model_baseline_combined.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_baseline_combined.h5")
    else:
        pass


"""
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对
的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，
将他们当成噪声点，泛化能力较强
"""


def fit_SVM_number():
    (x_train, y_train), (x_test, y_test) = load_data_baseline_number()
    y_train = reshape_label_classifier(y_train)
    y_test = reshape_label_classifier(y_test)
    clf_svm = svm.SVC(C=1, kernel='linear')
    clf_svm.fit(x_train, y_train)
    joblib.dump(clf_svm, "SVM(C=1)_number.m")
    classifierResult = clf_svm.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1

    print("accuracy rate: {}".format((len(y_test) - errorCount) / len(y_test) * 100))


def fit_SVM_letter():
    (x_train, y_train), (x_test, y_test) = load_data_baseline_letter()
    y_train = reshape_label_classifier(y_train)
    y_test = reshape_label_classifier(y_test)
    clf_svm = svm.SVC(C=1, kernel='linear')
    clf_svm.fit(x_train, y_train)
    joblib.dump(clf_svm, "SVM(C=1)_letter.m")
    classifierResult = clf_svm.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1

    print("accuracy rate: {}".format((len(y_test) - errorCount) / len(y_test) * 100))


def fit_KNN_number():
    (x_train, y_train), (x_test, y_test) = load_data_baseline_number()
    y_train = reshape_label_classifier(y_train)
    y_test = reshape_label_classifier(y_test)
    clf_knn = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
    clf_knn.fit(x_train, y_train)
    joblib.dump(clf_knn, "KNN(n=3)_number.m")
    classifierResult = clf_knn.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1

    print("accuracy rate: {}".format((len(y_test) - errorCount) / len(y_test) * 100))


def fit_KNN_letter():
    (x_train, y_train), (x_test, y_test) = load_data_baseline_letter()
    y_train = reshape_label_classifier(y_train)
    y_test = reshape_label_classifier(y_test)
    clf_knn = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
    clf_knn.fit(x_train, y_train)
    joblib.dump(clf_knn, "KNN(n=3)_letter.m")
    classifierResult = clf_knn.predict(x_test)
    errorCount = 0
    for i in range(len(y_test)):
        if classifierResult[i] != y_test[i]:
            errorCount += 1

    print("accuracy rate: {}".format((len(y_test) - errorCount) / len(y_test) * 100))
