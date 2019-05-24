from sklearn import svm
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from load_data import load_data_baseline_number

(x_train, y_train), (x_test, y_test) = load_data_baseline_number()

# 由于分类器的标签数据集格式与神经网络不一样，所以需要调整
# 分类器的标签就是一个个数字，而神经网络的是数字的one-hot编码(通过to_categorical转换过)
def reshape_label_classifier(_y):
    y_new = []
    for i in range(len(_y)):
        number = 0
        for j in range(len(_y[i])):
            if _y[i][j] == 1:
                number = j
                break
        y_new.append(number)

    return y_new


y_train = reshape_label_classifier(y_train)
y_test = reshape_label_classifier(y_test)
# knc = KNeighborsClassifier(n_neighbors=10)
# print("now")
# knc.fit(x_train, y_train)
# joblib.dump(knc, "knc.m")
# knc = joblib.load("knc.m")
# print("now")
# predict = knc.predict(x_test)
# print("now")
# print("accuracy_score: %.4lf" % accuracy_score(predict, y_test))
"""
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对
的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，
将他们当成噪声点，泛化能力较强
"""
# clf_svm = svm.SVC(C=1.0, kernel='linear')
# clf_svm.fit(x_train, y_train)
# joblib.dump(clf_svm, "svm.m")
clf_svm = joblib.load("svm.m")
print("1")
classifierResult = clf_svm.predict(x_test)
print("1")
errorCount = 0
for i in range(len(y_test)):
    if classifierResult[i] != y_test[i]:
        errorCount += 1

print("accuracy rate: {}".format((len(y_test)-errorCount) / len(y_test) * 100))