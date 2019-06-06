from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWidget(object):
    def setupUi(self, mainWidget):
        mainWidget.setObjectName("mainWidget")
        # mainWidget.resize(958, 479)
        mainWidget.resize(958, 520)
        mainWidget.setStyleSheet("QWidget#mainWidget{\n"
                                 "background-color:#FFF;\n"
                                 "border-radius:10px;\n"
                                 "}")
        self.clearButton = QtWidgets.QPushButton(mainWidget)
        self.clearButton.setGeometry(QtCore.QRect(805, 320, 113, 32))
        self.clearButton.setStyleSheet("border-radius:10px;\n"
                                       "background-color:#4C88F7;\n"
                                       "color:#FFF;")
        self.clearButton.setObjectName("clearButton")
        self.runButton = QtWidgets.QPushButton(mainWidget)
        self.runButton.setGeometry(QtCore.QRect(805, 250, 113, 32))
        self.runButton.setStyleSheet("border-radius:10px;\n"
                                     "background-color:#4C88F7;\n"
                                     "color:#FFF;")
        self.runButton.setObjectName("runButton")
        self.uploadButton = QtWidgets.QPushButton(mainWidget)
        self.uploadButton.setGeometry(QtCore.QRect(805, 180, 113, 32))
        self.uploadButton.setStyleSheet("border-radius:10px;\n"
                                        "background-color:#4C88F7;\n"
                                        "color:#FFF;")
        self.uploadButton.setObjectName("uploadButton")
        self.resultLineEdit = QtWidgets.QLineEdit(mainWidget)
        self.resultLineEdit.setGeometry(QtCore.QRect(785, 390, 156, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.resultLineEdit.setFont(font)
        self.resultLineEdit.setText("")
        self.resultLineEdit.setObjectName("resutLineEdit")
        self.algorithmCombo = QtWidgets.QComboBox(mainWidget)
        self.algorithmCombo.setGeometry(QtCore.QRect(805, 60, 120, 26))
        self.algorithmCombo.setStyleSheet("")
        self.algorithmCombo.setObjectName("algorithmCombo")
        self.algorithmCombo.addItem("")
        self.algorithmCombo.addItem("")
        self.algorithmCombo.addItem("")
        self.algorithmCombo.addItem("")
        self.targetCombo = QtWidgets.QComboBox(mainWidget)
        self.targetCombo.setGeometry(QtCore.QRect(805, 120, 120, 26))
        self.targetCombo.setObjectName("objectCombo")
        self.targetCombo.addItem("")
        self.targetCombo.addItem("")
        self.targetCombo.addItem("")
        self.label = QtWidgets.QLabel(mainWidget)
        self.label.setGeometry(QtCore.QRect(810, 40, 60, 18))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(mainWidget)
        self.label_2.setGeometry(QtCore.QRect(810, 100, 80, 18))
        self.label_2.setObjectName("label_2")
        self.line = QtWidgets.QFrame(mainWidget)
        self.line.setGeometry(QtCore.QRect(760, 0, 16, 501))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.photoLabel = QtWidgets.QLabel(mainWidget)
        self.photoLabel.setGeometry(QtCore.QRect(0, 0, 761, 471))
        self.photoLabel.setObjectName("photoLabel")

        self.retranslateUi(mainWidget)
        QtCore.QMetaObject.connectSlotsByName(mainWidget)

    def retranslateUi(self, mainWidget):
        _translate = QtCore.QCoreApplication.translate
        mainWidget.setWindowTitle(_translate("mainWidget", "Writing Pad"))
        self.clearButton.setText(_translate("mainWidget", "清空"))
        self.runButton.setText(_translate("mainWidget", "识别"))
        self.uploadButton.setText(_translate("mainWidget", "上传图片"))
        self.resultLineEdit.setPlaceholderText(_translate("mainWidget", "识别结果"))
        self.algorithmCombo.setItemText(0, _translate("mainWidget", "NN"))
        self.algorithmCombo.setItemText(1, _translate("mainWidget", "CNN"))
        self.algorithmCombo.setItemText(2, _translate("mainWidget", "SVM"))
        self.algorithmCombo.setItemText(3, _translate("mainWidget", "KNN"))
        self.targetCombo.setItemText(0, _translate("mainWidget", "数字"))
        self.targetCombo.setItemText(1, _translate("mainWidget", "字母"))
        self.targetCombo.setItemText(2, _translate("mainWidget", "数字+字母"))
        self.label.setText(_translate("mainWidget", "算法"))
        self.label_2.setText(_translate("mainWidget", "作用对象"))
        self.photoLabel.setText(_translate("mainWidget", " "))
