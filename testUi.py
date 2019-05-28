# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.run = QtWidgets.QPushButton(Form)
        self.run.setGeometry(QtCore.QRect(170, 270, 71, 21))
        self.run.setObjectName("run")
        self.output = QtWidgets.QTextBrowser(Form)
        self.output.setGeometry(QtCore.QRect(70, 10, 256, 161))
        self.output.setObjectName("output")
        self.model_name = QtWidgets.QTextEdit(Form)
        self.model_name.setGeometry(QtCore.QRect(50, 220, 71, 21))
        self.model_name.setObjectName("model_name")
        self.image_path = QtWidgets.QTextEdit(Form)
        self.image_path.setGeometry(QtCore.QRect(170, 220, 71, 21))
        self.image_path.setObjectName("image_path")
        self.target_name = QtWidgets.QTextEdit(Form)
        self.target_name.setGeometry(QtCore.QRect(280, 220, 71, 21))
        self.target_name.setObjectName("target_name")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.run.setText(_translate("Form", "Run"))

