from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QWidget, QPushButton
import sys
from testUi import *
from main import run

class Example(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(Example, self).__init__()
        self.setupUi(self)
        self.run.clicked.connect(self.runMain)

    def runMain(self):
        # img_path = r"images\A.jpg"
        # model_path = "models\model_convolution_letter.h5"
        # prediction = run(img_path, model_path, "convolution", "letter")
        # print(prediction)
        print("sss")

app = QtWidgets.QApplication(sys.argv)
myshow = Example()
myshow.show()
sys.exit(app.exec_())