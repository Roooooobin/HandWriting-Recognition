import sys
from UI_widgets import Ui_mainWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPen, QImage
from PyQt5.QtCore import Qt
from main import model_path_dic, algorithmName_sub, target_sub, run
from sklearn.externals import joblib
from keras.models import load_model

upload_tag = 0


class MainWindow(QMainWindow, Ui_mainWidget):  # 为了实现窗口的显示和业务逻辑分离，新建另一个调用窗口的文件
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.board = QPixmap(865, 485)
        self.board.fill(Qt.white)

        self.is_empty = True  # 空画板
        self.last_pos = QPoint(0, 0)
        self.current_pos = QPoint(0, 0)

        self.painter = QPainter()
        self.painter.setRenderHint(QPainter.Antialiasing)  # 反锯齿

        self.clearButton.clicked.connect(self.clear)
        self.uploadButton.clicked.connect(self.upload)
        self.runButton.clicked.connect(self.run)

    # 在此处识别，调用算法中的识别函数
    def run(self):
        algorithm = self.algorithmCombo.currentText()
        target = self.targetCombo.currentText()
        print(algorithm, target)
        # 先保存图片然后找到该图片识别
        image = self.board.toImage()
        # 这里可以改图片存储路径
        image_save_path = 'images/testImage.jpg'
        image.save(image_save_path)
        # 为了方便，识别时直接去存储路径下获取图片
        img_path = image_save_path
        # 由于尚未尝试用除NN外的算法去同时识别数字+字母，所以当选择其他算法时需要输出提示信息
        try:
            # 需要更名选择不同的model，可能会不存在，先load下来看是否报错
            model_path = model_path_dic[algorithm][target_sub[target]]
            if algorithm == "SVM" or algorithm == "KNN":
                model = joblib.load(model_path)
            else:
                model = load_model(model_path)
        except:  # 如果图片路径下没有相应的图片或是模型路径下没有相应的模型，报错
            QMessageBox.about(self, "提示", "暂不支持")
            self.resultLineEdit.setText("")
            self.board.fill(Qt.white)
        else:
            # 运行并返回预测结果
            prediction = run(img_path, model_path, algorithmName_sub[algorithm], target_sub[target])
            prediction = ' '.join([str(x) for x in prediction])
            print(prediction)
            self.resultLineEdit.setText(prediction)  # 将结果存放在LineEdit中
            self.board.load("result.png")
            self.update()

    def upload(self):
        filename = QFileDialog.getOpenFileName(None, 'open', ".")
        self.board.load(filename[0])
        self.update()

    def clear(self):
        self.board.fill(Qt.white)
        self.update()
        self.is_empty = True
        self.resultLineEdit.setText("")

    def paintEvent(self, paintEvent):
        self.painter.begin(self)
        self.painter.drawPixmap(0, 0, self.board)
        self.painter.end()

    def mouseReleaseEvent(self, QMouseEvent):
        self.is_empty = False

    def mousePressEvent(self, QMouseEvent):
        self.current_pos = QMouseEvent.pos()
        self.last_pos = self.current_pos

    def mouseMoveEvent(self, QMouseEvent):
        self.current_pos = QMouseEvent.pos()
        self.painter.begin(self.board)

        self.painter.setPen(QPen(Qt.black, 6))

        self.painter.drawLine(self.last_pos, self.current_pos)
        self.painter.end()
        self.last_pos = self.current_pos

        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())
