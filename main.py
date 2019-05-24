from fit_model import *
from utils import *
from predicting import predict

def fit_model():
    # fit_combined("baseline")
    # fit_letter("convolution")
    # fit_number("convolution")
    # fit_SVM_number()
    # fit_SVM_letter()
    # fit_KNN_number()
    fit_KNN_letter()

def run(imgPath, modelPath, method, target):
    # 计算边界
    borders = findBorderContours(imgPath)
    # 转换为MNIST格式（通过method来控制不同的格式转换）
    img_mnist = transMNIST(img_path, borders, method)
    # 得到预测结果
    predict_result = predict(modelPath, img_mnist, method)
    # 显示标记了测试结果的图片
    showResults(img_path, borders, target, predict_result)


if __name__ == "__main__":
    # fit_model()
    # 图片的路径
    # img_path = r"images\test2.png"
    img_path = r"images\test_combined2.jpg"
    # 模型的路径
    # model_path = "models\model_convolution_number.h5"
    model_path = "KNN(n=3)_letter.m"
    # model_path = "model_baseline1.h5"
    # run(img_path, model_path, "convolution", "number")
    run(img_path, model_path, "CLF", "letter")

