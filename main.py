from fit_model import fit_letter, fit_combined, fit_number
from utils import *
from predicting import predict

def fit_model():
    fit_combined("baseline")
    fit_letter("convolution")
    fit_number("convolution")

def run(imgPath, modelPath, method, target):
    # 计算边界
    borders = findBorderContours(imgPath)
    # 转换为MNIST格式
    img_mnist = transMNIST_convolution(img_path, borders, method)
    # 得到预测结果
    predict_result = predict(modelPath, img_mnist)
    # 显示标记了测试结果的图片
    showResults(img_path, borders, target, predict_result)


if __name__ == "__main__":
    # 图片的路径
    img_path = r"images\test2.png"
    # img_path = r"images\test_combined1.jpg"
    # 模型的路径
    model_path = "models\model_convolution_number.h5"
    # model_path = "model_baseline1.h5"
    run(img_path, model_path, "convolution", "number")

