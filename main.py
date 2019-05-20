from fit_model import fit_letter, fit
from utils import *
from predicting import predict

if __name__ == "__main__":
    # # fit("baseline")
    # # fit("convolution")
    # # for numbers
    # # 图片的路径
    # img_path = r"images\test3.png"
    img_path = r"images\test_combined1.jpg"
    # 模型的路径
    # model_path = "models\model_convolution.h5"
    model_path = "models\model_baseline.h5"
    # 计算边界
    borders = findBorderContours(img_path)
    # 转换为MNIST格式
    img_mnist = transMNIST2(img_path, borders)
    # 得到预测结果
    predict_result = predict(model_path, img_mnist)
    # 显示标记了测试结果的图片
    showResults(img_path, borders, predict_result)

    # for letters
    # img_path = r"images\test1.png"
    # model_path = "model_baseline.h5"
    # borders = findBorderContours(img_path)
    # img_mnist = transMNIST(img_path, borders)
    # predict_result = predict(model_path, img_mnist)
    # showResults(img_path, borders, predict_result)
