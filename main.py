from fit_model import *
from utils import *
from predicting import predict

def fit_model():
    fit_combined("convolution")
    fit_combined("baseline")
    fit_letter("convolution")
    fit_letter("baseline")
    fit_number("convolution")
    fit_number("baseline")
    fit_SVM_letter()
    fit_SVM_number()
    fit_KNN_letter()
    fit_KNN_number()

def run(imgPath, modelPath, method, target):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    borders = None
    flag = None
    # 首先判断是否为手写图片，如果是则寻找边界处理图片
    if img.shape != (28, 28):
        # 计算边界
        borders = findBorderContours(imgPath)
        # print(borders)
        # 转换为MNIST格式（通过method来控制不同的格式转换）
        img_mnist = transMNIST(imgPath, borders, method)
        flag = 0
    else:
        # 为正常的28*28的图片
        img_mnist = img.reshape(1, 28 * 28)
    # 显示标记了测试结果的图片
    # print(predict_result)
    # 得到预测结果
    prediction = predict(modelPath, img_mnist, method)
    if flag == 0:
        showResults(imgPath, borders, target, prediction)
    return transformResult(prediction, target)


if __name__ == "__main__":
    # 训练模型并保存
    # fit_model()

    # 图片的路径
    # img_path = r"images\test3.png"
    # img_path = r"images\number_test3.jpg"
    # img_path = r"images\letter_test2.jpg"
    # img_path = r"D:\data_images\train_letter\13\13_52.png"
    img_path = r"D:\data_images\train_number\train_4.bmp"

    # 模型的路径
    # model_path = "models\model_convolution_number.h5"
    model_path = r"C:\Users\robin\Desktop\Courses\models\KNN(n=3)_number.m"
    # model_path = "model_baseline1.h5"

    # 运行并返回预测结果
    prediction = run(img_path, model_path, "CLF", "number")
    print(prediction)
    # run(img_path, model_path, "CLF", "letter")
    # run_original_files(img_path, model_path, "baseline", "letter")
    # run_original_files(img_path, model_path)
