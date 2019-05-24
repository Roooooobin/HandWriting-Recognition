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

def run(imgPath, modelPath, method, target, prediction):
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
    predict(modelPath, img_mnist, method, prediction)
    if flag == 0:
        showResults(imgPath, borders, target, prediction)


if __name__ == "__main__":
    # 训练模型并保存
    # fit_model()

    # 图片的路径
    # img_path = r"images\test3.png"
    img_path = r"images\test_combined2.jpg"
    # img_path = r"D:\Letters DataSet\letters_train\13\13_52.png"

    # 模型的路径
    model_path = "models\model_baseline_letter.h5"
    # model_path = "KNN(n=3)_letter.m"
    # model_path = "model_baseline1.h5"

    prediction = []
    # 运行
    run(img_path, model_path, "baseline", "letter", prediction)
    print(prediction)
    # run(img_path, model_path, "CLF", "letter")
    # run_original_files(img_path, model_path, "baseline", "letter")
    # run_original_files(img_path, model_path)
