from fit_model import *
from utils import *
from predicting import predict

algorithmName_sub = {"CNN": "convolution", "NN": "baseline", "SVM": "CLF", "KNN": "CLF"}

model_path_dic = {"CNN": {"number": "models\model_convolution_number.h5",
                          "letter": "models\model_convolution_letter.h5"},
                  "NN": {"number": "models\model_baseline_number.h5",
                         "letter": "models\model_baseline_letter.h5",
                         "combined": "models\model_baseline_combine.h5"},
                  "SVM": {"number": r"C:\Users\robin\Desktop\Courses\models\SVM(C=1.0)_number.m",
                          "letter": r"C:\Users\robin\Desktop\Courses\models\SVM(C=1.0)_letter.m"},
                  "KNN": {"number": r"C:\Users\robin\Desktop\Courses\models\KNN(n=3)_number.m",
                          "letter": r"C:\Users\robin\Desktop\Courses\models\KNN(n=3)_letter.m"}
                  }

target_sub = {"数字": "number", "字母": "letter", "数字+字母": "combined"}


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
        cv2.imwrite("result.png", img)
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
    img_path = r"images\letter_test2.jpg"

    # 通过算法和识别对象选择模型的路径
    algorithm = "SVM"
    target = "letter"
    model_path = model_path_dic[algorithm][target]
    # 运行并返回预测结果
    prediction = run(img_path, model_path, algorithmName_sub[algorithm], target)
    print(prediction)
