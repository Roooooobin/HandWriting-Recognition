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

# def run(imgPath, modelPath, method, target):
#     # 计算边界
#     borders = findBorderContours(imgPath)
#     # 转换为MNIST格式（通过method来控制不同的格式转换）
#     img_mnist = transMNIST(imgPath, borders, method)
#     # 得到预测结果
#     predict_result = predict(modelPath, img_mnist, method)
#     # 显示标记了测试结果的图片
#     print(predict_result)
#     showResults(imgPath, borders, target, predict_result)

def run(imgPath, modelPath, method, target):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    borders = None
    flag = None
    # 首先判断是否为非手写图片，不是则寻找边界处理图片
    if img.shape != (28, 28):
        # 计算边界
        borders = findBorderContours(imgPath)
        print(borders)
        # 转换为MNIST格式（通过method来控制不同的格式转换）
        img_mnist = transMNIST(imgPath, borders, method)
        flag = 0
    else:
        img_mnist = img.reshape(1, 28 * 28)
    # 得到预测结果
    predict_result = predict(modelPath, img_mnist, method)
    # 显示标记了测试结果的图片
    print(predict_result)
    if flag == 0:
        prediction = showResults(imgPath, borders, target, predict_result)
        print(prediction)

# def run_original_files(imgPath, modelPath="", method="", target=""):
#     img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
#     print(img.shape)
#     if img.shape != (28, 28):
#         print(1)
#     img = img.reshape(1, 28 * 28)
#     img = img.astype('float32') / 255
#     model = load_model(modelPath)
#     prediction = model.predict(img)
#     print(chr(int(np.argmax(prediction))+65-1))


if __name__ == "__main__":
    # 训练模型并保存
    # fit_model()

    # 图片的路径
    # img_path = r"images\test3.png"
    # img_path = r"images\test_combined2.jpg"
    img_path = r"D:\Letters DataSet\letters_train\13\13_52.png"

    # 模型的路径
    model_path = "models\model_baseline_letter.h5"
    # model_path = "KNN(n=3)_letter.m"
    # model_path = "model_baseline1.h5"

    # 运行
    run(img_path, model_path, "baseline", "letter")
    # run(img_path, model_path, "CLF", "letter")
    # run_original_files(img_path, model_path, "baseline", "letter")
    # run_original_files(img_path, model_path)
