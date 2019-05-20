from utils import *
from predicting import predict

if __name__ == "__main__":
    # fit("baseline")
    # fit("convolution")
    # for numbers
    # img_path = r"images\test3.png"
    img_path = r"images\letter_test1.jpg"
    # model_path = "models\model_convolution.h5"
    model_path = "models\model_baseline_letter.h5"
    borders = findBorderContours(img_path)
    img_mnist = transMNIST2(img_path, borders)
    predict_result = predict(model_path, img_mnist)
    showResults(img_path, borders, predict_result)

    # for letters
    # img_path = r"images\test1.png"
    # model_path = "model_baseline.h5"
    # borders = findBorderContours(img_path)
    # img_mnist = transMNIST(img_path, borders)
    # predict_result = predict(model_path, img_mnist)
    # showResults(img_path, borders, predict_result)
