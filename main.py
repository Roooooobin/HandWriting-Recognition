from load_data import load_data_baseline
from utils import *
from predicting import predict

if __name__ == "__main__":
    # fit("baseline")
    # fit("convolution")
    (x_train, y_train), (x_test, y_test) = load_data_baseline()
    # img_path = "test1.png"
    # model_path = "model_convolution.h5"
    # borders = findBorderContours(img_path)
    # img_mnist = transMNIST(img_path, borders)
    # predict_result = predict(model_path, img_mnist)
    # showResults(img_path, borders, predict_result)
