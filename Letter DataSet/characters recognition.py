from predicting import predict
from utils import findBorderContours, transMNIST, showResults

img_path = "test2.jpg"
model_path = "../model_convolution.h5"
borders = findBorderContours(img_path)
img_mnist = transMNIST(img_path, borders)
print(img_mnist[0].shape)
res = predict(model_path, img_mnist)
showResults(img_path, borders, res)
print(res)