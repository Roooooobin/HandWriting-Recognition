import cv2
import numpy as np

# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img

# 寻找边缘，返回边框的左上角和右下角（利用cv2.findContours）
def findBorderContours(path, maxArea=100):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 去噪声
    img_denoised = cv2.fastNlMeansDenoising(img)
    img = accessBinary(img_denoised)
    # cv2.imwrite("img_denoised.png", img_denoised)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            border = [(x, y), (x + w, y + h)]
            borders.append(border)
    return borders

# transmit to MNIST format
def transMNIST(path, borders, method, size=(28, 28)):
    # 无符号整型uint8（0-255）
    if method == "convolution":
        imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    elif method == "baseline" or method == "CLF":
        imgData = np.zeros((len(borders), size[0] * size[0]), dtype='uint8')
    else: pass
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        if method == "convolution":
            targetImg = np.expand_dims(targetImg, axis=-1)
        elif method == "baseline" or method == "SVM":
            targetImg = targetImg.reshape(28 * 28)
        else: pass
        imgData[i] = targetImg
    print(imgData.shape)
    return imgData

# 由于分类器的标签数据集格式与神经网络不一样，所以需要调整
# 分类器的标签就是一个个数字，而神经网络的是数字的one-hot编码(通过to_categorical转换过)
def reshape_label_classifier(_y):
    y_new = []
    for i in range(len(_y)):
        number = 0
        for j in range(len(_y[i])):
            if _y[i][j] == 1:
                number = j
                break
        y_new.append(number)

    return y_new

# 显示结果及边框并保存（后续上到界面可能会有微调）
def showResults(path, borders, method,results=None):
    img = cv2.imread(path)
    # 绘制
    # print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (225, 105, 65))
        if results:
            # 通过method来控制不同的显示方式，绿色表示数字，红色表示字母
            if method == "combined":
                if results[i] > 10:
                    cv2.putText(img, chr(results[i]-11+65), border[0], cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 255), 1)
                else:
                    cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 1)
            elif method == "number":
                cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 1)
            elif method == "letter":
                cv2.putText(img, chr(results[i]-1+65), border[0], cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 255), 1)
            else: pass
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.namedWindow("test", 0)
    cv2.resizeWindow("test", 800, 600)
    cv2.imshow('test', img)
    # 保存图像
    cv2.imwrite("test1_result1.png", img)
    cv2.waitKey(0)