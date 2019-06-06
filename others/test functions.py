import numpy as np
import random as rd
#
# m = np.array([1, 2, 3])
# n = np.array([2, 3, 4])
# print(np.concatenate((m, n)))
# m = np.pad(m, (0, 7), 'constant')
# print(m)
# zeros = np.array([0] * 10)
# print(zeros)
#
# # 将标签扩展为10+26维
# def label_extended(label_number, label_letter):
#     for i in range(len(label_number)):
#         # 在number的label后添加26个0
#         label_number[i] = np.append(label_number[i], [0]*10)
#     zeros = np.array([0] * 10)
#     for i in range(len(label_letter)):
#         label_letter[i] = np.append(zeros, label_letter[i])
#
#
# m = np.array([1, 2, 3])
# m = np.append(m, [0] * 10)
# m1 = np.array([1, 2, 3])
# m = np.append(m, m1)
# print(type(m))
# n = np.array([1, 2, 3])
# n1 = np.array([1, 2, 3])
# n = np.vstack((n, n1))
# print(n)
# label_extended(m, n)
# print(m, n)
#
# label = np.array([[1, 2], [1, 3]])
# label_new = []
# for i in range(0, len(label)):
#     label_new.append(np.append(label[i], [0]*3, axis=0))
# label_new = np.array(label_new)
# print(label_new)
# print(type(label_new))

# label = np.array([[1, 2], [1, 3]])
# label2 = np.array([[1, 2], [1, 3]])
# l = list(label)
# print(type(np.array(l)))

# Tuple = [[(0, 1), (2, 3)], [(3, 2), (1, 2)], [(1, 2), (2, 1)]]
# Tuple = sorted(Tuple, key=lambda x: (x[0]))
# print(Tuple)

# def test(result):
#     result.append(1)
#
#
# t = []
# test(t)
# print(t)
#
# array = np.zeros((10, 3))
# for i in range(0, len(array)):
#     for j in range(0, len(array[0])):
#         array[i][j] = i + j
# print(array)
#
# sequence = list(range(0, len(array)))
# rd.shuffle(sequence)
# print(sequence)
# x_test_new = []
# y_test_new = []
# for i in sequence[:5]:
#     x_test_new.append(array[i])
#
# x_test_new = np.array(x_test_new)
# print(x_test_new)
#
# from datetime import datetime as dt

# import cv2
#
#
# array = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 7]])
# print(array[0:3, 0:2])
#
# def accessPixel(img):
#     height = img.shape[0]
#     width = img.shape[1]
#     for i in range(height):
#         for j in range(width):
#             img[i][j] = 255 - img[i][j]
#     return img
#
# def accessBinary(img, threshold=128):
#     img = accessPixel(img)
#     # 边缘膨胀，不加也可以
#     kernel = np.ones((3, 3), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
#     return img
#
#
# path = r"images\letter_test2.jpg"
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img = accessBinary(img)
# contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# borders = []
# for contour in contours:
#     # 将边缘拟合成一个矩形边框，简化保存矩形的左上角和右下角的坐标
#     x, y, w, h = cv2.boundingRect(contour)
#     if w * h > 200:
#         border = [(x, y), (x + w, y + h)]
#         borders.append(border)
# print(borders)
# for i, border in enumerate(borders):
#     print(border[0][1], border[1][1])
#     borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
#     print(borderImg.shape)
#     print(borderImg)
#     break
#
dic = {"a": 1, "b": 2}
print(dic["c"])