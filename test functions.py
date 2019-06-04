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

import cv2


array = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 7]])
print(array[1:3, 2:4])