import numpy as np
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

label = np.array([[1, 2], [1, 3]])
label2 = np.array([[1, 2], [1, 3]])
l = list(label)
print(type(np.array(l)))