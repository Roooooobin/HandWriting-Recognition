from keras.models import load_model
import numpy as np
from sklearn.externals import joblib


def predict(modelPath, imgMnist, method):
    result = []
    if method == "CLF":
        model = joblib.load(modelPath)
    else:
        model = load_model(modelPath)
    # astype将数组的数据类型从无符号整型转换为单精度浮点数
    img = imgMnist.astype('float32') / 255
    results = model.predict(img)
    # 分类器得到的就是一个个单一数字/字母
    if method == "CLF":
        for res in results:
            result.append(res)
    # 神经网络得到的是一组组softmax结果
    elif method == "baseline" or method == "convolution":
        for res in results:
            result.append(np.argmax(res))
            # 使用np.argmax找出其中值最大的参数作为预测数字
    # 返回预测的一个个结果
    return result
