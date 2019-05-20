from keras.models import load_model
import numpy as np
from keras import models

def predict(modelPath, imgMnist):
    model = load_model(modelPath)
    # astype将数组的数据类型从无符号整型转换为单精度浮点数
    print(imgMnist.shape)
    img = imgMnist.astype('float32') / 255
    results = model.predict(img)  # 得到的是softmax结果
    result_number = []
    for res in results:
        result_number.append(np.argmax(res))
        # 使用np.argmax找出其中值最大的参数作为预测数字
    return result_number
