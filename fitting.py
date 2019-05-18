from preparing import *
from modeling import *
from keras.models import load_model

def fit(method):
    if method == "convolution":
        (x_train, y_train), (x_test, y_test) = load_data_convolution()
        # model = model_convolution()
        # model.summary()
        # model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.1)

        model = load_model("model_convolution.h5")
        # 看下测试集的损失值和准确率
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_convolution.h5")

    elif method == "baseline":
        (x_train, y_train), (x_test, y_test) = load_data_baseline()
        model = model_baseline()
        model.summary()
        model.fit(x_train, y_train, epochs=20, batch_size=64)
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_baseline.h5")

        # model = load_model(""model_baseline.h5")
    else:
        pass