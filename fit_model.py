from load_data import *
from build_model import *
from keras.models import load_model

def fit_number(method):
    if method == "convolution":
        (x_train, y_train), (x_test, y_test) = load_data_convolution_number()
        # model = model_convolution()
        # model.summary()
        # model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.1)

        model = load_model("model_convolution.h5")
        # 看下测试集的损失值和准确率
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_convolution_number.h5")

    elif method == "baseline":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_number()
        model = model_baseline()
        model.summary()
        model.fit(x_train, y_train, epochs=10, batch_size=256)

        # model = load_model(""model_baseline.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_baseline_number.h5")
    else:
        pass

def fit_letter(method):
    if method == "baseline":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_letter()
        # showDatainPicture(x_train, y_train)
        model = model_baseline_letter()
        model.summary()
        model.fit(x_train, y_train, epochs=10, batch_size=256)

        # # model = load_model(""model_baseline.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_baseline_letter.h5")
    elif method == "convolution":
        (x_train, y_train), (x_test, y_test) = load_data_convolution_letter()
        # showDatainPicture(x_train, y_train)
        model = model_convolution_letter()
        model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.1)
        # model = load_model(""model_baseline.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_convolution_letter.h5")

def fit_combined(method):
    if method == "convolution":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_combined()
        model = model_convolution_combined()
        # model.summary()
        model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.1)

        # model = load_model("model_convolution.h5")
        # 看下测试集的损失值和准确率
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_convolution.h5")

    elif method == "baseline":
        (x_train, y_train), (x_test, y_test) = load_data_baseline_combined()
        model = model_baseline_combined()
        model.summary()
        model.fit(x_train, y_train, epochs=10, batch_size=256)

        # model = load_model(""model_baseline.h5")
        loss, accuracy = model.evaluate(x_test, y_test)
        print('loss {}, acc {}'.format(loss, accuracy))
        model.save("model_baseline1.h5")
    else:
        pass