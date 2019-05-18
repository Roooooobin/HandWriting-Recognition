from preparing import get_data
from modeling import  model_conv

def fit():
    x_train, x_test, y_train, y_test = get_data()
    # 定义模型
    model = model_conv()
    model.summary()
    his = model.fit(x_train, y_train, epochs=5, batch_size=256, validation_split=0.1)
    # 看下测试集的损失值和准确率
    loss, acc = model.evaluate(x_test, y_test)
    print('loss {}, acc {}'.format(loss, acc))
    model.save("my_mnist_model.h5")
    '''
        output:
                (60000, 28, 28, 1) (60000, 10)
                loss 0.02437469101352144, acc 0.9927
        测试集结果是99.27%，非常不错的模型
    '''