from keras.datasets import mnist
from keras import utils
import cv2
import numpy as np
from matplotlib import pyplot as plt

# opencv中ANN定义神经网络层
def create_ANN():
    ann = cv2.ml.ANN_MLP_create()
    # 设置神经网络层的结构 输入层为784 隐藏层为80 输出层为10
    ann.setLayerSizes(np.array([784, 64, 10]))
    # 设置网络参数为误差反向传播法
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    # 设置激活函数为sigmoid

    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    # 设置训练迭代条件
    # 结束条件为训练100次或者误差小于0.00001
    ann.setTermCriteria((cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 100, 0.0001))

    return ann


# 计算测试数据上的识别率
def evaluate_acc(ann, test_images, test_labels):
    # 采用的sigmoid激活函数，需要对结果进行置信度处理

    # 对于大于0.99的可以确定为1 对于小于0.01的可以确信为0

    test_ret = ann.predict(test_images)
    # 预测结果是一个元组
    test_pre = test_ret[1]
    # 可以直接最大值的下标 (10000,)
    test_pre = test_pre.argmax(axis=1)
    true_sum = (test_pre == test_labels)
    return true_sum.mean()


if __name__ == '__main__':
    # 直接使用Keras载入的训练数据(60000, 28, 28) (60000,)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 查看MNIST数据结构
    print(train_images.shape, test_images.shape)
    print(train_images[1])
    print(train_labels[1])
    plt.imshow(train_images[1])
    plt.show()

    # 变换数据的形状并归一化
    train_images = train_images.reshape(train_images.shape[0], -1)  # (60000, 784)
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape(test_images.shape[0], -1)
    test_images = test_images.astype('float32') / 255

    # 将标签变为one-hot形状 (60000, 10) float32
    train_labels = utils.to_categorical(train_labels)
    # 测试数据标签不用变为one-hot (10000,)
    test_labels = test_labels.astype(np.int32)

    # 定义神经网络模型结构
    ann = create_ANN()

    # 开始训练
    ann.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)
    # 在测试数据上测试准确率
    print(evaluate_acc(ann, test_images, test_labels))

    # 保存模型
    ann.save('mnist_ann.xml')
    # 加载模型
    myann = cv2.ml.ANN_MLP_load('mnist_ann.xml')