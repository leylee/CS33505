import numpy as np
# import cupy as np
from matplotlib import pyplot as plt
from random import shuffle

from util import GradientDescent
import pandas as pd
from sklearn import preprocessing


def sigmod(z):
    return 1.0 / (1.0 + np.exp(z))


def generate_2_dimension_data(number, mean_pos, mean_neg, proportion_pos, cov11=0.5, cov12=0.0, cov21=0.0, cov22=0.5
                              , scale_1_bias=0, scale_2_bias=0):
    """
    生成二维高斯分布的数据
    :param number: 数据数量
    :param mean_pos: 正例平均值
    :param mean_neg: 反例平均值
    :param proportion_pos: 正例比例
    :param cov11: 协方差矩阵
    :param cov12: 协方差矩阵
    :param cov21: 协方差矩阵
    :param cov22: 协方差矩阵
    :param scale_1_bias: 第一维方差偏置
    :param scale_2_bias: 第二维方差偏置
    :return: 二维高斯分布数据
    """
    assert (0 < proportion_pos < 1)
    x_sample = []
    y_sample = []
    number_pos = int(number * proportion_pos)
    number_neg = number - number_pos
    # mean_pos_bios = -0.3
    # mean_neg_bios = 0.5

    while True:
        if number_neg == 0 and number_pos == 0:
            break
        elif number_neg == 0:
            number_pos = number_pos - 1
            x1_temp, x2_temp = np.random.multivariate_normal(
                [mean_pos, mean_pos], [[cov11 + scale_1_bias, cov12],
                                       [cov21, cov22 + scale_2_bias]], 1).T
            x_sample.append([x1_temp[0], x2_temp[0]])
            y_sample.append(1)
        elif number_pos == 0:
            number_neg = number_neg - 1
            x1_temp, x2_temp = np.random.multivariate_normal(
                [mean_neg, mean_neg], [[cov11 + scale_1_bias, cov12],
                                       [cov21, cov22 + scale_2_bias]], 1).T
            x_sample.append([x1_temp[0], x2_temp[0]])
            y_sample.append(0)
        else:
            if np.random.randint(0, 2) == 0:
                number_neg = number_neg - 1
                x1_temp, x2_temp = np.random.multivariate_normal(
                    [mean_neg, mean_neg], [[cov11 + scale_1_bias, cov12],
                                           [cov21, cov22 + scale_2_bias]], 1).T
                x_sample.append([x1_temp[0], x2_temp[0]])
                y_sample.append(0)
            else:
                number_pos = number_pos - 1
                x1_temp, x2_temp = np.random.multivariate_normal(
                    [mean_pos, mean_pos], [[cov11 + scale_1_bias, cov12],
                                           [cov21, cov22 + scale_2_bias]], 1).T
                x_sample.append([x1_temp[0], x2_temp[0]])
                y_sample.append(1)
    return x_sample, np.array(y_sample)


def split_data(x_sample, y_sample, test_rate=0.3):
    """ 分开训练集与测试集
    Args: test_rate 为默认测试集占所有样本的比例
     """
    number = len(x_sample)
    number_test = int(number * test_rate)
    number_train = number - number_test
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    number_range = list(range(number))
    shuffle(number_range)
    for i in number_range:
        if number_test > 0:
            if number_train == 0 or np.random.randint(2) == 0:
                number_test = number_test - 1
                x_test.append(x_sample[i])
                y_test.append(y_sample[i])
            else:
                number_train = number_train - 1
                x_train.append(x_sample[i])
                y_train.append(y_sample[i])
        else:
            number_train = number_train - 1
            x_train.append(x_sample[i])
            y_train.append(y_sample[i])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def load_watermelon_data():
    """ 读取西瓜书3.3a的数据 """
    data_set = np.loadtxt("./watermelon-3.3.csv", delimiter=",")
    x_temp = data_set[:, 1:3]
    y_temp = data_set[:, 3]
    return x_temp, y_temp


def accuracy(__x_test, __y_test, beta):
    """ 计算在给定测试集上的分类准确度 """
    columns = len(__x_test)
    count = 0
    for index in range(columns):
        if sigmod(beta @ __x_test[index]) < 0.5 and __y_test[index] == 1:
            count = count + 1
        elif sigmod(beta @ __x_test[index]) > 0.5 and __y_test[index] == 0:
            count = count + 1
    return columns, columns - count, (1.0 * count) / columns


def draw_2_dimensions(x_sample, y_sample):
    """ 画出二维参数的样本 """
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(len(x_sample)):
        if y_sample[i] == 1:
            type1_x.append(x_sample[i][1])
            type1_y.append(x_sample[i][2])
        else:
            type2_x.append(x_sample[i][1])
            type2_y.append(x_sample[i][2])

    plt.scatter(type1_x, type1_y, facecolor="none", edgecolor="b", label="positive")
    plt.scatter(type2_x, type2_y, marker="x", c="r", label="negative")


def main():
    # 用于生成数据的测试
    gen_lambda = 0.1  # 惩罚项系数
    number_gen = 100  # 样本数量
    proportion_pos_gen = 0.5  # 正例比例
    mean_gen_pos = -0.5  # 正例基础均值
    mean_gen_neg = 1  # 反例基础均值
    generating_x, generating_y = generate_2_dimension_data(number_gen, mean_gen_pos, mean_gen_neg,
                                                           proportion_pos_gen,
                                                           # cov21=1,
                                                           )  # ,cov21=1, scale_pos1_bios=0.3,scale_neg1_bios=0.6)
    generating_x = np.c_[np.ones(len(generating_x)), generating_x]
    x_train_gen, y_train_gen, x_test_gen, y_test_gen = split_data(generating_x, generating_y)
    generating_rows, generating_columns = np.shape(x_train_gen)
    # 使用梯度下降进行测试  包含惩罚项
    ans_gdr_gen = GradientDescent.fitting(x_train_gen, y_train_gen, np.zeros(generating_columns), hyper=gen_lambda)
    # 使用梯度下降进行测试  不包含惩罚项
    ans_gd_gen = GradientDescent.fitting(x_train_gen, y_train_gen, np.zeros(generating_columns), hyper=0)
    print("Generating GDR:", ans_gdr_gen)  # 梯度下降法系数
    print("Generating GD:", ans_gd_gen)  # 梯度下降法系数(不含惩罚项)

    x_draw_gen = np.linspace(-3, 3)
    y_draw_gdr_gen = - (ans_gdr_gen[0] + ans_gdr_gen[1] * x_draw_gen) / ans_gdr_gen[2]
    y_draw_gd_gen = - (ans_gd_gen[0] + ans_gd_gen[1] * x_draw_gen) / ans_gd_gen[2]
    plt.figure()
    title = "GDR accuracy: %f" % (accuracy(x_test_gen, y_test_gen, ans_gdr_gen)[2])
    plt.subplot(121)
    plt.title(title + "\nTrain")
    plt.plot(x_draw_gen, y_draw_gdr_gen, label="GDR")
    plt.plot(x_draw_gen, y_draw_gd_gen, label="GD")
    draw_2_dimensions(x_train_gen, y_train_gen)
    plt.subplot(122)
    title = "GD accuracy: %f" % (accuracy(x_test_gen, y_test_gen, ans_gd_gen)[2])
    plt.title(title + "\nTest")
    plt.plot(x_draw_gen, y_draw_gdr_gen, label="GDR")
    plt.plot(x_draw_gen, y_draw_gd_gen, label="GD")
    draw_2_dimensions(x_test_gen, y_test_gen)

    print("Generating GDR accuracy:", accuracy(x_test_gen, y_test_gen, ans_gdr_gen)[2])
    print("Generating GD accuracy:", accuracy(x_test_gen, y_test_gen, ans_gd_gen)[2])

    # draw_2_dimensions(generating_x, generating_y)
    plt.legend()
    plt.show()


def test_uci():
    print("UCI dataset test")
    data_set = pd.read_csv("./mushrooms.csv")
    x = data_set.drop('class', axis=1)
    y = data_set['class']
    encoder_x = preprocessing.LabelEncoder()
    for col in x.columns:
        x[col] = encoder_x.fit_transform(x[col])
    encode_y = preprocessing.LabelEncoder()
    y = encode_y.fit_transform(y)
    x = pd.get_dummies(x, columns=x.columns, drop_first=True)
    x = np.array(x)
    x_scaled = preprocessing.scale(x)

    ms_lambda = np.exp(-8)  # mushroom 超参数
    mushroom_x = x_scaled
    mushroom_y = y
    mushroom_x = np.c_[np.ones(len(mushroom_x)), mushroom_x]
    x_train, y_train, x_test, y_test = split_data(mushroom_x, mushroom_y, test_rate=0.5)
    mushroom_rows, mushroom_columns = x_train.shape

    mushroom_ans_gdr = GradientDescent.fitting(x_train, y_train, np.zeros(mushroom_columns), hyper=ms_lambda, rate=5,
                                               delta=1e-6)
    mushroom_ans_gd = GradientDescent.fitting(x_train, y_train, np.zeros(mushroom_columns), hyper=0, rate=5,
                                               delta=1e-6)
    all_num, false_num, rate = accuracy(x_test, y_test, mushroom_ans_gdr)
    print("Mushrooms GDR accuracy:", rate)
    print(false_num, "false of all", all_num)
    all_num, false_num, rate = accuracy(x_test, y_test, mushroom_ans_gd)
    print("Mushrooms GD accuracy:", rate)
    print(false_num, "false of all", all_num)
    print("x_length = ", x.shape)


if __name__ == '__main__':
    main()
    test_uci()
