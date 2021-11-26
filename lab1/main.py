import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from util import AnalyticalSolution, GradientDescent, ConjugateGradient


def generate_sin(number: int, scale=0.2):
    x = np.linspace(0, 1, number)
    np.linspace
    t = np.sin(2 * pi * x) + np.random.normal(0, scale, x.shape)
    return x, t


def transform_vec_to_matrix(x: np.ndarray, degree):
    X = np.ones((degree + 1, x.size))
    for i in range(degree):
        X[i + 1] = X[i] * x
    return X.T


def predict(X, w):
    return X @ w
def main():
    over_fitting_params = (10, 50, 8)
    normal_params = (20, 50, 5)
    number_train, number_test, degree = normal_params
    # number_train = 20
    # number_test = 50
    # degree = 5
    x_training, y_training = generate_sin(number_train)
    x_test = np.linspace(0, 1, number_test)
    y_test = np.sin(2 * pi * x_test)

    X_training = transform_vec_to_matrix(x_training, degree)
    X_test = transform_vec_to_matrix(x_test, degree)

    plt.figure(figsize=(20, 10))
    title = "degree: " + str(degree) + ", number_train: " + str(number_train) + ", number_test: " + str(number_test)
    plt.title(title)
    plt.subplot(231)
    plt.ylim(-1.5, 1.5)
    plt.scatter(x_training, y_training, facecolor="none", edgecolor="b", label="training data")
    plt.plot(x_test, y_test, "g", label=r"$\sin(2\pi x)$")
    plt.plot(x_test, predict(X_test, AnalyticalSolution.fitting(X_training, y_training)), "r",
             label="analytical solution")
    plt.title(title)
    plt.legend()

    # 找到最优的超参数
    alphaTestList = []
    alphaTrainList = []
    alphaRange = range(-50, 1)
    for alpha_exp in alphaRange:
        w = AnalyticalSolution.fitting_with_regularization(X_training, y_training, np.exp(alpha_exp))
        f_x = predict(X_test, w)
        alphaTestList.append(AnalyticalSolution.RMS_error(f_x, y_test))
        alphaTrainList.append(AnalyticalSolution.RMS_error(y_training, predict(X_training, w)))
    best_alpha = alphaRange[np.where(alphaTestList == np.min(alphaTestList))[0][0]]
    print("best alpha = %f, err = %f" % (best_alpha, np.min(alphaTestList)))
    annotate = r"$\alpha = e^{%f}}$" % best_alpha
    plt.subplot(232)
    plt.ylabel("error")
    plt.ylim(0, 1)
    plt.xlabel(r"$\ln \alpha$")
    plt.annotate(annotate, (-30, 0.8))
    plt.plot(alphaRange, alphaTestList, 'o-', mfc="none", mec="b", ms=5, label='Test')
    plt.plot(alphaRange, alphaTrainList, 'o-', mfc="none", mec="r", ms=5, label='Train')
    plt.legend()

    # 求带惩罚的解析解
    w = AnalyticalSolution.fitting_with_regularization(X_training, y_training, np.exp(best_alpha))
    print("w_analytical_with_regulation(Analytical solution):\n",
          w)

    annotate = r"$\lambda = e^{" + str(best_alpha) + r"}$"
    plt.subplot(233)
    plt.ylim(-1.5, 1.5)
    plt.scatter(x_training, y_training, facecolor="none",
                edgecolor="b", label="training data")
    plt.plot(x_test, predict(X_test, w),
             "r", label="analytical with regulation")
    plt.plot(x_test, y_test, "g", label=r"$\sin(2\pi x)$")
    plt.annotate(annotate, xy=(0.3, -0.5))
    plt.legend()

    # 带惩罚项的梯度下降
    w_gradient = GradientDescent.fitting(X_training, y_training, np.exp(best_alpha), rate=0.1, w0=np.zeros(degree + 1), epsilon=1e-6)

    print("w_gradient(Gradient descent):\n", w_gradient)

    plt.subplot(234)
    plt.ylim(-1.5, 1.5)
    plt.scatter(x_training, y_training, facecolor="none",
                edgecolor="b", label="training data")
    plt.plot(x_test, y_test, "g", label=r"$\sin(2\pi x)$")
    plt.plot(x_test, predict(X_test, w),
             "r", label="Analytical with regulation")
    plt.plot(x_test, predict(X_test, w_gradient), "c", label="Gradient descent")
    plt.legend()

    # 共轭梯度法
    w_conjugate = ConjugateGradient.fitting(X_training, y_training, 1e-6, np.zeros(degree + 1), np.exp(best_alpha))

    print("w_conjugate(Conjugate gradient):\n", w_conjugate)

    plt.subplot(235)
    plt.ylim(-1.5, 1.5)
    plt.scatter(x_training, y_training, facecolor="none",
                edgecolor="b", label="training data")
    plt.plot(x_test, y_test, "g", label=r"$\sin(2\pi x)$")
    plt.plot(x_test, predict(X_test, w),
             "r", label="Analytical regulation")
    plt.plot(x_test, predict(X_test, w_conjugate), "m",
             label="Conjugate gradient")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
