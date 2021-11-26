import numpy as np


class AnalyticalSolution:
    @staticmethod
    def fitting(X: np.ndarray, y):
        """
        无惩罚项解析解
        :param X: 训练集矩阵
        :param y: 真实标记
        :return: 参数向量
        """
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    @staticmethod
    def fitting_with_regularization(X: np.ndarray, y, alpha):
        """
        带惩罚项解析解
        :param X: 训练集矩阵
        :param y: 真实标记
        :param alpha: 惩罚项系数
        :return: 参数向量
        """
        return np.linalg.solve(X.T @ X + alpha * np.identity(X.shape[1]), X.T @ y)

    @staticmethod
    def RMS_error(x: np.ndarray, y: np.ndarray):
        return np.sqrt(np.mean(np.square(x - y)))


class GradientDescent:
    @classmethod
    def __loss(cls, X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float):
        tmp = X @ w - y
        return 0.5 * np.mean(tmp.T @ tmp + alpha * np.inner(w, w))

    @classmethod
    def __derivative(cls, X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float):
        return X.T @ X @ w + alpha * w - X.T @ y

    @classmethod
    def fitting(cls, X: np.ndarray, y: np.ndarray, alpha: float, rate: float, w0: np.ndarray, epsilon: float):
        """
        梯度下降求解
        :param X: 训练集矩阵
        :param y: 真实标记
        :param alpha: 超参数
        :param rate: 学习率
        :param w0: 初始解 (一般为全 0)
        :return: 参数向量
        """
        lst_loss = cls.__loss(X, y, w0, alpha)
        k = 0
        lst_w = w0
        while True:
            w = lst_w - rate * cls.__derivative(X, y, lst_w, alpha)
            loss = cls.__loss(X, y, w, alpha)

            if np.abs(loss - lst_loss) < epsilon:
                return w
            else:
                k += 1
                if loss > lst_loss:
                    rate *= 0.5
                lst_loss = loss
                lst_w = w


class ConjugateGradient:
    @classmethod
    def fitting(cls, X: np.ndarray, y: np.ndarray, epsilon: float, w0: np.ndarray, alpha=1e-6):
        A = X.T @ X + np.identity(X.shape[1]) * alpha
        b = X.T @ y
        r_0 = b - A @ w0
        w = w0
        p = r_0
        k = 0
        print(p.size)
        print(X.size)
        while True:
            k = k + 1
            alpha = np.linalg.norm(r_0) ** 2 / (np.transpose(p) @ A @ p)
            w = w + alpha * p
            r = r_0 - alpha * A @ p  # 当前的残差
            # print(k, r)
            if np.linalg.norm(r) < epsilon:
                break
            beta = np.linalg.norm(r) ** 2 / np.linalg.norm(r_0) ** 2
            p = r + beta * p  # 下次的搜索方向
            r_0 = r
        return w
