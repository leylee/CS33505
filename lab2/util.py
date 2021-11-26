import numpy as np
# import cupy as np

class GradientDescent:
    """ 梯度下降法 """
    @classmethod
    def __sigmod(cls, z):
        return 1.0 / (1.0 + np.exp(z))

    @classmethod
    def __loss(cls, beta_t, x, y, hyper):
        ans = 0
        for i in range(len(x)):
            ans += (-y[i] * beta_t @ x[i] + np.log(1 + np.exp(beta_t @ x[i])))
        return (ans + 0.5 * hyper * beta_t @ beta_t) / len(x)
        # 此处m用于平衡loss 没有其他作用

    @classmethod
    def __derivative_beta(cls, beta_t, x, y, hyper):
        ans = np.zeros(len(x[0]))
        for i in range(len(x)):
            ans += (x[i] * (y[i] -
                                 (1.0 - cls.__sigmod(beta_t @ x[i]))))
        return (-1 * ans + hyper * beta_t) / len(x)

    @classmethod
    def fitting(cls, x, y, beta_0, hyper=0, rate=0.5, delta=1e-6):
        loss0 = cls.__loss(beta_0, x, y, hyper)
        k = 0
        beta = beta_0
        while True:
            beta_t = beta - rate * cls.__derivative_beta(beta, x, y, hyper)
            loss = cls.__loss(beta_t, x, y, hyper)
            if np.abs(loss - loss0) < delta:
                break
            else:
                k = k + 1
                print(k)
                print("loss:", loss)
                if loss > loss0:
                    rate *= 0.5
                # 进行学习率的衰减 得到的结果不正确??
                # 修改后答案正确 原因可能为hyper取值过大
                loss0 = loss
                beta = beta_t
        return beta

