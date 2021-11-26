<br/>
<br/>
<div style="text-align: center;"> <span style="font-size: x-large; "> 哈尔滨工业大学计算机科学与技术学院 </span></div>
<br/>
<br/>
<div style="text-align: center;"> <span style="font-size: xxx-large; "> 实验报告 </span></div>
<br/>
<br/>
<br/>
<div style="text-align: center;"> <span style="font-size: x-large; "> 
课程名称：机器学习 <br/>
课程类型：必修  <br/>
实验题目：多项式拟合正弦曲线
</span></div>
<br/>
<br/>
<div style="text-align: center;"> <span style="font-size: medium; "> 学号：1190501001 </span></div>
<div style="text-align: center;"> <span style="font-size: medium; "> 姓名：李恩宇 </span></div>

# 一、实验目的

掌握最小二乘法求解（无惩罚项的损失函数）、掌握加惩罚项（2范数）的损失函数优化、梯度下降法、共轭梯度法、理解过拟合、克服过拟合的方法(如加惩罚项、增加样本)

# 二、实验要求及实验环境

## 实验要求

1. 生成数据，加入噪声；
2. 用高阶多项式函数拟合曲线；
3. 用解析解求解两种loss的最优解（无正则项和有正则项）
4. 优化方法求解最优解（梯度下降，共轭梯度）；
5. 用你得到的实验数据，解释过拟合。
6. 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。
7. 语言不限，可以用matlab，python。求解解析解时可以利用现成的矩阵求逆。梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例如pytorch，tensorflow的自动微分工具。

## 实验环境

- Windows 11
- Python 3.7.2
- NumPy 1.17.4
- Matplotlib 3.1.2

# 三、设计思想（本程序中的用到的主要算法及数据结构）

## 1 数据生成

利用 $\sin(2\pi x)$ 生成样本, 其中 $x$ 在分布在 $[0, 1]$ 之间, 对于每一个目标值 $t=sin(2\pi x)$ 增加一个 $(0, 0.5)$ 的高斯噪声.

## 2 用高阶多项式函数拟合曲线 (解析解)

利用多项式函数进行学习. 假设多项式阶数为 $m$, 则预测函数为

$$
f(x, w) = \sum_{i=0}^{m} w_0 x^i

$$

将参数向量化:

$$
\mathbf x_i

$$

$$
\mathbf X = \begin{vmatrix} 1 & x_1 & \cdots & x_{1}^m\\ 1 & x_2 & \cdots & x_2^m\\ \vdots & \vdots & \ddots & \vdots\\ 1 & x_n & \cdots & x_n^m \end{vmatrix}

$$

$$
\mathbf w = \begin{vmatrix} w_0 \\ w_1 \\ \vdots \\ w_m \end{vmatrix}

$$

$$
\mathbf y = \begin{vmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{vmatrix}

$$

则预测函数可写为

$$
y(\mathrm x, w)

$$

均方误差如下式

$$
E(w)

$$

# 四、实验结果与分析

# 五、结论

# 六、参考文献

# 七、附录：源代码（带注释）
