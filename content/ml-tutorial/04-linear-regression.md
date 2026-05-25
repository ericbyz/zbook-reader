## 线性回归：用一条直线预测未来

> **核心问题**：给你一堆数据点，你怎么画一条"最好"的直线穿过它们？怎么定义"最好"？计算机怎么自动找到这条线？

---

### 0. 本章导览

线性回归是机器学习中最简单也最重要的算法。它简单到你可以用手算完成全部计算，却又深刻到蕴含了几乎所有 ML 模型的核心思想——损失函数、梯度下降、过拟合与欠拟合、偏差-方差权衡。

学完本章，你将能够：

1. 手算一组 3 个数据点的线性回归（正规方程 + 梯度下降两种方法）
2. 从零实现正规方程和梯度下降，理解每一行代码的数学含义
3. 独立完成一个完整的回归项目：数据探索 → 训练 → 评估 → 残差诊断 → 模型解释
4. 解释 MSE、RMSE、MAE、$R^2$ 的公式含义和适用场景
5. 看懂残差图、Q-Q 图、VIF，判断模型假设是否满足
6. 说出线性回归的 5 个核心假设以及违反后的修正手段

> 本章目标 1200+ 行，建议分 3-4 次读完。**代码部分请务必亲手跑一遍，手算部分请拿纸笔跟着算。**

前置章节：数学基础六章 + [数据预处理](./03-data-preprocessing.md)
下一章：[正则化回归 — Ridge 与 Lasso](./05-regularized-regression.md)

---

### 1. 生活例子：用一条直线预测房价

你是一个房产中介的学徒。师傅给你一本手册，上面记录了最近成交的几套房子的面积和价格：

| 面积 (m²) | 成交价 (万元) |
|-----------|-------------|
| 50 | 150 |
| 80 | 220 |
| 120 | 310 |
| 150 | 380 |

师傅问你："如果下一套房子面积是 100m²，你估计能卖多少钱？"

你拿出一张格子纸，把 4 个点画上去——横轴是面积，纵轴是价格。你发现这些点大致沿一条直线排列。你拿起尺子，凭感觉画了一条直线穿过这些点。在横轴 100 的位置垂直向上找到直线上的点，读出来——大概是 260 万。

师傅点点头："不错。但你画的这条线，是最好的吗？"

这个问题，就是线性回归要回答的全部。

**你不是在"创造"规则——你是在从数据中"发现"模式。** 面积和价格之间存在某种线性关系：价格 ≈ 某个系数 × 面积 + 某个基础价。线性回归的任务就是找到这个系数和基础价的最优值。

---

### 2. 直观理解：什么是一根"最好"的直线？

**直觉理解：** 想象桌上散落一把米粒。你拿一根筷子放上去，希望这根筷子"穿过"所有米粒的正中央。竖着看，每个米粒到筷子的垂直距离就是**误差**。把所有误差的平方加起来——这个总和最小的筷子位置，就是**线性回归的解**。

为什么用**垂直**距离而不是水平距离？因为你关心的是"对于给定的面积 x，我的预测价 ŷ 和真实价 y 差多远"——这个误差是沿着 y 轴方向的。

现实中的例子无处不在：

| 场景 | 输入特征 $x$ | 预测目标 $y$ |
|------|-------------|-------------|
| 房价预测 | 房屋面积、房间数、楼层 | 房价 |
| 健康评估 | 身高 | 体重 |
| 营销分析 | 广告投放金额 | 销售额 |
| 农业 | 施肥量 | 作物产量 |
| 能源 | 室外温度 | 空调耗电量 |

**核心直觉**：线性回归假设 $x$ 和 $y$ 之间存在一条直线关系。模型"学习"的过程，就是找到这条直线的**斜率 $w$** 和**截距 $b$**。

---

### 3. 形式化定义

线性回归假设目标值 $y$ 是输入特征 $x$ 的线性组合：

$$
\hat{y} = w x + b \quad \text{(一元线性回归)}
$$

其中：
- $\hat{y}$（读作"y hat"）是模型的预测值
- $w$（weight，权重）决定直线的**斜率**：$x$ 每增加 1 个单位，$\hat{y}$ 预计变化 $w$ 个单位
- $b$（bias，偏置）决定直线的**截距**：当 $x = 0$ 时的预测值

推广到 $p$ 个特征的多远线性回归：

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_p x_p + b
$$

用向量形式简洁表达（把偏置 $b$ 写成 $w_0$，并对应一个恒为 1 的特征 $x_0$）：

$$
\hat{y} = \mathbf{w}^\mathsf{T}\mathbf{x} = \sum_{j=0}^{p} w_j x_j, \quad x_0 = 1
$$

用矩阵形式表达整个数据集（$n$ 个样本）：

$$
\hat{\mathbf{y}} = X\mathbf{w} =
\underbrace{\begin{bmatrix}
1 & x_{11} & \cdots & x_{1p} \\
1 & x_{21} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \cdots & x_{np}
\end{bmatrix}}_{n \times (p+1)}
\underbrace{\begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_p \end{bmatrix}}_{(p+1) \times 1}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3.5 * X.squeeze() + 20 + np.random.randn(100) * 3

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot(X, 3.5 * X + 20, 'r-', linewidth=2, label='真实线 y = 3.5x + 20')
plt.xlabel('房屋面积 (m²)')
plt.ylabel('房价 (万元)')
plt.title('线性回归：用一条直线拟合数据')
plt.legend()
plt.show()
```

**应用连接：** 线性回归的思维贯穿机器学习全程——逻辑回归的决策边界是一条直线，SVM 用超平面分割数据，神经网络的最后一层通常是线性全连接层。理解"线性的"和"非线性的"之间的转换，是进阶深度学习的关键。

---

### 4. 最小二乘法：怎么定义"好"？

#### 4a. 直觉理解

你怎么判断一根筷子放得"好不好"？量一量每个米粒到筷子的**垂直距离**，把这些距离**平方**后加起来——总和越小，筷子放得越好。

这些垂直距离叫**残差**（residual）：$e_i = y_i - \hat{y}_i$。

"平方和最小"——这就是**最小二乘法**（Least Squares）的名字由来。

#### 4b. 为什么是平方？为什么不是绝对值？

| 理由 | 解释 |
|------|------|
| **可微性** | 平方函数 $f(e) = e^2$ 处处光滑可导，绝对值 $f(e) = |e|$ 在 $e = 0$ 处不可导。可微意味着能用梯度下降优雅地求解。 |
| **惩罚离群点** | 离直线 10 的点，平方后损失贡献 100；离直线 2 的点，平方后贡献 4。模型会"特别在意"远处的点——这既是优点也是缺点。 |
| **几何优雅性** | 最小化平方误差等价于求解线性方程组的最小二乘解，有闭式公式（正规方程），这在数学上非常优美。 |
| **统计解释** | 若误差服从正态分布 $\varepsilon \sim N(0, \sigma^2)$，最小二乘解就是最大似然估计。这条为线性回归提供了严谨的统计基础。 |

#### 4c. 形式化定义

**均方误差**（Mean Squared Error, MSE）：

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        = \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - (w x_i + b)\bigr)^2
$$

用矩阵形式表达（把 $b$ 并入 $\mathbf{w}$）：

$$
J(\mathbf{w}) = \frac{1}{n} \|X\mathbf{w} - \mathbf{y}\|^2
              = \frac{1}{n} (X\mathbf{w} - \mathbf{y})^\mathsf{T}(X\mathbf{w} - \mathbf{y})
$$

这是关于 $\mathbf{w}$ 的**二次函数**（抛物线在高维空间的推广），其几何形状是一个开口向上的"碗"。它只有一个全局最小值——这意味着线性回归的优化问题一定有唯一最优解。

#### 4d. 三种常用损失函数对比

| 损失函数 | 公式 | 优点 | 缺点 |
|----------|------|------|------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 可微，有闭式解 | 对离群点敏感 |
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 对离群点鲁棒 | 在 $e=0$ 处不可微 |
| **Huber** | $\begin{cases}\frac{1}{2}e^2 & |e| \le \delta \\ \delta(|e| - \frac{1}{2}\delta) & |e| > \delta\end{cases}$ | 兼顾两者 | 多一个超参数 $\delta$ |

---

### 5. 手算示例：3个数据点完整求解

这是本章最重要的部分。拿一张纸、一支笔，跟着算一遍。算完之后，你会发现线性回归不过如此。

**数据：** 3套房子的面积和价格：

| 序号 | 面积 $x$ (m²) | 价格 $y$ (万元) |
|------|-------------|----------------|
| 1 | 1 | 2 |
| 2 | 2 | 3 |
| 3 | 3 | 5 |

> 为简化计算，面积以"百平方米"为单位。

#### 5a. 正规方程手算

**第 1 步：构造设计矩阵 $X$ 和目标向量 $\mathbf{y}$**

给每个样本加上一列 1（对应偏置项 $b$）：

$$
X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}, \quad
\mathbf{w} = \begin{bmatrix} b \\ w \end{bmatrix}
$$

**第 2 步：计算 $X^\mathsf{T}X$**

$$
X^\mathsf{T}X = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix}
               \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}
             = \begin{bmatrix} 1+1+1 & 1+2+3 \\ 1+2+3 & 1+4+9 \end{bmatrix}
             = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}
$$

**第 3 步：计算 $(X^\mathsf{T}X)^{-1}$**

对于 $2 \times 2$ 矩阵 $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$，其逆矩阵为：

$$
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

这里 $ad - bc = 3 \times 14 - 6 \times 6 = 42 - 36 = 6$。

$$
(X^\mathsf{T}X)^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}
                     = \begin{bmatrix} \frac{14}{6} & -1 \\[4pt] -1 & \frac{1}{2} \end{bmatrix}
                     = \begin{bmatrix} \frac{7}{3} & -1 \\[4pt] -1 & 0.5 \end{bmatrix}
$$

**第 4 步：计算 $X^\mathsf{T}\mathbf{y}$**

$$
X^\mathsf{T}\mathbf{y} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix}
                         \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}
                       = \begin{bmatrix} 1\times2 + 1\times3 + 1\times5 \\ 1\times2 + 2\times3 + 3\times5 \end{bmatrix}
                       = \begin{bmatrix} 10 \\ 2+6+15 \end{bmatrix}
                       = \begin{bmatrix} 10 \\ 23 \end{bmatrix}
$$

**第 5 步：求解 $\mathbf{w} = (X^\mathsf{T}X)^{-1} X^\mathsf{T}\mathbf{y}$**

$$
\mathbf{w} = \begin{bmatrix} \frac{7}{3} & -1 \\[4pt] -1 & \frac{1}{2} \end{bmatrix}
             \begin{bmatrix} 10 \\ 23 \end{bmatrix}
           = \begin{bmatrix} \frac{7}{3} \times 10 + (-1) \times 23 \\[4pt] (-1) \times 10 + \frac{1}{2} \times 23 \end{bmatrix}
           = \begin{bmatrix} \frac{70}{3} - 23 \\[4pt] -10 + 11.5 \end{bmatrix}
           = \begin{bmatrix} \frac{70-69}{3} \\[4pt] 1.5 \end{bmatrix}
           = \begin{bmatrix} \frac{1}{3} \\[4pt] 1.5 \end{bmatrix}
$$

**结果：** $b = \frac{1}{3} \approx 0.333$，$w = 1.5$。

**回归直线：** $\hat{y} = 1.5x + \frac{1}{3}$

**验证：**

| $x$ | 预测 $\hat{y}$ | 真实 $y$ | 残差 $e$ | $e^2$ |
|-----|---------------|----------|---------|-------|
| 1 | $1.5 \times 1 + 0.333 = 1.833$ | 2 | $-0.167$ | $0.028$ |
| 2 | $1.5 \times 2 + 0.333 = 3.333$ | 3 | $+0.333$ | $0.111$ |
| 3 | $1.5 \times 3 + 0.333 = 4.833$ | 5 | $+0.167$ | $0.028$ |

MSE $= \frac{0.028 + 0.111 + 0.028}{3} = \frac{0.167}{3} \approx 0.056$

#### 5b. 梯度下降手算

现在用梯度下降找同样的解。我们从 $\mathbf{w} = [0, 0]^\mathsf{T}$ 出发（$b = 0$, $w = 0$），学习率 $\eta = 0.02$。

**梯度公式回顾：**

$$
\frac{\partial J}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i), \qquad
\frac{\partial J}{\partial w} = \frac{2}{n}\sum_{i=1}^{n} x_i (\hat{y}_i - y_i)
$$

**迭代 0**（初始化）：$b = 0$, $w = 0$

| $x$ | $\hat{y}$ | $y$ | $\hat{y} - y$ | $x \times (\hat{y} - y)$ |
|-----|----------|-----|--------------|-------------------------|
| 1 | $0 \times 1 + 0 = 0$ | 2 | $-2$ | $1 \times (-2) = -2$ |
| 2 | $0 \times 2 + 0 = 0$ | 3 | $-3$ | $2 \times (-3) = -6$ |
| 3 | $0 \times 3 + 0 = 0$ | 5 | $-5$ | $3 \times (-5) = -15$ |

$$
J = \frac{4+9+25}{3} = 12.667
$$

$$
\frac{\partial J}{\partial b} = \frac{2}{3}(-2-3-5) = \frac{2}{3} \times (-10) = -6.667
$$

$$
\frac{\partial J}{\partial w} = \frac{2}{3}(-2-6-15) = \frac{2}{3} \times (-23) = -15.333
$$

更新：
$$
b \leftarrow 0 - 0.02 \times (-6.667) = 0.133
$$
$$
w \leftarrow 0 - 0.02 \times (-15.333) = 0.307
$$

**迭代 1**：$b = 0.133$, $w = 0.307$

| $x$ | $\hat{y}$ | $y$ | $\hat{y} - y$ | $x \times (\hat{y} - y)$ |
|-----|----------|-----|--------------|-------------------------|
| 1 | $0.307 \times 1 + 0.133 = 0.440$ | 2 | $-1.560$ | $-1.560$ |
| 2 | $0.307 \times 2 + 0.133 = 0.747$ | 3 | $-2.253$ | $-4.507$ |
| 3 | $0.307 \times 3 + 0.133 = 1.053$ | 5 | $-3.947$ | $-11.840$ |

$$
J = \frac{(-1.560)^2 + (-2.253)^2 + (-3.947)^2}{3} = \frac{2.434 + 5.078 + 15.576}{3} = 7.696
$$

$$
\frac{\partial J}{\partial b} = \frac{2}{3}(-1.560-2.253-3.947) = \frac{2}{3} \times (-7.760) = -5.173
$$

$$
\frac{\partial J}{\partial w} = \frac{2}{3}(-1.560-4.507-11.840) = \frac{2}{3} \times (-17.907) = -11.938
$$

更新：
$$
b \leftarrow 0.133 - 0.02 \times (-5.173) = 0.133 + 0.103 = 0.237
$$
$$
w \leftarrow 0.307 - 0.02 \times (-11.938) = 0.307 + 0.239 = 0.545
$$

**迭代 2**：$b = 0.237$, $w = 0.545$

| $x$ | $\hat{y}$ | $y$ | $\hat{y} - y$ | $x \times (\hat{y} - y)$ |
|-----|----------|-----|--------------|-------------------------|
| 1 | $0.545 \times 1 + 0.237 = 0.782$ | 2 | $-1.218$ | $-1.218$ |
| 2 | $0.545 \times 2 + 0.237 = 1.327$ | 3 | $-1.673$ | $-3.347$ |
| 3 | $0.545 \times 3 + 0.237 = 1.872$ | 5 | $-3.128$ | $-9.383$ |

$$
J = \frac{1.483 + 2.800 + 9.785}{3} = 4.689
$$

$$
\frac{\partial J}{\partial b} = \frac{2}{3} \times (-6.019) = -4.013
$$

$$
\frac{\partial J}{\partial w} = \frac{2}{3} \times (-13.948) = -9.299
$$

更新：
$$
b \leftarrow 0.237 - 0.02 \times (-4.013) = 0.317
$$
$$
w \leftarrow 0.545 - 0.02 \times (-9.299) = 0.731
$$

**汇总：**

| 迭代 | $b$ | $w$ | $J$ (MSE) |
|------|-----|-----|-----------|
| 0 | 0.000 | 0.000 | 12.667 |
| 1 | 0.133 | 0.307 | 7.696 |
| 2 | 0.237 | 0.545 | 4.689 |
| 3 | 0.317 | 0.731 | — |
| ... | ... | ... | ... |
| 正规方程解 | 0.333 | 1.500 | 0.056 |

**关键观察：**
- 梯度下降从 $(0, 0)$ 出发，每步都向最优解 $(0.333, 1.5)$ 靠近
- 损失 $J$ 单调下降（从 12.67 → 7.70 → 4.69 → ...）
- 前几步下降很快（因为梯度大），后面越来越慢（越接近最优解，梯度越小）
- 只用了 3 步就到达了 $b \approx 0.317$, $w \approx 0.731$，距离最优解还有距离——需要更多迭代

> **如果你想继续算第 4 步**：$b=0.317$, $w=0.731$ → 预测 $[1.048, 1.779, 2.510]$ → 残差 $[-0.952, -1.221, -2.490]$ → $\frac{\partial J}{\partial b} = \frac{2}{3} \times (-4.663) = -3.109$ → $\frac{\partial J}{\partial w} = \frac{2}{3} \times (-0.952-2.442-7.470) = \frac{2}{3} \times (-10.864) = -7.243$ → $b \leftarrow 0.317 + 0.062 = 0.379$, $w \leftarrow 0.731 + 0.145 = 0.876$。

---

### 6. 正规方程：完整推导与解析解

#### 6a. 推导过程（标量版）

对于一元线性回归 $J(w, b) = \frac{1}{n}\sum_{i=1}^n (y_i - wx_i - b)^2$，最小值处偏导为零：

$$
\frac{\partial J}{\partial b} = -\frac{2}{n}\sum_{i=1}^n (y_i - wx_i - b) = 0
\quad\Longrightarrow\quad
b = \bar{y} - w\bar{x}
$$

$$
\frac{\partial J}{\partial w} = -\frac{2}{n}\sum_{i=1}^n x_i(y_i - wx_i - b) = 0
\quad\Longrightarrow\quad
w = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{\mathrm{Cov}(x, y)}{\mathrm{Var}(x)}
$$

其中 $\bar{x} = \frac{1}{n}\sum x_i$, $\bar{y} = \frac{1}{n}\sum y_i$ 是均值。

这个结果有漂亮的几何解释：$w$ 是 $x$ 和 $y$ 的协方差除以 $x$ 的方差，也就是说——**$x$ 每偏离均值 1 个单位，$y$ 偏离均值多少个单位**。

#### 6b. 推导过程（矩阵版）

从矩阵形式的损失函数出发：

$$
J(\mathbf{w}) = \frac{1}{n} (X\mathbf{w} - \mathbf{y})^\mathsf{T}(X\mathbf{w} - \mathbf{y})
$$

展开平方（用矩阵乘法分配律）：

$$
\begin{aligned}
J(\mathbf{w}) &= \frac{1}{n} \Bigl[ (X\mathbf{w})^\mathsf{T}X\mathbf{w} - (X\mathbf{w})^\mathsf{T}\mathbf{y} - \mathbf{y}^\mathsf{T}X\mathbf{w} + \mathbf{y}^\mathsf{T}\mathbf{y} \Bigr] \\
              &= \frac{1}{n} \Bigl[ \mathbf{w}^\mathsf{T}X^\mathsf{T}X\mathbf{w} - 2\mathbf{y}^\mathsf{T}X\mathbf{w} + \mathbf{y}^\mathsf{T}\mathbf{y} \Bigr]
\end{aligned}
$$

> 合并中间两项的原因：$\mathbf{y}^\mathsf{T}X\mathbf{w}$ 是一个标量，标量的转置等于自身，所以 $(X\mathbf{w})^\mathsf{T}\mathbf{y} = \mathbf{w}^\mathsf{T}X^\mathsf{T}\mathbf{y}$ 和 $\mathbf{y}^\mathsf{T}X\mathbf{w}$ 是同一个标量的两种写法。

对 $\mathbf{w}$ 求导（使用矩阵求导公式 $\frac{\partial}{\partial \mathbf{w}}(\mathbf{w}^\mathsf{T}A\mathbf{w}) = 2A\mathbf{w}$ 当 $A$ 对称，$\frac{\partial}{\partial \mathbf{w}}(\mathbf{a}^\mathsf{T}\mathbf{w}) = \mathbf{a}$）：

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n} \bigl[ 2X^\mathsf{T}X\mathbf{w} - 2X^\mathsf{T}\mathbf{y} \bigr]
                                         = \frac{2}{n} X^\mathsf{T}(X\mathbf{w} - \mathbf{y})
$$

令导数为零：

$$
X^\mathsf{T}X\mathbf{w} - X^\mathsf{T}\mathbf{y} = 0
\;\Longrightarrow\;
X^\mathsf{T}X\mathbf{w} = X^\mathsf{T}\mathbf{y}
$$

这个方程叫**正规方程**（Normal Equation）。如果 $X^\mathsf{T}X$ 可逆，两边左乘 $(X^\mathsf{T}X)^{-1}$：

$$
\boxed{\mathbf{w}^* = (X^\mathsf{T}X)^{-1} X^\mathsf{T}\mathbf{y}}
$$

这就是线性回归的**闭式解**——不需要迭代，一次性算出最优参数。

#### 6c. 什么时候正规方程不能用？

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| $p > n$（特征数 > 样本数） | $X^\mathsf{T}X$ 是 $p \times p$ 的秩不足矩阵，不可逆 | 正则化（Ridge/Lasso）或降维 |
| 多重共线性（特征高度相关） | $X^\mathsf{T}X$ 接近奇异，数值不稳定 | 删除冗余特征、正则化 |
| 特征数太大（$p > 10^4$） | 矩阵求逆 $O(p^3)$ 太慢 | 改用梯度下降 |

> **sklearn 的做法**：`sklearn.LinearRegression` 内部不用 `np.linalg.inv`，而是用 `scipy.linalg.lstsq`（基于 SVD 分解的最小二乘求解），数值稳定性更高，甚至在 $X^\mathsf{T}X$ 奇异时也能给出一个解。

---

### 7. 梯度下降：完整推导与迭代求解

#### 7a. 直觉理解

你深夜站在一座山的半山腰，四周漆黑一片。你唯一能感知的就是脚下地面的坡度——你往最陡的下坡方向走一步，到了新位置再重新感知坡度，再走一步。重复下去，你最终会走到山谷底。

这座山就是**损失函数 $J(\mathbf{w})$ 的地形图**，你的每一步就是**梯度下降的一次迭代**。

#### 7b. 梯度公式的推导

对 MSE 求 $\mathbf{w}$ 的梯度我们已经得到了（见第 6 节矩阵推导）：

$$
\nabla_{\mathbf{w}} J = \frac{2}{n} X^\mathsf{T}(X\mathbf{w} - \mathbf{y})
$$

这个梯度的**几何含义**非常直观：
- $X\mathbf{w} - \mathbf{y}$ 是预测值与真实值的差，即**残差向量** $\mathbf{e}$（$n \times 1$）
- 左乘 $X^\mathsf{T}$ 将这些残差"投影"回特征空间——对第 $j$ 个特征，梯度分量是 $\frac{2}{n}\sum_i x_{ij} e_i$
- 含义：如果一个特征的值和残差"同方向"（正相关），该特征的权重应该减小（梯度为正）；反之则增大

#### 7c. 更新规则

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot \nabla_{\mathbf{w}} J(\mathbf{w}^{(t)})
$$

其中 $\eta$（学习率）控制每一步的大小。这条公式就是你第 5b 节手算时反复执行的规则。

#### 7d. 三种梯度下降变体

| 变体 | 每次更新用多少样本 | 梯度计算 | 特点 | 适用场景 |
|------|------------------|---------|------|---------|
| **批量 GD** (Batch) | 全部 $n$ 个 | $\frac{2}{n}\sum_{i=1}^n$ | 稳定，精确，但每步开销 $O(n)$ | 小数据集 ($n \le 10^4$) |
| **随机 GD** (SGD) | 1 个 | $2(\hat{y}_i - y_i) \mathbf{x}_i$ | 更新快，但震荡剧烈 | 在线学习、流式数据 |
| **小批量 GD** (Mini-batch) | $m$ 个（如 32） | $\frac{2}{m}\sum_{i \in \text{batch}}$ | 兼顾速度与稳定性，GPU 友好 | 深度学习标配 |

#### 7e. 学习率的选择

学习率是最重要的超参数。一个直观的类比：

| 学习率 | 类比 | 后果 |
|--------|------|------|
| 太小 | 每次只走 1 毫米 | 天亮了还没走到山谷（收敛极慢） |
| 刚好 | 正常步幅行走 | 平稳到达山谷 |
| 太大 | 每次跨 10 米 | 在谷底来回震荡甚至飞出去（损失不降反升、发散为 NaN） |

**实用建议：** 从 0.001 开始尝试，对数搜索：0.001 → 0.003 → 0.01 → 0.03 → 0.1。观察损失曲线——如果前 10 步损失基本不变，加大学习率；如果损失震荡或上升，减小学习率。

#### 7f. 学习率衰减

固定学习率存在两难——太大在谷底震荡，太小收敛太慢。一个好的策略是**逐步减小**学习率：

```python
eta = lr_init / (1 + decay_rate * iteration)
```

这样前几步大步快走（快速接近目标），后几步小步微调（精确到达谷底）。这是第 5b 节手算用了固定 0.02 的局限性——如果后期衰减学习率，收敛会更快更稳。

---

### 8. Python 实现：从零手写两种方法

**这一节的目的不是让你以后用这些代码替代 sklearn，而是让你真正理解每一步计算在做什么。**

#### 8a. 正规方程实现

```python
import numpy as np

class NormalEquation:
    """正规方程：一步求解线性回归的最优参数"""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NormalEquation':
        """训练：求解 w = (XᵀX)⁻¹Xᵀy"""
        n = X.shape[0]
        X_b = np.c_[np.ones((n, 1)), X]       # 加偏置列 (全 1)
        
        # 这是第 5a 节手算的核心：w* = (XᵀX)⁻¹ Xᵀy
        self.w_ = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测：ŷ = X w"""
        n = X.shape[0]
        X_b = np.c_[np.ones((n, 1)), X]
        return X_b @ self.w_

# 验证：用第 5 节的 3 个点测试
X_3 = np.array([[1.0], [2.0], [3.0]])
y_3 = np.array([2.0, 3.0, 5.0])

model_ne = NormalEquation().fit(X_3, y_3)
b, w = model_ne.w_[0], model_ne.w_[1]
print(f"正规方程结果：b = {b:.4f}, w = {w:.4f}")   # b = 0.3333, w = 1.5000
# 和第 5a 节手算结果完全一致！
```

#### 8b. 梯度下降实现

```python
class GradientDescentLR:
    """批量梯度下降：迭代求解线性回归的最优参数"""
    
    def __init__(self, lr: float = 0.01, n_iters: int = 1000, 
                 verbose: bool = False):
        self.lr = lr
        self.n_iters = n_iters
        self.verbose = verbose
        self.history = []   # 记录每步的损失
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescentLR':
        n, m = X.shape
        X_b = np.c_[np.ones((n, 1)), X]       # 加偏置列
        m_with_bias = m + 1
        
        self.w_ = np.random.randn(m_with_bias) * 0.01  # 小随机初始化
        
        for i in range(self.n_iters):
            # 前向传播：计算预测值
            y_pred = X_b @ self.w_
            
            # 计算残差
            error = y_pred - y                 # n × 1
            
            # 计算梯度：∇J = (2/n) Xᵀ e
            # 这就是第 5b 节手算的 (2/n) * sum(error) 和 (2/n) * sum(x * error)
            grad = (2.0 / n) * (X_b.T @ error)  # (m+1) × 1
            
            # 梯度下降更新：w ← w - η·∇J
            self.w_ -= self.lr * grad
            
            # 记录损失
            loss = np.mean(error ** 2)
            self.history.append(loss)
            
            if self.verbose and i % 100 == 0:
                print(f"迭代 {i:4d}: MSE = {loss:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_b = np.c_[np.ones((n, 1)), X]
        return X_b @ self.w_

# 验证：用第 5 节的 3 个点测试
model_gd = GradientDescentLR(lr=0.02, n_iters=200, verbose=True)
model_gd.fit(X_3, y_3)
b_gd, w_gd = model_gd.w_[0], model_gd.w_[1]
print(f"梯度下降结果：b = {b_gd:.4f}, w = {w_gd:.4f}")
# 接近正规方程的结果 (b=0.3333, w=1.5000)
```

#### 8c. 完整对比与可视化

```python
# 生成更大的测试数据
np.random.seed(42)
X_big = np.random.rand(100, 1) * 10
y_big = 3.5 * X_big.squeeze() + 20 + np.random.randn(100) * 3

# 两种方法对比
ne = NormalEquation().fit(X_big, y_big)
gd_models = {}
for lr in [0.001, 0.01, 0.1]:
    gd = GradientDescentLR(lr=lr, n_iters=500).fit(X_big, y_big)
    gd_models[lr] = gd

print(f"{'方法':<20} {'b':>10} {'w':>10}")
print(f"{'-'*42}")
print(f"{'正规方程':<20} {ne.w_[0]:>10.4f} {ne.w_[1]:>10.4f}")
for lr, gd in gd_models.items():
    print(f"{'GD (lr=' + str(lr) + ')':<20} {gd.w_[0]:>10.4f} {gd.w_[1]:>10.4f}")
print(f"{'真实参数':<20} {20.0:>10.4f} {3.500:>10.4f}")

# 可视化收敛曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# 左图：不同学习率的损失下降曲线
colors = {0.001: 'green', 0.01: 'blue', 0.1: 'red'}
for lr, gd in gd_models.items():
    ax1.plot(gd.history[:100], color=colors[lr], label=f'lr={lr}')
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('MSE (对数尺度)')
ax1.set_title('不同学习率的收敛速度对比')
ax1.legend()
ax1.set_yscale('log')

# 右图：数据与拟合线
ax2.scatter(X_big, y_big, alpha=0.5, edgecolors='k', linewidth=0.3)
X_line = np.linspace(0, 10, 100).reshape(-1, 1)
ax2.plot(X_line, ne.predict(X_line), 'b-', linewidth=2, label='正规方程')
ax2.plot(X_line, gd_models[0.01].predict(X_line), '--', color='orange', 
         linewidth=2, label='GD (lr=0.01)')
ax2.plot(X_line, 3.5 * X_line.squeeze() + 20, 'r-', linewidth=2, 
         alpha=0.5, label='真实线')
ax2.set_xlabel('房屋面积 (m²)')
ax2.set_ylabel('房价 (万元)')
ax2.legend()
ax2.set_title('正规方程 vs 梯度下降：拟合结果几乎重合')
plt.tight_layout()
plt.show()
```

**关键结论：** 两种方法得到几乎相同的结果（在小数点后几位可能有微小差异），但获得结果的方式完全不同——正规方程一步到位（$O(p^3)$ 后结束），梯度下降步步逼近（可能需要数百步）。

---

### 9. 使用 sklearn：一行代码搞定

理解了原理之后，用 sklearn 就是一件非常简单的事：

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_big, y_big, test_size=0.2, random_state=42)

# sklearn 一行训练
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测和评估
y_pred = lr.predict(X_test)

print(f"sklearn 结果: b = {lr.intercept_:.4f}, w = {lr.coef_[0]:.4f}")
print(f"测试集 MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"测试集 R²:  {r2_score(y_test, y_pred):.4f}")
```

> `sklearn.LinearRegression` 内部使用 SVD 分解求解最小二乘问题，比直接矩阵求逆更稳定。但它做的事情本质上和第 8a 节的 `NormalEquation` 是完全一样的——求解正规方程。

---

### 10. 多元回归：从直线到超平面

#### 10a. 直觉理解

房价不只取决于面积，还和房间数、楼层、地段、房龄有关。多元线性回归就是在多维空间中用一个**超平面**（直线在高维的推广）去拟合数据点——原理不变，只是从一根筷子变成了一块木板。

#### 10b. 形式化定义

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_p x_p + b
$$

每个 $w_j$ 的**经济含义**（"控制其他变量不变"的解释）：

> **在其他特征不变的前提下，$x_j$ 每增加 1 个单位，$y$ 预计变化 $w_j$ 个单位。**

这句话的"控制其他变量不变"是关键——它是多元回归比一元回归强大的原因。一元回归中 $x$ 和 $y$ 的关系可能被其他混淆因素污染，而多元回归通过同时放入多个特征，尽可能分离出每个特征的**独立**贡献。

#### 10c. 特征缩放为什么是必须的

**场景：** 特征"年龄"范围是 [0, 100]，特征"年收入"范围是 [0, 1,000,000]。在梯度下降中，$w_{\text{年龄}}$ 的梯度分量受 0-100 的尺度控制，$w_{\text{收入}}$ 的梯度分量受 0-1,000,000 的尺度控制——后者的梯度分量是前者的 10,000 倍。

**后果：** 损失函数的等高线被拉成极细长的椭圆（称为"病态条件"，ill-conditioned）。梯度下降会在其中**之字形震荡**——沿着收入方向近乎垂直地跳来跳去，而沿着年龄方向几乎不动。收敛极慢，甚至不收敛。

**教训：** 用梯度下降前，**必须**把所有特征缩放到相近的尺度。最常用的方法是**标准化**（Z-score）：$x' = \frac{x - \mu}{\sigma}$，使每个特征的均值为 0、标准差为 1。

> 正规方程理论上不需要缩放——但由于数值精度原因，缩放后通常也更稳定。

#### 10d. Python 实现

```python
from sklearn.preprocessing import StandardScaler

# 构造多元数据：房价 = 3×面积 + 50×房间数 + 15×楼层 + 100 + 噪音
np.random.seed(42)
n = 300
X_multi = np.column_stack([
    np.random.rand(n) * 100 + 50,    # 面积: 50-150 m²
    np.random.randint(1, 6, n),      # 房间数: 1-5 间
    np.random.randint(1, 30, n),     # 楼层: 1-30 层
])
y_multi = (3.0 * X_multi[:, 0] 
           + 50.0 * X_multi[:, 1] 
           + 15.0 * X_multi[:, 2] 
           + 100.0 + np.random.randn(n) * 20)

# 特征缩放（关键步骤！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

# 手写梯度下降训练
model_multi = GradientDescentLR(lr=0.01, n_iters=1000)
model_multi.fit(X_scaled, y_multi)

# sklearn 对比
lr_sk = LinearRegression().fit(X_scaled, y_multi)

# 查看学习到的系数
feature_names = ['面积(m²)', '房间数', '楼层']
print(f"{'特征':<10} {'手写GD':>10} {'sklearn':>10}")
print('-' * 32)
for i, name in enumerate(feature_names):
    # 注意：我们加了一列 bias，所以系数从索引 1 开始
    print(f"{name:<10} {model_multi.w_[i+1]:>10.4f} {lr_sk.coef_[i]:>10.4f}")
print(f"{'截距b':<10} {model_multi.w_[0]:>10.4f} {lr_sk.intercept_:>10.4f}")
```

**标准化系数的比较：** 原始系数 $w_j$ 的单位依赖 $x_j$ 的单位（"万元/m²"和"万元/层"无法直接比较哪个特征更重要）。标准化后，所有特征在同一尺度上——系数的大小可以直接比较，"绝对值越大的特征对预测越重要"（虽然这没有树模型的重要性度量严谨，但在解释性场景中非常实用）。

---

### 11. 模型评估指标：MSE / RMSE / MAE / $R^2$

#### 11a. 直觉理解

"这个模型好不好"需要量化。但不同指标回答不同问题：

| 你问的问题是 | 该看 |
|-------------|------|
| "误差的平方平均有多大？" | MSE |
| "预测大概差多少？"（和 y 同单位） | RMSE |
| "平均误差是多少？" | MAE |
| "模型解释了多少比例的变异？" | $R^2$ |

#### 11b. 形式化定义与手算

用第 5 节的 3 点数据来手算每个指标。已知 $\hat{\mathbf{y}} = [1.833, 3.333, 4.833]$, $\mathbf{y} = [2, 3, 5]$。

| 指标 | 公式 | 计算过程 | 结果 |
|------|------|---------|------|
| SSE | $\sum (y_i - \hat{y}_i)^2$ | $0.167^2 + (-0.333)^2 + 0.167^2$ | 0.167 |
| **MSE** | $\frac{1}{n}\mathrm{SSE}$ | $\frac{0.167}{3}$ | **0.056** |
| **RMSE** | $\sqrt{\mathrm{MSE}}$ | $\sqrt{0.056}$ | **0.236** |
| **MAE** | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | $\frac{0.167 + 0.333 + 0.167}{3}$ | **0.222** |

**$R^2$ 手算：**

$$
\bar{y} = \frac{2 + 3 + 5}{3} = 3.333
$$

$$
\mathrm{SST} = \sum (y_i - \bar{y})^2 = (2-3.333)^2 + (3-3.333)^2 + (5-3.333)^2
             = 1.778 + 0.111 + 2.778 = 4.667
$$

$$
R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}} = 1 - \frac{0.167}{4.667} = 1 - 0.036 = \mathbf{0.964}
$$

> $R^2 = 0.964$ 的直观解释：**模型解释了数据中 96.4% 的变异，剩下 3.6% 是模型无法捕捉的噪声。**

**修正 $R^2$（Adjusted $R^2$）：** 普通 $R^2$ 的缺点是——只要加更多特征，$R^2$ 永远不会下降。修正 $R^2$ 对特征数量做惩罚：

$$
R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
$$

对 3 点示例（$n=3$, $p=1$）：$R^2_{\text{adj}} = 1 - \frac{0.036 \times 2}{3-1-1} = 1 - \frac{0.072}{1} = 0.928$

#### 11c. 各指标总结

| 指标 | 公式 | 值域 | 选它的场景 |
|------|------|------|-----------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | $[0, +\infty)$ | 训练优化（可微） |
| RMSE | $\sqrt{\mathrm{MSE}}$ | $[0, +\infty)$ | 汇报误差（与 y 同单位） |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | $[0, +\infty)$ | 有离群点的数据 |
| $R^2$ | $1 - \frac{\mathrm{SSE}}{\mathrm{SST}}$ | $(-\infty, 1]$ | 向非技术人员解释 |
| Adjusted $R^2$ | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | $(-\infty, 1]$ | 比较不同特征数的模型 |

#### 11d. Python 计算

```python
def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    """手动计算所有回归评估指标"""
    n = len(y_true)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    
    return {
        'MSE':  sse / n,
        'RMSE': np.sqrt(sse / n),
        'MAE':  np.mean(np.abs(y_true - y_pred)),
        'R²':   1 - sse / sst,
        'Adj R²': 1 - (1 - (1 - sse / sst)) * (n - 1) / (n - p - 1),
    }

# 验证：用第 5 节的 3 点数据
y_pred_3 = np.array([1.833, 3.333, 4.833])
y_true_3 = np.array([2.0, 3.0, 5.0])
for k, v in compute_all_metrics(y_true_3, y_pred_3, p=1).items():
    print(f"{k:8s}: {v:.4f}")
```

---

### 12. 线性回归的假设与诊断

#### 12a. 直觉理解

线性回归不是一个"万能模型"。它对数据有 **5 个核心假设**。满足假设时，系数估计是最优无偏的（Gauss-Markov 定理）；违反假设时，结论就不那么可靠。

做一次好的回归分析 = **建模 + 诊断 + 修正**。

#### 12b. 五大假设速查表

| # | 假设 | 一句话含义 | 违反后果 | 诊断工具 | 修正手段 |
|---|------|-----------|---------|---------|---------|
| 1 | **线性性** (Linearity) | $y$ 和 $X$ 的关系是线性的 | 欠拟合，系数有偏 | 残差 vs 拟合值图 | 多项式特征、Box-Cox 变换、非线性模型 |
| 2 | **独立性** (Independence) | 样本之间相互独立 | 标准误估计错误，p 值不对 | Durbin-Watson 检验 | 时间序列模型、GLS |
| 3 | **等方差性** (Homoscedasticity) | 误差的方差是常数 | 置信区间不可靠 | 残差图的"喇叭口"形状 | 加权最小二乘、对数变换 $y$ |
| 4 | **误差正态性** (Normality) | 误差 $\sim N(0, \sigma^2)$ | 小样本下 t 检验和 F 检验不可靠 | Q-Q 图、Shapiro-Wilk 检验 | 增大样本量（CLT）、稳健回归 |
| 5 | **无多重共线性** (No Multicollinearity) | 特征之间不高度相关 | 系数不稳定、解释困难 | VIF > 10 或 5 | 删除冗余特征、正则化（Ridge） |

#### 12c. 样本量对假设的影响

- $n \le 30$：**正态性假设很关键**——t 检验和置信区间依赖于正态性
- $n \ge 100$：中心极限定理开始起作用——即使原始误差不正态，$\hat{w}$ 的抽样分布也渐近正态
- $n \ge 10{,}000$：除**线性性和独立性**外，其他假设违反的影响大大减弱

#### 12d. 诊断实战

```python
from scipy import stats

# 用多元数据做诊断演示
residuals = y_multi - lr_sk.predict(X_scaled)
y_pred_sk = lr_sk.predict(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 图 1：残差 vs 拟合值（检查线性性 & 等方差性）
# 好的图应该像"噪点"——以 0 为中心均匀散布，没有弯曲线趋势，没有喇叭口形状
axes[0, 0].scatter(y_pred_sk, residuals, alpha=0.5, edgecolors='k', linewidth=0.3)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].set_xlabel('拟合值')
axes[0, 0].set_ylabel('残差')
axes[0, 0].set_title('残差 vs 拟合值\n(检查: 线性性 + 等方差性)')

# 图 2：Q-Q 图（检查误差正态性）
# 如果点紧贴 45° 对角线→误差近似正态；偏离对角线→非正态
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q 图\n(检查: 误差正态性)')

# 图 3：残差直方图
axes[1, 0].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1, 0].set_xlabel('残差')
axes[1, 0].set_title('残差分布直方图')

# 图 4：残差 vs 某个特征（检查该特征的线性性）
axes[1, 1].scatter(X_multi[:, 0], residuals, alpha=0.5, edgecolors='k', linewidth=0.3)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1, 1].set_xlabel('面积 (m²)')
axes[1, 1].set_ylabel('残差')
axes[1, 1].set_title('特征 vs 残差\n(检查: 该特征的线性性)')

plt.tight_layout()
plt.show()

# VIF 计算（简化版：通过检查特征间相关系数矩阵推断）
corr_matrix = np.corrcoef(X_multi.T)
np.fill_diagonal(corr_matrix, 0)  # 忽略自相关
print(f"特征间最大相关系数: {np.max(np.abs(corr_matrix)):.3f}")
if np.max(np.abs(corr_matrix)) > 0.9:
    print("⚠️  存在高度相关的特征对，可能存在多重共线性问题")
```

> 正式计算 VIF 需要：$VIF_j = \frac{1}{1 - R_j^2}$，其中 $R_j^2$ 是把特征 $j$ 对其他所有特征做回归的 $R^2$。VIF > 10 通常被认为是严重多重共线性的标志。可以用 `statsmodels` 的 `variance_inflation_factor` 函数计算。

#### 12e. 违反假设的虚构案例

**违反线性性：** 真实关系是抛物线 $y = x^2$，你硬用直线拟合。残差 vs 拟合值图会呈现明显的 **U 形趋势**——这是"欠拟合"的典型信号。此时应添加多项式特征 $x^2$，或使用树模型。

**违反等方差性：** 房价数据中，低价房预测误差 ±5 万，高价房预测误差 ±50 万——残差散点图呈现**喇叭口**形状（拟合值越大，残差散布越宽）。系数估计仍然无偏，但标准误的估计会出错→置信区间和 p 值不可靠。对 $y$ 取对数通常能缓解。

**违反多重共线性：** 你把"面积(m²)"和"面积(平方英尺)"同时放进模型。两个特征携带的信息几乎完全重叠→ $X^\mathsf{T}X$ 接近奇异→系数变得极端不稳定（增加一个样本，$w$ 可能从 3 变成 -3）。这种情况下，系数的**具体值**没有意义，只有它们的**联合效果**有意义。

---

### 13. 完整实战：加州房价预测

把前面所有知识串联成一个完整的机器学习项目。

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─── 第 1 步：加载数据 & 探索 ───
housing = fetch_california_housing()
X_ca, y_ca = housing.data, housing.target
feature_names = housing.feature_names

print(f"📊 样本数: {len(X_ca):,}, 特征数: {X_ca.shape[1]}")
print(f"💰 目标值范围: [{y_ca.min():.2f}, {y_ca.max():.2f}] (单位: 10万美元)")
print(f"\n各特征与房价的相关系数:")
for name, corr in zip(feature_names, 
                       np.corrcoef(np.c_[X_ca.T, y_ca].T)[-1, :-1]):
    bar = '█' * int(abs(corr) * 20)
    print(f"  {name:<8} {corr:+.3f} {bar}")

# ─── 第 2 步：划分训练/测试集 ───
X_train, X_test, y_train, y_test = train_test_split(
    X_ca, y_ca, test_size=0.2, random_state=42)
print(f"\n📋 训练集: {len(X_train):,} 样本, 测试集: {len(X_test):,} 样本")

# ─── 第 3 步：特征缩放 ───
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── 第 4 步：训练（sklearn + 手写对比） ───
# sklearn 版
lr_sk = LinearRegression().fit(X_train_s, y_train)
y_pred_sk = lr_sk.predict(X_test_s)

# 手写梯度下降版
lr_manual = GradientDescentLR(lr=0.01, n_iters=2000)
lr_manual.fit(X_train_s, y_train)
y_pred_manual = lr_manual.predict(X_test_s)

# ─── 第 5 步：评估对比 ───
print(f"\n📈 测试集评估对比:")
print(f"{'指标':<8} {'sklearn':>10} {'手写GD':>10}")
print('-' * 30)
for name, fn in [('MSE', mean_squared_error),
                  ('RMSE', lambda y, yp: np.sqrt(mean_squared_error(y, yp))),
                  ('MAE', mean_absolute_error),
                  ('R²', r2_score)]:
    print(f"{name:<8} {fn(y_test, y_pred_sk):>10.4f} {fn(y_test, y_pred_manual):>10.4f}")

# ─── 第 6 步：诊断（残差分析） ───
residuals_test = y_test - y_pred_sk
y_pred_test = y_pred_sk

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 残差 vs 拟合值
axes[0].scatter(y_pred_test, residuals_test, alpha=0.3, s=10)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('预测房价')
axes[0].set_ylabel('残差')
axes[0].set_title('残差 vs 预测值')

# Q-Q 图
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q 图')

# 预测 vs 真实
axes[2].scatter(y_test, y_pred_test, alpha=0.3, s=10)
axes[2].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[2].set_xlabel('真实房价')
axes[2].set_ylabel('预测房价')
axes[2].set_title('预测 vs 真实')

plt.tight_layout()
plt.show()

# ─── 第 7 步：解释系数 ───
print(f"\n🔍 标准化系数（绝对值越大 → 对预测越重要）:")
for name, coef in sorted(zip(feature_names, lr_sk.coef_), 
                         key=lambda x: -abs(x[1])):
    direction = "↑" if coef > 0 else "↓"
    bar = '█' * int(abs(coef) * 10)
    print(f"  {name:<20} {coef:>8.4f} {direction}  {bar}")
print(f"  截距: {lr_sk.intercept_:.4f}")
```

#### 完整流程总结

```
加载数据
  ↓
探索性分析（特征分布、相关性、目标值范围）
  ↓
切分训练/测试集
  ↓
特征缩放（StandardScaler）
  ↓
训练模型（sklearn 或手写）
  ↓
预测 & 评估（MSE, RMSE, MAE, R²）
  ↓
残差诊断（检查 5 大假设）
  ↓
解释系数 → 业务结论
```

---

### 14. 常见误区：新手最容易踩的 10 个坑

| # | 坑 | 现象 | 正确做法 |
|---|----|------|---------|
| 1 | **忘记特征缩放就用梯度下降** | 损失不降反升，或出现 NaN | 梯度下降前**必须** `StandardScaler` |
| 2 | **学习率太大** | 损失震荡、爆炸、发散为 NaN | 从 0.001 开始试，对数搜索 |
| 3 | **学习率太小** | 迭代几千次损失几乎不动 | 逐步加大，直到损失开始明显下降 |
| 4 | **在训练集上评估模型** | $R^2 = 0.99$，上线后 $R^2 = 0.7$ | 必须在**测试集**上评估，用 `train_test_split` |
| 5 | **不画残差图** | 误以为模型很好，实则违反假设 | 每次建模后必须画残差 vs 拟合值图 |
| 6 | **用 $R^2$ 跨数据集比较** | "在我的数据上 $R^2=0.9$，你的 $R^2=0.8$" | $R^2$ 依赖数据方差，不可跨数据集比较——用 RMSE/MAE 跨数据比较 |
| 7 | **直接解释未标准化的系数** | "楼层系数 15，面积系数 3，所以楼层比面积重要 5 倍" | 标准化后再比较系数大小 |
| 8 | **正规方程在小样本 + 多特征场景中标度问题** | $p > n$ 时程序崩溃 | 检查 $n$ 和 $p$ 的关系，$p > n$ 时用梯度下降 + 正则化 |
| 9 | **忽略多重共线性** | 系数符号和直觉相反（比如面积越大房价越低） | 检查 VIF，删除高度相关的特征 |
| 10 | **把相关系数当成因果关系** | "广告费相关系数 +0.8，所以增加广告费就能增加销售" | 相关性 ≠ 因果——可能有反向因果或遗漏变量 |

---

### 15. 思考题

#### 15a. 题目

**基础题（1-5）**

1. **手算验证：** 用正规方程手算以下 4 个点的线性回归：$(0, 1), (1, 2), (2, 3), (3, 5)$，写出完整的矩阵运算步骤。

2. **梯度方向：** 在手算的第 5b 节中，为什么前几步 $\frac{\partial J}{\partial w}$ 是负数？这意味着 $w$ 应该增大还是减小？

3. **学习率实验：** 如果你把第 5b 节的学习率从 0.02 改成 0.1，计算第 1 步更新后的 $b$ 和 $w$。会发生什么？

4. **指标计算：** 一个模型在 5 个测试点上的预测值为 $[2.1, 4.2, 6.0, 8.1, 10.3]$，真实值为 $[2, 4, 6, 8, 10]$。计算 MSE、RMSE、MAE 和 $R^2$。

5. **正规方程的适用条件：** 在什么情况下你不能用正规方程求解线性回归？（说出至少三种情况）

**进阶题（6-8）**

6. **梯度消失与收敛：** 第 5b 节手算中，你可以看到梯度的绝对值随迭代逐步减小。为什么？这导致什么问题？学习率衰减如何缓解？

7. **残差图诊断：** 你拿到一个回归模型，残差 vs 拟合值图呈现以下形态，分别说明问题是什么，以及你该怎么修正：
   - (a) U 形曲线
   - (b) 喇叭口形状（拟合值越大残差越分散）
   - (c) 残差整体偏离 0（大部分残差为正）

8. **$R^2$ 的陷阱：** 你在一条回归直线上加了一个完全随机生成的特征"幸运数字"（取值范围 0-100），$R^2$ 从 0.750 变成了 0.751。请问这意味着模型变好了吗？修正 $R^2$ 会怎么变？（假设 $n = 100$, 原 $p = 3$）

**综合题（9-10）**

9. **完整推导：** 从 MSE 损失函数 $J(w, b) = \frac{1}{n}\sum(y_i - wx_i - b)^2$ 出发，通过求偏导令其为零，推导出一元线性回归系数的标量公式：$w = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$ 和 $b = \bar{y} - w\bar{x}$。

10. **实战分析：** 以下是一个房价模型的标准化系数输出。请分析每个系数的大小和符号是否合理，并指出可能存在多重共线性的特征对，给出你的理由。

```
特征              标准化系数
面积(m²)          +0.52
房间数            +0.38
总面积(平方英尺)   +0.49
房龄(年)          -0.15
到最近学校距离(km) -0.22
邮编_encoded      +0.01
```

---

#### 15b. 解答

<details>
<summary><b>点击展开完整解答</b></summary>

**第 1 题：**

$$
X = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix} 1 \\ 2 \\ 3 \\ 5 \end{bmatrix}
$$

$$
X^\mathsf{T}X = \begin{bmatrix} 4 & 6 \\ 6 & 14 \end{bmatrix}, \quad
X^\mathsf{T}\mathbf{y} = \begin{bmatrix} 11 \\ 23 \end{bmatrix}
$$

$(X^\mathsf{T}X)^{-1} = \frac{1}{4 \times 14 - 6 \times 6}\begin{bmatrix}14 & -6 \\ -6 & 4\end{bmatrix} = \frac{1}{20}\begin{bmatrix}14 & -6 \\ -6 & 4\end{bmatrix}$

$\mathbf{w} = \frac{1}{20}\begin{bmatrix}14 & -6 \\ -6 & 4\end{bmatrix}\begin{bmatrix}11 \\ 23\end{bmatrix} = \frac{1}{20}\begin{bmatrix}154-138 \\ -66+92\end{bmatrix} = \frac{1}{20}\begin{bmatrix}16 \\ 26\end{bmatrix} = \begin{bmatrix}0.80 \\ 1.30\end{bmatrix}$

回归直线：$\hat{y} = 1.3x + 0.8$

**第 2 题：** 前几步预测值都远小于真实值，所以残差全部为负。梯度公式 $\frac{\partial J}{\partial w} = \frac{2}{n}\sum x_i(\hat{y}_i - y_i)$ 中，$\hat{y}_i - y_i$ 全部为负，再乘以正的 $x_i$，求和仍是负数。负梯度意味着 $w$ 的负方向是下降方向→$w$ 应该**增大**（负梯度 × 负学习率 = 正增量）。

**第 3 题：** 迭代 0: $b=0, w=0$, $\frac{\partial J}{\partial b} = -6.667$, $\frac{\partial J}{\partial w} = -15.333$

$b \leftarrow 0 - 0.1 \times (-6.667) = 0.667$
$w \leftarrow 0 - 0.1 \times (-15.333) = 1.533$

更新后 $b=0.667$, $w=1.533$，已经非常接近最优解 $(0.333, 1.500)$——但第二步就会**过冲**：

第二步预测: $[2.200, 3.733, 5.267]$，残差: $[+0.200, +0.733, +0.267]$

$\frac{\partial J}{\partial b} = \frac{2}{3} \times 1.200 = 0.800$（变正了！）
$\frac{\partial J}{\partial w} = \frac{2}{3}(0.200+1.467+0.800) = 1.644$（也变正了！）

$b \leftarrow 0.667 - 0.1 \times 0.800 = 0.587$
$w \leftarrow 1.533 - 0.1 \times 1.644 = 1.369$

> 学习率太大→"冲过了"最优解→残差变正→梯度变号→往回走→可能来回震荡。

**第 4 题：**

| $i$ | $y_i$ | $\hat{y}_i$ | $e_i$ | $e_i^2$ | $|e_i|$ |
|-----|-------|------------|-------|---------|--------|
| 1 | 2 | 2.1 | -0.1 | 0.01 | 0.1 |
| 2 | 4 | 4.2 | -0.2 | 0.04 | 0.2 |
| 3 | 6 | 6.0 | 0.0 | 0.00 | 0.0 |
| 4 | 8 | 8.1 | -0.1 | 0.01 | 0.1 |
| 5 | 10 | 10.3 | -0.3 | 0.09 | 0.3 |

$\mathrm{MSE} = \frac{0.01+0.04+0+0.01+0.09}{5} = \frac{0.15}{5} = 0.03$

$\mathrm{RMSE} = \sqrt{0.03} \approx 0.173$

$\mathrm{MAE} = \frac{0.1+0.2+0+0.1+0.3}{5} = \frac{0.7}{5} = 0.14$

$\bar{y} = 6$, $\mathrm{SST} = 16+4+0+4+16 = 40$

$R^2 = 1 - \frac{0.15}{40} = 1 - 0.00375 = 0.996$

**第 5 题：**
1. **特征数 > 样本数**（$p > n$）：$X^\mathsf{T}X$ 是 $p \times p$ 矩阵但秩 $\le n < p$→不可逆
2. **多重共线性**：某些特征是其他特征的线性组合→ $X^\mathsf{T}X$ 奇异
3. **特征数太大**（$p > 10^4$）：矩阵求逆 $O(p^3)$ → 计算时间不可接受
4. **内存不够**：存储 $p \times p$ 矩阵需要 $O(p^2)$ 内存

**第 6 题：** 梯度 $|\nabla J|$ 随迭代减小是因为越接近最优解，残差 $|\hat{y}_i - y_i|$ 越小→梯度分量 $\frac{2}{n}\sum x_i(\hat{y}_i - y_i)$ 越小。这导致"最后 1% 的路要走 90% 的时间"——越接近终点，每步前进的距离越小。

学习率衰减通过逐步降低 $\eta$ 来补偿：前期大 $\eta$ 大步快走，后期小 $\eta$ 小步微调——使得整个过程的实际步长维持在合理范围。常用衰减策略：$\eta_t = \frac{\eta_0}{1 + \lambda t}$ 或 $\eta_t = \eta_0 \times 0.95^{t/100}$。

**第 7 题：**
- **(a) U 形曲线：** 违反线性性假设——$x$ 和 $y$ 之间存在非线性关系。修正：添加多项式特征（$x^2, x^3$），对 $x$ 或 $y$ 做 Box-Cox 变换，或改用非线性模型（树模型、GAM）。
- **(b) 喇叭口形状：** 违反等方差性（异方差）——不同预测值区间的误差方差不同。修正：对 $y$ 做对数变换（$\log(y)$），或使用加权最小二乘（WLS），为大方差区域的样本分配更小的权重。
- **(c) 残差整体偏离 0：** 模型系统性高估或低估——可能是缺失了重要特征，或偏置项被错误约束。修正：添加缺失的特征，或检查截距项是否被正则化误约束。

**第 8 题：** 模型没有变好。$R^2$ 从 0.750 涨到 0.751 是因为公式特性——$R^2$ 永远不会因增加特征而下降（即使新增的特征是随机噪声）。

计算修正 $R^2$：
原模型：$R^2_{\text{adj}} = 1 - \frac{(1-0.750)(100-1)}{100-3-1} = 1 - \frac{0.250 \times 99}{96} = 1 - 0.258 = 0.742$
新模型：$R^2_{\text{adj}} = 1 - \frac{(1-0.751)(100-1)}{100-4-1} = 1 - \frac{0.249 \times 99}{95} = 1 - 0.259 = 0.741$

修正 $R^2$ 反而**下降了**（0.742 → 0.741），正确反映了新增特征是无用噪声这一事实。

**第 9 题：**

对 $J(w, b) = \frac{1}{n}\sum(y_i - wx_i - b)^2$ 求偏导：

$$
\frac{\partial J}{\partial b} = \frac{1}{n}\sum 2(y_i - wx_i - b) \cdot (-1) = -\frac{2}{n}\sum(y_i - wx_i - b)
$$

令其为零：

$$
\sum(y_i - wx_i - b) = 0 \;\Longrightarrow\; \sum y_i - w\sum x_i - nb = 0 \;\Longrightarrow\; \bar{y} - w\bar{x} - b = 0
$$

$$
\boxed{b = \bar{y} - w\bar{x}}
$$

对 $w$ 求偏导：

$$
\frac{\partial J}{\partial w} = \frac{1}{n}\sum 2(y_i - wx_i - b) \cdot (-x_i) = -\frac{2}{n}\sum x_i(y_i - wx_i - b)
$$

令其为零，代入 $b = \bar{y} - w\bar{x}$：

$$
\begin{aligned}
\sum x_i(y_i - wx_i - \bar{y} + w\bar{x}) &= 0 \\
\sum x_i y_i - w\sum x_i^2 - \bar{y}\sum x_i + w\bar{x}\sum x_i &= 0 \\
\sum x_i y_i - w\sum x_i^2 - n\bar{x}\bar{y} + wn\bar{x}^2 &= 0
\end{aligned}
$$

整理：

$$
\begin{aligned}
w(\sum x_i^2 - n\bar{x}^2) &= \sum x_i y_i - n\bar{x}\bar{y} \\
w &= \frac{\sum x_i y_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2}
\end{aligned}
$$

注意到恒等式 $\sum(x_i - \bar{x})(y_i - \bar{y}) = \sum x_i y_i - n\bar{x}\bar{y}$ 且 $\sum(x_i - \bar{x})^2 = \sum x_i^2 - n\bar{x}^2$：

$$
\boxed{w = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{\mathrm{Cov}(x, y)}{\mathrm{Var}(x)}}
$$

**第 10 题：**

系数分析：

- **面积 (+0.52)**：合理——面积大，房价通常高。符号正确。
- **房间数 (+0.38)**：合理——房间多，房价高。但系数比面积低，可能因为面积已经捕获了大部分"大小"信息。
- **总面积(平方英尺) (+0.49)**：**极度可疑！** "面积(m²)"和"总面积(平方英尺)"测的是同一件事（房屋大小），只是单位不同（1 m² ≈ 10.76 ft²）。二者理论上完全线性相关。这就是**多重共线性的典型案例**。应该删除其中一个。
- **房龄 (-0.15)**：合理——房子越老，价格越低。
- **到最近学校距离 (-0.22)**：合理——离学校越远，价格越低（好学区溢价）。
- **邮编_encoded (+0.01)**：系数非常小——可能这个特征编码方式有问题（邮编是类别型特征，不应该简单做数值编码），或者它携带的信息已经被其他特征（如距离学校）解释掉了。

特征对提示：**面积(m²)** 和 **总面积(平方英尺)** 高度共线（VIF 会远超 10），应删除一个（保留"面积(m²)"，因为它和房间数等其他特征的单位体系一致）。

</details>

---

### 16. 本章小结

你从"用一条直线预测房价"这个生活场景出发，系统地走完了线性回归的完整学习路径：

```
生活例子（预测房价）
    ↓
直观理解（筷子穿过米粒）
    ↓
形式化定义（ŷ = wx + b）
    ↓
手算（3 个点 × 2 种方法，每一步都写下来）
    ↓
Python 实现（从零手写 + sklearn）
    ↓
多元回归（从直线到超平面 + 特征缩放）
    ↓
评估指标（MSE / RMSE / MAE / R² 的含义和手算）
    ↓
假设诊断（残差图 / Q-Q / VIF）
    ↓
完整实战（加州房价预测）
    ↓
常见误区（10 个新手踩过的坑）
    ↓
思考题（10 道 + 完整解答）
```

**线性回归看似简单，却蕴含了机器学习的全部核心思想：** 损失函数定义"好"的标准，参数是模型从数据中学到的内容，梯度下降是迭代优化的通用范式，评估指标告诉你模型靠不靠谱，假设诊断为模型的可靠性提供统计担保。

理解线性回归 ≠ 只会调 `sklearn.LinearRegression()`，而是理解了**所有 ML 模型的底层逻辑**。逻辑回归的 sigmoid 后面接的是一个线性组合；神经网络的最后一层通常是线性全连接层；SVM 的核函数本质上是把"非线性的"映射成"高维线性的"。

**你已经站在了 ML 大厦的入口。后面的每一章，都是在这座地基上添砖加瓦。**

---

下一步：[正则化回归 — Ridge 与 Lasso](./05-regularized-regression.md) — 当特征太多、数据太小时，如何让线性回归不"过拟合"？
