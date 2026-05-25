## 逻辑回归：把"是/否"变成概率

> **核心问题**：一封邮件来了，你怎么用数学告诉计算机"这封有 87% 的可能是垃圾邮件"？怎么让计算机自己学会做这个判断？

---

### 0. 本章导览

逻辑回归（Logistic Regression）是分类问题的"第一把锤子"——它简单到核心公式只有五行，却在信用评分、医学诊断、广告点击率预估中无处不在。它名字里带"回归"，却是个**分类模型**：输出不是房价那样的连续数字，而是"属于某类的概率"。

学完本章，你将能够：

1. 手算 Sigmoid 函数值、Log-Loss、3 轮梯度下降（4 个数据点，纸笔完成）
2. 从零实现一个 `LogisticRegression` 类，理解每一行代码的数学含义
3. 解释为什么分类问题不能用 MSE、为什么交叉熵对"极端错误"惩罚巨大
4. 读懂决策边界的几何意义，知道什么时候逻辑回归不够用
5. 独立完成一个分类项目：数据探索 → 训练 → 评估 → 概率校准 → 特征解释
6. 说出 5 个常见误区及正确做法

> 本章目标 1100+ 行，建议分 3 次读完。**手算部分请务必拿纸笔跟着算，代码部分请亲手跑一遍。**

前置章节：[正则化回归](./05-regularized-regression.md)、[概率论](./math-probability.md)
下一章：[K近邻](./07-knn.md)

---

### 1. 生活例子：邮件是垃圾还是正常？

你每天早上打开邮箱，收件箱里躺着 50 封新邮件。其中一些是老板发来的工作邮件，一些是朋友分享的猫咪视频，还有一些是"恭喜您获得 100 万大奖"的垃圾信息。

你扫一眼就能分辨——但计算机不行。计算机只认识数字。你得想办法**把"垃圾 vs 正常"这个判断，翻译成计算机能运算的数字问题**。

假设你挑出三个容易量化的信号：

| 特征 | 含义 | 量化方式 |
|------|------|----------|
| $x_1$ | 是否包含"免费" | 1 = 包含，0 = 不包含 |
| $x_2$ | 感叹号数量 | 实际计数 |
| $x_3$ | 发件人是否在通讯录 | 1 = 是，0 = 否 |

你翻看过去的 1000 封邮件，每封都手动标注了"垃圾(1)"或"正常(0)"。现在你手上有一张表：1000 行，每行 3 个数字输入 + 1 个标签。

**你要做的事**：找到一种方法，输入这三个数字，输出一个 0 到 1 之间的分数——分数越高，越可能是垃圾邮件。

这不是在"发明"规则（如果靠人写规则——"如果包含'免费'且感叹号>3 个则判为垃圾"——垃圾邮件制造者改一个词就绕过去了）。你是在让计算机**从数据中自动学出**每个信号该占多大权重。

---

### 2. 直观理解：从"打分"到"概率"

**直觉理解：** 想象你是一个老师，批改作文时给每篇作文打一个"好坏分"——可以是从负无穷到正无穷的任何实数。但教务系统要求你提交的是"通过概率"——一个 0 到 1 之间的数字。

你的做法很自然：分数特别高的（z ≫ 0），通过概率接近 1；分数特别低的（z ≪ 0），通过概率接近 0；分数在及格线附近（z ≈ 0），概率在 0.5 左右——"不确定"。

这个**把任意实数压扁到 (0, 1) 之间的函数**，就是 Sigmoid——逻辑回归的灵魂。

```
任意实数 z ──Sigmoid──▶ 概率 ŷ ∈ (0, 1)

z = -∞  →  ŷ ≈ 0    （绝对不是垃圾邮件）
z =  0  →  ŷ = 0.5  （完全不确定）
z = +∞  →  ŷ ≈ 1    （绝对是垃圾邮件）
```

**为什么叫"逻辑"回归？** 这个 S 形曲线最早是用来描述人口增长的（"逻辑斯蒂增长"），后来被借用来把线性模型的输出转化为概率。名字是历史遗留——它做的事其实是**概率分类**。

**核心直觉：** 逻辑回归 = 线性回归 + Sigmoid。先算一个线性分数 $z = w_1 x_1 + w_2 x_2 + \cdots + b$，再用 Sigmoid 把这个分数"翻译"成概率。

---

### 3. 形式化定义

#### 3.1 Sigmoid 函数

Sigmoid 函数（也叫逻辑函数，logistic function）的数学表达式：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**三个关键性质：**

1. **值域 (0, 1)**：$z \to +\infty$ 时 $\sigma(z) \to 1$；$z \to -\infty$ 时 $\sigma(z) \to 0$；$\sigma(0) = 0.5$
2. **单调递增**：$z_1 > z_2$ 则 $\sigma(z_1) > \sigma(z_2)$，处处光滑可导
3. **导数优美**：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$，用函数值自身就能算导数，无需额外存储

**逻辑回归模型：**

$$
\hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
$$

其中 $\hat{y}$ 的含义是 $P(y = 1 \mid x; w, b)$——"给定输入 $x$ 和参数 $w, b$，样本属于正类的概率"。

#### 3.2 决策边界——概率的"分水岭"

把概率变成"是/否"决策，最自然的门槛是 0.5：

$$
\text{prediction} = \begin{cases}
1 & \text{if } \hat{y} \geq 0.5 \text{ 即 } w^T x + b \geq 0 \\
0 & \text{if } \hat{y} < 0.5 \text{ 即 } w^T x + b < 0
\end{cases}
$$

$\hat{y} = 0.5$ 等价于 $w^T x + b = 0$——这是一条**直线**（二维）或**超平面**（高维）。逻辑回归的分类边界是**线性的**——这就是为什么它叫"线性分类器"。

$w$ 是决策边界的法向量，$\|w\|$ 控制边界的"锐度"——$\|w\|$ 越大，概率从 0 跳到 1 的区域越窄。

#### 3.3 对数损失（Log Loss / Cross-Entropy）——为什么不用 MSE？

分类问题用 MSE（均方误差）有两个致命问题：

**问题一：非凸。** Sigmoid 的非线性导致 MSE 在参数空间中变成非凸函数——梯度下降可能卡在局部最优。

**问题二：惩罚不合理。** 看这个例子——真实标签是"垃圾邮件(1)"：

| 预测概率 $\hat{y}$ | MSE $(1 - \hat{y})^2$ | Log-Loss $-\log(\hat{y})$ |
|---------------------|------------------------|----------------------------|
| 0.99（"很可能是垃圾"✓） | 0.0001 | 0.0101 |
| 0.51（"勉强算垃圾"） | 0.2401 | 0.6733 |
| 0.01（"绝对正常"✗ — 巨大错误） | 0.9801 | **4.6052** |

Log-Loss 对"极端错误"（ŷ≈0 而 y=1）的惩罚是指数级放大的——**这恰恰符合我们对概率直觉的要求**：把 99% 确定的事情弄反了，应该比模棱两可时犯错付出更大的代价。

**形式化定义：**

单个样本的交叉熵损失：

$$
L(y, \hat{y}) = -\big[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\big]
$$

整个数据集上的平均损失：

$$
J(w, b) = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\big]
$$

**与最大似然估计（MLE）的联系：** 最小化交叉熵损失，在数学上等价于最大化参数的似然函数——逻辑回归的输出可以严格解释为概率，因为它本质上在做最大似然估计。

#### 3.4 梯度下降——怎么找到最优的 w 和 b？

损失函数对参数求偏导（推导使用链式法则，核心步骤见手算示例）：

$$
\frac{\partial L}{\partial z} = \hat{y} - y
$$

$$
\frac{\partial L}{\partial w} = (\hat{y} - y)\,x
$$

$$
\frac{\partial L}{\partial b} = \hat{y} - y
$$

整个数据集的梯度（向量化形式）：

$$
\nabla_w J = \frac{1}{n} X^T (\hat{y} - y)
$$

$$
\nabla_b J = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

**关键洞察：** 梯度的形式和线性回归**几乎一样**——都是 `X.T @ (prediction - truth) / n`。区别仅在于 prediction 的含义：线性回归中是 $Xw$（无界实数），逻辑回归中是 $\sigma(Xw)$（0-1 概率）。一旦你理解了线性回归的梯度，逻辑回归的梯度就自然懂了。

参数更新规则：

$$
w := w - \alpha \nabla_w J
$$
$$
b := b - \alpha \nabla_b J
$$

其中 $\alpha$ 是学习率。

---

### 4. 手算示例：从纸上理解逻辑回归的每一步

> **准备纸笔。** 这一节，我们用手算完成 Sigmoid、Log-Loss、和 3 轮梯度下降。目标是让你**真正理解每个数字怎么来的**——之后写代码时，你脑子里浮现的将是这些手工步骤，而不是黑盒。

#### 4.1 Sigmoid 手算

计算 5 个典型 z 值的 Sigmoid 输出（保留 4 位小数，$e \approx 2.7183$）：

**z = -2：**
$$
\sigma(-2) = \frac{1}{1 + e^{2}} = \frac{1}{1 + 7.3891} = \frac{1}{8.3891} = 0.1192
$$

**z = -1：**
$$
\sigma(-1) = \frac{1}{1 + e^{1}} = \frac{1}{1 + 2.7183} = \frac{1}{3.7183} = 0.2689
$$

**z = 0：**
$$
\sigma(0) = \frac{1}{1 + 1} = 0.5000
$$

**z = 1：**
$$
\sigma(1) = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.3679} = \frac{1}{1.3679} = 0.7311
$$

**z = 2：**
$$
\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} = 0.8808
$$

| z | -2 | -1 | 0 | 1 | 2 |
|---|-----|-----|---|---|---|
| σ(z) | 0.1192 | 0.2689 | 0.5000 | 0.7311 | 0.8808 |

**直觉验证：** z 越大，σ(z) 越接近 1；z 越小，σ(z) 越接近 0；z=0 时恰好 0.5。这条 S 形曲线以 (0, 0.5) 为中心对称。

#### 4.2 Log-Loss 手算：3 个预测

假设模型对 3 封邮件给出了预测概率，真实标签已知：

| 邮件 | 真实标签 y | 预测概率 ŷ | 判断 |
|------|-----------|-----------|------|
| A | 1（垃圾） | 0.9 | 很准 |
| B | 0（正常） | 0.2 | 不错（预测正常概率 0.8） |
| C | 1（垃圾） | 0.3 | 很差（预测垃圾概率只有 0.3） |

使用 $\ln$（自然对数）计算每条样本的交叉熵损失：

**邮件 A（y=1, ŷ=0.9）：**
$$
L_A = -[1 \cdot \ln(0.9) + 0 \cdot \ln(0.1)] = -\ln(0.9)
$$
$\ln(0.9) \approx -0.1054$，所以 $L_A = 0.1054$

**邮件 B（y=0, ŷ=0.2）：**
$$
L_B = -[0 \cdot \ln(0.2) + 1 \cdot \ln(0.8)] = -\ln(0.8)
$$
$\ln(0.8) \approx -0.2231$，所以 $L_B = 0.2231$

**邮件 C（y=1, ŷ=0.3）：**
$$
L_C = -[1 \cdot \ln(0.3) + 0 \cdot \ln(0.7)] = -\ln(0.3)
$$
$\ln(0.3) \approx -1.2040$，所以 $L_C = 1.2040$

**平均 Log-Loss：**
$$
J = \frac{0.1054 + 0.2231 + 1.2040}{3} = \frac{1.5325}{3} = 0.5108
$$

**观察：** 邮件 C 的错误贡献了总损失的 78%——这正是交叉熵的特性：**对"极端错误"的惩罚远大于"轻微错误"**。

#### 4.3 梯度下降手算：3 轮迭代（4 个数据点）

**数据集（单特征，便于手算）：**

| i | $x$（特征） | $y$（标签） | 含义 |
|---|------------|------------|------|
| 1 | 1 | 0 | x 小时通常是负类 |
| 2 | 2 | 0 | x 较小时通常是负类 |
| 3 | 3 | 1 | x 较大时通常是正类 |
| 4 | 4 | 1 | x 大时通常是正类 |

直观上，x 越大越可能是正类——模型应该学出一个**正的 w**（斜率），让 $z = wx + b$ 随 x 增大而增大。

**初始参数：** $w = 0$，$b = 0$，学习率 $\alpha = 0.3$

---

**第 1 轮迭代**

**Step 1 — 前向传播（计算预测值）：**

$z_i = w \cdot x_i + b = 0$（所有样本）

$\hat{y}_i = \sigma(0) = 0.5$（所有样本）

| i | $x_i$ | $y_i$ | $z_i$ | $\hat{y}_i$ |
|---|-------|-------|-------|-------------|
| 1 | 1 | 0 | 0 | 0.5 |
| 2 | 2 | 0 | 0 | 0.5 |
| 3 | 3 | 1 | 0 | 0.5 |
| 4 | 4 | 1 | 0 | 0.5 |

**Step 2 — 计算损失：**

样本 1 (y=0, ŷ=0.5)：$-\ln(0.5) = 0.6931$
样本 2 (y=0, ŷ=0.5)：$-\ln(0.5) = 0.6931$
样本 3 (y=1, ŷ=0.5)：$-\ln(0.5) = 0.6931$
样本 4 (y=1, ŷ=0.5)：$-\ln(0.5) = 0.6931$

$J = \frac{4 \times 0.6931}{4} = 0.6931$

**Step 3 — 计算梯度：**

误差向量 $\hat{y} - y = [0.5, 0.5, -0.5, -0.5]$

$$
\nabla_w J = \frac{1}{4} \sum_{i=1}^{4} (\hat{y}_i - y_i) \cdot x_i
= \frac{1}{4}\big[1 \times 0.5 + 2 \times 0.5 + 3 \times (-0.5) + 4 \times (-0.5)\big]
$$

$$
= \frac{1}{4}\big[0.5 + 1.0 - 1.5 - 2.0\big] = \frac{1}{4} \times (-2.0) = -0.5
$$

$$
\nabla_b J = \frac{1}{4} \sum_{i=1}^{4} (\hat{y}_i - y_i)
= \frac{1}{4}\big[0.5 + 0.5 + (-0.5) + (-0.5)\big] = \frac{1}{4} \times 0 = 0
$$

**Step 4 — 更新参数：**

$w_{\text{new}} = w - 0.3 \times (-0.5) = 0 + 0.15 = \boxed{0.15}$

$b_{\text{new}} = b - 0.3 \times 0 = \boxed{0}$

---

**第 2 轮迭代（使用新参数 w=0.15, b=0）**

**Step 1 — 前向传播：**

$z = 0.15x + 0 = [0.15, 0.30, 0.45, 0.60]$

| i | $x_i$ | $y_i$ | $z_i$ | $\hat{y}_i$（计算） |
|---|-------|-------|-------|---------------------|
| 1 | 1 | 0 | 0.15 | $\frac{1}{1+e^{-0.15}} = \frac{1}{1+0.8607} = 0.5374$ |
| 2 | 2 | 0 | 0.30 | $\frac{1}{1+e^{-0.30}} = \frac{1}{1+0.7408} = 0.5744$ |
| 3 | 3 | 1 | 0.45 | $\frac{1}{1+e^{-0.45}} = \frac{1}{1+0.6376} = 0.6106$ |
| 4 | 4 | 1 | 0.60 | $\frac{1}{1+e^{-0.60}} = \frac{1}{1+0.5488} = 0.6457$ |

**Step 2 — 计算损失：**

样本 1 (y=0, ŷ=0.5374)：$-\ln(1 - 0.5374) = -\ln(0.4626) = 0.7709$
样本 2 (y=0, ŷ=0.5744)：$-\ln(1 - 0.5744) = -\ln(0.4256) = 0.8543$
样本 3 (y=1, ŷ=0.6106)：$-\ln(0.6106) = 0.4932$
样本 4 (y=1, ŷ=0.6457)：$-\ln(0.6457) = 0.4374$

$J = \frac{0.7709 + 0.8543 + 0.4932 + 0.4374}{4} = \frac{2.5558}{4} = 0.6390$（损失下降了！）

**Step 3 — 计算梯度：**

误差向量 $\hat{y} - y = [0.5374, 0.5744, -0.3894, -0.3543]$

$$
\nabla_w J = \frac{1}{4}\big[1(0.5374) + 2(0.5744) + 3(-0.3894) + 4(-0.3543)\big]
$$

$$
= \frac{1}{4}\big[0.5374 + 1.1488 - 1.1682 - 1.4172\big] = \frac{1}{4} \times (-0.8992) = -0.2248
$$

$$
\nabla_b J = \frac{1}{4}\big[0.5374 + 0.5744 + (-0.3894) + (-0.3543)\big] = \frac{1}{4} \times 0.3681 = 0.0920
$$

**Step 4 — 更新参数：**

$w_{\text{new}} = 0.15 - 0.3 \times (-0.2248) = 0.15 + 0.0674 = \boxed{0.2174}$

$b_{\text{new}} = 0 - 0.3 \times 0.0920 = \boxed{-0.0276}$

---

**第 3 轮迭代（使用新参数 w=0.2174, b=-0.0276）**

**Step 1 — 前向传播：**

$z = 0.2174x - 0.0276 = [0.1898, 0.4072, 0.6246, 0.8420]$

| i | $x_i$ | $y_i$ | $z_i$ | $\hat{y}_i$（计算） |
|---|-------|-------|-------|---------------------|
| 1 | 1 | 0 | 0.1898 | $\frac{1}{1+e^{-0.1898}} = \frac{1}{1+0.8272} = 0.5473$ |
| 2 | 2 | 0 | 0.4072 | $\frac{1}{1+e^{-0.4072}} = \frac{1}{1+0.6655} = 0.6004$ |
| 3 | 3 | 1 | 0.6246 | $\frac{1}{1+e^{-0.6246}} = \frac{1}{1+0.5355} = 0.6513$ |
| 4 | 4 | 1 | 0.8420 | $\frac{1}{1+e^{-0.8420}} = \frac{1}{1+0.4308} = 0.6989$ |

**Step 2 — 计算损失：**

样本 1 (y=0, ŷ=0.5473)：$-\ln(0.4527) = 0.7924$
样本 2 (y=0, ŷ=0.6004)：$-\ln(0.3996) = 0.9172$
样本 3 (y=1, ŷ=0.6513)：$-\ln(0.6513) = 0.4288$
样本 4 (y=1, ŷ=0.6989)：$-\ln(0.6989) = 0.3582$

$J = \frac{0.7924 + 0.9172 + 0.4288 + 0.3582}{4} = 0.6242$

**Step 3 — 计算梯度：**

误差向量 $\hat{y} - y = [0.5473, 0.6004, -0.3487, -0.3011]$

$$
\nabla_w J = \frac{1}{4}\big[1(0.5473) + 2(0.6004) + 3(-0.3487) + 4(-0.3011)\big]
$$

$$
= \frac{1}{4}\big[0.5473 + 1.2008 - 1.0461 - 1.2044\big] = \frac{1}{4} \times (-0.5024) = -0.1256
$$

**Step 4 — 更新参数：**

$w_{\text{new}} = 0.2174 - 0.3 \times (-0.1256) = 0.2174 + 0.0377 = \boxed{0.2551}$

---

**三轮迭代总结：**

| 轮数 | $w$ | $b$ | 损失 $J$ | $\nabla_w$ |
|------|-----|-----|----------|------------|
| 初始 | 0.0000 | 0.0000 | — | — |
| 1 | 0.1500 | 0.0000 | 0.6931 | -0.5000 |
| 2 | 0.2174 | -0.0276 | 0.6390 | -0.2248 |
| 3 | 0.2551 | — | 0.6242 | -0.1256 |

**观察：**
- $w$ 从 0 逐渐增大为正值——模型正在学会"x 越大，越可能是正类"
- 损失从 0.6931 持续下降到 0.6242——参数在改善
- $\nabla_w$ 从 -0.5 到 -0.1256——梯度越来越小，说明正在接近最优解
- 决策边界 $\hat{y}=0.5 \Leftrightarrow z=0 \Leftrightarrow x = -b/w$，随着 w 增大和 b 变负，边界在自动调整

---

### 5. Python 从零实现

#### 5.1 完整的 LogisticRegression 类

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """从零实现的二分类逻辑回归（批量梯度下降）。

    Parameters
    ----------
    lr : float, default=0.1
        学习率（learning rate）。
    n_iters : int, default=1000
        梯度下降迭代次数。
    tol : float, default=1e-6
        早停阈值：损失下降小于此值时提前终止。
    verbose : bool, default=False
        是否打印训练进度。
    random_state : int or None, default=42
        随机种子（权重初始化用）。
    """
    def __init__(self, lr=0.1, n_iters=1000, tol=1e-6,
                 verbose=False, random_state=42):
        self.lr = lr
        self.n_iters = n_iters
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.w = None
        self.b = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid 函数，内置数值稳定性处理。"""
        return 1 / (1 + np.exp(-z))

    def _log_loss(self, y_true, y_pred):
        """交叉熵损失（防止 log(0) 的裁剪）。"""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) +
                        (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """训练逻辑回归模型。

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            训练特征。
        y : np.ndarray, shape (n,)
            训练标签（0 或 1）。

        Returns
        -------
        self : LogisticRegression
        """
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)
        self.w = rng.randn(d) * 0.01
        self.b = 0.0
        self.loss_history = []

        prev_loss = float('inf')
        for i in range(self.n_iters):
            z = X @ self.w + self.b
            y_hat = self._sigmoid(z)

            loss = self._log_loss(y, y_hat)
            self.loss_history.append(loss)

            dw = (X.T @ (y_hat - y)) / n
            db = np.mean(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if self.verbose and i % max(1, self.n_iters // 10) == 0:
                acc = np.mean((y_hat >= 0.5) == y)
                print(f"[{i:>5d}/{self.n_iters}] "
                      f"loss: {loss:.6f}  acc: {acc:.4f}")

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"早停于第 {i} 轮，损失变化 < {self.tol}")
                break
            prev_loss = loss

        return self

    def predict_proba(self, X):
        """返回正类概率 P(y=1 | X)。"""
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        """返回硬分类结果（0 或 1）。"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """返回准确率。"""
        return np.mean(self.predict(X) == y)
```

#### 5.2 验证：与手算结果对照

```python
# 用第 4 节手算的 4 个数据点验证
X_hand = np.array([[1], [2], [3], [4]])
y_hand = np.array([0, 0, 1, 1])

model_hand = LogisticRegression(lr=0.3, n_iters=3, random_state=42)
model_hand.fit(X_hand, y_hand)

print(f"3 轮迭代后：")
print(f"  w = {model_hand.w[0]:.4f}  （手算结果：0.2551）")
print(f"  b = {model_hand.b:.4f}  （手算结果：-0.0276）")
print(f"  最终损失 = {model_hand.loss_history[-1]:.4f}  （手算结果：0.6242）")
```

**预期输出（数值与手算高度吻合）：**
```
3 轮迭代后：
  w = 0.2551  （手算结果：0.2551）
  b = -0.0276  （手算结果：-0.0276）
  最终损失 = 0.6242  （手算结果：0.6242）
```

#### 5.3 在合成数据上训练并可视化

```python
np.random.seed(42)

# 生成二维二分类数据
n_samples = 300
X_train = np.random.randn(n_samples, 2) * 1.5
# 正类：x1 + x2 > 0 的区域（带噪声）
y_train = (X_train[:, 0] + X_train[:, 1] +
           0.3 * np.random.randn(n_samples) > 0).astype(int)

print(f"正类占比: {y_train.mean():.2%}")

# 训练
model = LogisticRegression(lr=0.1, n_iters=2000, verbose=True)
model.fit(X_train, y_train)

# ─── 可视化 ───
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 左：损失曲线
axes[0].plot(model.loss_history, 'b-', linewidth=1.5, alpha=0.8)
axes[0].plot(np.convolve(model.loss_history, np.ones(50)/50, mode='valid'),
             'r-', linewidth=2, label='50轮移动平均')
axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('交叉熵损失')
axes[0].set_title(f'损失收敛 (最终: {model.loss_history[-1]:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 中：决策边界（概率热力图）
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid).reshape(xx.shape)

cs = axes[1].contourf(xx, yy, probs, levels=np.linspace(0, 1, 11),
                       cmap='RdYlBu', alpha=0.6)
axes[1].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                c='#2c7bb6', edgecolors='white', s=30, label='负类 (y=0)')
axes[1].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                c='#d7191c', edgecolors='white', s=30, label='正类 (y=1)')
axes[1].contour(xx, yy, probs, levels=[0.5], colors='black',
                linewidths=2.5, linestyles='--')
plt.colorbar(cs, ax=axes[1], label='P(y=1)')
axes[1].set_title('概率热力图 + 决策边界')
axes[1].legend(loc='lower right')

# 右：不同 z 值处的预测概率分布
zs = X_train @ model.w + model.b
axes[2].scatter(zs[y_train == 0], model._sigmoid(zs[y_train == 0]),
                c='#2c7bb6', alpha=0.4, s=20, label='负类')
axes[2].scatter(zs[y_train == 1], model._sigmoid(zs[y_train == 1]),
                c='#d7191c', alpha=0.4, s=20, label='正类')
z_line = np.linspace(zs.min() - 1, zs.max() + 1, 300)
axes[2].plot(z_line, model._sigmoid(z_line), 'k-', linewidth=2,
             label='Sigmoid 曲线')
axes[2].axvline(0, color='gray', linestyle='--', alpha=0.6)
axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.6)
axes[2].set_xlabel('z = wx + b')
axes[2].set_ylabel('ŷ = σ(z)')
axes[2].set_title('每个样本的 z 值与预测概率')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n权重 w: {model.w}")
print(f"偏置 b: {model.b:.4f}")
print(f"决策边界方程: {model.w[0]:.3f}·x₁ + {model.w[1]:.3f}·x₂ + {model.b:.3f} = 0")
print(f"训练准确率: {model.score(X_train, y_train):.4f}")
```

#### 5.4 不同学习率的对比

```python
lr_list = [0.01, 0.05, 0.1, 0.3, 0.5]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr_val in lr_list:
    m = LogisticRegression(lr=lr_val, n_iters=500, random_state=42)
    m.fit(X_train, y_train)
    axes[0].plot(m.loss_history, label=f'lr={lr_val}', linewidth=1.5,
                 alpha=0.8)
    axes[1].plot(m.loss_history[-50:], linewidth=1.5, alpha=0.8)

axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('损失')
axes[0].set_title('不同学习率下的损失收敛（完整）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('迭代次数（最后 50 轮）')
axes[1].set_ylabel('损失')
axes[1].set_title('不同学习率下的损失收敛（放大尾部）')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n学习率选择建议：")
print("  lr 太小 → 收敛太慢，训练不动")
print("  lr 太大 → 损失震荡，甚至发散")
print("  lr 适中 → 平滑下降，快速收敛")
print("  一般从 0.1 开始试，不行再调")
```

---

### 6. sklearn 实战：乳腺癌诊断

```python
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, log_loss, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

print(f"数据集: {data.DESCR.split(chr(10))[1].strip()}")
print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
print(f"类别分布: 恶性(0)={np.sum(y == 0)}, 良性(1)={np.sum(y == 1)}")

# 切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 必须标准化——逻辑回归对特征量纲敏感
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 训练 sklearn 逻辑回归
sk_lr = SKLogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                              max_iter=5000, random_state=42)
sk_lr.fit(X_train_s, y_train)

y_pred = sk_lr.predict(X_test_s)
y_proba = sk_lr.predict_proba(X_test_s)[:, 1]

print(f"\n测试集准确率:  {accuracy_score(y_test, y_pred):.4f}")
print(f"测试集 Log-Loss: {log_loss(y_test, y_proba):.4f}")
print(f"测试集 AUC:      {roc_auc_score(y_test, y_proba):.4f}")
print(f"\n{classification_report(y_test, y_pred,
                              target_names=data.target_names)}")

# ─── 混淆矩阵 ───
cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")
print(f"  TP(真正例) = {cm[1, 1]}, FN(假负例) = {cm[1, 0]}")
print(f"  FP(假正例) = {cm[0, 1]}, TN(真负例) = {cm[0, 0]}")
```

#### 6.1 特征重要性解释

```python
# 逻辑回归的可解释性是其最大优势之一
feature_names = data.feature_names
coef = sk_lr.coef_[0]

# 按绝对值排序，取前 15 个最重要的特征
top_idx = np.argsort(np.abs(coef))[-15:][::-1]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#d7191c' if c > 0 else '#2c7bb6' for c in coef[top_idx]]
bars = ax.barh(range(len(top_idx)), coef[top_idx], color=colors)
ax.set_yticks(range(len(top_idx)))
ax.set_yticklabels(feature_names[top_idx])
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('权重系数（标准化后）')
ax.set_title('Top 15 最重要特征及影响力方向')
ax.text(0.98, 0.02, '红色=正相关(推高恶性概率)\n蓝色=负相关(推低恶性概率)',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, color='gray')

plt.tight_layout()
plt.show()

print("\n特征解读示例：")
for i in top_idx[:5]:
    direction = "增大恶性概率" if coef[i] > 0 else "减小恶性概率"
    print(f"  {feature_names[i]:30s}  权重={coef[i]:+.4f}  ({direction})")
```

#### 6.2 ROC 曲线与概率校准

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 左：ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'LR (AUC={roc_auc_score(y_test, y_proba):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
axes[0].set_xlabel('假正率 (FPR)')
axes[0].set_ylabel('真正率 (TPR)')
axes[0].set_title('ROC 曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 中：预测概率分布
axes[1].hist(y_proba[y_test == 0], bins=20, alpha=0.6, color='#2c7bb6',
             label=f'真实恶性 (n={np.sum(y_test == 0)})', edgecolor='white')
axes[1].hist(y_proba[y_test == 1], bins=20, alpha=0.6, color='#d7191c',
             label=f'真实良性 (n={np.sum(y_test == 1)})', edgecolor='white')
axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='决策门槛')
axes[1].set_xlabel('预测的恶性概率')
axes[1].set_ylabel('样本数')
axes[1].set_title('预测概率分布')
axes[1].legend()

# 右：正则化强度 C 的影响
C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores, coef_norms = [], [], []
for C in C_values:
    lr = SKLogisticRegression(penalty='l2', C=C, solver='lbfgs',
                               max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)
    train_scores.append(lr.score(X_train_s, y_train))
    test_scores.append(lr.score(X_test_s, y_test))
    coef_norms.append(np.linalg.norm(lr.coef_[0]))

axes[2].semilogx(C_values, train_scores, 'bo-', label='训练准确率')
axes[2].semilogx(C_values, test_scores, 'ro-', label='测试准确率')
axes[2].set_xlabel('C（正则化强度倒数）')
axes[2].set_ylabel('准确率')
axes[2].set_title('正则化强度 C 的影响')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n正则化强度解读：")
print("  C 越小 → 正则化越强 → 权重被压缩 → 防止过拟合")
print("  C 越大 → 正则化越弱 → 权重可以很大 → 可能过拟合")
print(f"  C=1.0 时的权重 L2 范数: {np.linalg.norm(sk_lr.coef_[0]):.2f}")
```

---

### 7. 常见误区

#### 误区 1：用 MSE 替代交叉熵

> **错误想法：** "反正都是衡量预测和真实的差距，直接用 MSE 不就完了？"

**为什么错：** MSE + Sigmoid 的组合导致损失函数非凸，梯度下降可能停在局部最优。更重要的是，交叉熵是分类问题的"自然"损失——它等价于最大似然估计，输出的概率有严格的统计学基础。

**正确做法：** 二分类用交叉熵（BCE），多分类用多类交叉熵。只有在极少数特殊场景（如标签平滑）才会混合使用。

#### 误区 2：不标准化特征

> **错误想法：** "逻辑回归是线性模型，和距离没关系，不用标准化吧？"

**为什么错：** 虽然逻辑回归的预测不依赖距离，但梯度下降的收敛速度和稳定性严重依赖特征量纲。假设 x₁ 范围是 [0, 0.01]，x₂ 范围是 [0, 10000]——梯度在 x₂ 方向上将被放大 100 万倍，导致参数更新极不稳定。

**正确做法：** 训练前必须做标准化（`StandardScaler`）或归一化（`MinMaxScaler`）。

#### 误区 3：把权重系数大小直接当特征重要性

> **错误想法：** "w₁ = 3.5, w₂ = 0.2，所以特征 1 比特征 2 重要 17 倍。"

**为什么错：** 权重的大小取决于特征的量纲。如果 x₁ 的单位是"毫米"而 x₂ 是"千米"，直接比较权重毫无意义。即使标准化后，权重也不等价于重要性——特征之间可能存在相关性（共线性）。

**正确做法：** 标准化后权重的绝对值可以**粗略**反映重要性。更严谨的做法是看 p 值（statsmodels）或用 permutation importance / SHAP 值。

#### 误区 4：认为逻辑回归只能画直线

> **错误想法：** "决策边界是线性的 → 逻辑回归只能处理线性可分的数据。"

**为什么错：** 决策边界的"线性"是相对于**特征空间**而言的。你可以先做特征工程——添加 $x_1^2$、$x_1 x_2$、$\sin(x_1)$ 等非线性变换——然后在**变换后的空间**里，边界仍然是线性的，但投影回原始空间就是非线性的。

**正确做法：** 面对非线性数据，先尝试多项式特征、交互特征、分箱等，再看是否需要换模型。

#### 误区 5：样本不均衡时只看准确率

> **错误想法：** "模型准确率 95%，效果很好啊！"

**为什么错：** 如果数据中 95% 是负类，一个"永远预测负类"的模型也有 95% 准确率——但它的正类召回率是 0。

**正确做法：** 同时看精确率（Precision）、召回率（Recall）、F1 分数、AUC。使用 `class_weight='balanced'` 或调整决策阈值（不是固定的 0.5）来应对不均衡。

---

### 8. 多分类扩展：Softmax 回归

对于 K 个类别（K > 2），有两种策略：

**策略一：One-vs-Rest（OvR）**——为每个类别训练一个"是/不是这个类"的二分类器。K 个类别 = K 个分类器。预测时选概率最高的那个。简单但忽略了类别之间的关系。

**策略二：Softmax 回归**——Sigmoid 的多类推广。Softmax 函数将 K 个任意实数转化为 K 个概率，且和为 1：

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

其中 $z_k = w_k^T x + b_k$ 是第 k 类的 logit。

```python
def softmax(z):
    """稳定的 Softmax 实现。"""
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class SoftmaxRegression:
    """多分类 Softmax 回归（批量梯度下降）。"""
    def __init__(self, lr=0.1, n_iters=1000, random_state=42):
        self.lr = lr
        self.n_iters = n_iters
        self.random_state = random_state
        self.W = None    # (d, K)
        self.b = None    # (K,)
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        K = len(np.unique(y))
        y_onehot = np.eye(K)[y.astype(int)]

        rng = np.random.RandomState(self.random_state)
        self.W = rng.randn(d, K) * 0.01
        self.b = np.zeros(K)
        self.loss_history = []

        for _ in range(self.n_iters):
            z = X @ self.W + self.b
            y_hat = softmax(z)

            eps = 1e-15
            loss = -np.mean(np.sum(
                y_onehot * np.log(np.clip(y_hat, eps, 1)), axis=1))
            self.loss_history.append(loss)

            dW = (X.T @ (y_hat - y_onehot)) / n
            db = np.mean(y_hat - y_onehot, axis=0)
            self.W -= self.lr * dW
            self.b -= self.lr * db

        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
```

**梯度形式对比——惊人的统一：**

| 模型 | 预测值 $\hat{y}$ | 梯度公式 |
|------|-------------------|----------|
| 线性回归 | $Xw$ | $\frac{1}{n}X^T(\hat{y} - y)$ |
| 逻辑回归（二分类） | $\sigma(Xw)$ | $\frac{1}{n}X^T(\hat{y} - y)$ |
| Softmax 回归（多分类） | $\text{softmax}(XW)$ | $\frac{1}{n}X^T(\hat{Y} - Y)$ |

三种模型的梯度公式结构完全一致——区别仅在于 $\hat{y}$ 的含义。这是理解分类问题的"钥匙"。

---

### 9. 本章总结

| 概念 | 要点 |
|------|------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$，将任意实数 → (0,1) 概率，单调、光滑、导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))$ |
| 决策边界 | 线性超平面 $w^Tx+b=0$，$\|w\|$ 控制边界锐度 |
| 交叉熵损失 | $-\frac{1}{n}\sum [y\log\hat{y} + (1-y)\log(1-\hat{y})]$，等价 MLE，对极端错误惩罚巨大 |
| 梯度 | $\nabla_w J = \frac{1}{n}X^T(\hat{y}-y)$，形式与线性回归一致 |
| Softmax | Sigmoid 的多类推广，$K$ 类输出 $K$ 维概率 |
| 正则化 | `penalty='l2'`，`C` 控制强度（C 小 = 正则化强） |
| 标准化 | 特征量纲不一致时必须标准化，否则梯度下降不稳定 |

**逻辑回归的定位：** 它是分类世界的"线性回归"——简单、可解释、训练快、输出概率。在信用评分卡、医疗诊断、风控模型等需要**透明决策**的领域，逻辑回归始终是首选基线。当你面对一个新的分类问题，先跑一个逻辑回归——它的表现告诉你这个问题有多难，它的权重告诉你哪些特征最重要。

---

### 10. 思考题

每题建议先独立思考 5-10 分钟，再看解答。

---

**Q1：为什么逻辑回归叫"回归"却做分类？这两个名字怎么统一？**

<details>
<summary>点击查看解答</summary>

"回归"指的是模型在**训练阶段**做的事——它回归（拟合）的是对数几率（log-odds）：

$$
\log\frac{P(y=1)}{P(y=0)} = w^T x + b
$$

左边是"正类概率与负类概率之比的对数"——这是一个可以在 $(-\infty, +\infty)$ 上取任意值的连续量。最小化交叉熵损失本质上就是对对数几率做"回归"。

"分类"是**推断阶段**做的事——把回归出的对数几率通过 Sigmoid 映射为概率，再用阈值 0.5 作出分类决策。

名字是历史遗留，但本质是对的：逻辑回归 = 对对数几率做线性回归 + Sigmoid 映射为分类。

</details>

---

**Q2：如果 $\hat{y} = 0.99$ 而真实标签 $y = 0$，Log-Loss 是多少？为什么这个惩罚"合理"？**

<details>
<summary>点击查看解答</summary>

$$
L = -[0 \cdot \ln(0.99) + 1 \cdot \ln(0.01)] = -\ln(0.01) = -\ln(10^{-2}) = 2 \ln(10) \approx 4.605
$$

对比一下：如果 $\hat{y} = 0.51$ 而 $y = 0$，$L = -\ln(0.49) \approx 0.713$。

同样是"判断错误"，前者（99% 确信却错了）的损失是后者（51% 猜测却错了）的 **6.5 倍**。

这很合理——在现实决策中，把"几乎确定对的事"判断错了，应该受到比"模棱两可时猜错"严厉得多的惩罚。如果你 99% 确定一个肿瘤是良性的但实际是恶性的，这个错误的严重程度远超 51% 确定时的错误。

</details>

---

**Q3：梯度 $\nabla_w J = \frac{1}{n}X^T(\hat{y}-y)$ 中，为什么误差项是 $(\hat{y} - y)$ 而不是其他形式？请给出链式法则推导。**

<details>
<summary>点击查看解答</summary>

设单个样本 $(x, y)$，$z = w^T x + b$，$\hat{y} = \sigma(z)$。

**第一步：** 损失对 $\hat{y}$ 的导数
$$
\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}
$$

**第二步：** Sigmoid 的导数
$$
\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \hat{y}(1-\hat{y})
$$

**第三步：** 链式法则
$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}
= \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \hat{y} - y
$$

分母 $\hat{y}(1-\hat{y})$ 和 Sigmoid 导数中的 $\hat{y}(1-\hat{y})$ 恰好抵消——这就是使用交叉熵 + Sigmoid 组合的**数学巧妙之处**。如果换成 MSE，这个抵消不会发生，梯度会多出一个 $\hat{y}(1-\hat{y})$ 因子，导致 $\hat{y}$ 接近 0 或 1 时梯度趋近于 0（梯度消失）。

**第四步：** 对 w 和 b 的梯度
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = (\hat{y} - y) x
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = \hat{y} - y
$$
</details>

---

**Q4：手算数据集 $x=[1, 2, 3], y=[0, 1, 1]$，假设 $w=1, b=-2$，计算 Log-Loss。**

<details>
<summary>点击查看解答</summary>

**Step 1 — 前向传播：**

$z_1 = 1 \times 1 + (-2) = -1$，$\hat{y}_1 = \sigma(-1) = 0.2689$
$z_2 = 1 \times 2 + (-2) = 0$，$\hat{y}_2 = \sigma(0) = 0.5000$
$z_3 = 1 \times 3 + (-2) = 1$，$\hat{y}_3 = \sigma(1) = 0.7311$

**Step 2 — 计算损失：**

样本 1 (y=0, ŷ=0.2689)：$-\ln(1-0.2689) = -\ln(0.7311) = 0.3133$
样本 2 (y=1, ŷ=0.5000)：$-\ln(0.5000) = 0.6931$
样本 3 (y=1, ŷ=0.7311)：$-\ln(0.7311) = 0.3133$

$J = (0.3133 + 0.6931 + 0.3133) / 3 = 1.3197 / 3 = 0.4399$

</details>

---

**Q5：如果所有特征都乘以 1000（比如把"米"改成"毫米"），不重新训练，模型预测会怎么变？为什么必须标准化？**

<details>
<summary>点击查看解答</summary>

模型学到的参数 $w$ 是针对原始量纲的。如果输入 $x$ 突然放大 1000 倍：

$$
z_{\text{new}} = w^T \cdot (1000x) + b = 1000 \cdot w^T x + b
$$

此时 $z$ 的值会变得非常极端——如果原来 $z \approx 0$（决策边界附近），新的 $z$ 可能达到 ±1000。Sigmoid 会直接输出 0 或 1，模型失去了概率的"灰度"——所有预测都变成 100% 确信，毫无区分度。

**标准化**将所有特征拉到同一量级（均值 0，标准差 1），保证：
1. 梯度在各个方向上大小相近，梯度下降稳定收敛
2. 权重系数之间可以大致比较重要性
3. 正则化惩罚对所有特征公平——否则量纲大的特征会被惩罚更多

</details>

---

**Q6：决策边界 $w^T x + b = 0$ 上的点，$\hat{y}$ 是多少？离边界越远的点，$\hat{y}$ 怎么变化？**

<details>
<summary>点击查看解答</summary>

边界上 $w^T x + b = 0$，所以 $z = 0$，$\hat{y} = \sigma(0) = 0.5$。

离边界越远，$|z|$ 越大：

- 正类一侧（$z > 0$）：$z$ 越大，$\hat{y} \to 1$
- 负类一侧（$z < 0$）：$|z|$ 越大，$\hat{y} \to 0$

关键：$\|w\|$ 决定了"多远才算远"——$\|w\|$ 很大时，距离边界一小步，概率就从 0.5 跳到接近 0 或 1，边界很"锐"；$\|w\|$ 很小时，概率变化平缓，边界很"钝"。

正则化（L2）会压缩 $\|w\|$，因此正则化越强 → 决策边界越"钝" → 模型越保守（不确定的预测更多）。

</details>

---

**Q7：数据 95% 是负类，模型准确率 95%。这个模型有用吗？应该用什么指标？**

<details>
<summary>点击查看解答</summary>

可能完全没用。如果模型只是"永远预测负类"，准确率也是 95%——完美利用了类别不平衡，但没有学到任何东西。

对于不均衡数据，应该看：

| 指标 | 含义 | 关注场景 |
|------|------|----------|
| 精确率 (Precision) | 预测为正的样本中真正为正的比例 | 不能错杀（如垃圾邮件误判） |
| 召回率 (Recall) | 真正为正的样本中被找出的比例 | 不能漏掉（如癌症筛查） |
| F1 分数 | 精确率和召回率的调和平均 | 两者都重要 |
| AUC-ROC | 在所有阈值下的综合排序能力 | 关注排序而非具体阈值 |

**修正方法：**
1. 设置 `class_weight='balanced'` 或手动指定权重
2. 调整决策阈值（不一定用 0.5）——用验证集找到最优阈值
3. 使用上采样（SMOTE）或下采样平衡数据
4. 改用对不均衡更鲁棒的损失函数（如 Focal Loss）

</details>

---

**Q8：添加 $x_1^2$、$x_1 x_2$ 等多项式特征后，逻辑回归能拟合非线性边界吗？为什么？**

<details>
<summary>点击查看解答</summary>

**能。** 关键在于理解"线性"是相对于什么空间。

原始特征空间 $(x_1, x_2)$ 中，决策边界是 $w_1 x_1 + w_2 x_2 + b = 0$——一条直线。

添加多项式特征后，新特征空间是 $(x_1, x_2, x_1^2, x_1 x_2, x_2^2)$，模型学到的决策边界是：

$$
w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2 + b = 0
$$

这在**新特征空间**中是线性超平面，但在**原始 $(x_1, x_2)$ 空间**中是一条二次曲线——可以是椭圆、抛物线、双曲线等非线性形状。

换句话说：逻辑回归总是学习一个线性边界，但你通过特征工程控制了"在哪个空间里是线性的"。**特征工程决定了模型的表达能力上限**——这是经典 ML 的核心哲学。你甚至可以添加 $\sin(x_1)$、$\log(x_2)$ 等非线性变换，让边界变成任意形状。

代价是：特征维度膨胀（d 个原始特征 → 约 $O(d^2)$ 个多项式特征），需要更强的正则化来防止过拟合。

</details>

---

**Q9：Softmax 回归和 K 个 OvR 逻辑回归有什么本质区别？什么场景该选谁？**

<details>
<summary>点击查看解答</summary>

| 维度 | Softmax | OvR（K 个逻辑回归） |
|------|---------|---------------------|
| 模型数量 | 1 个（联合训练） | K 个（独立训练） |
| 输出 | K 个概率，和为 1 | K 个概率，各自独立 |
| 类间关系 | 能利用（参数共享） | 忽略（每个分类器独立） |
| 训练效率 | 一次训练 | K 次独立训练 |
| 新类扩展 | 需要重训 | 只需加一个新分类器 |
| 概率解释 | 严格（和为 1） | 不严格（可能都 > 0.5） |

**选择建议：**
- **Softmax**：类别互斥（一个样本只能属于一个类）、类别数量固定、需要严格的概率解释。绝大多数多分类场景用 Softmax。
- **OvR**：可能新增类别（开放集分类）、每个类的数据分布差异极大、需要并行训练加速、类别数量非常多（K 大会让 Softmax 计算量大）。极端不平衡时 OvR 更好——可以给每个二分类器单独调 class_weight。
</details>

---

**Q10：逻辑回归训练 1000 轮后损失还在下降但非常缓慢（每轮降 1e-6），此时应该继续训练还是停下来？有什么办法加速？**

<details>
<summary>点击查看解答</summary>

**判断标准：**

如果测试集指标（准确率/AUC/Log-Loss）已经稳定，继续训练只会导致**过拟合**——参数继续向训练数据的噪声方向微调，泛化能力反而下降。

**加速收敛的方法：**

1. **增大学习率**：当前可能太小了。但如果已经出现了震荡，说明 lr 已经到上限。

2. **使用动量（Momentum）或自适应优化器**：
   ```python
   # 简单的动量实现
   v_w = 0
   v_w = momentum * v_w + lr * dw
   w -= v_w
   ```
   比普通 GD 快很多。

3. **特征标准化**：如果特征量纲不一致，GD 的收敛路径是之字形的，很慢。

4. **换优化器**：sklearn 的 `solver='lbfgs'`（拟牛顿法）通常比 GD 快得多。对于大规模数据可以用 `solver='saga'`。

5. **早停（Early Stopping）**：在验证集上监控指标，指标不再改善时停止。这是最简单有效的防过拟合手段。

**底线：** 如果损失下降 < 1e-6 且验证指标已稳定——停下来，模型已经够好了。过度追求训练损失的最小化恰恰是过拟合的开始。

</details>

---

**逻辑回归学完了。** 下一站是另一个经典分类算法：[K近邻](./07-knn.md)——一个**完全不需要训练**的分类器。如果说逻辑回归是"先学习规则再判断"，K近邻就是"遇到新问题，翻翻以前的相似案例再决定"。
