# 神经网络入门：从感知机到反向传播——全手工推演

> **本章在 ML 学习路径中的位置**：神经网络与深度学习第一站。如果说前十四章的各种算法是在各自的领域里单打独斗，那么神经网络就是一种"元算法"——它用一个统一的结构（层层堆叠的神经元）去逼近任何你想要的函数。本章从最原始的感知机出发，用纸笔手工推演 AND/OR/XOR，设定具体权重一步步算出 2→4→1 网络的前向传播和反向传播的每一个梯度值，然后用 numpy 从零实现整个训练流程。
>
> 前置章节：[支持向量机](./08-svm.md)、[逻辑回归](./06-logistic-regression.md)、[最优化](./math-optimization.md)
> 下一章：[CNN与RNN基础](./16-cnn-rnn-basics.md)

---

神经网络在 2012 年 AlexNet 在 ImageNet 上一战成名后，已成为机器学习领域最耀眼的明星。ChatGPT、Stable Diffusion、AlphaFold——几乎所有令人震撼的 AI 成果背后，都是神经网络的某种变体。但剥离华丽的包装，神经网络的核心思想极其朴素：**一堆简单的计算单元，层层堆叠，就能逼近任何函数**。

本章的独特之处：我们不满足于"看懂公式"，而是要**亲手算一遍**。你将用具体数字走完感知机的 AND/OR/XOR 推演，用指定权重完成 2→4→1 网络的前向传播，再用链式法则逐项算出反向传播的 14 个梯度值，最后把这些手工推导变成可运行的 numpy 代码。读完本章，"反向传播"对你而言将不再是一个抽象概念，而是一组你亲手算过的数字。

读完本章你将能够：

- 手工推演感知机在 AND/OR/XOR 上的完整计算，理解线性不可分的数学本质
- 用指定的具体权重，一步一步完成 2→4→1 MLP 的前向传播
- 用链式法则手工计算同一网络上所有 14 个参数的梯度
- 用数值梯度检验验证手工梯度的正确性
- 从零用 numpy 实现完整的 MLP 训练循环，解决 XOR 问题
- 回答 10 道思考题，彻底消化神经网络的每一个核心概念

---

## 1. 神经网络的直觉

### 直觉理解

1943 年，神经生理学家 McCulloch 和数学家 Pitts 发表了一篇划时代的论文，提出了人工神经元的第一个数学模型。灵感来自大脑：人的大脑由约 860 亿个**神经元**组成，每个神经元接收来自其他神经元的电信号，当输入信号累计超过某个阈值时，该神经元"激活"（fire），向下一批神经元输出信号。

**单个神经元做的事情**：接收一串数字 $x_1, x_2, \dots, x_n$，给每个数字乘上一个权重，把所有结果加起来，再过一个"开关函数"——输出 1 还是 0。这个简单到几乎微不足道的计算单元，就是一切深度神经网络的"原子"。

**但真正的魔力来自层叠**。单个神经元只能画一条直线（线性分类器），但把几百个、几千个神经元串起来、叠起来，这个网络就能表达极其复杂的函数——就像单个乐高积木很简单，但堆起来可以搭城堡。这背后是一条深刻的数学定理：

> **万能近似定理（Universal Approximation Theorem）**：一个包含至少一个隐藏层、使用非线性激活函数的前馈神经网络，只要有足够多的隐藏神经元，就能以任意精度逼近任何连续函数。

### 应用连接

从本章的最小 MLP（不到 100 行代码、5 个神经元）到 GPT-4（约 1.8 万亿参数），遵循的是**完全相同**的原理——前向传播、损失函数、反向传播、梯度下降。把本章的核心概念吃透，你就拿到了理解一切深度学习模型的钥匙。

---

## 2. 感知机——手工推演 AND、OR、XOR

### 2.1 感知机的数学定义

1957 年，Frank Rosenblatt 造出了世界上第一台"学习机器"——Mark I 感知机。它只有一层神经元，能做最简单的二分类。感知机的数学表达极其简洁：

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(w^T x + b)
$$

其中 $f$ 是**阶跃函数**（step function）：

$$
f(z) = \begin{cases} 1, & z \ge 0 \\[4pt] 0, & z < 0 \end{cases}
$$

当加权和 $z = w^T x + b \ge 0$ 时输出 1（正类），否则输出 0（负类）。几何上，$w^T x + b = 0$ 定义了一条直线（二维）或超平面（高维），感知机就是用这条线把空间切成两半。

感知机的参数更新规则（感知机学习算法）：

$$
w \leftarrow w + \eta \cdot (y_{true} - y_{pred}) \cdot x
$$
$$
b \leftarrow b + \eta \cdot (y_{true} - y_{pred})
$$

这个更新规则极富几何直觉：只有当预测错了（$y_{true} \neq y_{pred}$），才更新权重。更新方向是使分界线向正确分类的方向旋转——错了就往对的方向挪一挪，对了就不动。这就是最早的"从错误中学习"的数学实现。

### 2.2 手工推演：AND 门

AND 逻辑的真值表：`(0,0)→0`, `(0,1)→0`, `(1,0)→0`, `(1,1)→1`。只有两个输入都为 1 时才输出 1。

我们手动设定一组权重和偏置，然后逐条验证。取 $w_1 = 0.5,\ w_2 = 0.5,\ b = -0.7$：

| 输入 $(x_1, x_2)$ | $z = w_1 x_1 + w_2 x_2 + b$ | $z \ge 0$? | 预测 $\hat{y}$ | 真实 $y$ | ✓/✗ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| (0, 0) | $0.5 \times 0 + 0.5 \times 0 - 0.7 = -0.7$ | 否 | 0 | 0 | ✓ |
| (0, 1) | $0.5 \times 0 + 0.5 \times 1 - 0.7 = -0.2$ | 否 | 0 | 0 | ✓ |
| (1, 0) | $0.5 \times 1 + 0.5 \times 0 - 0.7 = -0.2$ | 否 | 0 | 0 | ✓ |
| (1, 1) | $0.5 \times 1 + 0.5 \times 1 - 0.7 = +0.3$ | **是** | 1 | 1 | ✓ |

四条全对。几何解释：直线 $0.5x_1 + 0.5x_2 - 0.7 = 0$（即 $x_1 + x_2 = 1.4$）将 (1,1) 划到正侧，(0,0)、(0,1)、(1,0) 划到负侧——一条斜线完美分开。

### 2.3 手工推演：OR 门

OR 逻辑的真值表：`(0,0)→0`, `(0,1)→1`, `(1,0)→1`, `(1,1)→1`。只要有一个输入为 1 就输出 1。

取 $w_1 = 0.5,\ w_2 = 0.5,\ b = -0.2$：

| 输入 $(x_1, x_2)$ | $z = w_1 x_1 + w_2 x_2 + b$ | $z \ge 0$? | 预测 $\hat{y}$ | 真实 $y$ | ✓/✗ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| (0, 0) | $0.5 \times 0 + 0.5 \times 0 - 0.2 = -0.2$ | 否 | 0 | 0 | ✓ |
| (0, 1) | $0.5 \times 0 + 0.5 \times 1 - 0.2 = +0.3$ | **是** | 1 | 1 | ✓ |
| (1, 0) | $0.5 \times 1 + 0.5 \times 0 - 0.2 = +0.3$ | **是** | 1 | 1 | ✓ |
| (1, 1) | $0.5 \times 1 + 0.5 \times 1 - 0.2 = +0.8$ | **是** | 1 | 1 | ✓ |

四条全对。注意到 AND 和 OR 只差了一个偏置值——AND 的阈值高（需要两个输入都强），OR 的阈值低（一个就够）。偏置 $b$ 控制着神经元"激活的难易程度"。

### 2.4 手工推演：XOR——感知机的滑铁卢

XOR（异或）逻辑的真值表：`(0,0)→0`, `(0,1)→1`, `(1,0)→1`, `(1,1)→0`。"相同为 0，不同为 1"。

我们先尝试找一组能用的权重。试 $w_1 = 1.0,\ w_2 = 0.5,\ b = -0.3$：

| 输入 $(x_1, x_2)$ | $z$ | $z \ge 0$? | 预测 | 真实 | ✓/✗ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| (0, 0) | $1 \times 0 + 0.5 \times 0 - 0.3 = -0.3$ | 否 | 0 | 0 | ✓ |
| (0, 1) | $1 \times 0 + 0.5 \times 1 - 0.3 = +0.2$ | 是 | 1 | 1 | ✓ |
| (1, 0) | $1 \times 1 + 0.5 \times 0 - 0.3 = +0.7$ | 是 | 1 | 1 | ✓ |
| (1, 1) | $1 \times 1 + 0.5 \times 1 - 0.3 = +1.2$ | 是 | 1 | 0 | **✗** |

(1,1) 错了！它被分到了正类，但 XOR 要求 (1,1) 是 0。

再试另一组参数 $w_1 = 0.5,\ w_2 = -1.0,\ b = 0.3$：

| 输入 | $z$ | 预测 | 真实 | ✓/✗ |
|:---:|:---:|:---:|:---:|:---:|
| (0, 0) | $+0.3$ → 1 | 1 | 0 | **✗** |

第一行就错了。

**反证法证明 XOR 线性不可分**。假设存在 $w_1, w_2, b$ 能完美分类 XOR 的四个点，则必须同时满足：

$$
\begin{aligned}
(0,0):\quad & b < 0 \quad &\text{(输出应为 0)} \\[4pt]
(0,1):\quad & w_2 + b \ge 0 \quad &\text{(输出应为 1)} \\[4pt]
(1,0):\quad & w_1 + b \ge 0 \quad &\text{(输出应为 1)} \\[4pt]
(1,1):\quad & w_1 + w_2 + b < 0 \quad &\text{(输出应为 0)}
\end{aligned}
$$

由 $b < 0$ 和 $w_2 + b \ge 0$ 推出 $w_2 \ge -b > 0$，所以 $w_2 > 0$。
由 $b < 0$ 和 $w_1 + b \ge 0$ 推出 $w_1 \ge -b > 0$，所以 $w_1 > 0$。

现在考虑第四个不等式：
$$w_1 + w_2 + b = (w_1 + b) + w_2$$

因为 $w_1 + b \ge 0$ 且 $w_2 > 0$，所以 $(w_1 + b) + w_2 > 0$。但这与 $w_1 + w_2 + b < 0$ **矛盾**！

因此，不存在任何 $(w_1, w_2, b)$ 能同时满足四个约束。**XOR 是线性不可分的**。

几何直觉：在二维平面上画 XOR 的四个点——(0,0) 蓝、(1,0) 红、(0,1) 红、(1,1) 蓝。红蓝两点在对角线上。一条直线最多把平面切成两半，而你需要同时把 (1,0) 和 (0,1) 分到一边、(0,0) 和 (1,1) 分到另一边——这在二维空间里做不到。解决方法：**加一层隐藏层**，把输入映射到一个新的"表示空间"，让 XOR 在那个空间里变得线性可分。

### 2.5 感知机训练过程手工推演：AND 门的一轮学习

为了理解感知机"从错误中学习"的具体机制，我们手算 AND 门训练的第一轮。初始值 $w_1 = w_2 = 0,\ b = 0$，学习率 $\eta = 0.1$。

**epoch 1，第 1 步：输入 (0, 0)，标签 0**

$$
z = 0 \times 0 + 0 \times 0 + 0 = 0, \quad \hat{y} = 1\ (\text{因为 } z \ge 0)
$$

预测 1 ≠ 标签 0 → **判错，更新**：
$$
\begin{aligned}
w_1 &\leftarrow 0 + 0.1 \times (0 - 1) \times 0 = 0 \\
w_2 &\leftarrow 0 + 0.1 \times (0 - 1) \times 0 = 0 \\
b &\leftarrow 0 + 0.1 \times (0 - 1) = -0.1
\end{aligned}
$$

**epoch 1，第 2 步：输入 (0, 1)，标签 0**

$$
z = 0 \times 0 + 0 \times 1 - 0.1 = -0.1, \quad \hat{y} = 0
$$

预测 0 = 标签 0 → **正确，不更新**。权重保持 $(0, 0), -0.1$。

**epoch 1，第 3 步：输入 (1, 0)，标签 0**

$$
z = 0 \times 1 + 0 \times 0 - 0.1 = -0.1, \quad \hat{y} = 0
$$

预测 0 = 标签 0 → **正确，不更新**。

**epoch 1，第 4 步：输入 (1, 1)，标签 1**

$$
z = 0 \times 1 + 0 \times 1 - 0.1 = -0.1, \quad \hat{y} = 0
$$

预测 0 ≠ 标签 1 → **判错，更新**：
$$
\begin{aligned}
w_1 &\leftarrow 0 + 0.1 \times (1 - 0) \times 1 = 0.1 \\
w_2 &\leftarrow 0 + 0.1 \times (1 - 0) \times 1 = 0.1 \\
b &\leftarrow -0.1 + 0.1 \times (1 - 0) = 0.0
\end{aligned}
$$

一轮结束，权重从 $(0, 0), 0$ 变为 $(0.1, 0.1), 0.0$。现在验证：(1,1) 得 $0.1+0.1+0=0.2 \ge 0 \to 1$ ✓，(0,0) 得 $0 \to 1$ ✗。虽然第二轮 (0,0) 又会把偏置推回负值——感知机就是在"错误—纠正—新错误—再纠正"的循环中逐步逼近正确边界的。这就是最早的"从错误中学习"的完整过程。

### 2.6 Python 验证：感知机在 AND 上成功、XOR 上失败

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """从零实现感知机"""
    def __init__(self, lr: float = 0.1, n_iter: int = 100):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0.0
        self.errors = []

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            n_errors = 0
            for xi, yi in zip(X, y):
                y_pred = 1 if (np.dot(xi, self.w) + self.b) >= 0 else 0
                if y_pred != yi:
                    update = self.lr * (yi - y_pred)
                    self.w += update * xi
                    self.b += update
                    n_errors += 1
            self.errors.append(n_errors)
        return self


# ---- 测试：AND 逻辑（线性可分，感知机成功） ----
np.random.seed(42)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

p = Perceptron(lr=0.1, n_iter=20)
p.fit(X_and, y_and)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 左：AND 决策边界
axes[0].scatter(X_and[:, 0], X_and[:, 1], c=y_and, s=200, cmap='bwr', edgecolors='k')
x_boundary = np.array([-0.2, 1.2])
if abs(p.w[1]) > 1e-8:
    axes[0].plot(x_boundary, -(p.w[0] * x_boundary + p.b) / p.w[1], 'k-', linewidth=2)
axes[0].set_xlim(-0.2, 1.2); axes[0].set_ylim(-0.2, 1.2)
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')
axes[0].set_title('感知机：AND 逻辑 ✓', fontsize=12)

# 中：收敛曲线
axes[1].plot(range(len(p.errors)), p.errors, 'b-o', markersize=4)
axes[1].set_xlabel('迭代次数'); axes[1].set_ylabel('错误次数')
axes[1].set_title('感知机收敛曲线', fontsize=12)
axes[1].grid(True, alpha=0.3)

# 右：XOR 问题——感知机失败
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

axes[2].scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
                c='blue', s=200, edgecolors='k', label='0')
axes[2].scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
                c='red', s=200, edgecolors='k', label='1')
for w1, w2, bias in [(1, 0.5, -0.3), (-1, 0.5, 0.8), (0.5, 1, -0.5), (0.5, -1, 0.5)]:
    xb = np.linspace(-0.3, 1.3, 50)
    axes[2].plot(xb, -(w1 * xb + bias) / (w2 + 1e-8), 'gray', alpha=0.3)
axes[2].set_xlim(-0.3, 1.3); axes[2].set_ylim(-0.3, 1.3)
axes[2].set_xlabel('x1'); axes[2].set_ylabel('x2')
axes[2].set_title('XOR 问题：没有一条直线能分开 ✗', fontsize=12)
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"AND: 学习到的权重={p.w}, 偏置={p.b:.3f}")
print(f"AND 预测: {p.predict(X_and)}")
```

**AND 门**的四个点中，输出为 1（红）和输出为 0（蓝）的点可以被一条直线完美分开——这是线性可分问题，单层感知机轻松解决。

**XOR 门**则完全不同：四个点在平面上形成一个对角线格局，你找不到任何一条直线能把蓝色和红色分开。这是线性不可分问题的经典代表——也是 1969 年 Minsky 和 Papert 在《感知机》一书中用来"杀死"单层感知机研究的证据。

### 应用连接

感知机的参数更新规则 $w \leftarrow w + \eta \cdot (y - \hat{y}) \cdot x$ 已经是梯度下降的雏形——沿着"减少错误的方向"更新参数。但阶跃函数的导数要么是 0 要么不存在（在 0 处不可导），使得真正的梯度下降无法应用。下一节：用光滑的、处处可导的**激活函数**替换阶跃函数。

---

## 3. 激活函数

### 直觉理解：为什么需要非线性

如果神经网络每一层都只是线性变换（矩阵乘法），那么无论堆多少层，最终等价于一个单层线性变换：

$$
h_1 = W_1 x;\quad h_2 = W_2 h_1 = W_2 W_1 x = W_{eq} x
$$

$W_2 W_1$ 仍然是一个矩阵——堆来堆去，还是一个线性变换。**非线性激活函数的使命就是打破这种"堆了等于白堆"的线性传递链**，让每一层能学到不同的特征层次。

### 形式化定义与 Python 验证

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)


x = np.linspace(-4, 4, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

functions = [
    (sigmoid, 'Sigmoid: σ(x)=1/(1+e⁻ˣ)', 'r'),
    (tanh, 'Tanh: 2σ(2x)−1', 'g'),
    (relu, 'ReLU: max(0, x)', 'b'),
    (leaky_relu, 'LeakyReLU: max(0.01x, x)', 'm'),
    (elu, 'ELU: x>0?x:α(eˣ−1)', 'c'),
    (swish, 'Swish: x·σ(x)', 'orange'),
]

for ax, (fn, title, color) in zip(axes, functions):
    y = fn(x)
    dy = np.gradient(y, x[1] - x[0])
    ax.plot(x, y, color, linewidth=2, label='f(x)')
    ax.plot(x, dy, color, linestyle='--', alpha=0.5, label="f'(x)")
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle('常用激活函数及其导数', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 函数对比

| 激活函数 | 公式 | 输出范围 | 导数 | 优点 | 缺点 |
|----------|------|:---:|------|------|------|
| **Sigmoid** | $\sigma(x)=\frac{1}{1+e^{-x}}$ | (0,1) | $\sigma(x)(1-\sigma(x))$ | 光滑、输出有界 | 梯度消失；非零中心 |
| **Tanh** | $\tanh(x)$ | (-1,1) | $1-\tanh^2(x)$ | 零中心 | 梯度消失 |
| **ReLU** | $\max(0,x)$ | [0,∞) | $[x>0]$ | 无梯度消失；计算快 | 负区死亡 |
| **LeakyReLU** | $\max(\alpha x,x)$ | (-∞,∞) | $[x>0]+\alpha[x\le0]$ | 解决死亡问题 | 多超参数 |
| **ELU** | $x[x>0]+\alpha(e^x-1)[x\le0]$ | (-α,∞) | 见公式 | 输出均值近 0 | 计算量大 |
| **Swish** | $x\cdot\sigma(x)$ | (-∞,∞) | $\sigma(x)+x\sigma(x)(1-\sigma(x))$ | 自门控 | 计算量大 |

本章后续手工推演中，隐藏层使用 **Tanh**（导数优雅：$1-a^2$），输出层使用 **Sigmoid**（输出概率解释）。

```python
# 梯度消失演示
x_deep = np.linspace(-8, 8, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label, fn, col in [('Sigmoid', sigmoid, 'red'), ('ReLU', relu, 'blue')]:
    y = fn(x_deep)
    dy = np.gradient(y, x_deep[1] - x_deep[0])
    axes[0].plot(x_deep, y, col, linewidth=2)
    axes[1].plot(x_deep, dy, col, linewidth=2, label=f"{label} 导数")

axes[0].set_title('函数值'); axes[1].set_title('导数值（链式法则的因子）')
for ax in axes:
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.2)
axes[1].legend()
plt.suptitle('梯度消失问题的根源', fontweight='bold')
plt.tight_layout()
plt.show()
```

**历史转折**：2012 年 AlexNet 用 ReLU 替代了 Sigmoid/Tanh，训练速度提升数倍——ReLU 在正区导数恒为 1，梯度传播不衰减。从阶跃函数 → Sigmoid → Tanh → ReLU，激活函数的进化史就是深度学习从"学不动"到"能学很深"的进化史。

### 3.1 激活函数导数详细推导

反向传播中要用到每个激活函数的导数，这里集中推导以备查阅：

**Sigmoid**：$\sigma(x) = \frac{1}{1+e^{-x}}$

$$
\begin{aligned}
\sigma'(x) &= \frac{d}{dx}(1+e^{-x})^{-1} = -(1+e^{-x})^{-2} \cdot (-e^{-x}) \\
           &= \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} \\
           &= \sigma(x) \cdot (1 - \sigma(x))
\end{aligned}
$$

这是梯度计算中最优美的结果之一——Sigmoid 的导数可以用自身的函数值表示。

**Tanh**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

（推导可用商的导数法则或利用 $\tanh(x) = 2\sigma(2x)-1$ 和上面 Sigmoid 导数的结果。）

**ReLU**：$\text{ReLU}(x) = \max(0, x)$

$$
\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x < 0 \\ \text{未定义}, & x = 0 \end{cases}
$$

在实际实现中，$x=0$ 处的导数通常取 0 或 1（不影响训练结果，因为浮点数几乎不会恰好落在 0 上）。

**LeakyReLU**：$\text{LReLU}(x) = \max(\alpha x, x)$

$$
\text{LReLU}'(x) = \begin{cases} 1, & x > 0 \\ \alpha, & x \le 0 \end{cases}
$$

$\alpha$ 通常取 0.01。当 $\alpha=0$ 时退化为 ReLU。

**ELU**：$\text{ELU}(x) = \begin{cases} x, & x > 0 \\ \alpha(e^x - 1), & x \le 0 \end{cases}$

$$
\text{ELU}'(x) = \begin{cases} 1, & x > 0 \\ \alpha e^x = \text{ELU}(x) + \alpha, & x \le 0 \end{cases}
$$

ELU 负区导数仍为正（只要 $x > -\infty$），不会出现梯度完全为零的情况。

### 3.2 AI 寒冬与深度学习的复兴——一段简史

理解激活函数的进化，需要了解神经网络经历的三次起伏：

| 时期 | 事件 | 激活函数 |
|:---|:---|:---|
| 1957 | Rosenblatt 发明感知机 | 阶跃函数 |
| 1969 | Minsky & Papert 证明感知机连 XOR 都学不会 | — |
| 1970s | **第一次 AI 寒冬**：神经网络研究经费被砍 | — |
| 1986 | Rumelhart, Hinton & Williams 发表反向传播算法 | Sigmoid |
| 1990s | **第二次 AI 寒冬**：SVM 崛起，Sigmoid 的梯度消失使深层网络无法训练 | — |
| 2012 | AlexNet + ReLU + GPU 在 ImageNet 上碾压传统方法 | ReLU |
| 2014-至今 | GAN、ResNet、Transformer、GPT 相继问世 | ReLU/GELU/Swish |

两次寒冬的根源都指向同一个技术瓶颈：**学不动**。第一次是因为单层结构学不了 XOR，第二次是因为 Sigmoid 的梯度消失让深层网络学不了。ReLU 解决了第二次瓶颈，打开了深度学习的大门。今天的 GELU（Transformer 中的标配）本质上是 Swish 的一种近似——激活函数的进化仍在继续。

### 3.3 应用连接

在本章的反向传播手工推演中，隐藏层使用 Tanh（$g'(z) = 1-a^2$，可以用已缓存的 $a$ 直接计算），输出层使用 Sigmoid（$\sigma'(z) = \hat{y}(1-\hat{y})$，同样无需重新计算）。这种"前向缓存值反向复用"的技巧是所有框架自动微分的核心优化策略。

---

## 4. 多层感知机前向传播：2→4→1 全手工推演

### 4.1 网络结构

本章的核心手工推演对象：一个两层 MLP（输入层不计入层数）。

```
输入层 (2)  →  隐藏层 (4, Tanh)  →  输出层 (1, Sigmoid)
   x₁,x₂    →    h₁,h₂,h₃,h₄     →       ŷ
```

数学表达：

$$
\begin{aligned}
z^{[1]} &= x\,W^{[1]} + b^{[1]} \quad &\text{(2)→(4) 线性变换} \\
a^{[1]} &= \tanh(z^{[1]}) \quad &\text{隐藏层激活} \\
z^{[2]} &= a^{[1]}\,W^{[2]} + b^{[2]} \quad &\text{(4)→(1) 线性变换} \\
\hat{y} = a^{[2]} &= \sigma(z^{[2]}) \quad &\text{输出层 Sigmoid}
\end{aligned}
$$

其中 $x$ 是 1×2 的行向量（单个样本），$W^{[1]}$ 是 2×4，$b^{[1]}$ 是 1×4，$W^{[2]}$ 是 4×1，$b^{[2]}$ 是 1×1。

### 4.2 设定具体权重

为了手工推演，我们设定以下权重（这些值精心挑选，使得中间计算量适中且可追溯）：

**第一层权重矩阵** $W^{[1]}$（2×4）：

$$
W^{[1]} = \begin{bmatrix}
 0.8 & -0.5 &  0.3 & -0.7 \\
-0.4 &  0.9 & -0.6 &  0.2
\end{bmatrix}
$$

**第一层偏置** $b^{[1]}$（1×4）：

$$
b^{[1]} = \begin{bmatrix} 0.1 & -0.2 & 0.0 & 0.3 \end{bmatrix}
$$

**第二层权重矩阵** $W^{[2]}$（4×1）：

$$
W^{[2]} = \begin{bmatrix}
 0.6 \\
-0.3 \\
 0.5 \\
-0.4
\end{bmatrix}
$$

**第二层偏置** $b^{[2]}$（1×1）：

$$
b^{[2]} = \begin{bmatrix} 0.1 \end{bmatrix}
$$

### 4.3 输入样本 $x = [1, 0]$，真实标签 $y = 1$ 的完整前向传播

我们选 XOR 问题中的一条样本：输入 (1, 0)，目标输出 1（XOR 表中第二行）。

#### 步骤 1：计算隐藏层加权和 $z^{[1]}$

$z^{[1]} = x \cdot W^{[1]} + b^{[1]}$，逐元素计算：

$$
\begin{aligned}
z^{[1]}_1 &= 1 \times 0.8 + 0 \times (-0.4) + 0.1 = 0.9 \\
z^{[1]}_2 &= 1 \times (-0.5) + 0 \times 0.9 - 0.2 = -0.7 \\
z^{[1]}_3 &= 1 \times 0.3 + 0 \times (-0.6) + 0.0 = 0.3 \\
z^{[1]}_4 &= 1 \times (-0.7) + 0 \times 0.2 + 0.3 = -0.4
\end{aligned}
$$

所以 $z^{[1]} = [0.9,\; -0.7,\; 0.3,\; -0.4]$。

#### 步骤 2：隐藏层激活 $a^{[1]} = \tanh(z^{[1]})$

逐元素计算 tanh（$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$）：

$$
\begin{aligned}
a^{[1]}_1 &= \tanh(0.9): \quad e^{0.9} \approx 2.4596,\; e^{-0.9} \approx 0.4066 \\
          &= \frac{2.4596 - 0.4066}{2.4596 + 0.4066} = \frac{2.0530}{2.8662} \approx 0.7163 \\[8pt]
a^{[1]}_2 &= \tanh(-0.7) = -\tanh(0.7) \approx -0.6044 \\[8pt]
a^{[1]}_3 &= \tanh(0.3) \approx 0.2913 \\[8pt]
a^{[1]}_4 &= \tanh(-0.4) = -\tanh(0.4) \approx -0.3799
\end{aligned}
$$

所以 $a^{[1]} = [0.7163,\; -0.6044,\; 0.2913,\; -0.3799]$。

#### 步骤 3：计算输出层加权和 $z^{[2]}$

$z^{[2]} = a^{[1]} \cdot W^{[2]} + b^{[2]}$：

$$
\begin{aligned}
z^{[2]} &= 0.7163 \times 0.6 + (-0.6044) \times (-0.3) + 0.2913 \times 0.5 + (-0.3799) \times (-0.4) + 0.1 \\
        &= 0.4298 + 0.1813 + 0.1457 + 0.1520 + 0.1 \\
        &= 1.0088
\end{aligned}
$$

#### 步骤 4：输出层激活 $\hat{y} = \sigma(z^{[2]})$

$$
\hat{y} = \frac{1}{1 + e^{-1.0088}} \approx \frac{1}{1 + 0.3647} = 0.7328
$$

#### 步骤 5：计算损失（MSE）

$$
L = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.7328 - 1.0)^2 = \frac{1}{2} \times 0.07138 = 0.03569
$$

**前向传播总结**：输入 [1, 0] 经过 2→4→1 网络，预测值 $\hat{y} = 0.7328$，与真实值 1.0 的差距体现在损失 0.03569 上。接下来，我们要通过反向传播计算这个损失对 14 个可学习参数中每一个的梯度。

### 4.4 输入样本 $x = [0, 1]$ 的快速验证（供对比）

对第二个 XOR 样本 (0, 1)，目标 1：

$$
\begin{aligned}
z^{[1]} &= [0 \times 0.8 + 1 \times (-0.4) + 0.1,\; 0 \times (-0.5) + 1 \times 0.9 - 0.2,\; 0 \times 0.3 + 1 \times (-0.6) + 0.0,\; 0 \times (-0.7) + 1 \times 0.2 + 0.3] \\
        &= [-0.3,\; 0.7,\; -0.6,\; 0.5] \\[4pt]
a^{[1]} &= [\tanh(-0.3),\; \tanh(0.7),\; \tanh(-0.6),\; \tanh(0.5)] \\
        &\approx [-0.2913,\; 0.6044,\; -0.5370,\; 0.4621] \\[4pt]
z^{[2]} &= (-0.2913)(0.6) + (0.6044)(-0.3) + (-0.5370)(0.5) + (0.4621)(-0.4) + 0.1 \\
        &= -0.1748 - 0.1813 - 0.2685 - 0.1848 + 0.1 = -0.7094 \\[4pt]
\hat{y} &= \sigma(-0.7094) \approx 0.3299
\end{aligned}
$$

对于 (0, 1)，预测 0.3299——远低于目标 1.0。

### 4.5 输入样本 $x = [1, 1]$ 的完整前向传播

对第三个 XOR 样本 (1, 1)，目标 0（XOR 表中"相同为 0"）：

**步骤 1：** $z^{[1]} = x W^{[1]} + b^{[1]}$

$$
\begin{aligned}
z^{[1]}_1 &= 1 \times 0.8 + 1 \times (-0.4) + 0.1 = 0.5 \\
z^{[1]}_2 &= 1 \times (-0.5) + 1 \times 0.9 - 0.2 = 0.2 \\
z^{[1]}_3 &= 1 \times 0.3 + 1 \times (-0.6) + 0.0 = -0.3 \\
z^{[1]}_4 &= 1 \times (-0.7) + 1 \times 0.2 + 0.3 = -0.2
\end{aligned}
$$

**步骤 2：** $a^{[1]} = \tanh(z^{[1]})$

$$
\begin{aligned}
a^{[1]}_1 &= \tanh(0.5) \approx 0.4621 \\
a^{[1]}_2 &= \tanh(0.2) \approx 0.1974 \\
a^{[1]}_3 &= \tanh(-0.3) \approx -0.2913 \\
a^{[1]}_4 &= \tanh(-0.2) \approx -0.1974
\end{aligned}
$$

**步骤 3：** $z^{[2]} = a^{[1]} W^{[2]} + b^{[2]}$

$$
\begin{aligned}
z^{[2]} &= 0.4621 \times 0.6 + 0.1974 \times (-0.3) + (-0.2913) \times 0.5 + (-0.1974) \times (-0.4) + 0.1 \\
        &= 0.2773 - 0.0592 - 0.1457 + 0.0790 + 0.1 = 0.2514
\end{aligned}
$$

**步骤 4：** $\hat{y} = \sigma(0.2514) \approx 0.5625$

**步骤 5：** $L = \frac{1}{2}(0.5625 - 0)^2 = 0.1582$

预测 0.5625，而目标为 0——误差很大（损失 0.1582 vs 之前的 0.0357）。网络对 (1,1) 的预测最差。

### 4.6 四个样本前向传播汇总表

| 输入 $(x_1, x_2)$ | $z^{[1]}$ | $a^{[1]}$ (tanh) | $z^{[2]}$ | $\hat{y}$ | 真实 $y$ | 损失 $\frac{1}{2}(\hat{y}-y)^2$ |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| (0, 0) | [0.1, -0.2, 0.0, 0.3] | [0.100, -0.197, 0.0, 0.291] | 0.2576 | 0.5641 | 0 | 0.1591 |
| (0, 1) | [-0.3, 0.7, -0.6, 0.5] | [-0.291, 0.604, -0.537, 0.462] | -0.7094 | 0.3299 | 1 | 0.2245 |
| (1, 0) | [0.9, -0.7, 0.3, -0.4] | [0.716, -0.604, 0.291, -0.380] | **1.0088** | **0.7328** | 1 | **0.0357** |
| (1, 1) | [0.5, 0.2, -0.3, -0.2] | [0.462, 0.197, -0.291, -0.197] | 0.2514 | 0.5625 | 0 | 0.1582 |

**总损失（样本平均）**：$\frac{0.1591 + 0.2245 + 0.0357 + 0.1582}{4} = 0.1444$。

当前随机权重表现很差——(0,1) 和 (1,0) 应该有高预测但分别只有 0.33 和 0.73，(0,0) 和 (1,1) 应该有低预测但都约 0.56。训练的目标就是通过反向传播和梯度下降，让四个预测分别趋近 0, 1, 1, 0。

### 4.7 附：MSE vs 交叉熵损失

对于二分类问题，两种主流损失函数对比：

**MSE（均方误差）**：

$$L_{MSE} = \frac{1}{2}(\hat{y} - y)^2, \qquad \frac{\partial L_{MSE}}{\partial \hat{y}} = \hat{y} - y$$

配合 Sigmoid 输出：$\frac{\partial L_{MSE}}{\partial z} = (\hat{y} - y) \cdot \hat{y}(1-\hat{y})$。

**交叉熵（Cross-Entropy）**：

$$L_{CE} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})], \qquad \frac{\partial L_{CE}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

配合 Sigmoid 输出时，有一个优美的化简：

$$\frac{\partial L_{CE}}{\partial z} = \hat{y} - y$$

为什么交叉熵 + Sigmoid 是二分类的标准配方？两个原因：

1. **梯度不衰减**：当 $\hat{y} \to 0$ 或 $\hat{y} \to 1$ 时，Sigmoid 的导数 $\hat{y}(1-\hat{y}) \to 0$。MSE 的梯度被这个因子"掐死"——预测越极端，学习越慢。交叉熵的 $\frac{\partial L}{\partial z} = \hat{y} - y$ 完全不受 Sigmoid 导数影响。用第 5 节的数值为例：若用交叉熵，$\delta^{[2]} = \hat{y} - y = -0.2672$，不含 0.1958 的衰减因子。

2. **概率解释**：交叉熵来自最大似然估计——$L_{CE} = -\log P(y|x;\theta)$，直接优化模型在真实标签上的概率。MSE 则没有这个概率解释。

本章坚持用 MSE 做手工推演，因为它的链式法则步骤更直观（多了一个 $\frac{\partial L}{\partial \hat{y}}$ 的中间环节），让反向传播的每一步更清晰可辨。**实际项目中，二分类输出层请用 Sigmoid + 交叉熵。**

### 4.8 应用连接：前向传播总结

10 次运算（4 加权和 + 4 tanh + 1 加权和 + 1 sigmoid），四个样本总损失 0.1444。当前网络表现很差，这恰恰给了反向传播发挥的空间——下一节，我们将选取 (1,0) 样本，在同一组权重上手工推导全部 14 个参数的梯度。

---

## 5. 反向传播：全手工计算 14 个梯度

### 5.1 链式法则回顾

反向传播的本质是**链式法则系统化执行**。对复合函数 $L = f(g(h(x)))$，其导数：

$$
\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}
$$

在神经网络中，我们从输出端的损失 $L$ 出发，沿着计算图逐层往回传播梯度信号。每个节点接收"上游传来的梯度"，乘以本节点的"局部梯度"，得到该节点对输入的梯度，继续往下传。同时，对于有权重的节点，还要计算损失对该权重的梯度供参数更新使用。

**本节继续使用第 4 节完全相同的网络和样本**：$x = [1, 0]$，$y = 1$，已缓存的前向值：

$$
\begin{aligned}
x &= [1,\; 0] \\
z^{[1]} &= [0.9,\; -0.7,\; 0.3,\; -0.4] \\
a^{[1]} &= [0.7163,\; -0.6044,\; 0.2913,\; -0.3799] \\
z^{[2]} &= 1.0088 \\
\hat{y} &= 0.7328 \\
L &= 0.03569
\end{aligned}
$$

需要求梯度的参数共有 **14 个**：$W^{[1]}$ 的 8 个元素 + $b^{[1]}$ 的 4 个元素 + $W^{[2]}$ 的 1 个元素 + $b^{[2]}$ 的 1 个元素。下面逐一计算。

### 5.2 第一步：损失对输出的梯度

MSE 损失 $L = \frac{1}{2}(\hat{y} - y)^2$：

$$
\boxed{\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 0.7328 - 1.0 = -0.2672}
$$

### 5.3 第二步：$\hat{y}$ 对 $z^{[2]}$ 的梯度（Sigmoid 导数）

Sigmoid 的导数：$\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$。

$$
\frac{\partial \hat{y}}{\partial z^{[2]}} = \hat{y}(1 - \hat{y}) = 0.7328 \times 0.2672 = 0.1958
$$

**输出层的"误差信号"** $\delta^{[2]}$（损失对 $z^{[2]}$ 的梯度）：

$$
\boxed{\delta^{[2]} = \frac{\partial L}{\partial z^{[2]}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}} = (-0.2672) \times 0.1958 = -0.05231}
$$

### 5.4 第三步：$z^{[2]}$ 对 $W^{[2]}$ 和 $b^{[2]}$ 的梯度

$z^{[2]} = a^{[1]} \cdot W^{[2]} + b^{[2]}$：

对 $W^{[2]}$（4×1 矩阵）的每个元素：$\frac{\partial z^{[2]}}{\partial W^{[2]}_i} = a^{[1]}_i$。

对 $b^{[2]}$：$\frac{\partial z^{[2]}}{\partial b^{[2]}} = 1$。

应用链式法则：

$$
\begin{aligned}
\frac{\partial L}{\partial W^{[2]}_1} &= a^{[1]}_1 \cdot \delta^{[2]} = 0.7163 \times (-0.05231) = \boxed{-0.03747} \\[4pt]
\frac{\partial L}{\partial W^{[2]}_2} &= a^{[1]}_2 \cdot \delta^{[2]} = (-0.6044) \times (-0.05231) = \boxed{+0.03162} \\[4pt]
\frac{\partial L}{\partial W^{[2]}_3} &= a^{[1]}_3 \cdot \delta^{[2]} = 0.2913 \times (-0.05231) = \boxed{-0.01524} \\[4pt]
\frac{\partial L}{\partial W^{[2]}_4} &= a^{[1]}_4 \cdot \delta^{[2]} = (-0.3799) \times (-0.05231) = \boxed{+0.01987} \\[4pt]
\frac{\partial L}{\partial b^{[2]}} &= 1 \cdot \delta^{[2]} = \boxed{-0.05231}
\end{aligned}
$$

### 5.5 第四步：损失对隐藏层输出 $a^{[1]}$ 的梯度

链式法则：$\frac{\partial L}{\partial a^{[1]}} = \delta^{[2]} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}$。

由于 $z^{[2]} = a^{[1]} \cdot W^{[2]} + b^{[2]}$，所以 $\frac{\partial z^{[2]}}{\partial a^{[1]}_i} = W^{[2]}_i$（这是一个标量对向量的梯度，结果是一个行向量）。

具体来说，$\delta^{[2]}$ 是标量 -0.05231，它通过 $W^{[2]}$ 的转置"分发"到 $a^{[1]}$ 的每个元素：

$$
\begin{aligned}
\frac{\partial L}{\partial a^{[1]}_1} &= \delta^{[2]} \cdot W^{[2]}_1 = (-0.05231) \times 0.6 = \boxed{-0.03139} \\[4pt]
\frac{\partial L}{\partial a^{[1]}_2} &= \delta^{[2]} \cdot W^{[2]}_2 = (-0.05231) \times (-0.3) = \boxed{+0.01569} \\[4pt]
\frac{\partial L}{\partial a^{[1]}_3} &= \delta^{[2]} \cdot W^{[2]}_3 = (-0.05231) \times 0.5 = \boxed{-0.02616} \\[4pt]
\frac{\partial L}{\partial a^{[1]}_4} &= \delta^{[2]} \cdot W^{[2]}_4 = (-0.05231) \times (-0.4) = \boxed{+0.02093}
\end{aligned}
$$

### 5.6 第五步：$a^{[1]}$ 对 $z^{[1]}$ 的梯度（Tanh 导数）

Tanh 的导数：$\tanh'(z) = 1 - \tanh^2(z) = 1 - (a^{[1]})^2$。

$$
\begin{aligned}
\frac{\partial a^{[1]}_1}{\partial z^{[1]}_1} &= 1 - (0.7163)^2 = 1 - 0.5131 = 0.4869 \\[4pt]
\frac{\partial a^{[1]}_2}{\partial z^{[1]}_2} &= 1 - (-0.6044)^2 = 1 - 0.3653 = 0.6347 \\[4pt]
\frac{\partial a^{[1]}_3}{\partial z^{[1]}_3} &= 1 - (0.2913)^2 = 1 - 0.0848 = 0.9152 \\[4pt]
\frac{\partial a^{[1]}_4}{\partial z^{[1]}_4} &= 1 - (-0.3799)^2 = 1 - 0.1443 = 0.8557
\end{aligned}
$$

**隐藏层的误差信号** $\delta^{[1]}$（逐元素乘积 $\odot$）：

$$
\begin{aligned}
\delta^{[1]}_1 = \frac{\partial L}{\partial z^{[1]}_1}
    &= \frac{\partial L}{\partial a^{[1]}_1} \cdot \frac{\partial a^{[1]}_1}{\partial z^{[1]}_1}
    = (-0.03139) \times 0.4869 = \boxed{-0.01528} \\[4pt]
\delta^{[1]}_2 &= 0.01569 \times 0.6347 = \boxed{+0.00996} \\[4pt]
\delta^{[1]}_3 &= (-0.02616) \times 0.9152 = \boxed{-0.02394} \\[4pt]
\delta^{[1]}_4 &= 0.02093 \times 0.8557 = \boxed{+0.01791}
\end{aligned}
$$

### 5.7 第六步：$z^{[1]}$ 对 $W^{[1]}$ 和 $b^{[1]}$ 的梯度

$z^{[1]} = x \cdot W^{[1]} + b^{[1]}$。对 $W^{[1]}_{jk}$（第 $j$ 个输入到第 $k$ 个隐藏神经元）：

$$
\frac{\partial z^{[1]}_k}{\partial W^{[1]}_{jk}} = x_j
$$

所以 $\frac{\partial L}{\partial W^{[1]}_{jk}} = x_j \cdot \delta^{[1]}_k$。

$$W^{[1]} = \begin{bmatrix} W_{11} & W_{12} & W_{13} & W_{14} \\ W_{21} & W_{22} & W_{23} & W_{24} \end{bmatrix}$$

对于输入 $x = [1, 0]$（注意 $x_2 = 0$）：

$$
\begin{aligned}
\frac{\partial L}{\partial W^{[1]}_{11}} &= x_1 \cdot \delta^{[1]}_1 = 1 \times (-0.01528) = \boxed{-0.01528} \\[4pt]
\frac{\partial L}{\partial W^{[1]}_{12}} &= x_1 \cdot \delta^{[1]}_2 = 1 \times 0.00996 = \boxed{+0.00996} \\[4pt]
\frac{\partial L}{\partial W^{[1]}_{13}} &= x_1 \cdot \delta^{[1]}_3 = 1 \times (-0.02394) = \boxed{-0.02394} \\[4pt]
\frac{\partial L}{\partial W^{[1]}_{14}} &= x_1 \cdot \delta^{[1]}_4 = 1 \times 0.01791 = \boxed{+0.01791} \\[4pt]
\frac{\partial L}{\partial W^{[1]}_{21}} &= x_2 \cdot \delta^{[1]}_1 = 0 \times (-0.01528) = \boxed{0.00000} \\[4pt]
\frac{\partial L}{\partial W^{[1]}_{22}} &= x_2 \cdot \delta^{[1]}_2 = 0 \times 0.00996 = \boxed{0.00000} \\[4pt]
\frac{\partial L}{\partial W^{[1]}_{23}} &= x_2 \cdot \delta^{[1]}_3 = 0 \times (-0.02394) = \boxed{0.00000} \\[4pt]
\frac{\partial L}{\partial W^{[1]}_{24}} &= x_2 \cdot \delta^{[1]}_4 = 0 \times 0.01791 = \boxed{0.00000}
\end{aligned}
$$

对 $b^{[1]}$（每个偏置对应一个隐藏神经元）：

$$
\frac{\partial L}{\partial b^{[1]}_k} = \delta^{[1]}_k
$$

$$
\begin{aligned}
\frac{\partial L}{\partial b^{[1]}_1} &= \boxed{-0.01528} \\[4pt]
\frac{\partial L}{\partial b^{[1]}_2} &= \boxed{+0.00996} \\[4pt]
\frac{\partial L}{\partial b^{[1]}_3} &= \boxed{-0.02394} \\[4pt]
\frac{\partial L}{\partial b^{[1]}_4} &= \boxed{+0.01791}
\end{aligned}
$$

### 5.8 14 个梯度完整汇总

| # | 参数 | 梯度值 | 所在层 |
|:--:|:---|:---:|:---|
| 1 | $W^{[1]}_{11}$ | **-0.01528** | 隐藏层权重 |
| 2 | $W^{[1]}_{12}$ | +0.00996 | 隐藏层权重 |
| 3 | $W^{[1]}_{13}$ | **-0.02394** | 隐藏层权重 |
| 4 | $W^{[1]}_{14}$ | +0.01791 | 隐藏层权重 |
| 5 | $W^{[1]}_{21}$ | 0.00000 | 隐藏层权重 |
| 6 | $W^{[1]}_{22}$ | 0.00000 | 隐藏层权重 |
| 7 | $W^{[1]}_{23}$ | 0.00000 | 隐藏层权重 |
| 8 | $W^{[1]}_{24}$ | 0.00000 | 隐藏层权重 |
| 9 | $b^{[1]}_1$ | -0.01528 | 隐藏层偏置 |
| 10 | $b^{[1]}_2$ | +0.00996 | 隐藏层偏置 |
| 11 | $b^{[1]}_3$ | **-0.02394** | 隐藏层偏置 |
| 12 | $b^{[1]}_4$ | +0.01791 | 隐藏层偏置 |
| 13 | $W^{[2]}_1$ | **-0.03747** | 输出层权重 |
| 14 | $b^{[2]}$ | **-0.05231** | 输出层偏置 |

**关键观察**：

1. **梯度最大的是 $b^{[2]}$（-0.05231）和 $W^{[2]}_1$（-0.03747）**——输出层参数离损失最近，梯度信号最强。
2. **$W^{[1]}$ 的第二行全为 0**——因为当前样本的 $x_2 = 0$，所以从 $x_2$ 出发的权重得不到更新信号。这解释了为什么需要多个样本进行批量训练：不同样本激活不同的输入维度。
3. **负梯度意味着参数需要增大**（因为梯度下降是 $w \leftarrow w - \eta \cdot \text{grad}$，负梯度使得 $w$ 增加）。
4. **隐藏层神经元 3 的梯度（-0.02394）比神经元 1（-0.01528）更大**——神经元 3 对当前损失的"贡献"（或者说"责任"）更大。

### 5.9 应用连接

你刚才逐项算出的这 14 个数字，就是 PyTorch 的 `loss.backward()` 在底层做的事——一模一样的链式法则，只不过它自动完成了。

### 5.9 对比样本：$x = [1, 1]$ 的反向传播关键值

为了理解不同样本如何贡献不同的梯度信号，我们快速计算 (1,1) 样本（目标 $y=0$，预测 $\hat{y}=0.5625$）的关键梯度值，与 (1,0) 样本做对比。

| 计算步骤 | $x=[1,0],\ y=1$ | $x=[1,1],\ y=0$ |
|:---|:---:|:---:|
| $\hat{y}$ | 0.7328 | 0.5625 |
| $\frac{\partial L}{\partial \hat{y}} = \hat{y}-y$ | **-0.2672** | **+0.5625** |
| $\sigma'(z) = \hat{y}(1-\hat{y})$ | 0.1958 | 0.2461 |
| $\delta^{[2]} = (\hat{y}-y)\sigma'(z)$ | **-0.05231** | **+0.1384** |
| $\frac{\partial L}{\partial b^{[2]}} = \delta^{[2]}$ | **-0.05231** | **+0.1384** |
| $\frac{\partial L}{\partial W^{[2]}_1} = a^{[1]}_1 \cdot \delta^{[2]}$ | -0.03747 | 0.4621×0.1384=**+0.0640** |

**关键对比**：对于 (1,0)（目标 1，预测 0.73 偏低），$\delta^{[2]}$ 是**负值**——梯度下降会让 $b^{[2]}$ 增大（因为减去负值），从而增大输出。对于 (1,1)（目标 0，预测 0.56 偏高），$\delta^{[2]}$ 是**正值**——梯度下降会让 $b^{[2]}$ 减小，从而减小输出。两个样本的梯度**方向相反**，体现了"有错就改、方向不同"的核心机制。

批量训练时，这两个样本的梯度求平均：$\bar{\delta}^{[2]} = (-0.05231 + 0.1384)/2 = +0.04305$。整个 batch 的净效果取决于哪个样本的错误信号更强。

### 5.10 Python 验证手工计算

```python
import numpy as np

# 验证第 5 节的手工梯度计算
W1 = np.array([[0.8, -0.5, 0.3, -0.7],
               [-0.4, 0.9, -0.6, 0.2]])
b1 = np.array([0.1, -0.2, 0.0, 0.3])
W2 = np.array([[0.6], [-0.3], [0.5], [-0.4]])
b2 = np.array([0.1])

x = np.array([[1.0, 0.0]])   # 行向量，形状 (1, 2)
y = np.array([1.0])

# 前向传播
z1 = x @ W1 + b1           # [[0.9, -0.7, 0.3, -0.4]]
a1 = np.tanh(z1)           # [[0.7163, -0.6044, 0.2913, -0.3799]]
z2 = a1 @ W2 + b2          # [[1.0088]]
a2 = 1 / (1 + np.exp(-z2)) # [[0.7328]]

# 反向传播
dL_da2 = (a2 - y)                                    # [[-0.2672]]
da2_dz2 = a2 * (1 - a2)                              # [[0.1958]]
dL_dz2 = dL_da2 * da2_dz2                             # [[-0.05231]] = δ[2]
dL_dW2 = a1.T @ dL_dz2                                # (4,1) 列向量
dL_db2 = np.sum(dL_dz2, axis=0)                      # [-0.05231]

dL_da1 = dL_dz2 @ W2.T                               # (1,4) 行向量
da1_dz1 = 1 - a1 ** 2                                 # tanh 导数
dL_dz1 = dL_da1 * da1_dz1                             # δ[1]
dL_dW1 = x.T @ dL_dz1                                 # (2,4)
dL_db1 = np.sum(dL_dz1, axis=0)                       # (4,)

print("=== 验证手工计算值 ===")
print(f"δ[2]  = {dL_dz2[0,0]:.5f}  (手算: -0.05231)")
print(f"dL/dW2[0] = {dL_dW2[0,0]:.5f}  (手算: -0.03747)")
print(f"dL/dW2[1] = {dL_dW2[1,0]:.5f}  (手算: 0.03162)")
print(f"dL/dW2[2] = {dL_dW2[2,0]:.5f}  (手算: -0.01524)")
print(f"dL/dW2[3] = {dL_dW2[3,0]:.5f}  (手算: 0.01987)")
print(f"dL/db2  = {dL_db2[0]:.5f}  (手算: -0.05231)")
print(f"δ[1]  = [{dL_dz1[0,0]:.5f}, {dL_dz1[0,1]:.5f}, {dL_dz1[0,2]:.5f}, {dL_dz1[0,3]:.5f}]")
print(f"      手算: [-0.01528, 0.00996, -0.02394, 0.01791]")
```

运行这段代码，你会看到 numpy 的计算结果与第 5 节的手工值在小数点后四位完全一致。这就是"手工推导 → 代码验证"的闭环——确信自己理解了反向传播的每一个细节。

### 5.11 应用连接

数值梯度检验（下节）会用有限差分验证这些手工梯度是否正确：如果相对差异小于 $10^{-6}$，你就知道自己真正懂了反向传播。

---

## 6. 从零实现 numpy MLP + 数值梯度检验

### 6.1 数值梯度检验的原理

数值梯度用有限差分近似偏导数：

$$
\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta_i + \varepsilon) - f(\theta_i - \varepsilon)}{2\varepsilon}
$$

这是**中心差分**，误差为 $O(\varepsilon^2)$。如果用前向差分 $\frac{f(\theta_i+\varepsilon)-f(\theta_i)}{\varepsilon}$，误差只有 $O(\varepsilon)$，精度差一个数量级。$\varepsilon$ 通常取 $10^{-5}$。

### 6.2 完整实现

```python
import numpy as np

def numerical_gradient(f, params, eps=1e-5):
    """对参数向量 params 中的每个元素，用中心差分近似偏导数"""
    grads = np.zeros_like(params)
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        f_plus = f(params)
        params[i] = orig - eps
        f_minus = f(params)
        grads[i] = (f_plus - f_minus) / (2 * eps)
        params[i] = orig  # 恢复原始值
    return grads


class MLP:
    """两层 MLP：隐藏层 Tanh + 输出层 Sigmoid + MSE 损失"""
    def __init__(self, input_size, hidden_size, output_size, lr=0.5):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = 1 / (1 + np.exp(-np.clip(self.Z2, -500, 500)))
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        dZ2 = (self.A2 - y) * self.A2 * (1 - self.A2) / m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - self.A1 ** 2)  # tanh 导数

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return dW1, db1, dW2, db2

    def step(self, grads):
        dW1, db1, dW2, db2 = grads
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred.flatten() - y_true) ** 2) / 2


# ---- 数值梯度检验：验证手工梯度与你刚才的手算一致 ----
np.random.seed(0)
mlp = MLP(2, 3, 1, lr=0.5)

X_test = np.array([[0.3, -0.7], [0.8, 0.2]])
y_test = np.array([0, 1])

# 手工梯度
_ = mlp.forward(X_test)
grads_manual = mlp.backward(X_test, y_test)

# 数值梯度：将参数展平，对损失函数做有限差分
def param_to_loss(flat_params):
    idx = 0
    for name in ['W1', 'b1', 'W2', 'b2']:
        arr = getattr(mlp, name)
        size = arr.size
        setattr(mlp, name, flat_params[idx:idx+size].reshape(arr.shape))
        idx += size
    pred = mlp.forward(X_test)
    return mlp.mse_loss(pred, y_test)

flat_params = np.concatenate([mlp.W1.ravel(), mlp.b1.ravel(),
                              mlp.W2.ravel(), mlp.b2.ravel()])
grads_numerical = numerical_gradient(param_to_loss, flat_params)

# 展平手工梯度
grads_manual_flat = np.concatenate([grads_manual[0].ravel(), grads_manual[1].ravel(),
                                     grads_manual[2].ravel(), grads_manual[3].ravel()])

diff = np.linalg.norm(grads_numerical - grads_manual_flat) / (
    np.linalg.norm(grads_numerical) + np.linalg.norm(grads_manual_flat) + 1e-8
)

print(f"手工梯度 vs 数值梯度的相对差异: {diff:.2e}")
print(f"{'梯度验证通过 ✓' if diff < 1e-6 else '梯度验证失败 ✗'}")

# 打印前几个梯度做肉眼对比
print(f"\n前 5 个数值梯度: {grads_numerical[:5]}")
print(f"前 5 个手工梯度: {grads_manual_flat[:5]}")
```

数值梯度检验是深度学习开发中最重要的调试技术。PyTorch 的 `torch.autograd.gradcheck` 就是自动化的这个过程。当你的反向传播实现正确时，相对差异通常在 $10^{-7}$ 到 $10^{-9}$ 量级。

---

## 7. 训练神经网络：XOR 从"学不会"到"学会了"

### 完整训练循环

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0., 1., 1., 0.])   # XOR

mlp = MLP(input_size=2, hidden_size=4, output_size=1, lr=0.5)

N_EPOCHS = 2000
losses = []
for epoch in range(N_EPOCHS):
    pred = mlp.forward(X)
    loss = mlp.mse_loss(pred, y)
    losses.append(loss)
    grads = mlp.backward(X, y)
    mlp.step(grads)

# ---- 结果 ----
pred_final = mlp.forward(X).flatten()
print("XOR 训练结果:")
for i in range(4):
    print(f"  Input {X[i]}: True={y[i]:.0f}, Pred={pred_final[i]:.4f}")

# ---- 可视化 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 左：损失曲线
axes[0].plot(losses, color='steelblue', linewidth=1)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('训练损失曲线', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].text(0.6, 0.9, f'最终损失: {losses[-1]:.6f}',
             transform=axes[0].transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 中：决策边界（用密集网格可视化）
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = mlp.forward(grid).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.7)
axes[1].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=150,
                edgecolors='k', linewidths=2, label='0')
axes[1].scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=150,
                edgecolors='k', linewidths=2, label='1')
axes[1].contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2.5)
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
axes[1].set_title('MLP 决策边界（XOR 问题）', fontsize=13)
axes[1].legend()
axes[1].set_xlim(-0.5, 1.5); axes[1].set_ylim(-0.5, 1.5)

# 右：隐藏层表示（学到的特征变换）
A1 = np.tanh(X @ mlp.W1 + mlp.b1)
axes[2].scatter(A1[y == 0, 0], A1[y == 0, 1], c='blue', s=150,
                edgecolors='k', linewidths=2, label='0')
axes[2].scatter(A1[y == 1, 0], A1[y == 1, 1], c='red', s=150,
                edgecolors='k', linewidths=2, label='1')
axes[2].set_xlabel('隐藏神经元 1 的输出'); axes[2].set_ylabel('隐藏神经元 2 的输出')
axes[2].set_title('隐藏层表示空间（XOR 变得线性可分）', fontsize=13)
axes[2].legend()
axes[2].grid(True, alpha=0.2)

plt.suptitle('MLP 训练 XOR：从"学不会"到"学会了"', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n隐藏层输出（前两个神经元）:")
for i in range(4):
    print(f"  Input {X[i]} -> Hidden ({A1[i, 0]:.4f}, {A1[i, 1]:.4f}) -> Pred {pred_final[i]:.4f}")
```

右图是理解神经网络最关键的图示之一——隐藏层将原始空间中线性不可分的 XOR 映射到了一个**新空间**，在新空间中四个点变得线性可分。这就是"表示学习"（representation learning）的可视化：神经网络不仅在学习分类边界，更重要的是在学习**如何重新表示数据**。

---

## 8. PyTorch 对比：一行 `backward()` 代替你的 30 行

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

losses_pt = []
for epoch in range(2000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()          # ← 框架自动计算所有梯度
    optimizer.step()         # ← 框架内置的梯度下降
    losses_pt.append(loss.item())

with torch.no_grad():
    pred = model(X)
    print("PyTorch XOR 训练结果:")
    for i in range(4):
        print(f"  Input {X[i].tolist()}: True={y[i].item():.0f}, "
              f"Pred={pred[i].item():.4f}")

plt.figure(figsize=(8, 4))
plt.plot(losses_pt, color='steelblue', linewidth=1)
plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)')
plt.title('PyTorch 训练 XOR——与纯 numpy 实现等价')
plt.grid(True, alpha=0.3)
plt.show()
```

对比你手写的 MLP 和 PyTorch 版本：

- 前向传播：`mlp.forward(X)` ↔ `model(X)` ——几乎一样
- 反向传播：你手写了 30 行链式法则 ↔ `loss.backward()` ——一行
- 参数更新：你手写了 4 行 `-= lr * grad` ↔ `optimizer.step()` ——一行

**框架没有魔法**——`backward()` 内部执行的就是第 5 节你一笔一笔算出的那六个步骤。理解手工实现让你不再是框架的"调包侠"，而是知道底层在发生什么。

---

## 9. 权重初始化策略

### 9.1 为什么不能初始化为零

如果在第 5 节的手工推演中，所有 $W^{[1]}$ 初始化为零矩阵、$b^{[1]}$ 为零向量会怎样？

$$
z^{[1]} = x \cdot \mathbf{0} + \mathbf{0} = [0, 0, 0, 0]
$$

所有 $z^{[1]}_i = 0$，所以 $a^{[1]}_i = \tanh(0) = 0$（全部相同）。

接着 $z^{[2]} = a^{[1]} \cdot W^{[2]} + b^{[2]} = \mathbf{0} \cdot W^{[2]} + 0 = 0$，$\hat{y} = \sigma(0) = 0.5$。

反向传播时，由于所有隐藏神经元输出相同，$\delta^{[1]}$ 的所有分量也相同，这导致 **$W^{[1]}$ 的每一行都收到完全相同的梯度**——所有隐藏神经元永远做同样的事，网络退化成只有一个有效神经元。这就是**对称性问题**（symmetry breaking problem）。

### 9.2 常见初始化方法

| 方法 | 公式 | 适用激活函数 | 原理 |
|------|------|:---:|------|
| **Xavier (Glorot)** | $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$ | Tanh, Sigmoid | 保持前向和反向传播的方差一致 |
| **He (Kaiming)** | $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$ | ReLU, LeakyReLU | 考虑 ReLU 将一半输出置 0 |
| **LeCun** | $W \sim \mathcal{N}(0, \frac{1}{n_{in}})$ | SELU | 保持输入方差 |

```python
# 不同初始化的 Python 实现对比
def init_weights(shape, method='he'):
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else 1
    if method == 'xavier':
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif method == 'he':
        std = np.sqrt(2.0 / fan_in)
    elif method == 'lecun':
        std = np.sqrt(1.0 / fan_in)
    else:
        std = 0.01  # 简单小随机数
    return np.random.randn(*shape) * std
```

本章 MLP 使用 He 初始化（`np.sqrt(2.0 / input_size)`），因为 Xavier 更适合 Tanh 而我们的实现也用了 Tanh 隐藏层——两者在实践中通常都能正常工作。

---

## 10. 训练动态：一次梯度更新后发生了什么

为了感受"学习"的全过程，我们在第 5 节算出的梯度上执行**一次梯度下降**（学习率 $\eta = 0.5$），看看参数如何变化。

### 更新公式

$$
\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta L
$$

### 输出层参数更新

| 参数 | 旧值 | 梯度 × η | 新值 |
|:---|:---:|:---:|:---:|
| $W^{[2]}_1$ | 0.6 | $-0.5 \times (-0.03747) = +0.01874$ | **0.6187** |
| $W^{[2]}_2$ | -0.3 | $-0.5 \times 0.03162 = -0.01581$ | **-0.3158** |
| $W^{[2]}_3$ | 0.5 | $-0.5 \times (-0.01524) = +0.00762$ | **0.5076** |
| $W^{[2]}_4$ | -0.4 | $-0.5 \times 0.01987 = -0.00994$ | **-0.4099** |
| $b^{[2]}$ | 0.1 | $-0.5 \times (-0.05231) = +0.02616$ | **0.1262** |

### 隐藏层参数更新

| 参数 | 旧值 | 梯度 × η | 新值 |
|:---|:---:|:---:|:---:|
| $W^{[1]}_{11}$ | 0.8 | $-0.5 \times (-0.01528) = +0.00764$ | **0.8076** |
| $W^{[1]}_{12}$ | -0.5 | $-0.5 \times 0.00996 = -0.00498$ | **-0.5050** |
| $W^{[1]}_{13}$ | 0.3 | $-0.5 \times (-0.02394) = +0.01197$ | **0.3120** |
| $W^{[1]}_{14}$ | -0.7 | $-0.5 \times 0.01791 = -0.00896$ | **-0.7090** |
| $W^{[1]}_{21}$ | -0.4 | $-0.5 \times 0.0 = 0.0$ | **-0.4** (不变!) |
| $W^{[1]}_{22}$ | 0.9 | 0.0 | **0.9** (不变!) |
| $W^{[1]}_{23}$ | -0.6 | 0.0 | **-0.6** (不变!) |
| $W^{[1]}_{24}$ | 0.2 | 0.0 | **0.2** (不变!) |
| $b^{[1]}_1$ | 0.1 | $-0.5 \times (-0.01528) = +0.00764$ | **0.1076** |
| $b^{[1]}_2$ | -0.2 | $-0.5 \times 0.00996 = -0.00498$ | **-0.2050** |
| $b^{[1]}_3$ | 0.0 | $-0.5 \times (-0.02394) = +0.01197$ | **0.0120** |
| $b^{[1]}_4$ | 0.3 | $-0.5 \times 0.01791 = -0.00895$ | **0.2910** |

### 更新后重新前向传播

用新参数重新计算 (1,0) 的预测：

$$
\begin{aligned}
z^{[1]}_{new} &= [1 \times 0.8076 + 0 \times (-0.4) + 0.1076,\; \dots] \\
              &= [0.9152,\; -0.7050,\; 0.3120,\; -0.4090] \\
a^{[1]}_{new} &\approx [0.723,\; -0.608,\; 0.302,\; -0.388] \\
z^{[2]}_{new} &= 0.723 \times 0.6187 + (-0.608) \times (-0.3158) + 0.302 \times 0.5076 + (-0.388) \times (-0.4099) + 0.1262 \\
              &= 0.4473 + 0.1920 + 0.1533 + 0.1590 + 0.1262 = 1.0778 \\
\hat{y}_{new} &= \sigma(1.0778) \approx 0.7460
\end{aligned}
$$

**旧预测 0.7328 → 新预测 0.7460**。向目标值 1.0 靠近了 0.013，方向正确。一次梯度下降让预测向正确方向移动了一小步——重复几千次，网络就学会了 XOR。

### 10.1 训练中的常见问题与对策

理解一次梯度更新后，我们总结神经网络训练中三大常见问题及其解决方案：

**① 学习率选择**

| 学习率 | 现象 | 损失曲线 |
|:---:|:---|:---|
| 太大 (如 10) | 参数一步跳太远，越过最优点 | 损失振荡或直接 NaN |
| 适中 (如 0.5) | 稳步下降 | 平滑衰减 |
| 太小 (如 0.0001) | 每次更新几乎不动 | 损失缓慢到几乎水平 |

**经验法则**：从 0.01 或 0.1 开始尝试，对数尺度搜索（0.001 → 0.01 → 0.1 → 1.0）。

**② 激活函数死亡（Dying ReLU）**

当隐藏层用 ReLU 时，如果某个神经元对**所有训练样本**的输出都是 0（$z \le 0$），其梯度也恒为 0——该神经元永久失效。解决方案：
- 使用 LeakyReLU 或 ELU（负区也有小梯度）
- 降低学习率（避免权重被推到全负区）
- 使用更好的初始化（He 初始化专为 ReLU 设计）

**③ 过拟合与正则化**

4 个样本的 XOR 不存在过拟合问题。但在真实数据上，神经网络极易过拟合——损失在训练集上持续下降，验证集上却不降反升。对策：
- **早停（Early Stopping）**：监控验证损失，开始上升时停止训练
- **L2 正则化**：损失函数加 $\lambda \|W\|^2$，限制权重大小
- **Dropout**：训练时随机"关掉"一部分神经元，迫使网络学习冗余表示
- **数据增强**：扩充训练数据（图像旋转、噪声注入等）

### 10.2 PyTorch 中的梯度验证

用 PyTorch 复现第 5 节的手工梯度计算，确认框架和手工推导一致：

```python
import torch

W1 = torch.tensor([[0.8, -0.5, 0.3, -0.7],
                   [-0.4, 0.9, -0.6, 0.2]], requires_grad=True)
b1 = torch.tensor([0.1, -0.2, 0.0, 0.3], requires_grad=True)
W2 = torch.tensor([[0.6], [-0.3], [0.5], [-0.4]], requires_grad=True)
b2 = torch.tensor([0.1], requires_grad=True)

x = torch.tensor([[1.0, 0.0]])
y = torch.tensor([1.0])

# 前向传播
z1 = x @ W1 + b1
a1 = torch.tanh(z1)
z2 = a1 @ W2 + b2
a2 = torch.sigmoid(z2)
loss = 0.5 * (a2 - y) ** 2

loss.backward()

print("PyTorch 自动梯度 vs 手工计算:")
print(f"  dL/dW2[0]: {W2.grad[0,0].item():.5f} (手算: -0.03747)")
print(f"  dL/db2:    {b2.grad[0].item():.5f} (手算: -0.05231)")
print(f"  dL/dW1[0,0]: {W1.grad[0,0].item():.5f} (手算: -0.01528)")
print(f"  dL/dW1[0,2]: {W1.grad[0,2].item():.5f} (手算: -0.02394)")
print(f"  dL/dW1[1,0]: {W1.grad[1,0].item():.5f} (手算: 0.00000)")
```

PyTorch 输出应与第 5 节的手工值完全一致——这就是 `loss.backward()` 替你完成的所有工作。

---

## 11. sklearn 实战：MLPClassifier

`sklearn` 的 `MLPClassifier` 封装了成熟的多层感知机分类器，背后是自适应学习率和早停（early stopping）。在小型结构化数据上，它往往是比手写 PyTorch 更实用的选择。

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X, y = digits.data, digits.target

print(f"数据形状: {X.shape}")       # (1797, 64) —— 8×8 像素展平
print(f"类别数: {len(np.unique(y))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 两个隐藏层：100 和 50 个神经元
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42,
    verbose=False,
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])}")

# ---- 权重可视化 ----
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].plot(mlp.loss_curve_, color='steelblue', linewidth=1)
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('训练损失曲线', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

vmax = np.max(np.abs(mlp.coefs_[0][:, :6]))
for i in range(6):
    ax = axes[(i) // 3 + 1, (i) % 3] if i < 6 else axes.flatten()[i]
    weight_img = mlp.coefs_[0][:, i].reshape(8, 8)
    im = ax.imshow(weight_img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'隐藏层神经元 {i+1} 的权重', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('MLPClassifier 在手写数字上的训练', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

权重可视化图中，每个 8×8 的小图是一个隐藏层神经元学到的"模板"——有些寻找圆形模式（数字 0 和 8 的中心），有些寻找竖线（数字 1）。这正是神经网络逐层提取特征的直观证据。

### MLPClassifier 关键参数

| 参数 | 含义 | 建议 |
|------|------|------|
| `hidden_layer_sizes` | 各隐藏层神经元数 | 从单层开始，逐渐增加复杂度 |
| `activation` | 激活函数 | `'relu'` 安全默认；`'tanh'` 适合小网络 |
| `solver` | 优化器 | `'adam'` 默认最佳；`'sgd'` 更可解释 |
| `alpha` | L2 正则化强度 | 0.0001 起步，过拟合时增大 |
| `learning_rate_init` | 初始学习率 | 默认 0.001，不收敛时减小 |
| `max_iter` | 最大迭代次数 | 默认 200，不够时增大 |
| `early_stopping` | 是否启用早停 | True 时自动用验证集防止过拟合 |

---

## 10. 本章小结

从感知机的一条直线到 MLP 的非线性决策边界，你亲手实现了"深度学习"在 2012 年爆发之前就已经完整的核心框架。重温你在本章走过的路：

1. **感知机手工推演**——手算 AND/OR/XOR，用反证法证明 XOR 线性不可分
2. **激活函数**——非线性是深度学习的"灵魂"，没了它堆再多层也是线性的
3. **MLP 前向传播手工推演**——2→4→1 网络，输入 (1,0)，9 次运算得出 $\hat{y}=0.7328$
4. **反向传播手工推演**——链式法则逐项计算 14 个参数梯度，全部分解到小数点后四位
5. **数值梯度检验**——用中心差分验证反向传播实现正确性
6. **训练循环**——前向算损失，反向算梯度，梯度下降更新参数，周而复始
7. **框架**——自动微分解放双手，让你专注于架构设计

这些概念放到 2025 年的 GPT-4 和 Stable Diffusion 上，**本质完全一样**。唯一的区别是：它们的隐藏层有几千层而不是 1 层，神经元有几十亿个而不是 4 个，输入是整段文本/整张图片而不是两个数字。但反向传播、梯度下降、激活函数的原理完全不变。

---

## 11. 思考题

**1. XOR 线性不可分的数学本质**

在 2.4 节的反证法中，我们证明了不存在 $(w_1, w_2, b)$ 能同时满足 XOR 的四个约束。如果把问题换成 XNOR（"相同为 1，不同为 0"），XNOR 是线性可分的吗？请用类似的方法证明或证伪。

**2. 感知机的几何解释**

感知机学习算法 $w \leftarrow w + \eta(y - \hat{y})x$ 中，当 $y=1$ 而 $\hat{y}=0$ 时，更新为 $w \leftarrow w + \eta x$。请从几何角度解释：为什么向 $x$ 方向移动 $w$ 能改善对这个样本的分类？

**3. 权重初始化为零的后果**

如果 MLP 的全部权重和偏置都初始化为 0，会发生什么？（提示：考虑对称性和梯度）为什么我们需要随机初始化？

**4. 隐藏层最小容量**

要解决 XOR 问题，隐藏层最少需要几个神经元？设计一个只含**1 个**隐藏神经元（但使用合适的激活函数）的网络，它能否解决 XOR？如果能，请给出权重；如果不能，请证明。

**5. RBF 核 vs 隐藏层**

SVM 通过 RBF 核将数据映射到高维空间使 XOR 线性可分。MLP 的隐藏层做的是类似的事——学习一个新的表示。这两种"空间变换"有什么本质区别？（提示：一个是被动映射，一个是主动学习）

**6. 线性激活函数的退化**

如果本章的 MLP 把隐藏层的 Tanh 替换为恒等函数 $g(z) = z$，不管隐藏层有多少个神经元，整个网络会退化成什么？请用矩阵运算严格推导。

**7. 梯度消失的具体数值**

在第 5 节的反向传播中，我们看到 $\delta^{[2]} = -0.05231$。如果网络有 10 个隐藏层且每层都使用 Sigmoid（其导数最大值为 0.25），从输出端传到第一隐藏层的梯度信号会衰减多少倍？给出最坏情况的衰减倍数。

**8. 学习率的上下界**

在第 5 节中，$W^{[2]}_1$ 的梯度是 -0.03747。如果学习率 $\eta = 10$，更新后 $W^{[2]}_1$ 会变为多少？这可能导致什么问题？为什么学习率通常远小于 1？

**9. 对输入的梯度**

推导 $\frac{\partial L}{\partial x}$（损失对输入数据的梯度），使用第 5 节的网络结构和前向缓存值算出具体数值。这个梯度有一个重要应用——你知道是什么吗？（提示：对抗样本）

**10. 批量训练 vs 单样本训练**

在第 5 节中，$x_2 = 0$ 导致 $W^{[1]}$ 第二行全为 0。如果用批量训练（一次输入全部 4 个 XOR 样本），$W^{[1]}$ 第二行还会全是 0 吗？请定性分析，并说明批量训练相对单样本训练的优缺点。

---

---

## 12. 思考题解答

### 题 1：XNOR 是线性可分的吗？

**答**：XNOR 的真值表是 `(0,0)→1`, `(0,1)→0`, `(1,0)→0`, `(1,1)→1`。把四个点画在平面上：(0,0) 和 (1,1) 是正类（红），(0,1) 和 (1,0) 是负类（蓝）。这和 XOR 完全一样——只是标签翻转了。用同样的反证法：

假设存在 $(w_1, w_2, b)$ 能分类 XNOR，则：
$$
\begin{aligned}
(0,0):\quad & b \ge 0 \quad\text{(输出应为 1)} \\
(0,1):\quad & w_2 + b < 0 \quad\text{(输出应为 0)} \\
(1,0):\quad & w_1 + b < 0 \quad\text{(输出应为 0)} \\
(1,1):\quad & w_1 + w_2 + b \ge 0 \quad\text{(输出应为 1)}
\end{aligned}
$$

由 $b \ge 0$ 和 $w_2 + b < 0$ 推出 $w_2 < -b \le 0$，所以 $w_2 < 0$。同理 $w_1 < 0$。但 $w_1 + w_2 + b \ge 0$，代入 $b \ge 0$ 和 $w_1, w_2 < 0$：两个负数之和加一个非负数不可能 $\ge 0$？等等，不一定——例如 $w_1=-0.1$, $w_2=-0.1$, $b=1$，则 $-0.1-0.1+1=0.8 \ge 0$ 是可能的。

实际上 XNOR **也是线性不可分的**。从几何上看：XNOR 中红点和蓝点的分布与 XOR 完全相同（只是颜色互换），仍然是对角线格局。严格证明：对 XOR 的四个不等式做变量替换 $w'_1 = -w_1$, $w'_2 = -w_2$, $b' = -b$，不等式组变为 $b' < 0$, $w'_2 + b' \ge 0$, $w'_1 + b' \ge 0$, $w'_1 + w'_2 + b' < 0$——回到了 XOR 的反证形式，仍然矛盾。**结论：XNOR 和 XOR 一样是线性不可分的。**

---

### 题 2：感知机更新的几何解释

**答**：当 $y=1,\ \hat{y}=0$ 时，说明真实标签是正类但被分到了负类侧（$w^T x + b < 0$）。更新 $w \leftarrow w + \eta x$ 后，新的内积变为：

$$(w + \eta x)^T x = w^T x + \eta \|x\|^2$$

内积增加了 $\eta \|x\|^2 > 0$，这意味着决策边界的法向量向 $x$ 方向旋转，使得 $x$ 下次被分到正类侧的概率增大。几何上，**权重向量 $w$ 是决策边界的法向量**，把它往误分类样本的方向推，就是让分界线"弯腰"去接纳这个样本。

---

### 题 3：权重初始化为零的后果

**答**：如果所有权重为零：

1. **前向传播**：$z^{[1]} = x \cdot \mathbf{0} + \mathbf{0} = \mathbf{0}$，所有隐藏神经元输出相同（tanh(0)=0），输出 $\hat{y} = \sigma(0) = 0.5$。
2. **反向传播**：$\frac{\partial L}{\partial W^{[1]}_{ij}} = x_i \cdot \delta^{[1]}_j$。由于所有隐藏神经元对称，$\delta^{[1]}$ 各分量相等，所以 $W^{[1]}$ 的每一列梯度完全相同。
3. **参数更新**：每个隐藏神经元收到完全相同的更新，更新后仍然全部相同——**对称性永不打破**，网络退化为只有一个有效神经元。
4. **偏置为零不会造成对称性**问题（偏置可以初始化为零），因为偏置不参与"对称性"——每个神经元的偏置独立更新，梯度来自各自的 $\delta$。

这就是必须随机初始化的原因：打破神经元之间的对称性，让不同的隐藏神经元学习不同的特征。

---

### 题 4：隐藏层最少需要几个神经元？

**答**：**至少需要 2 个隐藏神经元**。

经典构造（用阶跃函数，可推广到 Tanh/Sigmoid）：

- 神经元 1：检测 $(x_1 \text{ OR } x_2)$。$w = [1, 1], b = -0.5$，输出 $h_1 = \text{step}(x_1 + x_2 - 0.5)$。
- 神经元 2：检测 $(x_1 \text{ AND } x_2)$。$w = [1, 1], b = -1.5$，输出 $h_2 = \text{step}(x_1 + x_2 - 1.5)$。
- 输出层：$\hat{y} = \text{step}(h_1 - 2h_2 - 0.5)$，实现 $\text{XOR} = \text{OR AND NOT AND}$。

验证：(0,0)→$h_1$=0,$h_2$=0→$\hat{y}$=0 ✓；(0,1)→$h_1$=1,$h_2$=0→$\hat{y}$=1 ✓；(1,0)→$h_1$=1,$h_2$=0→$\hat{y}$=1 ✓；(1,1)→$h_1$=1,$h_2$=1→$\hat{y}$=0 ✓。

**1 个隐藏神经元不能解决 XOR**：单个隐藏神经元 $h = g(w_1 x_1 + w_2 x_2 + b)$，输出 $\hat{y} = \sigma(w_o h + b_o)$。将第一个式子代入第二个，由于 $\sigma$ 是单调函数，决策边界仍然是 $x_1, x_2$ 平面上的某条单调曲线——无法产生 XOR 需要的非单调边界。严格来说，1 个神经元产生的隐藏表示是一维的，而在一条线上，你无法用任何单调函数将交替排列的两类点分开。

---

### 题 5：RBF 核 vs 隐藏层

**答**：两者都将数据映射到新空间使 XOR 线性可分，但本质完全不同：

- **RBF 核（SVM）**：映射是**固定的、预先定义的**。$\phi(x) = [\dots, e^{-\gamma\|x - x_i\|^2}, \dots]$，映射函数由核函数和人造地标（支持向量）决定，不随训练改变。SVM 只学习在这个固定空间里的线性分类器参数。
- **MLP 隐藏层**：映射是**可学习的、自适应的**。$\phi(x) = g(W^{[1]} x + b^{[1]})$，$W^{[1]}$ 和 $b^{[1]}$ 通过反向传播和梯度下降来学习。网络自己"发现"什么样的空间变换对当前任务最有利。

这反映了两种范式：SVM 是"先定好表示空间再分类"（表示固定），MLP 是"表示和分类一起学"（端到端学习）。后者的灵活性更大，但也需要更多数据和计算。

---

### 题 6：线性激活函数的退化

**答**：设 $g(z) = z$（恒等函数）。前向传播：

$$a^{[1]} = z^{[1]} = x W^{[1]} + b^{[1]}$$
$$\hat{y} = a^{[2]} = a^{[1]} W^{[2]} + b^{[2]} = (x W^{[1]} + b^{[1]})W^{[2]} + b^{[2]}$$
$$= x(W^{[1]}W^{[2]}) + (b^{[1]}W^{[2]} + b^{[2]})$$
$$= xW' + b'$$

其中 $W' = W^{[1]}W^{[2]}$（2×1 矩阵），$b' = b^{[1]}W^{[2]} + b^{[2]}$（标量）。不管隐藏层有多少神经元，$W^{[1]}W^{[2]}$ 永远只是一个 2×1 的矩阵——整个网络**等价于一个单层线性模型**（即没有隐藏层的感知机，只是激活函数不同）。堆再多层也是线性变换，这就是为什么非线性激活函数是必需的。

---

### 题 7：梯度消失的最坏衰减

**答**：10 层且每层用 Sigmoid，Sigmoid 导数最大值出现在 $z=0$ 处：$\sigma'(0) = 0.5 \times (1-0.5) = 0.25$。

链式法则从输出端传到第一层要经过 10 次局部导数乘法：
$$\frac{\partial L}{\partial z^{[1]}} = \frac{\partial L}{\partial z^{[10]}} \cdot \prod_{l=2}^{10} \frac{\partial z^{[l]}}{\partial z^{[l-1]}}$$

每一层的局部梯度包含 $\sigma'(z) \cdot W$，在最坏情况下（$z=0$ 使导数最大，但 $\|W\|$ 也小），梯度信号每层衰减 0.25 倍。10 层后：
$$0.25^{10} = \frac{1}{4^{10}} = \frac{1}{1,048,576} \approx 9.5 \times 10^{-7}$$

**衰减约 100 万倍**。第 10 层的一个梯度信号传到第 1 层时几乎为零——这就是"梯度消失"的灾难。ReLU 解决了这个问题（正区导数恒为 1），这也是为什么深度网络几乎都用 ReLU 及其变体。

---

### 题 8：学习率过大导致发散

**答**：$\eta = 10$ 时，$W^{[2]}_1$ 的更新量 = $-10 \times (-0.03747) = +0.3747$。

新 $W^{[2]}_1 = 0.6 + 0.3747 = 0.9747$。参数在**单步**内就改变了 62%，远大于正常训练中每步 0.1%~1% 的变化量。后果：

1. **发散（divergence）**：参数在损失曲面上跳来跳去，每次更新都"跳过"了最优解，损失不降反升。
2. **振荡（oscillation）**：在最优解两侧来回跳动，永远无法收敛。
3. **NaN**：数值溢出导致参数变成 NaN。

学习率通常远小于 1（如 0.1, 0.01, 0.001），本质上是控制"每次从错误中学习多少"。太大则震荡发散，太小则收敛极慢。第 10.1 节中 $\eta=0.5$ 只改变了参数约 1%~3%，这是健康的学习速度。

---

### 题 9：对输入的梯度与对抗样本

**答**：使用第 5 节的网络和缓存值：

$$\frac{\partial L}{\partial x} = \delta^{[1]} \cdot (W^{[1]})^T$$

$$
\begin{aligned}
\frac{\partial L}{\partial x_1} &= \delta^{[1]}_1 W^{[1]}_{11} + \delta^{[1]}_2 W^{[1]}_{12} + \delta^{[1]}_3 W^{[1]}_{13} + \delta^{[1]}_4 W^{[1]}_{14} \\
&= (-0.01528)(0.8) + (0.00996)(-0.5) + (-0.02394)(0.3) + (0.01791)(-0.7) \\
&= -0.01222 - 0.00498 - 0.00718 - 0.01254 = -0.03692 \\[4pt]
\frac{\partial L}{\partial x_2} &= \delta^{[1]}_1 W^{[1]}_{21} + \delta^{[1]}_2 W^{[1]}_{22} + \delta^{[1]}_3 W^{[1]}_{23} + \delta^{[1]}_4 W^{[1]}_{24} \\
&= (-0.01528)(-0.4) + (0.00996)(0.9) + (-0.02394)(-0.6) + (0.01791)(0.2) \\
&= 0.00611 + 0.00896 + 0.01436 + 0.00358 = 0.03301
\end{aligned}
$$

所以 $\frac{\partial L}{\partial x} = [-0.0369,\; 0.0330]$。

**应用——对抗样本（adversarial examples）**：如果我们想"欺骗"网络，让网络把输入 (1,0)（真实标签 1）预测成 0，我们可以沿**梯度的反方向**修改输入（因为我们想让损失增大，使预测偏离标签）：

$$x_{adv} = x - \varepsilon \cdot \frac{\partial L}{\partial x} = [1 - \varepsilon(-0.0369),\; 0 - \varepsilon(0.0330)]$$

取 $\varepsilon = 1$：$x_{adv} = [1.037, -0.033]$。这个对人眼几乎不可见的扰动，却可能让网络的预测从 0.7328 大幅偏离 1.0。这就是对抗攻击（FGSM）的数学原理——也是深度学习安全性的核心挑战。

---

### 题 10：批量训练 vs 单样本训练

**答**：第 5 节中 $x_2=0$ 导致 $W^{[1]}$ 第二行梯度全为 0，只因为当前样本的 $x_2$ 分量为 0。如果同时输入 4 个 XOR 样本做批量训练：

$$\frac{\partial L_{batch}}{\partial W^{[1]}} = \frac{1}{4}\sum_{k=1}^{4} \frac{\partial L^{(k)}}{\partial W^{[1]}}$$

样本 (0,1) 的 $x_2=1$，它对 $W^{[1]}$ 第二行产生非零梯度；样本 (1,1) 同理。所以**批量训练时 $W^{[1]}$ 第二行不会全是 0**。

| 维度 | 批量（Batch, m=4） | 单样本（SGD, m=1） | 小批量（Mini-batch, m=2） |
|:---|:---|:---|:---|
| 梯度估计 | 精确（全体平均） | 高方差（每个样本噪声大） | 折中 |
| 收敛速度（每 epoch） | 慢（每次更新用全部数据） | 快（每样本更新一次） | 适中 |
| 收敛稳定性 | 最稳定 | 容易震荡 | 适中 |
| 内存 | 需载入全部数据 | 最少 | 可调 |
| 逃逸局部最优 | 难 | 噪声有助于跳出 | 较好 |
| 并行化 | 容易（大矩阵乘法） | 难（一次一个样本） | 较好 |

**实践建议**：小批量（mini-batch size=32/64/128）是最常用的折中方案，兼顾了梯度稳定性、内存效率和收敛速度。

---

> **下一步：[CNN与RNN基础](./16-cnn-rnn-basics.md)**——从全连接网络扩展到专门处理图像和序列的网络架构
