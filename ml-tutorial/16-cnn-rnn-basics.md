# CNN与RNN基础：处理图像和序列

> **本节在 ML 学习路径中的位置**：神经网络从"通用函数逼近器"走向"专业结构化数据处理器"的关键一章。CNN 让网络看见空间的局部模式，RNN 让网络记住时间的先后依赖——这是理解一切现代深度学习模型（ResNet、LSTM、Transformer）的必要前置。
>
> 前置章节：[神经网络入门](./15-neural-networks-intro.md)
> 下一章：[模型评估与调参](./17-model-evaluation.md)

> **核心问题**：全连接网络把数据当成一堆互不相干的数字，但图像像素之间有空间关系，文本单词之间有时序依赖。CNN 和 RNN 各自用什么机制解决了这两个问题？卷积核滑过图像时到底在算什么？RNN 的隐状态如何一步步传递记忆？

前面的 MLP（多层感知机）是一把**通用锤子**——不管输入什么，全部拍扁成一维向量，然后全连接层一视同仁地处理。这在对表格数据时没问题，但现实中的很多数据有**结构**：图像里相邻的像素往往属于同一个物体，文本里后面的词依赖于前面的词。CNN 通过**局部感受野 + 权重共享**捕捉空间模式，RNN 通过**隐状态**维持时序记忆——它们分别是处理空间和序列数据的"专用工具"。

读完本章你将能够：

- **手算**卷积操作：给定一个 $4\times4$ 的图像和一个 $2\times2$ 的卷积核，写出输出特征图的每一个元素是怎么算出来的
- **手算**池化操作：给定一个 $4\times4$ 的特征图，写出 $2\times2$ 最大池化的完整过程
- **手算** RNN 的时间展开：给定权重和输入序列，一步步算出 $h_1 \to h_2 \to h_3$ 的隐状态变化
- 用 numpy 从零实现卷积、池化和简单 RNN 单元
- 理解 LSTM 三扇门和 GRU 两扇门各自解决了什么问题
- 用预训练 CNN 做迁移学习，用 LSTM 做序列预测
- 判断一个任务该用 CNN / RNN 还是传统 ML

---

## 1. 为什么全连接网络不够？

### 直觉理解

想象把一张 $224\times224$ 的彩色图像展平成一个 $150528$ 维的向量。MLP 的第一层如果有 $1000$ 个神经元，光是权重矩阵就有 **1.5 亿个参数**——而且**每个像素都被当作独立的特征**。左上角的一只猫耳朵和右下角的另一只猫耳朵对 MLP 来说是两个完全不相关的东西。MLP 不知道"耳朵"这个局部模式可以出现在图像的**任何位置**，它必须分别学习每个位置的耳朵——这既不划算，也不符合视觉的物理规律。

再看文本："这部电影拍得精妙绝伦"和"这部电影拍得一塌糊涂"——关键区别只是最后的几个字。但前面的"这部电影拍得"完全一样。MLP 把它们当成两组不相关的特征输入，不知道"后面的词在前面词的基础上展开"这个时序结构。更致命的是：100 个词的句子和 200 个词的句子，MLP 必须强制截断或填充到相同长度，因为它只接受固定长度的输入。

### 形式化定义

给定 MLP 层 $h = \sigma(Wx + b)$：

- **空间盲**：如果 $x$ 中的像素被打乱顺序，只要排列一致（同时打乱训练和测试），MLP 能学到相同的模式。这说明它根本不理解"相邻像素"这个概念——对 MLP 来说，输入只是一个毫无结构的高维向量。
- **参数爆炸**：输入 $C \times H \times W$（通道 $\times$ 高 $\times$ 宽），展开后维度为 $CHW$。每一层的权重数正比于 $CHW \times \text{output\_dim}$，对中等大小的图像就已失控。
- **无长度泛化**：MLP 接受固定长度的输入，无法自然处理变长序列。

> **核心矛盾**：MLP 把结构化数据当成"一袋子数字"，而我们需要的模型应该**知道哪些数字在空间/时间上是邻居**。

### Python 代码验证：为什么打乱像素不影响 MLP

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target  # shape (1797, 64) — 8×8 展平

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 情况A：正常像素顺序
mlp_a = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
mlp_a.fit(X_train, y_train)
acc_a = mlp_a.score(X_test, y_test)

# 情况B：对每个样本的像素做相同的随机排列
perm = np.random.RandomState(42).permutation(64)
X_train_shuffled = X_train[:, perm]
X_test_shuffled = X_test[:, perm]

mlp_b = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
mlp_b.fit(X_train_shuffled, y_train)
acc_b = mlp_b.score(X_test_shuffled, y_test)

print(f"原始像素顺序的测试准确率:  {acc_a:.4f}")
print(f"像素全部打乱后的测试准确率: {acc_b:.4f}")
print(f"差异: {abs(acc_a - acc_b):.6f}  ← 几乎不变！")

# 可视化：原图和打乱后的图
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(digits.images[0], cmap='gray')
axes[0].set_title('原始 8×8 图像')
shuffled_img = digits.images[0].ravel()[perm].reshape(8, 8)
axes[1].imshow(shuffled_img, cmap='gray')
axes[1].set_title('像素打乱后 —— MLP 看着一样')
plt.suptitle('MLP 看不到空间结构：打乱像素不影响它学习')
plt.tight_layout(); plt.show()
```

### 应用连接

| 数据类型 | MLP 的问题 | 需要的结构 |
|---------|-----------|-----------|
| 图像 | 找不到局部模式，参数太多 | 局部感受野 + 权重共享（CNN） |
| 文本/时间序列 | 不知道先后顺序，无法处理变长 | 隐状态 + 循环/注意力（RNN/Transformer） |
| 表格数据 | 无问题 | MLP 或传统 ML 即可 |

---

## 2. 卷积神经网络 (CNN)

### 核心思想

CNN 用两个精巧的设计解决了 MLP 的空间盲：

1. **局部感受野（Local Receptive Field）**：每个神经元不看全部输入，只看一个小窗口（比如 $3\times3$）。这模仿了生物视觉皮层中"一个神经元只对视野中的一小块区域敏感"的机制。
2. **权重共享（Weight Sharing）**：同一个卷积核（也叫 filter）滑过整张图像的所有位置——检测左上角边缘的核也被用来检测右下角的边缘。这意味着无论猫耳朵出现在哪里，同一个核都能发现它，因为核不关心位置，只关心模式。

这两个设计做了三件关键的事：
- **(a) 参数大幅减少**：一个 $3\times3$ 的卷积核只有 9 个参数（加偏置 10 个），不管输入是 $28\times28$ 还是 $224\times224$。对比之下，MLP 对 $224\times224$ 输入的全连接层第一层就有 $150528 \times \text{neurons}$ 个参数。
- **(b) 平移不变性**：猫耳朵不管在左上角还是右下角，同一个核都能检测到它。因为核滑过整个图像，每个位置都在进行相同的模式匹配。
- **(c) 层级抽象**：浅层核学习边缘和纹理，中层核学习形状（由边缘组成），深层核学习物体（由形状组成）。这就是神经网络的"视觉理解"一步步建立起来的过程。

### 形式化定义

一个 2D 卷积操作：给定输入 $X \in \mathbb{R}^{H \times W}$ 和卷积核 $K \in \mathbb{R}^{k \times k}$，输出特征图 $Y$ 的第 $i$ 行第 $j$ 列为：

$$
Y_{ij} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} K_{mn} \cdot X_{i+m,\; j+n}
$$

翻译成人话：把卷积核放在输入的第 $(i, j)$ 位置（核的左上角对齐输入的 $(i, j)$），对应元素逐个相乘再全部加起来，就是输出 $(i, j)$ 位置的值。

**超参数定义**：
- **步长（stride）** $s$：卷积核每次滑动的步数。$s=1$ 时输出尺寸 $\approx H - k + 1$，$s=2$ 时起到降采样作用。
- **填充（padding）** $p$：在输入四周补 $p$ 圈零，用于控制输出尺寸。`same` padding 让输出和输入尺寸一致，需补 $p = (k-1)/2$ 圈（假设 $k$ 是奇数且 $s=1$）。
- **输出尺寸公式**：$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$，$W_{out}$ 同理。

**多通道扩展**：输入有 $C_{in}$ 个通道（如 RGB 的 3 个），每个卷积核也有 $C_{in}$ 个通道——对每个输入通道都有一个 $k\times k$ 的二维核，把 $C_{in}$ 个通道的卷积结果加起来得到 1 个输出通道。用 $C_{out}$ 个不同的卷积核（每个核有独立的 $C_{in} \times k \times k$ 个参数），就得到 $C_{out}$ 个输出通道。

### 手算：$2\times2$ 卷积核在 $4\times4$ 图像上的完整过程

下面我们**逐位置、逐元素**地手算整个卷积过程。设置：输入 $4\times4$，核 $2\times2$，stride=1，padding=0。输出将是 $3\times3$。

**给定数据**：

$$
X = \begin{bmatrix}
1 & 2 & 3 & 0 \\
0 & 1 & 2 & 3 \\
3 & 0 & 1 & 2 \\
2 & 3 & 0 & 1
\end{bmatrix}
\qquad
K = \begin{bmatrix}
1 & 0 \\
2 & 1
\end{bmatrix}
$$

**输出尺寸**：$H_{out} = \frac{4 - 2}{1} + 1 = 3$，所以输出 $Y$ 是 $3\times3$ 矩阵。

---

**位置 $(0,0)$**——核左上角对齐 $X[0,0]$，覆盖 $X$ 的第 0-1 行、第 0-1 列：

$$
\underbrace{\begin{bmatrix}1 & 2 \\ 0 & 1\end{bmatrix}}_{\text{输入区域 }X[0:2,\,0:2]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
1 \times 1 &= 1 \\
2 \times 0 &= 0 \\
0 \times 2 &= 0 \\
1 \times 1 &= 1
\end{aligned}
$$

$$
Y[0,0] = 1 + 0 + 0 + 1 = 2
$$

---

**位置 $(0,1)$**——核右滑 1 步，对齐 $X[0,1]$，覆盖 $X$ 的第 0-1 行、第 1-2 列：

$$
\underbrace{\begin{bmatrix}2 & 3 \\ 1 & 2\end{bmatrix}}_{\text{输入区域 }X[0:2,\,1:3]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
2 \times 1 &= 2 \\
3 \times 0 &= 0 \\
1 \times 2 &= 2 \\
2 \times 1 &= 2
\end{aligned}
$$

$$
Y[0,1] = 2 + 0 + 2 + 2 = 6
$$

---

**位置 $(0,2)$**——核再右滑 1 步，覆盖 $X$ 的第 0-1 行、第 2-3 列：

$$
\underbrace{\begin{bmatrix}3 & 0 \\ 2 & 3\end{bmatrix}}_{\text{输入区域 }X[0:2,\,2:4]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
3 \times 1 &= 3 \\
0 \times 0 &= 0 \\
2 \times 2 &= 4 \\
3 \times 1 &= 3
\end{aligned}
$$

$$
Y[0,2] = 3 + 0 + 4 + 3 = 10
$$

---

**位置 $(1,0)$**——核下滑 1 步，回最左，覆盖 $X$ 的第 1-2 行、第 0-1 列：

$$
\underbrace{\begin{bmatrix}0 & 1 \\ 3 & 0\end{bmatrix}}_{\text{输入区域 }X[1:3,\,0:2]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
0 \times 1 &= 0 \\
1 \times 0 &= 0 \\
3 \times 2 &= 6 \\
0 \times 1 &= 0
\end{aligned}
$$

$$
Y[1,0] = 0 + 0 + 6 + 0 = 6
$$

---

**位置 $(1,1)$**——核居中，覆盖 $X$ 的第 1-2 行、第 1-2 列：

$$
\underbrace{\begin{bmatrix}1 & 2 \\ 0 & 1\end{bmatrix}}_{\text{输入区域 }X[1:3,\,1:3]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
1 \times 1 &= 1 \\
2 \times 0 &= 0 \\
0 \times 2 &= 0 \\
1 \times 1 &= 1
\end{aligned}
$$

$$
Y[1,1] = 1 + 0 + 0 + 1 = 2
$$

---

**位置 $(1,2)$**——核在第一行最后一个位置，覆盖 $X$ 的第 1-2 行、第 2-3 列：

$$
\underbrace{\begin{bmatrix}2 & 3 \\ 1 & 2\end{bmatrix}}_{\text{输入区域 }X[1:3,\,2:4]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
2 \times 1 &= 2 \\
3 \times 0 &= 0 \\
1 \times 2 &= 2 \\
2 \times 1 &= 2
\end{aligned}
$$

$$
Y[1,2] = 2 + 0 + 2 + 2 = 6
$$

---

**位置 $(2,0)$**——核再下滑，覆盖 $X$ 的第 2-3 行、第 0-1 列：

$$
\underbrace{\begin{bmatrix}3 & 0 \\ 2 & 3\end{bmatrix}}_{\text{输入区域 }X[2:4,\,0:2]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
3 \times 1 &= 3 \\
0 \times 0 &= 0 \\
2 \times 2 &= 4 \\
3 \times 1 &= 3
\end{aligned}
$$

$$
Y[2,0] = 3 + 0 + 4 + 3 = 10
$$

---

**位置 $(2,1)$**——核右下角前一步，覆盖 $X$ 的第 2-3 行、第 1-2 列：

$$
\underbrace{\begin{bmatrix}0 & 1 \\ 3 & 0\end{bmatrix}}_{\text{输入区域 }X[2:4,\,1:3]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
0 \times 1 &= 0 \\
1 \times 0 &= 0 \\
3 \times 2 &= 6 \\
0 \times 1 &= 0
\end{aligned}
$$

$$
Y[2,1] = 0 + 0 + 6 + 0 = 6
$$

---

**位置 $(2,2)$**——核到达右下角，覆盖 $X$ 的第 2-3 行、第 2-3 列：

$$
\underbrace{\begin{bmatrix}1 & 2 \\ 0 & 1\end{bmatrix}}_{\text{输入区域 }X[2:4,\,2:4]}
\odot
\underbrace{\begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}}_{\text{核 }K}
\quad\Longrightarrow\quad
\begin{aligned}
1 \times 1 &= 1 \\
2 \times 0 &= 0 \\
0 \times 2 &= 0 \\
1 \times 1 &= 1
\end{aligned}
$$

$$
Y[2,2] = 1 + 0 + 0 + 1 = 2
$$

---

**最终输出**：

$$
Y = \begin{bmatrix}
2 & 6 & 10 \\
6 & 2 & 6 \\
10 & 6 & 2
\end{bmatrix}
$$

**观察**：原图中间有一条"暗带"（第 1-2 行中间的 1→2→1，被 $K = \begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}$ 这个不对称核在滑动过程中检出并放大了变化）。$Y[0,2]$ 和 $Y[2,0]$ 的值最大（=10），反映的是那些含 3 较多的区域被核的第二行（系数 2）放大了。

> **核心直觉**：卷积不是魔法，就是对每个窗口做"加权求和"。10 分钟玩下来，你应该能闭着眼睛算出任何一个核在任何位置上的响应——这就是"看懂"卷积的第一步。

### Python：从零实现二维卷积

```python
import numpy as np

def conv2d_naive(x, kernel, stride=1, padding=0):
    """从零实现 2D 卷积（单通道版本）。
    
    Args:
        x: 输入矩阵, shape (H, W)
        kernel: 卷积核, shape (kH, kW)
        stride: 步长
        padding: 零填充圈数
    
    Returns:
        out: 特征图, shape (out_H, out_W)
    """
    H, W = x.shape
    kH, kW = kernel.shape
    
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)),
                   mode='constant', constant_values=0)
    
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    out = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            i_start = i * stride
            j_start = j * stride
            region = x[i_start:i_start + kH, j_start:j_start + kW]
            out[i, j] = np.sum(region * kernel)
    
    return out

# --- 验证：用刚才手算的矩阵 ---
X = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
], dtype=float)
K = np.array([
    [1, 0],
    [2, 1]
], dtype=float)

Y = conv2d_naive(X, K, stride=1, padding=0)
print("手写卷积输出:")
print(Y)
print(f"\n与手算结果一致: {np.allclose(Y, [[2, 6, 10], [6, 2, 6], [10, 6, 2]])}")

# --- 边缘检测实战 ---
from scipy import signal
import matplotlib.pyplot as plt

# 建一张 9×9 的图：左边暗(0)，右边亮(1)
img = np.zeros((9, 9))
img[:, 4:] = 1.0  # 竖直线分界

# 竖边检测核（Sobel x 方向近似）
edge_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

my_out = conv2d_naive(img, edge_kernel, stride=1, padding=0)
scipy_out = signal.correlate2d(img, edge_kernel, mode='valid')

print(f"\n手写 vs scipy 最大误差: {np.max(np.abs(my_out - scipy_out)):.10f}")

# 水平边检测
h_edge = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
h_out = conv2d_naive(img, h_edge)

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原图（左暗右亮）')
axes[1].imshow(my_out, cmap='RdBu')
axes[1].set_title('竖边检测 → 看到分界线！')
im = axes[2].imshow(h_out, cmap='RdBu')
axes[2].set_title('横边检测 → 几乎没有（正确）')
plt.colorbar(im, ax=axes[2])
axes[3].imshow(np.abs(my_out) > 1, cmap='gray')
axes[3].set_title('阈值化：找到的竖边')
plt.suptitle('手写卷积：边缘检测验证')
plt.tight_layout(); plt.show()
```

### 池化层

卷积提取了一大堆特征图后，一个自然而然的操作是**降采样**。道理很简单：一个 $2\times2$ 区域里的最大值或平均值，已经足够代表这个区域的主要特征了。降采样带来三重好处：**减少计算量**（每降一半尺寸，后续层的计算量减少 75%）、**增大感受野**（同样大小的核能看到更大的原始图像区域）、**抑制过拟合**（丢失了一些精确位置信息，反而让模型更关注"模式是否存在"而非"模式在哪个像素上"）。

#### 手算：$2\times2$ 最大池化在 $4\times4$ 特征图上

用卷积手算中得到的 $Y$ 矩阵作为输入。池化窗口 $2\times2$，stride=2（不重叠）：

$$
\text{输入 }Y = \begin{bmatrix}
2 & 6 & 10 \\
6 & 2 & 6 \\
10 & 6 & 2
\end{bmatrix}
$$

等一下——$Y$ 是 $3\times3$，stride=2 的情况下只有输出 $1\times1$，不够展示。我们换用原始的 $4\times4$ 输入做池化：

$$
X = \begin{bmatrix}
1 & 2 & 3 & 0 \\
0 & 1 & 2 & 3 \\
3 & 0 & 1 & 2 \\
2 & 3 & 0 & 1
\end{bmatrix}
\qquad
\text{池化窗口 } 2\times2,\; \text{stride}=2
$$

**输出尺寸**：$\frac{4 - 2}{2} + 1 = 2$，输出是 $2\times2$。

---

**窗口 $(0,0)$**——覆盖第 0-1 行、第 0-1 列：

$$
\begin{bmatrix}1 & 2 \\ 0 & 1\end{bmatrix}
\quad\Longrightarrow\quad
\max(1, 2, 0, 1) = 2
$$

---

**窗口 $(0,1)$**——覆盖第 0-1 行、第 2-3 列：

$$
\begin{bmatrix}3 & 0 \\ 2 & 3\end{bmatrix}
\quad\Longrightarrow\quad
\max(3, 0, 2, 3) = 3
$$

---

**窗口 $(1,0)$**——覆盖第 2-3 行、第 0-1 列：

$$
\begin{bmatrix}3 & 0 \\ 2 & 3\end{bmatrix}
\quad\Longrightarrow\quad
\max(3, 0, 2, 3) = 3
$$

---

**窗口 $(1,1)$**——覆盖第 2-3 行、第 2-3 列：

$$
\begin{bmatrix}1 & 2 \\ 0 & 1\end{bmatrix}
\quad\Longrightarrow\quad
\max(1, 2, 0, 1) = 2
$$

---

**最终输出**：

$$
\text{maxpool}(X) = \begin{bmatrix}
2 & 3 \\
3 & 2
\end{bmatrix}
$$

同样是 $4\times4 \to 2\times2$，如果用**平均池化**：

$$
\text{avgpool}(X) = \begin{bmatrix}
\frac{1+2+0+1}{4} & \frac{3+0+2+3}{4} \\[4pt]
\frac{3+0+2+3}{4} & \frac{1+2+0+1}{4}
\end{bmatrix}
= \begin{bmatrix}
1.0 & 2.0 \\
2.0 & 1.0
\end{bmatrix}
$$

> **直觉**：最大池化保留"最明显"的特征——适合捕获边缘、角点等强信号；平均池化做平滑——适合保留整体趋势。CNN 中 max pooling 远更常用，因为在特征检测的语境下，"有没有这个模式"比"这个模式平均有多强"更有信息量。

#### Python：从零实现池化

```python
def max_pool2d(x, pool_size=2, stride=2):
    """从零实现最大池化。"""
    H, W = x.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            i_start = i * stride
            j_start = j * stride
            out[i, j] = np.max(x[i_start:i_start + pool_size,
                                   j_start:j_start + pool_size])
    return out

def avg_pool2d(x, pool_size=2, stride=2):
    """从零实现平均池化。"""
    H, W = x.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            i_start = i * stride
            j_start = j * stride
            out[i, j] = np.mean(x[i_start:i_start + pool_size,
                                   j_start:j_start + pool_size])
    return out

# 验证手算结果
X = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
], dtype=float)

print("最大池化:")
print(max_pool2d(X))
print("\n平均池化:")
print(avg_pool2d(X))
print(f"\n与手算一致: {np.allclose(max_pool2d(X), [[2, 3], [3, 2]])}")
```

### 典型 CNN 架构：从 LeNet 到 ResNet

一个简化版的 CNN 流水线大概是这样的：

```
输入 28×28 → 卷积(5×5×8) → 28×28×8 → 最大池化(2×2) → 14×14×8
            → 卷积(5×5×16) → 14×14×16 → 最大池化(2×2) → 7×7×16
            → Flatten → 784维向量 → 全连接(128) → 全连接(10) → Softmax
```

整个过程的可视化直觉是：**边缘检测 → 形状组合 → 物体识别**。每一层都在前一层的基础上构建更高层的抽象——浅层看到线条，中层看到轮廓，深层看到"这是一张脸"。

下面是 CNN 发展史上的四座里程碑：

| 架构 | 年份 | 核心创新 | Top-5 错误率 |
|------|------|----------|-------------|
| LeNet-5 | 1998 | CNN 鼻祖，手写数字识别 | — |
| AlexNet | 2012 | ReLU + Dropout + GPU，点燃深度学习革命 | 15.3% |
| VGG-16 | 2014 | 用堆叠的小核($3\times3$)代替一个大核($7\times7$)，更深更省参数 | 7.3% |
| ResNet | 2015 | 残差连接(skip connection: $y = F(x) + x$)，让 152 层网络也能稳定训练 | 3.57% |

**关键洞察**：AlexNet 证明了"大数据 + GPU + CNN"的组合碾压传统方法；VGG 证明了"更深 = 更好"的规律；ResNet 的残差连接解决了"超过 20 层就训不动了"的退化问题——identity shortcut 让梯度可以绕过卷积层直接流回浅层，相当于给梯度铺了一条高速公路。

---

## 3. CNN 实战：迁移学习

迁移学习（Transfer Learning）是 CNN 实战中最常用的技术——不用十个 GPU 训练 ImageNet，直接"白嫖"别人练好的视觉底层特征。道理很简单：ImageNet 有一千多万张标注图片，在它上面训练出来的卷积核已经学会了"什么样的边缘、纹理、形状组合在一起构成物体"——这个知识对几乎所有视觉任务都有用。

```python
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch

# 加载预训练的 ResNet-18（在 ImageNet 1000 类上训练过的）
model = models.resnet18(pretrained=True)

# 冻结所有卷积层：不更新预训练权重
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层：从 1000 类 → 10 类（CIFAR-10）
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)

# 只用新加入的 fc 层是可训练的
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"可训练参数: {trainable:,} | 冻结参数: {frozen:,} "
      f"| 冻结比例: {frozen/(frozen+trainable)*100:.1f}%")

# 准备 CIFAR-10（CIFAR 是 32×32，ResNet 接受 224×224，需要 resize）
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = CIFAR10(root='./data', train=False, download=True, transform=transform)
print(f"训练集: {len(train_set)} 张 | 类别: {train_set.classes}")
```

> 迁移学习的本质：**底层特征可以复用**。就像学会了骑自行车的人学骑摩托车会快很多——平衡感（底层能力）已经内化了，只需学习摩托车特有的操作（顶层映射）。CNN 的浅层学到的是 Gabor 滤波器和颜色检测器——这些对所有视觉任务都通用。

---

## 4. 循环神经网络 (RNN)

### 核心思想

CNN 处理的是空间上的"邻居关系"，RNN 处理的则是时间上的"先后关系"。当你看一句话——"我今天早上去超市买了..."——读到"买了"的时候，你的理解已经整合了"今天早上"和"超市"的信息。这就是 RNN 想做的事。

RNN 的核心发明是一个**隐状态（hidden state）$h_t$**。它在处理序列的第 $t$ 个时间步时，同时接收两样东西：当前输入 $x_t$ 和上一个时间步的隐状态 $h_{t-1}$。$h_{t-1}$ 里"压缩存储"了从 $x_1$ 到 $x_{t-1}$ 的全部历史信息：

$$
h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)
$$

这个简洁的公式实现了**对过去的记忆与更新**：$W_h h_{t-1}$ 是"旧记忆的过滤版"，$W_x x_t$ 是"新输入的编码版"，$\tanh$ 把两者加起来并压缩到 $(-1, 1)$。

### 时间展开（Unrolled View）

RNN 的优雅之处在于：**它是"同一个网络在每一个时间步上重复应用"**。时间展开后可以看作一个深度等于序列长度的网络——但所有时间步共享同一组参数 $W_h, W_x, b$（这是 RNN 能处理变长序列的关键）。

输入序列 $x_1, x_2, \ldots, x_T$ 通过 RNN：

$$
\begin{aligned}
h_1 &= \tanh(W_h h_0 + W_x x_1 + b) \\
h_2 &= \tanh(W_h h_1 + W_x x_2 + b) \\
h_3 &= \tanh(W_h h_2 + W_x x_3 + b) \\
&\;\;\vdots \\
h_T &= \tanh(W_h h_{T-1} + W_x x_T + b) \\[4pt]
y_T &= W_y h_T + b_y \quad \text{（输出层）}
\end{aligned}
$$

### 手算：RNN 时间展开 3 步，逐步骤计算隐状态

下面是 RNN 最核心的**纸上计算**——给定一切参数和输入，一步步算出 $h_1, h_2, h_3$。建议你拿张纸跟算一遍。

**设定**：单维输入、单维隐状态（为了手算可行）。权重和输入如下：

$$
W_h = 0.5,\quad W_x = 1.0,\quad b = 0,\quad h_0 = 0
$$

$$
x_1 = 0.5,\quad x_2 = -0.3,\quad x_3 = 2.0
$$

> 注释：$W_h = 0.5$ 意味着每一步只保留前一隐状态的一半；$W_x = 1.0$ 意味着当前输入不衰减地进入计算；$b = 0$ 简化计算。

---

**时间步 1**：计算 $h_1$

$$
\begin{aligned}
z_1 &= W_h \cdot h_0 + W_x \cdot x_1 + b \\
    &= 0.5 \times 0 + 1.0 \times 0.5 + 0 \\
    &= 0 + 0.5 + 0 \\
    &= 0.5 \\[8pt]
h_1 &= \tanh(z_1) = \tanh(0.5) \\
    &= \frac{e^{0.5} - e^{-0.5}}{e^{0.5} + e^{-0.5}} \\
    &= \frac{1.6487 - 0.6065}{1.6487 + 0.6065} \\
    &= \frac{1.0422}{2.2552} \approx 0.4621
\end{aligned}
$$

$h_1 \approx 0.4621$ 编码了"看到第一个输入 0.5"之后的记忆。

---

**时间步 2**：计算 $h_2$

$$
\begin{aligned}
z_2 &= W_h \cdot h_1 + W_x \cdot x_2 + b \\
    &= 0.5 \times 0.4621 + 1.0 \times (-0.3) + 0 \\
    &= 0.2311 - 0.3 + 0 \\
    &= -0.0689 \\[8pt]
h_2 &= \tanh(z_2) = \tanh(-0.0689) \\
    &= \frac{e^{-0.0689} - e^{0.0689}}{e^{-0.0689} + e^{0.0689}} \\
    &= \frac{0.9334 - 1.0713}{0.9334 + 1.0713} \\
    &= \frac{-0.1379}{2.0047} \approx -0.0688
\end{aligned}
$$

$h_2 \approx -0.0688$ 接近零——说明了什么？来自 $h_1$ 的正记忆（0.2311）被当前的负输入（$-0.3$）几乎完全抵消了。RNN 在这一步"忘了"前面的正信号。

---

**时间步 3**：计算 $h_3$

$$
\begin{aligned}
z_3 &= W_h \cdot h_2 + W_x \cdot x_3 + b \\
    &= 0.5 \times (-0.0688) + 1.0 \times 2.0 + 0 \\
    &= -0.0344 + 2.0 + 0 \\
    &= 1.9656 \\[8pt]
h_3 &= \tanh(z_3) = \tanh(1.9656) \\
    &= \frac{e^{1.9656} - e^{-1.9656}}{e^{1.9656} + e^{-1.9656}} \\
    &= \frac{7.141 - 0.1400}{7.141 + 0.1400} \\
    &= \frac{7.001}{7.281} \approx 0.9615
\end{aligned}
$$

$h_3 \approx 0.9615$——当前输入 $2.0$ 主导了隐状态。前一隐状态 $-0.0688$ 乘上 $W_h = 0.5$ 后只剩 $-0.0344$，在 $2.0$ 面前几乎可以忽略。

---

**总结手算过程**：

| 时间步 | 输入 $x_t$ | 线性和 $z_t$ | 隐状态 $h_t$ | 解读 |
|:------|:----------|:------------|:-----------|:-----|
| 0 | — | — | 0 | 初始状态 |
| 1 | 0.5 | 0.5 | **0.4621** | 记住"正 0.5" |
| 2 | -0.3 | -0.0689 | **-0.0688** | 正负抵消，近乎遗忘 |
| 3 | 2.0 | 1.9656 | **0.9615** | 大正输入主导，接近饱和 |

**核心观察**：$W_h = 0.5$ 意味着每一步来自历史的信息衰减一半。如果没有新的强输入"续命"，历史记忆很快就会消失——这就是朴素 RNN **梯度消失**的根源（见下一节）。同时注意，$h_3$ 不仅依赖 $x_3$，也"依稀记得"前两步——$x_2$ 的负号通过 $W_h \times h_2$ 传递给了 $h_3$（虽然只剩 $\frac{1}{4}$ 的影响）。

### Python：从零实现简单 RNN 单元

```python
class SimpleRNN:
    """从零实现的朴素 RNN 单元。
    
    前向公式：h_t = tanh(W_h·h_{t-1} + W_x·x_t + b)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        scale = np.sqrt(6.0 / (hidden_dim + input_dim))
        self.W_x = np.random.uniform(-scale, scale, (hidden_dim, input_dim))
        self.W_h = np.random.uniform(-scale, scale, (hidden_dim, hidden_dim))
        self.b   = np.zeros((hidden_dim, 1))
        
        self.W_y = np.random.uniform(-scale, scale, (output_dim, hidden_dim))
        self.b_y = np.zeros((output_dim, 1))
    
    def forward(self, x_sequence):
        """前向传播。x_sequence 是 list of (input_dim, 1) 向量。"""
        T = len(x_sequence)
        hidden_dim = self.W_h.shape[0]
        
        h = np.zeros((hidden_dim, 1))  # 初始隐状态 = 0
        h_states = [h.copy()]
        
        for t in range(T):
            h = np.tanh(self.W_h @ h + self.W_x @ x_sequence[t] + self.b)
            h_states.append(h.copy())
        
        y = self.W_y @ h + self.b_y
        return y, h_states

# --- 验证手算结果 ---
# 用和手算完全相同的参数创建 RNN
rnn_verify = SimpleRNN(input_dim=1, hidden_dim=1, output_dim=1)
rnn_verify.W_x = np.array([[1.0]])
rnn_verify.W_h = np.array([[0.5]])
rnn_verify.b   = np.array([[0.0]])

x_seq = [np.array([[0.5]]), np.array([[-0.3]]), np.array([[2.0]])]
_, states = rnn_verify.forward(x_seq)

for i, s in enumerate(states):
    print(f"h_{i} = {s[0, 0]:.4f}")
print(f"\n与手算一致: h1={abs(states[1][0,0] - 0.4621) < 0.001}, "
      f"h2={abs(states[2][0,0] - (-0.0688)) < 0.001}, "
      f"h3={abs(states[3][0,0] - 0.9615) < 0.001}")

# --- 演示：教 RNN 记住"两个数的和" ---
np.random.seed(42)
rnn = SimpleRNN(input_dim=1, hidden_dim=8, output_dim=1)

def make_seq(a, b):
    return [np.array([[a]]), np.array([[b]]), np.array([[0.0]])]  # 0 = 终止符

X_train = [make_seq(a, b) for a, b in [(2, 3), (5, 7), (1, 4), (8, 2), (6, 9)]]
y_train = [np.array([[a + b]]) for a, b in [(2, 3), (5, 7), (1, 4), (8, 2), (6, 9)]]

for i, (seq, target) in enumerate(zip(X_train[:3], y_train[:3])):
    pred, states = rnn.forward(seq)
    print(f"输入 [a={seq[0][0,0]}, b={seq[1][0,0]}, 终止符] "
          f"→ 预测 {pred[0,0]:.3f}, 真实 {target[0,0]}")
    print(f"  隐状态: h0={states[0][0,0]:.3f} "
          f"→ h1={states[1][0,0]:.3f} "
          f"→ h2={states[2][0,0]:.3f} "
          f"→ h3={states[3][0,0]:.3f}")

print("\n注意隐状态 h 在每个时间步都在更新。"
      "如果用 MLP，a 和 b 必须同时输入——RNN 可以先看 a，再看 b，再出结果。")
```

### 应用连接

RNN 的"记忆"能力天然适用于：

- **时间序列预测**：股票价格、天气模式、传感器数据——下一个值依赖于过去的值
- **自然语言处理**：机器翻译、文本生成、情感分析——下一个词依赖于上文
- **语音识别**：声波 → 文字——每个音素都依赖于前面的音素
- **视频理解**：连续帧的动作识别——当前帧的动作由前面帧的动作决定

> **局限性**：朴素 RNN 的实际"记忆"只延续几个时间步——序列超过 10-20 步后，早期信息几乎完全消失。这是**梯度消失**导致的（见下节）。在手算中我们已经看到了：$W_h = 0.5$ 时，经过 3 步，$x_1$ 对 $h_3$ 的影响只剩 $\frac{1}{4}$。经过 10 步就是 $\frac{1}{1024}$——几乎为零。

---

## 5. LSTM 与 GRU：解决长程记忆

### 梯度消失——朴素 RNN 的死穴

通过时间展开，RNN 像一个极深的网络。反向传播时，梯度从 $h_T$ 一路乘到 $h_1$。每一步乘上 $W_h$ 的导数（对 tanh 来说是 $\text{diag}(1 - h_t^2)$，其元素都 ≤ 1）。如果 $W_h$ 的特征值小于 1，连乘几十次后梯度趋近于 0——这就是**梯度消失**。

在手算例子中，$W_h = 0.5$ 且 $\tanh'(z) \approx 1$（当 $z$ 接近 0 时），所以每反向传播一步，梯度大约缩为原来的 0.5 倍。3 步后剩 $\frac{1}{8}$，10 步后剩约 $\frac{1}{1000}$，30 步后剩约 $\frac{1}{10^9}$——早期输入对最终输出的影响在梯度层面完全消失，网络**只能学会利用最后几个词的信息**。

**直觉案例**：句子"我五年前在巴黎买的那个红色背包"——要预测下一个词是什么，RNN 需要"记住"五年前、巴黎、红色、背包——但这些词在序列中的位置差异很大。朴素 RNN 到了句子末尾，"五年前"和"巴黎"的信息已经在梯度传递中蒸发殆尽了。

### LSTM：三扇门的记忆高速公路

LSTM（长短期记忆网络，1997）的核心创新是引入了**细胞状态（cell state）$c_t$**——一条独立于隐状态的"记忆高速公路"。在这条高速公路上，信息可以几乎无衰减地跨越数十个时间步。三个门（gate）控制信息的流入、保留和流出：

- **遗忘门（Forget Gate）**：$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ —— $\sigma$ 输出 0-1，决定"旧记忆要忘记多少"。1 = 全部保留，0 = 全部丢弃。
- **输入门（Input Gate）**：$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ ——"新信息要写入细胞状态多少"
- **输出门（Output Gate）**：$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ ——"细胞状态的内容要输出给隐状态多少"

另外有一个**候选记忆** $\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$，代表"这一时间步新产生的信息"。

细胞状态和隐状态的更新方程为：

$$
\begin{aligned}
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

$\odot$ 是按元素相乘。**关键观察**：当遗忘门 $f_t \approx 1$ 且输入门 $i_t \approx 0$ 时，$c_t \approx c_{t-1}$——记忆原封不动地通过细胞状态传递下去。反向传播时，梯度通过 $c_t$ 这个"加法门"向后传递，**不经过 $\tanh$ 的导数压缩**——这就是梯度不消失的核心原因。

> **类比**：朴素 RNN 的隐状态像一根蜡烛——每一步都在燃烧，光越来越弱。LSTM 的细胞状态像一条传送带——东西放在上面就可以一直传下去，门控机制决定什么时候往上传/往下拿。

### GRU：LSTM 的精简版

GRU（门控循环单元，2014）把 LSTM 的三扇门精简为两扇，去掉了独立的细胞状态，让隐状态同时承担"记忆存储"和"当前输出"两个角色：

- **重置门（Reset Gate）**：$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$ —— "对历史内容打多少折扣？"$r_t = 0$ 时，当前状态完全忽略历史。
- **更新门（Update Gate）**：$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ —— "保留旧记忆还是接收新信息？"$z_t = 1$ 时完全保留旧状态，$z_t = 0$ 时完全更新为新状态。

$$
\begin{aligned}
\tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

GRU 的参数量比 LSTM 少约 1/3，实践中效果通常不相上下。选择哪个更多是工程权衡——LSTM 表达力略强，GRU 训练稍快。

### LSTM vs GRU 直观对比

| 维度 | 朴素 RNN | LSTM | GRU |
|:----|:--------|:-----|:----|
| 门数量 | 0 | 3（遗忘、输入、输出） | 2（重置、更新） |
| 是否有独立的记忆通道 | 否（$h_t$ 身兼二职） | 是（$c_t$ 专管记忆） | 否（$h_t$ 身兼二职） |
| 梯度消失 | 严重 | 基本解决 | 基本解决 |
| 参数量（比） | 1× | $\approx 4\times$ | $\approx 3\times$ |
| 长程依赖（>50步） | 失败 | 可以 | 通常可以 |

### Python：LSTM 做时间序列预测

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # 用最后时间步的输出做预测

# 生成正弦波数据
def generate_sine_data(seq_len=50, n_samples=500):
    np.random.seed(42)
    X_data, y_data = [], []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2 * np.pi)
        x_seq = np.linspace(start, start + 4 * np.pi, seq_len + 1)
        wave = np.sin(x_seq)
        X_data.append(wave[:seq_len])
        y_data.append(wave[seq_len])
    X = np.array(X_data).reshape(n_samples, seq_len, 1)
    y = np.array(y_data).reshape(n_samples, 1)
    return X, y

X, y = generate_sine_data()

split = int(0.8 * len(X))
X_train, X_test = (torch.tensor(X[:split], dtype=torch.float32),
                    torch.tensor(X[split:], dtype=torch.float32))
y_train, y_test = (torch.tensor(y[:split], dtype=torch.float32),
                    torch.tensor(y[split:], dtype=torch.float32))

model = LSTMPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 40 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
        print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.6f} | "
              f"Test Loss: {test_loss.item():.6f}")

print(f"\n训练完成。LSTM 学会了根据前 50 个正弦波点预测第 51 个点。")

import matplotlib.pyplot as plt
model.eval()
with torch.no_grad():
    demo_pred = model(X_test[:10])

plt.figure(figsize=(8, 4))
plt.plot(y_test[:10].numpy(), 'o-', label='真实值', markersize=6)
plt.plot(demo_pred[:10].numpy(), 's--', label='LSTM 预测', markersize=6)
plt.xlabel('样本编号'); plt.ylabel('sin(x) 值')
plt.title('LSTM 正弦波预测（前 50 点 → 第 51 点）')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

**输出示例**：经过 200 轮训练，测试 MSE 降到 0.0001 以下——LSTM 精确捕捉了正弦波的周期性。

---

## 6. 应用场景全览

### CNN 的应用版图

| 任务 | 输入 | 输出 | 典型架构 |
|------|------|------|---------|
| 图像分类 | 单张图片 → | 类别标签 | ResNet, EfficientNet |
| 目标检测 | 单张图片 → | 框坐标 + 类别 | YOLO, Faster R-CNN |
| 语义分割 | 单张图片 → | 每个像素的类别 | U-Net, Mask R-CNN |
| 人脸识别 | 人脸照片 → | 身份嵌入向量 | FaceNet, ArcFace |

### RNN/LSTM 的应用版图

| 任务 | 输入 | 输出 | 架构 |
|------|------|------|------|
| 文本分类 | 词序列 → | 情感/主题标签 | LSTM/GRU + 分类头 |
| 机器翻译 | 源语言句子 → | 目标语言句子 | Seq2Seq + Attention |
| 时间序列预测 | 历史 N 个点 → | 未来 M 个点 | LSTM/GRU + 全连接 |
| 语音识别 | 音频帧序列 → | 文字序列 | BiLSTM + CTC |

### Python：快速中文情感分析示例

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "这部电影太好了 演得非常棒", "垃圾电影 浪费时间 差评", "还行吧 一般般 没什么亮点",
    "超级好看 推荐 五星好评", "太烂了 不想再看第二遍", "特效不错 剧情稍微弱了点",
    "看完很感动 值得一看", "烂片 简直就是骗钱的", "比预想的好 算是合格",
    "无语 这种片也能上映", "质量很高 演员演技在线", "失望 浪费了两小时",
    "经典之作 反复看了好多遍", "还行 打发时间可以", "不要看 非常雷人",
    "剧情紧凑 节奏不错", "太拖沓了 看到一半就想走", "画面很美 配乐很棒",
    "比第一部差远了", "值回票价 推荐观看",
]
labels = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
X_bow = vectorizer.fit_transform(texts)

lr = LogisticRegression(random_state=42)
lr.fit(X_bow, labels)

print("词袋 + 逻辑回归 (baseline):")
for text, label in zip(texts[:5], labels[:5]):
    pred = lr.predict(vectorizer.transform([text]))[0]
    print(f"  '{text[:12]}...' → 真实:{'正面' if label else '负面'}, "
          f"预测:{'正面' if pred else '负面'}")

print("\n在真实项目中，你会用 LSTM/GRU 处理这类任务：")
print("1. 分词 → 词嵌入(embedding) → LSTM → 全连接 → Softmax(二分类)")
print("2. 或者直接上预训练模型: BERT/Ernie 微调效果远好于 LSTM")
print("   (但 LSTM 计算量更小，适合资源有限的场景)")
```

---

## 7. 何时使用 CNN / RNN vs 传统 ML

### 决策框架

按数据类型选架构：

| 你的数据特征 | 推荐方案 | 理由 |
|------------|---------|------|
| 表格数据（特征 < 1000，样本 < 1万） | SVM / RandomForest / XGBoost | CNN/RNN 是杀鸡用牛刀 |
| 表格数据（特征 > 1000 或样本 > 10万） | MLP | 简单架构 + 足够表达能力 |
| 图像数据（小规模，几百张） | 传统 ML + 手工特征 + 迁移学习 | 从头训练 CNN 会过拟合 |
| 图像数据（上千张以上） | CNN（或迁移学习） | 局部模式 + 层级特征 |
| 序列数据（短序列 < 20 步） | 传统时序模型 / MLP | RNN 优势不明显 |
| 序列数据（长序列，依赖远） | LSTM / GRU / Transformer | 需要长程记忆 |
| 序列数据 + 大计算资源 | Transformer | 几乎全面碾压 LSTM |

### 当传统 ML 比深度学习更好时

1. **数据量小**（< 1000 样本）：CNN/RNN 参数多、需要大量数据。Random Forest / SVM 在小数据上偏差更大、不易过拟合。
2. **特征已经有明确意义**：比如"用户年龄、收入、最近登录次数"——这些数字本身已经编码了结构化含义，CNN 的"局部感受野"毫无用武之地。
3. **可解释性是硬要求**：银行信贷审批需要解释为什么拒绝——决策树可以给出一串规则，CNN 给不出。
4. **部署环境受限**：手机本地推理要求模型极小（< 1MB）。逻辑回归 / 小决策树比任何深度网络都轻。

> **一句话总结**：数据结构决定架构选择。CNN/RNN 不是万能的——它们是针对空间/序列结构化数据设计的专门工具。你的数据没有这种结构，就不要强行套用。**先看数据长什么样，再选模型。不是先选模型再看数据能不能塞进去。**

---

## 8. 本章小结

本章从"MLP 为什么不够"出发，系统讲解了 CNN 和 RNN 两种处理结构化数据的神经网络架构。核心脉络：

1. **MLP 的致命盲区**——空间盲（打乱像素不影响训练）和时间盲（不知道词序），因为全连接层把所有输入当成"没有位置信息的一袋子数字"。

2. **CNN = 局部感受野 + 权重共享**——手算了一个 $2\times2$ 核在 $4\times4$ 图像上滑过的全部 9 个位置，证明了卷积不是魔法：就是对每个窗口做加权求和。同样的核检测所有位置的同一模式，所以猫耳朵在哪都能被找到。

3. **池化 = 压缩 + 增大感受野**——手算了 $2\times2$ 最大池化和平均池化，感受了两种策略的不同哲学：max 抓住"最明显的存在"，avg 抓住"整体的趋势"。

4. **RNN = 隐状态传递记忆**——手算了 3 个时间步的完整隐状态演化，亲眼看到 $W_h = 0.5$ 时历史信息每一步衰减一半——这就是梯度消失的微观根源。

5. **LSTM 的三扇门（遗忘、输入、输出）和 GRU 的两扇门（重置、更新）**解决了长程依赖——细胞状态 $c_t$ 像一条传送带，信息可以在上面几乎无衰减地跨越几十个时间步。

6. **实践的智慧**——预训练做迁移学习（白嫖 ImageNet 特征）、序列数据优先试 LSTM（够用了再考虑 Transformer）、数据不够多时传统 ML 可能更好。

下一步：[模型评估与调参](./17-model-evaluation.md)——有了模型之后，如何系统地评估它并找到最佳超参数配置。

---

## 9. 思考题

### Q1：全连接层和卷积层的参数数量对比——$3\times3$ 核的威力

输入是一张 $32\times32\times3$ 的 RGB 图像。输出需要 $10$ 个 $32\times32$ 的特征图。如果(a)用全连接层（后面 reshape 成 $32\times32\times10$）和(b)用 $3\times3$ 卷积（padding=same）分别需要多少参数？差距是多少倍？为什么卷积能做到参数这么少还不损失表达能力？

**答案：**

- (a) 全连接：输入 $32\times32\times3 = 3072$ 维，输出 $32\times32\times10 = 10240$ 维。需参数量 = $3072 \times 10240 + 10240 = 31,467,520 + 10,240 = \mathbf{31,477,760}$（约 3148 万）。
- (b) 卷积：每个核的参数量 = $3\times3\times3 + 1 = 28$（三个通道，一个偏置）。需要 10 个核，总参数量 = $28 \times 10 = \mathbf{280}$。

差距为 $31477760 / 280 \approx \mathbf{112,420}$ 倍。卷积能做到这么少的参数是因为**权重共享**——同一个核对图像的所有位置使用完全相同的权重。全连接层必须为"左上角第 3 行第 5 列的像素"和"右下角第 30 行第 30 列的像素"各自学习独立的连接权重，而卷积只学一种模式（比如"检测竖边"），然后扫遍全图去匹配——不管位置在哪里。

这种"参数少"不是弱点，反而是优势：参数少意味着**归纳偏置强**（更强的空间结构先验），更不容易过拟合，需要更少的训练数据就能泛化。

---

### Q2：手算 strided 卷积输出尺寸

输入尺寸 $7\times7$，卷积核 $3\times3$，填充 padding=1。分别计算 stride=1、stride=2、stride=3 的输出尺寸，并说明 stride=3 为何在工程中几乎不会这么设。

**答案：**

输出尺寸公式：$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$

- stride=1：$H_{out} = \lfloor (7 + 2 - 3) / 1 \rfloor + 1 = \lfloor 6 \rfloor + 1 = \mathbf{7}$（same padding 效果）
- stride=2：$H_{out} = \lfloor (7 + 2 - 3) / 2 \rfloor + 1 = \lfloor 3 \rfloor + 1 = \mathbf{4}$
- stride=3：$H_{out} = \lfloor (7 + 2 - 3) / 3 \rfloor + 1 = \lfloor 2 \rfloor + 1 = \mathbf{3}$

stride=3 基本不用于普通卷积，因为 $s > k$ 时卷积核会**跳过**输入中的某些像素（每个输入像素不一定被核覆盖到），丢失信息。Stride=3 常见于**第一个卷积层的快速降采样**（如 ResNet 的开头用 7×7 stride=2 核快速缩小尺寸），但不是 stride=3 配 3×3 核——因为 3×3 核 stride=3 意味着覆盖窗口之间完全没有重叠，每个像素只被一个输出位置看到，而且输入边缘的像素（如第 1、2、4、5 行）根本不参与任何窗口的中心位置计算，浪费了大量信息。

---

### Q3：平均池化 vs 最大池化——你会在哪种场景选谁？

各举一个场景，说明应该用 max pooling 还是 average pooling，并给出理由。

**答案：**

- **场景 A（用 max pooling）**：图像分类中检测是否存在某个物体。比如判断图片里有没有猫——猫耳朵的边缘是一个强信号，我们需要知道"有没有这个模式"而非"这个模式平均有多强"。max pooling 对单个像素的高响应极其敏感，非常适合捕获稀疏的、局部的强信号特征。

- **场景 B（用 average pooling）**：医学图像分割中需要对整个区域做平滑预测。比如 CT 扫描中判断某个区域是否是肿瘤——边缘处可能会有噪声（单个像素的异常高值），用 average 可以平滑这些噪声，得到更稳定的区域表示。

- **场景 C（全局 average pooling, GAP）**：分类网络的最后一层，在最后一个卷积层之后、全连接层之前，用 GAP 将 $H\times W\times C$ 的特征图压缩为 $1\times1\times C$ 的向量。这比先 Flatten 再接全连接层的方案参数量少得多（GAP 本身零参数），同时强制每个通道的激活图与最终分类一一对应，可解释性更强（CAM 可视化就依赖 GAP）。

---

### Q4：如果给 CNN 的第一层放一个全是 1 的大核（如 $7\times7$），它会学到什么？如果放一个小核（如 $3\times3$），它会学到什么？

**答案：**

全是 1 的 $7\times7$ 核等价于对 $7\times7$ 区域做**均值滤波**——输出是对局部区域亮度的平滑。这是一种**低通滤波器**，会抹平细节（边缘模糊化），保留了物体的大致轮廓和整体亮度分布。在 CNN 的第一层用全是 1 的核相当于在问"这 7×7 区域平均有多亮/暗？"——这在某些任务（如光照归一化）中有用，但对检测物体的精细特征帮助不大。

$3\times3$ 核（即使是随机初始化的）可以表达更丰富的局部模式——竖边、横边、斜边、角点、纹理——因为这些模式的空间尺度通常就是 3-5 个像素。更关键的是，**两个 $3\times3$ 核堆叠的感受野等于一个 $5\times5$ 核**，但参数量更少（$2\times9 = 18$ vs $25$），且中间夹了一层非线性激活，表达能力更强。这就是 VGG 用全部 $3\times3$ 核而不用大核的原因。

---

### Q5：为什么朴素 RNN 的权重 $W_h$ 的特征值必须小于 1（不能大于 1）？

设 $W_h$ 的最大特征值为 $\lambda_{\max}$。如果 $\lambda_{\max} > 1$，会发生什么？如果 $\lambda_{\max} < 1$ 呢？从梯度传播的角度解释。

**答案：**

时间展开后，从 $h_T$ 到 $h_1$ 的梯度大约经过 $T$ 次乘以 $W_h$（忽略 $\tanh'$ 的影响，它 ≤ 1，只会让梯度更小或不变）：

- **$\lambda_{\max} > 1$（梯度爆炸）**：每反向传播一步，梯度大致乘以 $\lambda_{\max}$。10 步后梯度放大 $\lambda_{\max}^{10}$ 倍，50 步后放大 $\lambda_{\max}^{50}$ 倍——梯度呈指数级爆炸，导致参数更新步长巨大、损失函数剧烈震荡或直接变为 NaN，模型无法收敛。工程上用**梯度裁剪**（gradient clipping）来缓解：当梯度范数超过阈值时按比例缩小。

- **$\lambda_{\max} < 1$（梯度消失）**：每反向传播一步，梯度缩小为原来的 $\lambda_{\max}$ 倍，$T$ 步后只剩 $\lambda_{\max}^T$。这就是朴素 RNN 的死穴——早期输入对损失函数的影响在训练中几乎为零，网络只能学到短程依赖。

理想情况是 $\lambda_{\max} \approx 1$——这样梯度在反向传播中保持稳定。LSTM 之所以解决了梯度消失，就是因为细胞状态的梯度路径中 $f_t$ 接近 1 时，这条路径上的乘法因子 ≈ 1（而非 RNN 中总是 × $W_h$ 且过 $\tanh$），信息得以"原样"传递很远。

---

### Q6：LSTM 中，当遗忘门 $f_t = 1$ 且输入门 $i_t = 0$ 时，细胞状态如何变化？这对应什么"认知行为"？

**答案：**

当 $f_t = 1, i_t = 0$ 时：

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t = 1 \cdot c_{t-1} + 0 \cdot \tilde{c}_t = c_{t-1}
$$

细胞状态完全不变——上一个时间步的记忆被原封不动地复制到当前时间步。对应的"认知行为"是：**当前输入 $x_t$ 完全不相关，不应该影响当前对上下文的记忆**。

**实例**：在阅读"这个苹果——它又红又圆——非常好吃"时，读到括号内的"它又红又圆"时，LSTM 可以设置 $f_t \approx 1, i_t \approx 0$（暂且不更新主记忆，因为这是对"苹果"的补充描述），等读到"非常好吃"时再打开输入门，把口感信息写入细胞状态。

另一个极端情况：如果 $f_t = 0, i_t = 1$，则 $c_t = \tilde{c}_t$——旧记忆被完全丢弃，细胞状态被全新的信息替代。这对应于"话题转换"的场景。

---

### Q7：GRU 的门数比 LSTM 少一个，为什么效果通常不相上下？

**答案：**

GRU 的设计核心是把 LSTM 的"遗忘门 + 输入门"这对互补操作合并为一个**更新门** $z_t$（以及它的互补值 $1 - z_t$）：

- LSTM 中：$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$，$f_t$ 和 $i_t$ 是两个独立参数决定的 sigmoid 值，可以出现"既不忘也不记"的暧昧状态（比如 $f_t = 0.4, i_t = 0.5$，$c_t \neq c_{t-1}$ 也 $\neq \tilde{c}_t$）。
- GRU 中：$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$，$z_t$ 在 0-1 之间，$(1-z_t)$ 和 $z_t$ 之和恒为 1——这是一个**凸组合**，旧状态和新候选状态之间是"此消彼长"的关系。

GRU 效果通常不相上下的原因：

1. **凸组合已足够**：在实践中，"保留旧记忆 vs 接收新信息"这个权衡用一个自由度（$z_t$）来控制已经够用了，不需要独立的 $f_t$ 和 $i_t$ 两个自由度。
2. **更少的参数 = 更不容易过拟合**：在小数据集上 GRU 可能反而优于 LSTM。
3. **重置门 $r_t$ 提供了灵活性**：虽然更新只用一个门，但重置门允许当前步选择性地忽略历史，给模型保留了"聚焦当下"的能力。

不过在大规模数据集上，LSTM 的表达力优势有时会显现出来。选择 LSTM 还是 GRU 更多是经验和调参，没有决定性的理论优势。

---

### Q8：用你手算过卷积的那张 $4\times4$ 矩阵 $X$，如果用 padding=1（一圈零填充）后再做 $2\times2$ stride=1 卷积，输出尺寸变成多少？计算新矩阵 $(0,0)$ 和 $(2,2)$ 位置的卷积值。

**答案：**

原始 $X$：
$$
X = \begin{bmatrix}
1 & 2 & 3 & 0 \\
0 & 1 & 2 & 3 \\
3 & 0 & 1 & 2 \\
2 & 3 & 0 & 1
\end{bmatrix}
$$

padding=1 后变成 $6\times6$（四周各加一圈 0），核 $K$ 不变：

$$
X_{\text{pad}} = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 2 & 3 & 0 & 0 \\
0 & 0 & 1 & 2 & 3 & 0 \\
0 & 3 & 0 & 1 & 2 & 0 \\
0 & 2 & 3 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

输出尺寸：$H_{out} = \frac{4 + 2\times1 - 2}{1} + 1 = 5$，输出是 $5\times5$。

位置 $(0,0)$：覆盖 padding 后的第 0-1 行、0-1 列——全是零。

$$
Y[0,0] = 0\times1 + 0\times0 + 0\times2 + 0\times1 = \mathbf{0}
$$

位置 $(2,2)$：覆盖 padding 后的第 2-3 行、2-3 列，即原始 $X$ 的第 1-2 行、1-2 列：

$$
\begin{bmatrix}1 & 2 \\ 0 & 1\end{bmatrix} \odot \begin{bmatrix}1 & 0 \\ 2 & 1\end{bmatrix}
= 1\times1 + 2\times0 + 0\times2 + 1\times1 = \mathbf{2}
$$

---

### Q9：你手算了 $W_h = 0.5$ 的 RNN。如果 $W_h$ 改为 $0.1$ 或 $0.9$，对 $h_3$ 的值和"记忆能力"分别有什么影响？用同一个输入序列 $[0.5, -0.3, 2.0]$ 估算。

**答案：**

重新手算。原 $W_h = 0.5$ 时：$h_1 = 0.4621, h_2 = -0.0688, h_3 = 0.9615$。

**$W_h = 0.1$（弱记忆）**：

$$
\begin{aligned}
h_1 &= \tanh(0.1 \times 0 + 1.0 \times 0.5) = \tanh(0.5) = 0.4621 \\
h_2 &= \tanh(0.1 \times 0.4621 + 1.0 \times (-0.3)) = \tanh(-0.2538) \approx -0.2485 \\
h_3 &= \tanh(0.1 \times (-0.2485) + 1.0 \times 2.0) = \tanh(1.9752) \approx 0.9620
\end{aligned}
$$

$h_3$ 从 0.9615 变成 0.9620——几乎不变。因为 $W_h = 0.1$ 时，$x_1$ 对 $h_3$ 的影响经过两步衰减后只剩 $0.1^2 = 1\%$，$x_2$ 的影响只剩 $0.1 = 10\%$——RNN 几乎变成了一个"只看当前输入"的网络，没有记忆能力。

**$W_h = 0.9$（强记忆）**：

$$
\begin{aligned}
h_1 &= \tanh(0.9 \times 0 + 1.0 \times 0.5) = 0.4621 \\
h_2 &= \tanh(0.9 \times 0.4621 + 1.0 \times (-0.3)) = \tanh(0.1159) \approx 0.1154 \\
h_3 &= \tanh(0.9 \times 0.1154 + 1.0 \times 2.0) = \tanh(2.1039) \approx 0.9715
\end{aligned}
$$

$h_3$ 从 0.9615 变成 0.9715——升高了，因为 $x_1$ 的正信号（经过 $0.9^2 = 81\%$ 保留）和 $x_2$ 的负信号（经过 $0.9 = 90\%$ 保留）更完整地传到了 $h_3$。$W_h = 0.9$ 意味着历史信息几乎不失真地传递（每步保留 90%），记忆能力强，但代价是梯度反向传播时也乘 $0.9$，经过 50 步还剩 $0.9^{50} \approx 0.5\%$——仍然会消失，只是比 $0.5^{50}$ 好得多。

---

### Q10：以下三个任务，你分别选 CNN、RNN/LSTM 还是传统 ML（Random Forest / XGBoost）？简要说明理由。

(a) 根据用户近 6 个月的刷卡记录（金额、商户类型、时间戳）预测下个月是否会逾期。数据量 5 万条，每条 200 个时间步。

(b) 手机拍照识别 10 种植物叶子。你只有每种 20 张照片，共 200 张。

(c) 根据 50 个结构化特征（年龄、收入、城市、历史购买品类等）预测用户是否会点击某个广告。数据量 200 万条。

**答案：**

**(a) 序列预测 → LSTM/GRU**。刷卡记录是典型的时间序列——"最近在赌场刷了三笔大额"这个模式的时序依赖很强，朴素 MLP 无法自然地建模时间先后关系。LSTM/GRU 可以处理变长的 200 步序列，并通过门控机制捕捉"某类异常消费的累积效应"。如果数据量更大（百万级），可以考虑 Transformer。

**(b) 少量图像 → 迁移学习（CNN + 预训练）**。200 张图片从头训练 CNN 必然过拟合。正确的做法是：(1) 加载 ImageNet 预训练的 ResNet/MobileNet，(2) 冻结卷积层，(3) 只训练顶层分类器。如果还不够，可以做数据增强（旋转、裁剪、颜色抖动）和少量微调顶层卷积。极端小数据时，传统 CV 特征（SIFT + SVM）也可能更好。

**(c) 大量结构化数据 → XGBoost 或 MLP**。200 万条、50 个特征——数据量很大，但特征之间没有空间或时序结构，CNN/RNN 的归纳偏置是累赘。XGBoost 在处理表格数据上的效率通常高于 MLP（更快、调参更简单、自动处理缺失值）。但 200 万条的规模也适合 MLP，尤其当特征之间有复杂的非线性交互时。两条路线都可以试，用验证集选最优。

关键原则：**先看数据有没有空间/时序结构（有→CNN/RNN），再看数据量（少→传统 ML 或迁移学习），最后根据规模和算力在深度方法和传统方法之间权衡。**

---

> **下一步：[模型评估与调参](./17-model-evaluation.md)**——学会了 CNN 和 RNN 怎么工作，接下来学如何系统地评估它们并调到最佳状态。
