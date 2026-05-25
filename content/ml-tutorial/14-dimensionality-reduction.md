## 降维：压缩数据，保留信息

> **核心问题**：给你 5 个二维点——(1,2), (2,4), (3,6), (4,8), (5,10)——它们几乎都在一条直线上。你能不能只用**一个数**来描述每个点，同时又能在需要的时候把它"还原"成原来的二维坐标？如果可以，怎么找到这个"最不丢信息"的方向？

---

### 0. 本章导览

降维是机器学习中唯一一个**主动丢弃信息**的技术——这听起来很疯狂，但正是这种"有意为之的损失"让你能看见高维数据的形状、滤掉噪声、加速模型训练。

本章不做"调包侠"。你将**亲手**对一个只有 5 个点的数据集完成完整的 PCA 计算：中心化→协方差矩阵→特征分解→投影到 1D→重建。每一步都用笔算的精度展开，然后用 Python 验证。在此基础上我们再讨论 t-SNE、SVD、sklearn 实战。

学完本章，你将能够：

1. **手算** PCA 的完整六步流程：5 个二维点 → 1 个维度，每步写出数值
2. 解释为什么协方差矩阵的特征向量就是"方差最大的方向"
3. 用特征值比例选择最优主成分数量 K
4. 理解 PCA 与 SVD 的等价性，知道为何 sklearn 内部用 SVD 而非协方差分解
5. 区分 t-SNE 和 PCA 的适用场景——以及 t-SNE 为什么**不能**放进建模流水线
6. 在 sklearn 中完整应用 PCA 管道，并对比降维前后的模型效果

> 本章目标 1200+ 行。**核心手算部分请拿纸笔跟着算。**

前置章节：数学基础六章（尤其是[线性代数](./math-linear-algebra.md)） + [聚类](./13-clustering.md)
下一章：[神经网络入门](./15-neural-networks-intro.md)

---

### 1. 生活例子：100 列的 Excel，能画出它的形状吗？

你是公司的新晋数据分析师。市场部给你一个 Excel 文件：2000 个客户，每个客户有 100 个属性——年龄、收入、消费频次、最近购买日期、浏览页面数、平均停留时长……你想"看一看"这群客户长什么样。

你把数据丢进 `plt.scatter()`，报错了 —— scatter 只能画 2 维或 3 维。100 维的数据对人眼来说是完全不可见的世界——你只能盯着数字，却看不到"形状"。

更麻烦的是，你怀疑这 100 个属性里有大量冗余——"月消费金额"和"年消费金额"基本上说的是同一件事，"注册天数"和"活跃天数"高度相关。2000 个客户可能只分布在 5-6 个"有效方向"上，其余 90+ 维都是噪声。

降维就是解决这个问题的数学工具：

| 动机 | 具体问题 | 降维如何解决 |
|------|---------|------------|
| **可视化** | 100 维数据无法画图 | 降至 2 或 3 维，肉眼观察聚类、异常点 |
| **去噪** | 大量维度只是测量误差 | 保留方差大的主方向，丢弃低方差 → 零噪声维度 |
| **加速** | 特征每多 1 维，训练时间线性增长 | 压缩到低维后模型训练和推理显著更快 |
| **缓解过拟合** | 样本数 n=200 但特征数 p=500 | 降维 → 减少有效参数数量 |

**两种范式：特征选择 vs 特征提取。** 特征选择（Feature Selection）从原始 100 列里挑出最有用的 5 列——优点是"我知道保留的是哪 5 列"，缺点是可能丢失分散在多列中的信号。特征提取（Feature Extraction，即降维）通过数学变换把 100 列组合成 5 个全新的"超级列"——新列不可直接解释（你不会说"超级列 3 代表什么"），但能更高效地压缩信息。PCA 属于后者。

```python
import numpy as np
import matplotlib.pyplot as plt

# 验证维度灾难：维度越高，所有点"距离"越趋于相等
np.random.seed(42)
n_points = 500

dims, ratios = [], []
for d in [1, 2, 5, 10, 20, 50, 100]:
    X = np.random.randn(n_points, d)
    dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    mask = dists > 0
    ratio = dists[mask].min() / dists[mask].max()
    dims.append(d)
    ratios.append(ratio)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(dims, ratios, 'b-o', markersize=8)
ax1.set_xlabel('维度 d'); ax1.set_ylabel('最近距 / 最远距')
ax1.set_title('维度灾难：d↑ → 所有距离趋同 → "近邻"无意义')

ax2.text(0.3, 0.5, '100维数据\n↘\n只剩一行数字\n\x1b[0m',
         transform=ax2.transAxes, fontsize=22, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='#FFEEEE', alpha=0.9))
ax2.set_title('高维数据 = 看不见形状的数据')
ax2.axis('off')
plt.tight_layout(); plt.show()
```

---

### 2. 直观理解：PCA 到底在做什么？

#### 2a. 蚂蚁爬棍子

**直觉理解：** 想象一群蚂蚁沿着一根倾斜的棍子爬动——有的在棍子顶部，有的在底部，有的在中间。从三维空间看，每只蚂蚁有一个 (x, y, z) 坐标。但如果你观察够久会发现：所有蚂蚁几乎都在**同一条直线**上——这根棍子的方向。PCA 做的事就是找到这根"棍子"（第一主成分），然后把每只蚂蚁的位置从 (x, y, z) 压缩成**一个数**——沿棍子走了多远。

生活化的比喻：你给一群人拍了张合照，记录下他们的身高、臂展、鞋码。你发现这三个特征高度相关——个子高的人臂展也长、鞋码也大。与其存三个数，不如用一个叫"体型"的综合指标来描述每个人。PCA 就是自动算出这个"体型"指标——而且保证它是所有可能的一维指标中**保留信息最多**的那个。

#### 2b. 形式化定义

PCA 寻找数据**方差最大的方向**。为什么是方差？因为方差 = 信息。如果全部数据在某个方向上完全一致（方差 = 0），沿着这个方向你分不清任何两个点——它没有"区分度"，即没有信息。PCA 要找的就是让投影后的点尽可能**散开**的方向。

PCA 的六步流程：

1. **中心化**：$\mathbf{X} \leftarrow \mathbf{X} - \bar{\mathbf{x}}$（每列减去该列均值，把数据"搬到原点"）
2. **协方差矩阵**：$\mathbf{C} = \frac{1}{n-1} \mathbf{X}^{\mathsf{T}}\mathbf{X}$——每对特征间的"共同变化趋势"
3. **特征分解**：解 $\mathbf{C}\mathbf{v}_i = \lambda_i \mathbf{v}_i$，得到特征向量 $\mathbf{v}_i$（主成分方向）和 $\lambda_i$（该方向的方差大小）
4. **选前 K 个最大 $\lambda$** 对应的 $\mathbf{v}$，组成投影矩阵 $\mathbf{V}_K$
5. **降维**：$\mathbf{Z} = \mathbf{X} \mathbf{V}_K$（每个样本的新坐标：原来 p 维，现在 K 维）
6. **重建**：$\hat{\mathbf{X}} = \mathbf{Z} \mathbf{V}_K^{\mathsf{T}}$（注意：$\hat{\mathbf{X}} \neq \mathbf{X}$，除非你把所有成分都保留了）

---

### 3. 手算 PCA：5 个点，从头算到尾 ⭐

这是本章的核心。放下鼠标，拿起纸笔，一步一步跟着算。

#### 3a. 数据：5 个恰到好处的二维点

我们精心挑选了 5 个点——它们**完全共线**（全部落在直线 $y = 2x$ 上）：

$$(1, 2), \quad (2, 4), \quad (3, 6), \quad (4, 8), \quad (5, 10)$$

之所以选完全共线的点，是因为这样 PCA 的行为最"极端"也最清晰：第一主成分会精确地落在 $y=2x$ 方向上（能捕获 100% 的方差），第二主成分方差为 0——就是那条直线的垂直方向。算完之后你会对 PCA 的几何含义刻骨铭心。

```
y
10 ┤                                    ✦(5,10)
 9 ┤
 8 ┤                              ✦(4,8)
 7 ┤
 6 ┤                        ✦(3,6)
 5 ┤
 4 ┤                  ✦(2,4)
 3 ┤
 2 ┤            ✦(1,2)
 1 ┤
  └─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴── x
   1   2   3   4   5   6   7   8   9  10
```

**一眼就能看出：** 所有点沿斜线延伸，垂直斜线的方向完全没有任何"散开"——这正是 PCA 要捕捉的结构。

#### 3b. 第 1 步：中心化（去均值）

先把数据"搬到原点"。求每列的均值：

$$\bar{x} = \frac{1 + 2 + 3 + 4 + 5}{5} = \frac{15}{5} = 3$$

$$\bar{y} = \frac{2 + 4 + 6 + 8 + 10}{5} = \frac{30}{5} = 6$$

每个点减去均值 $(\bar{x}, \bar{y}) = (3, 6)$：

$$
\begin{aligned}
(1, 2) - (3, 6) &= (-2, -4) \\
(2, 4) - (3, 6) &= (-1, -2) \\
(3, 6) - (3, 6) &= (\;0, \;\;0) \\
(4, 8) - (3, 6) &= (\;1, \;\;2) \\
(5, 10) - (3, 6) &= (\;2, \;\;4)
\end{aligned}
$$

中心化后的数据矩阵（$5 \times 2$）：

$$\mathbf{X}_{\text{centered}} = \begin{bmatrix}
-2 & -4 \\
-1 & -2 \\
 0 &  0 \\
 1 &  2 \\
 2 &  4
\end{bmatrix}$$

注意其规律：**第二列 = 第一列 × 2**——这是共线数据在中心化后留下的痕迹，也为后面的特征分解埋下了伏笔。

**验证中心化是否正确：** 每列的和应当为 0。
第一列：$(-2) + (-1) + 0 + 1 + 2 = 0$ ✓
第二列：$(-4) + (-2) + 0 + 2 + 4 = 0$ ✓

#### 3c. 第 2 步：计算协方差矩阵

原始数据有 2 个特征（列），所以协方差矩阵是 $2 \times 2$ 的：

$$\mathbf{C} = \frac{1}{n-1} \mathbf{X}_{\text{centered}}^{\mathsf{T}} \mathbf{X}_{\text{centered}} \;=\; \frac{1}{4} \cdot \mathbf{X}_{\text{centered}}^{\mathsf{T}} \mathbf{X}_{\text{centered}}$$

先算 $\mathbf{X}^{\mathsf{T}}\mathbf{X}$ 的四个元素（注：以下用 $\mathbf{X}$ 简写 $\mathbf{X}_{\text{centered}}$）：

**元素 (1,1) — 特征 $x$ 的方差相关：**

$$\sum x_i^2 = (-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 4 + 1 + 0 + 1 + 4 = 10$$

$$C_{11} = \frac{10}{4} = 2.5$$

**元素 (1,2) 和 (2,1) — $x$ 与 $y$ 的协方差相关（对称矩阵，二者相等）：**

$$\sum x_i y_i = (-2)(-4) + (-1)(-2) + 0 \cdot 0 + 1 \cdot 2 + 2 \cdot 4 = 8 + 2 + 0 + 2 + 8 = 20$$

$$C_{12} = C_{21} = \frac{20}{4} = 5.0$$

**元素 (2,2) — 特征 $y$ 的方差相关：**

$$\sum y_i^2 = (-4)^2 + (-2)^2 + 0^2 + 2^2 + 4^2 = 16 + 4 + 0 + 4 + 16 = 40$$

$$C_{22} = \frac{40}{4} = 10.0$$

整理成矩阵：

$$\boxed{\mathbf{C} = \begin{bmatrix} 2.5 & 5.0 \\ 5.0 & 10.0 \end{bmatrix}}$$

**关键观察：** 第二行 = 第一行 × 2，第二列 = 第一列 × 2——这正是共线数据在协方差矩阵中的表现形式：矩阵的秩为 1（不是满秩），说明有一个方向的方差为零。

**直觉连接：** $C_{11} = 2.5$ 告诉你 x 方向上有多少"散开"，$C_{22} = 10.0$ 告诉你 y 方向上更多。$C_{12} = 5.0$ 告诉你：x 变大时 y 也变大（正相关），而且相关性很强（$5.0$ 相对 $2.5$ 和 $10.0$ 来说很大）。

#### 3d. 第 3 步：特征分解 — 找到"棍子"的方向

协方差矩阵 $\mathbf{C}$ 的特征向量就是我们要找的主成分方向，特征值就是该方向上的方差大小。

解特征方程 $\det(\mathbf{C} - \lambda\mathbf{I}) = 0$：

$$\det\!\begin{pmatrix} 2.5 - \lambda & 5.0 \\ 5.0 & 10.0 - \lambda \end{pmatrix} = 0$$

$$(2.5 - \lambda)(10.0 - \lambda) - 5.0 \times 5.0 = 0$$

展开：

$$25.0 - 2.5\lambda - 10.0\lambda + \lambda^2 - 25.0 = 0$$

$$\lambda^2 - 12.5\lambda = 0$$

$$\lambda(\lambda - 12.5) = 0$$

两个特征值从天而降：

$$\boxed{\lambda_1 = 12.5, \quad \lambda_2 = 0}$$

**解读：** $\lambda_1 = 12.5$ 意味着沿第一主成分方向的方差为 12.5——这是所有的信息所在的方向。$\lambda_2 = 0$ 意味着沿着第二主成分方向**方差为 0**——所有点在这个方向上完全重合，没有任何区分度。这完美印证了"数据完全共线"的事实。

**解释方差比（Explained Variance Ratio）：**

$$\text{PC1：} \frac{12.5}{12.5 + 0} = 100\% \qquad \text{PC2：} \frac{0}{12.5 + 0} = 0\%$$

第一主成分捕获了 **100%** 的方差——降维到 1D 时零信息损失。

---

**找特征向量 — 对 $\lambda_1 = 12.5$：**

解 $(\mathbf{C} - 12.5\mathbf{I})\mathbf{v}_1 = \mathbf{0}$：

$$\begin{bmatrix} 2.5 - 12.5 & 5.0 \\ 5.0 & 10.0 - 12.5 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} -10 & 5 \\ 5 & -2.5 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \mathbf{0}$$

取第一行：$-10v_{11} + 5v_{12} = 0 \implies 5v_{12} = 10v_{11} \implies v_{12} = 2v_{11}$

所以第一主成分的方向是 $[1,\; 2]^{\mathsf{T}}$。（乘以任何非零常数仍是特征向量；我们取最简单的形式）

**这个结果有多美妙？** 原始数据在直线 $y = 2x$ 上，而 PCA 求出的第一主成分恰恰好就是 $(1, 2)$ 方向——斜率 $2/1 = 2$，完全吻合！PCA 不是猜出来的，是**算**出来的。

归一化（转成单位向量，方便后续投影）：

$$\|\mathbf{v}_1\| = \sqrt{1^2 + 2^2} = \sqrt{5}$$

$$\boxed{\mathbf{v}_1 = \begin{bmatrix} 1/\sqrt{5} \\[4pt] 2/\sqrt{5} \end{bmatrix} \approx \begin{bmatrix} 0.4472 \\ 0.8944 \end{bmatrix}}$$

---

**找特征向量 — 对 $\lambda_2 = 0$：**

解 $\mathbf{C}\mathbf{v}_2 = \mathbf{0}$：

$$\begin{bmatrix} 2.5 & 5.0 \\ 5.0 & 10.0 \end{bmatrix} \begin{bmatrix} v_{21} \\ v_{22} \end{bmatrix} = \mathbf{0}$$

取第一行：$2.5v_{21} + 5.0v_{22} = 0 \implies v_{21} = -2v_{22}$

取 $v_{22} = 1$，得 $v_{21} = -2$，方向 $[-2,\; 1]^{\mathsf{T}}$。

归一化：

$$\|\mathbf{v}_2\| = \sqrt{(-2)^2 + 1^2} = \sqrt{5}$$

$$\boxed{\mathbf{v}_2 = \begin{bmatrix} -2/\sqrt{5} \\[4pt] \;1/\sqrt{5} \end{bmatrix} \approx \begin{bmatrix} -0.8944 \\ \;\;0.4472 \end{bmatrix}}$$

**验证正交性：** $\mathbf{v}_1 \cdot \mathbf{v}_2 = (1)(-2) + (2)(1) = -2 + 2 = 0$ ✓。两个主成分相互垂直，分别捕捉独立的"变化方向"。

**几何解释：**

```
       y
       ↑
       │           ✦ PC1 = (1,2)/√5  → 数据延伸的方向
       │         ✦
       │       ✦          ← 所有点都沿这条线
       │     ✦
       │   ✦
       │ ✦
       │
 ──────┼──────────→ x
       │
       │    ← PC2 ⟂ PC1：垂直于数据方向, 方差=0
```

#### 3e. 第 4 步：选 K，做投影矩阵

我们要降到 1 维（K=1）。投影矩阵 $\mathbf{V}_K$ 就是取最大的特征值对应的特征向量：

$$\mathbf{V}_1 = \mathbf{v}_1 = \begin{bmatrix} 1/\sqrt{5} \\[2pt] 2/\sqrt{5} \end{bmatrix}$$

#### 3f. 第 5 步：降维 — 把 2D 压成 1D

对每个中心化后的样本 $\mathbf{x}_i$，计算它沿 PC1 的投影（= 内积）：

$$\begin{aligned}
z_1 &= (-2,\; -4) \cdot \begin{bmatrix} 1/\sqrt{5} \\ 2/\sqrt{5} \end{bmatrix} = \frac{-2 \times 1 + (-4) \times 2}{\sqrt{5}} = \frac{-2 - 8}{\sqrt{5}} = \frac{-10}{\sqrt{5}} = -2\sqrt{5} \approx -4.472 \\[8pt]
z_2 &= (-1,\; -2) \cdot \begin{bmatrix} 1/\sqrt{5} \\ 2/\sqrt{5} \end{bmatrix} = \frac{-1 - 4}{\sqrt{5}} = \frac{-5}{\sqrt{5}} = -\sqrt{5} \approx -2.236 \\[8pt]
z_3 &= (0,\; 0) \cdot \begin{bmatrix} 1/\sqrt{5} \\ 2/\sqrt{5} \end{bmatrix} = 0 \\[8pt]
z_4 &= (1,\; 2) \cdot \begin{bmatrix} 1/\sqrt{5} \\ 2/\sqrt{5} \end{bmatrix} = \frac{1 + 4}{\sqrt{5}} = \frac{5}{\sqrt{5}} = \sqrt{5} \approx 2.236 \\[8pt]
z_5 &= (2,\; 4) \cdot \begin{bmatrix} 1/\sqrt{5} \\ 2/\sqrt{5} \end{bmatrix} = \frac{2 + 8}{\sqrt{5}} = \frac{10}{\sqrt{5}} = 2\sqrt{5} \approx 4.472
\end{aligned}$$

**压缩完成！** 原来 5 个样本 × 2 个特征（共 10 个数），现在 5 个样本 × 1 个特征（共 5 个数）。压缩比 50%。

$$\boxed{\mathbf{Z} = \begin{bmatrix} -2\sqrt{5} \\ -\sqrt{5} \\ 0 \\ \sqrt{5} \\ 2\sqrt{5} \end{bmatrix}}$$

**观察：** 降维后的坐标从左到右均匀分布——你丢失了"y 坐标"（垂直于 PC1 的方向），但沿 PC1 方向的相对位置完美保留。

#### 3g. 第 6 步：重建 — 从 1D 回到 2D

重建公式：$\hat{\mathbf{X}} = \mathbf{Z} \mathbf{V}_K^{\mathsf{T}}$，即为每个 $z$ 值重新"拉"回二维空间，方向沿 $\mathbf{v}_1$：

$$\begin{aligned}
\hat{\mathbf{x}}_1 &= z_1 \cdot \mathbf{v}_1^{\mathsf{T}} = (-2\sqrt{5}) \times \begin{bmatrix} 1/\sqrt{5} & 2/\sqrt{5} \end{bmatrix} = \begin{bmatrix} -2 & -4 \end{bmatrix} \\[4pt]
\hat{\mathbf{x}}_2 &= (-\sqrt{5}) \times \begin{bmatrix} 1/\sqrt{5} & 2/\sqrt{5} \end{bmatrix} = \begin{bmatrix} -1 & -2 \end{bmatrix} \\[4pt]
\hat{\mathbf{x}}_3 &= 0 \times \begin{bmatrix} 1/\sqrt{5} & 2/\sqrt{5} \end{bmatrix} = \begin{bmatrix} 0 & 0 \end{bmatrix} \\[4pt]
\hat{\mathbf{x}}_4 &= \sqrt{5} \times \begin{bmatrix} 1/\sqrt{5} & 2/\sqrt{5} \end{bmatrix} = \begin{bmatrix} 1 & 2 \end{bmatrix} \\[4pt]
\hat{\mathbf{x}}_5 &= 2\sqrt{5} \times \begin{bmatrix} 1/\sqrt{5} & 2/\sqrt{5} \end{bmatrix} = \begin{bmatrix} 2 & 4 \end{bmatrix}
\end{aligned}$$

这些是**中心化坐标系**下的重建值。加上均值 $(\bar{x}, \bar{y}) = (3, 6)$ 还原到原始坐标系：

$$\begin{aligned}
(1):\; (-2, -4) + (3, 6) &= (1, 2) \\
(2):\; (-1, -2) + (3, 6) &= (2, 4) \\
(3):\; (0, 0) + (3, 6) &= (3, 6) \\
(4):\; (1, 2) + (3, 6) &= (4, 8) \\
(5):\; (2, 4) + (3, 6) &= (5, 10)
\end{aligned}$$

**每一个重建点都精确等于原始点！** 这是因为 $\lambda_2 = 0$——第二主成分方向上的信息就是零，丢弃它不损失任何东西。

#### 3h. 手算小结：回顾全程

把上面所有的计算压缩成一张流程图：

```
原始数据 (5×2)              中心化 (5×2)           协方差矩阵 (2×2)
  (1,  2)                     (-2, -4)               ┌──────────┐
  (2,  4)         减去         (-1, -2)    计算        │ 2.5   5.0│
  (3,  6)  ────►  均值  ────►  ( 0,  0)  ────►       │          │
  (4,  8)        (3, 6)       ( 1,  2)    C=XᵀX/4    │ 5.0  10.0│
  (5, 10)                     ( 2,  4)               └──────────┘
                                                          │
                                              特征分解    │  det(C-λI)=0
                                                          ↓
  1D 坐标                   投影                   ┌──────────────┐
  -2√5 ≈ -4.472            沿 [1,2]/√5            │ λ₁=12.5  100%│
  -√5  ≈ -2.236            方向压缩                │ λ₂= 0      0%│
    0             ◄────                            └──────────────┘
   √5  ≈  2.236
   2√5 ≈  4.472                                    v₁=[1,2]/√5
                                                   v₂=[-2,1]/√5
```

**你在纸上亲手完成了 PCA 的全过程。** 下次当你调用 `sklearn.decomposition.PCA(n_components=1).fit_transform(X)` 时，你知道 `sklearn` 在背后做了什么。

#### 3h₂. 扩展：给完美共线数据加一点噪声

上面是"完美共线"的理想情况——$\lambda_2 = 0$，降维到 1D 零损失。现实数据永远有噪声。我们把每个 y 坐标加上一个微小的随机扰动：

扰动后的数据（示意）：$(1, 2.1), (2, 3.9), (3, 6.1), (4, 7.8), (5, 10.2)$

这些点不再完美落在直线上了，但**几乎**还在——肉眼依然能看出 $y \approx 2x$ 的趋势。现在 PCA 会怎样？

**中心化：** 均值 = $(3, 6.02)$。中心化坐标大致为 $(-2, -3.92), (-1, -2.12), (0, 0.08), (1, 1.78), (2, 4.18)$。

**协方差矩阵：** 手工计算时你会发现：
- $C_{11}$ 基本不变（≈ 2.5）—— x 方向没动
- $C_{12}$ 基本不变（≈ 5.0）—— 相关性方向不变
- $C_{22}$ **略微增大**（≈ 10.0 → ≈ 10.1）—— y 方向多了噪声方差

**特征值：** $\lambda_1 \approx 12.6$（基本不变），$\lambda_2$ 不再是 0，而是一个小的正数（≈ 0.01-0.1 量级）。

**关键洞察：** PC1 的方向几乎是 $(1, 2)$——和原来几乎一样；$\lambda_2$ 虽然 > 0，但相比 $\lambda_1$ 微不足道。PCA 自动将噪声分离到了低方差的第二成分中。丢弃 PC2 → 丢弃噪声，保留信号。**这就是 PCA 去噪的基本原理。**

#### 3i. Python 验证：让计算机重做一遍

```python
import numpy as np

# ─── 原始数据 ───
X = np.array([[1,  2],
              [2,  4],
              [3,  6],
              [4,  8],
              [5, 10]], dtype=float)

# ─── Step 1: 中心化 ───
mean = X.mean(axis=0)                           # [3., 6.]
X_centered = X - mean
print("均值:", mean)
print("中心化后:\n", X_centered)
print("列和验证:", X_centered.sum(axis=0))       # 应该 ≈ [0, 0]

# ─── Step 2: 协方差矩阵 ───
n = X.shape[0]
C = (X_centered.T @ X_centered) / (n - 1)
print("\n协方差矩阵 C:\n", C)
# 应输出: [[2.5 5.0], [5.0 10.0]]

# ─── Step 3: 特征分解 ───
eigvals, eigvecs = np.linalg.eigh(C)             # eigh: Hermitian矩阵, 升序
idx = np.argsort(eigvals)[::-1]                  # 降序索引
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print("\n特征值 (降序):", eigvals)               # [12.5, 0.]
print("特征向量 (列):\n", eigvecs)
# 应输出: [[0.4472, -0.8944], [0.8944, 0.4472]]  (≈ [1,2]/√5 和 [-2,1]/√5)

# 解释方差比
evr = eigvals / eigvals.sum()
print(f"PC1 解释方差比: {evr[0]:.1%}")           # 100.0%
print(f"PC2 解释方差比: {evr[1]:.1%}")           # 0.0%

# ─── Step 4 & 5: 降维到 1D ───
V1 = eigvecs[:, :1]                              # 只取第一主成分，形状 (2, 1)
Z = X_centered @ V1                              # (5, 2) @ (2, 1) → (5, 1)
print("\n降维后 Z (1D):")
for i, z in enumerate(Z.ravel()):
    print(f"  点{i+1}: z = {z:.4f}  (理论: {np.array([-2*np.sqrt(5), -np.sqrt(5), 0, np.sqrt(5), 2*np.sqrt(5)])[i]:.4f})")

# ─── Step 6: 重建 ───
X_recon = Z @ V1.T + mean                       # 加回均值
print("\n重建 vs 原始 (应完全一致):")
print("原始:\n", X)
print("重建:\n", X_recon)
print("最大重建误差:", np.max(np.abs(X - X_recon)))  # 应 ≈ 0

# ─── 使用 sklearn 验证 ───
from sklearn.decomposition import PCA
pca_sk = PCA(n_components=1)
Z_sk = pca_sk.fit_transform(X)
print(f"\nsklearn 降维结果: {Z_sk.ravel()}")
print(f"sklearn PC1 方向: {pca_sk.components_[0]}")
print(f"sklearn 解释方差比: {pca_sk.explained_variance_ratio_[0]:.1%}")
```

**运行这段代码，你会看到：** 每一个数值都和我们手算的结果一模一样。你不再是被动地"相信书上说的"——你是亲手算过，用代码验证过的。

---

### 4. PCA 的数学本质

#### 4a. 为什么协方差矩阵的特征向量就是"方差最大方向"？

这是一个你应该问但很多人不问的问题。答案全在数学中：

设 $\mathbf{w}$ 是任意一个单位方向向量。将数据投影到 $\mathbf{w}$ 上：投影值 = $\mathbf{X}\mathbf{w}$（中心化后）。投影后数据的方差为：

$$\text{Var}(\mathbf{X}\mathbf{w}) = \frac{1}{n-1}(\mathbf{X}\mathbf{w})^{\mathsf{T}}(\mathbf{X}\mathbf{w}) = \mathbf{w}^{\mathsf{T}}\left(\frac{\mathbf{X}^{\mathsf{T}}\mathbf{X}}{n-1}\right)\mathbf{w} = \mathbf{w}^{\mathsf{T}}\mathbf{C}\mathbf{w}$$

**PCA 的优化问题：** 寻找一个单位向量 $\mathbf{w}$，使得 $\mathbf{w}^{\mathsf{T}}\mathbf{C}\mathbf{w}$ 最大化。

这恰恰是**瑞利商（Rayleigh Quotient）**的形式，而瑞利商的最大值 = 最大特征值 $\lambda_{\max}$，在 $\mathbf{w} = \mathbf{v}_{\max}$（最大特征值对应的特征向量）处取得。

这就是为什么特征向量 → 主成分方向，特征值 → 该方向的方差大小。不是巧合——是线性代数定理的必然结果。

**一步步展开瑞利商的推导（拿纸跟着写）：**

对于中心化数据 $\mathbf{X}$（均值已减），给定单位方向 $\mathbf{w}$（$\|\mathbf{w}\| = 1$），投影值为向量 $\mathbf{z} = \mathbf{X}\mathbf{w}$。投影方差：

$$\text{Var}(\mathbf{z}) = \frac{1}{n-1}\sum_{i=1}^n (z_i - \bar{z})^2$$

因为 $\mathbf{X}$ 已中心化，$\bar{z} = \frac{1}{n}\sum z_i = \frac{1}{n}\sum \mathbf{x}_i \cdot \mathbf{w} = (\frac{1}{n}\sum \mathbf{x}_i) \cdot \mathbf{w} = \mathbf{0} \cdot \mathbf{w} = 0$。所以：

$$\text{Var}(\mathbf{z}) = \frac{1}{n-1} \mathbf{z}^{\mathsf{T}}\mathbf{z} = \frac{1}{n-1} (\mathbf{X}\mathbf{w})^{\mathsf{T}}(\mathbf{X}\mathbf{w}) = \mathbf{w}^{\mathsf{T}}\left(\frac{\mathbf{X}^{\mathsf{T}}\mathbf{X}}{n-1}\right)\mathbf{w} = \mathbf{w}^{\mathsf{T}}\mathbf{C}\mathbf{w}$$

其中 $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^{\mathsf{T}}\mathbf{X}$ 正是协方差矩阵。所以 PCA 的优化问题是：

$$\max_{\mathbf{w}} \; \mathbf{w}^{\mathsf{T}} \mathbf{C} \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^{\mathsf{T}}\mathbf{w} = 1$$

构造拉格朗日函数：$\mathcal{L} = \mathbf{w}^{\mathsf{T}}\mathbf{C}\mathbf{w} - \lambda(\mathbf{w}^{\mathsf{T}}\mathbf{w} - 1)$

求 $\nabla_{\mathbf{w}}\mathcal{L} = 2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0 \implies \boxed{\mathbf{C}\mathbf{w} = \lambda\mathbf{w}}$

**这就是为什么特征向量就是主成分方向。** 无需求助于任何"黑箱直觉"——从优化问题的 KKT 条件一步就能推出来。

此时投影方差 = $\mathbf{w}^{\mathsf{T}}\mathbf{C}\mathbf{w} = \mathbf{w}^{\mathsf{T}}(\lambda\mathbf{w}) = \lambda$。特征值就是该方向的方差大小。

**第二主成分怎么找？** 在 $\mathbf{w}_2 \perp \mathbf{w}_1$ 的约束下重复同样的优化——得到第二大的特征值对应的特征向量。依次类推到第 K 个。

#### 4b. PCA 与 SVD 的等价性

有时候你不想构造 $p \times p$ 的协方差矩阵（例如 $p = 100,000$ 时矩阵太大）。这时可以用 SVD 绕过：

对中心化后的数据矩阵 $\mathbf{X}$（$n \times p$）做截断 SVD：

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{\mathsf{T}}$$

其中：
- $\mathbf{V}$ 的列 = 主成分方向（和协方差分解中得到的特征向量完全一致，仅可能有符号差异）
- $\mathbf{\Sigma}$ 的对角线平方 $/ (n-1)$ = 特征值
- $\mathbf{U} \mathbf{\Sigma}$ = $\mathbf{Z}$（降维后的坐标）

**为什么 sklearn 内部用 SVD？** 因为 SVD 的数值稳定性远优于协方差矩阵分解。当某些特征值接近零时，协方差矩阵求逆可能爆炸，而 SVD 能干净地处理零（或接近零）的奇异值。

```python
# 验证 PCA = SVD
X_c = X - X.mean(axis=0)
n = X_c.shape[0]

# SVD 路线
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
components_svd = Vt.T                             # V 的列 = 主成分
eigvals_svd = S**2 / (n - 1)                       # 特征值

# 协方差路线
C = np.cov(X_c, rowvar=False)
eigvals_cov, eigvecs_cov = np.linalg.eigh(C)
idx_cov = np.argsort(eigvals_cov)[::-1]

print("SVD 特征值:", eigvals_svd)
print("Cov 特征值:", eigvals_cov[idx_cov])
print(f"最大差异: {np.max(np.abs(eigvals_svd - eigvals_cov[idx_cov])):.2e}")
# 符号可能不同，但方向一致
for k in range(min(2, S.size)):
    dot = np.abs(np.dot(components_svd[:, k], eigvecs_cov[:, idx_cov[k]]))
    print(f"PC{k+1}: |svd·cov 内积| = {dot:.6f}")
```

#### 4c. PCA 的另一个视角：最小化重建误差

PCA 有两个等价的定义：

| 视角 | 定义 | 数学表达 |
|------|------|---------|
| **方差最大化** | 找到投影方向，使投影后方差最大 | $\max_{\mathbf{w}} \text{Var}(\mathbf{X}\mathbf{w})$ |
| **重建误差最小化** | 找到 K 维超平面，使投影距离平方和最小 | $\min \|\mathbf{X} - \hat{\mathbf{X}}\|_F^2$ |

这两个定义被证明是等价的——就像"把箭射得最远的方向"和"把靶心打得最近的方向"在理想条件下指向同一个方向。前者告诉 PCA "要保留什么"，后者告诉 PCA "要舍弃什么"——一体两面。

---

### 5. 选择主成分数量 K

PCA 不会自动告诉你保留几个主成分。你需要自己决策。

#### 5a. 碎石图（Scree Plot）

把特征值从大到小画成柱状图，观察"肘部"——斜率从陡峭变为平缓的拐点。

#### 5b. 选 K 的四种策略

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| **肘部法** | 看累计方差曲线变平的位置 | 快速判断，有一定主观性 |
| **95% 阈值** | 选第一个累计方差比 ≥ 95% 的 K | 需要量化保证信息留存 |
| **Kaiser 准则** | 舍去特征值 < 1 的成分（仅用于标准化后数据） | 每个成分至少解释一个原始特征的方差 |
| **下游任务驱动** | 选使下游模型表现最好的 K | 最实用：效果说话 |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_bc, y_bc = data.data, data.target
X_scaled = StandardScaler().fit_transform(X_bc)

pca_full = PCA().fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 左：碎石图
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, color='steelblue', alpha=0.6)
axes[0].plot(range(1, len(cumsum) + 1), cumsum, 'r-o', markersize=4, label='累计方差比')
axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95% 阈值')
axes[0].set_xlabel('主成分序号'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('碎石图 (Scree Plot)'); axes[0].legend()

# 右：前 10 个成分的柱状图
k_95 = np.argmax(cumsum >= 0.95) + 1
for i, evr in enumerate(pca_full.explained_variance_ratio_[:10]):
    color = 'coral' if i < k_95 else 'lightgray'
    axes[1].bar(i + 1, evr, color=color, alpha=0.8)
    axes[1].text(i + 1, evr + 0.005, f'{evr:.3f}\n(cum:{cumsum[i]:.3f})',
                 ha='center', va='bottom', fontsize=6.5)
axes[1].set_xlabel('主成分序号'); axes[1].set_ylabel('解释方差比')
axes[1].set_title(f'前 10 个主成分 (K={k_95} 达到 95%)')

plt.tight_layout(); plt.show()
print(f"原始特征数: {X_bc.shape[1]}, 保留 95% 方差需 {k_95} 个主成分")
```

**"95% 方差"不是物理定律——** 如果你的下游任务对噪声极其敏感（如医疗诊断），你可能希望保留 99% 来降低信息丢失；如果是为了去噪（主动丢弃高频噪声），80% 就够了。K 是**工程决策**，不是数学结论。

#### 5c. 应用：用 PCA 估计数据的"内在维度"

PCA 不仅用于降维——它还能回答一个更根本的问题：**我的数据到底有几维？**

一个有 100 个原始特征的数据集，如果前 3 个主成分就累积了 95% 的方差，那说明原始数据真正"有用"的维度只有 3 个——其余 97 个维度混合了冗余 + 噪声。"特征的数量"不等于"信息的维度"。

这在实践中有一个惊艳的应用——**异常检测**：如果一个新样本在 PCA 低维子空间上的重建误差非常大（即它在前几个主成分中由低维坐标重建回来和原始坐标差异很大），说明这个样本不在"正常数据"分布的主流方向上，很可能就是一个异常点。

```python
# PCA 异常检测示意：重建误差大 → 可能是异常
def pca_anomaly_score(X_test, pca_model, n_components):
    Z = pca_model.transform(X_test)[:, :n_components]
    X_recon = Z @ pca_model.components_[:n_components, :] + pca_model.mean_
    return np.sum((X_test - X_recon)**2, axis=1)

# 正常数据 → 小误差；异常数据偏离主方向 → 大误差
```

---

### 6. 从零实现 PCA

用我们手算的公式，写出一个完整的 PCA 类。阅读每一行，确认它和我们纸上写的步骤一一对应。

```python
class PCA_FromScratch:
    """从零实现 PCA — 完全匹配手算的六步流程"""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> 'PCA_FromScratch':
        # Step 1: 中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Step 2: 协方差矩阵
        cov = np.cov(X_centered, rowvar=False)

        # Step 3: 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 4: 取前 K 个
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Step 5: 投影
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        # Step 6: 重建
        return Z @ self.components_.T + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ─── 在手算数据上验证我们的实现 ───
pca_ours = PCA_FromScratch(n_components=1)
Z_ours = pca_ours.fit_transform(X)
print("手写 PCA 降维:", Z_ours.ravel())
print("均值:", pca_ours.mean_)
print("PC1:", pca_ours.components_.ravel())
```

---

### 7. t-SNE：专为可视化而生

#### 7a. 直觉理解

PCA 问："谁离谁最远？"——它关心的是**全局结构**。

t-SNE 问："谁是谁的邻居？"——它关心的是**局部结构**。

t-SNE 的核心类比：社交网络嵌入。两个人如果不认识，距离多远无所谓；但如果 A 是 B 的邻居，B 是 C 的邻居，降维后这个邻居关系必须保留。算法把高维距离转化为概率——两个点距离越近，它们互为邻居的概率越高——然后在低维空间里不断移动点，直到低维的邻居概率分布尽可能匹配高维的分布。

整个过程很像物理模拟：同类点之间互相吸引（引力），不同类点之间互相排斥（斥力），经过上百轮迭代后达到平衡状态。

#### 7b. 关键参数 perplexity

perplexity（困惑度）大致等于"每个点预期的邻居数量"。

| perplexity | 行为 | 问题 |
|-----------|------|------|
| 太小（2-5） | 只看紧邻，关注微观结构 | 丢失全局聚类结构 |
| 适中（15-50） | 平衡局部与全局 | 推荐范围 |
| 太大（>100） | 趋于全局 PCA | 丢失局部细节 |

#### 7c. 最重要的警告

> **t-SNE 的坐标轴没有任何含义。** PC1 有"数据方差最大方向"这个明确的解释；t-SNE 的横轴和纵轴是纯粹"做"出来的可视化坐标，不能用于特征提取或下游建模。**t-SNE 只用于探索性可视化，绝不可放入机器学习流水线。**

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# PCA vs t-SNE 对比
X_pca = PCA(n_components=2, random_state=42).fit_transform(X_digits)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_digits)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
scatter_kw = dict(c=y_digits, cmap='tab10', s=12, alpha=0.7)

for ax, X_2d, title in zip(axes,
                           [X_pca, X_tsne],
                           ['PCA (线性，全局结构)', 't-SNE (非线性，局部结构)']):
    ax.scatter(X_2d[:, 0], X_2d[:, 1], **scatter_kw)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

plt.colorbar(axes[1].collections[0], ax=axes, label='数字类别 (0-9)')
plt.tight_layout(); plt.show()

# 不同 perplexity 对比
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, perp in zip(axes.flat, [5, 15, 30, 80]):
    X_t = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X_digits)
    ax.scatter(X_t[:, 0], X_t[:, 1], c=y_digits, cmap='tab10', s=10, alpha=0.7)
    ax.set_title(f'perplexity = {perp}')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('perplexity 参数对 t-SNE 的影响', fontsize=14)
plt.tight_layout(); plt.show()
```

**应用连接：** 在探索性数据分析中，t-SNE 是发现标签错误的利器。如果一个点在 t-SNE 图中被"别人"包围（周围所有邻居都是另一类），这个样本很可能是标注错误的，值得手工检查。

---

### 8. 其他降维方法简介

| 方法 | 核心思想 | 与 PCA 的关系 | 适用场景 |
|------|---------|-------------|---------|
| **TruncatedSVD** | 截断奇异值分解 | 不需要中心化，能处理稀疏矩阵 | TF-IDF 文本矩阵、推荐系统评分矩阵 |
| **LDA** | 最大化类间 / 最小化类内 | **有监督**，利用标签找最佳投影 | 分类前的有监督降维（最多 C-1 维） |
| **Autoencoder** | 神经网络自编码 | 可学习非线性映射 | 图像去噪、异常检测、生成模型 |

**LDA 的直觉：** PCA 找方差最大的方向——但这个方向可能是噪声最大的方向。LDA 说：我有标签在手，应该找**最能区分不同类**的方向。想象一班学生：PCA 可能按身高排序（方差大），LDA 按考试成绩排序（区分性强）。

```python
# TruncatedSVD 演示：文本矩阵降维
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "机器学习 数据 算法 模型 训练",
    "深度学习 神经网络 卷积 循环",
    "股票 投资 金融市场 交易",
    "房价 地段 面积 贷款 首付",
    "机器学习 深度学习 神经网络 模型 训练",
    "股票 基金 理财 投资 收益"
]
tfidf = TfidfVectorizer().fit_transform(docs)
svd = TruncatedSVD(n_components=2, random_state=42)
docs_2d = svd.fit_transform(tfidf)

print(f"TF-IDF 形状: {tfidf.shape} (稀疏矩阵)")
plt.figure(figsize=(7, 5))
plt.scatter(docs_2d[:, 0], docs_2d[:, 1], s=100, alpha=0.7, edgecolors='k')
for i, (x, y) in enumerate(docs_2d):
    plt.annotate(f'文档{i}', (x, y), xytext=(5, 5), textcoords='offset points')
plt.xlabel('SVD 分量 1'); plt.ylabel('SVD 分量 2')
plt.title('TruncatedSVD：ML/AI 文档 vs 金融文档 自动分离')
plt.grid(True, alpha=0.3); plt.show()
```

---

### 9. sklearn 实战：PCA 管道

核心问题：**降维一定会损害模型性能吗？** 答案取决于数据。冗余和噪声多 → PCA 帮助聚焦信号，性能可能不降反升。每个维度都独立携带信息 → 降维就是净损失。

```python
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ─── 加载与准备 ───
data = load_breast_cancer()
X, y = data.data, data.target
print(f"样本: {X.shape[0]}, 特征: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ─── 基线：不用 PCA ───
lr_base = LogisticRegression(max_iter=5000, random_state=42)
lr_base.fit(X_train_s, y_train)
base_acc = accuracy_score(y_test, lr_base.predict(X_test_s))
print(f"\n基线 (30 维) 测试准确率: {base_acc:.4f}")

# ─── 不同 K 的对比 ───
ks = list(range(1, X.shape[1] + 1, 2))
test_accs, cv_scores = [], []

for k in ks:
    pca = PCA(n_components=k, random_state=42)
    X_tr_pca = pca.fit_transform(X_train_s)
    X_te_pca = pca.transform(X_test_s)

    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_tr_pca, y_train)
    test_accs.append(accuracy_score(y_test, lr.predict(X_te_pca)))
    cv_scores.append(cross_val_score(lr, X_tr_pca, y_train, cv=5).mean())

# ─── 可视化 ───
pca_full = PCA(random_state=42).fit(X_train_s)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
k_95 = np.argmax(cumsum >= 0.95) + 1

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].bar(range(1, len(cumsum)+1), pca_full.explained_variance_ratio_,
           color='steelblue', alpha=0.6, label='单分量')
axes[0].plot(range(1, len(cumsum)+1), cumsum, 'r-o', ms=4, label='累计')
axes[0].axhline(y=0.95, color='gray', ls='--', alpha=0.7)
axes[0].axvline(x=k_95, color='green', ls='--', alpha=0.7, label=f'K={k_95}')
axes[0].set_xlabel('主成分数'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('解释方差比分布'); axes[0].legend(fontsize=8)

axes[1].plot(ks, test_accs, 'b-o', ms=5, label='测试准确率')
axes[1].plot(ks, cv_scores, 'r-s', ms=5, label='CV 准确率')
axes[1].axhline(y=base_acc, color='gray', ls='--', alpha=0.7, label=f'基线 ({base_acc:.3f})')
axes[1].set_xlabel('主成分数 K'); axes[1].set_ylabel('准确率')
axes[1].set_title('降维对模型性能的影响'); axes[1].legend(fontsize=8)

X_tr_2d = pca_full.transform(X_train_s)[:, :2]
axes[2].scatter(X_tr_2d[:, 0], X_tr_2d[:, 1],
                c=y_train, cmap='coolwarm', s=20, alpha=0.6)
axes[2].set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%})')
axes[2].set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.1%})')
axes[2].set_title('PC1-PC2 空间中的数据分布')

plt.tight_layout(); plt.show()

best_k = ks[np.argmax(cv_scores)]
print(f"\n最优 K = {best_k} (CV 最高)")
print(f"K={k_95} 保留 95% 方差")
```

---

### 10. 常见误区

| 误区 | 真相 |
|------|------|
| "PCA 自动去除了不重要特征" | PCA 去除的是**方差小**的方向，不等于**对任务不重要**的方向。一个方差极小的基因标记可能正是分类的关键信号。 |
| "t-SNE 可以替代 PCA 做降维" | t-SNE 没有 `.transform()` 方法能处理新数据，坐标轴没有意义，不能放进建模流水线。 |
| "PCA 前不需要标准化" | 如果特征量纲不同（身高 cm vs 收入 万元），PCA 会被方差大的特征"劫持"。**永远先 StandardScaler。** |
| "主成分越多越好" | 超过一定数量后新增成分主要携带噪声，反而降低模型泛化能力。 |
| "重建误差为零说明降维无损" | 当你 K = 原始维度 p 时重建误差为零——但这等于没降维。降维必然有损失，关键是损失的是信号还是噪声。 |
| "降维后特征重新正交 → 没有多重共线性了" | 主成分虽然相互正交，但它们是从原始特征线性组合而来的。如果原始数据共线性严重，PC1 本身就已经高度混合了那些共线特征——可解释性丧失是 PCA 的隐性代价。 |
| "PCA 后可以直接看散点图判断分类难易" | PCA 只看方差不看标签。PC1-PC2 散点图上一团糟不意味着分类不可行——可能 PC3 或 PC4 就刚好把两个类分开了，只是你没画出来。**永远不要仅凭 PCA 可视化否定一个分类任务。** |

**应用连接 — 生产部署：** PCA 是一个**有状态变换**。部署时你必须连同拟合好的 scaler 和 pca 对象一起保存（`joblib.dump`），推理时用**完全相同的**变换参数（`joblib.load` → `scaler.transform() → pca.transform()`）。如果手算时用的是训练集的均值，推理时必须沿用训练集的均值——这是所有人入门 PCA 时最容易漏掉的工程细节。

---

### 11. 思考题

#### 11a. 题目

**基础题（1-5）**

1. **手算 PCA：** 给定 3 个二维点 $(0, 0), (2, 1), (4, 2)$，请按照本章 3a-3h 的流程，完整地手算出：(a) 均值 (b) 中心化后的坐标 (c) 协方差矩阵 (d) 特征值和特征向量 (e) 降到 1D 的坐标 (f) 重建回 2D 的坐标。你注意到了什么？

2. **方差不变量：** 证明 PCA 变换前后**总方差不变**——即原始数据的各特征方差之和 = 所有特征值之和。用手算数据验证：原始 $(x, y)$ 两列的方差之和是多少？特征值之和是多少？

3. **特征值的含义：** 如果对某个数据集做 PCA，发现 $\lambda_1 = 50, \lambda_2 = 30, \lambda_3 = 20$（总共 3 个特征），请问：降到 2 维会丢失多少百分比的信息？如果降到 1 维呢？

4. **投影方向的符号：** 本章手算中 PC1 是 $(1, 2)$，如果特征向量取 $(-1, -2)$（反方向），降维后的 z 值和重建结果会怎样变化？这对 PCA 的最终用途（可视化、下游建模）有影响吗？

5. **标准化与 PCA：** 如果不先做标准化（StandardScaler），直接对包含"身高（cm, 均值 170, 标准差 20）"和"收入（万元, 均值 30, 标准差 15）"两个特征的数据做 PCA，第一主成分会大致指向哪个方向？为什么标准化能修正这个问题？

**进阶题（6-8）**

6. **PCA 与去噪：** 对本章手算的 5 个完美共线点添加随机噪声——给每个 y 坐标加上 $\mathcal{N}(0, 0.3^2)$ 的扰动。重新手算协方差矩阵（提示：扰动后数据不再完美共线，$\lambda_2 > 0$）。原 $\lambda_2 = 0$ 变成了什么？PC1 方向变化了多少？这说明 PCA 如何自动"忽略"小噪声？

7. **SVD 与 PCA 的数值差异：** 写出代码验证：当特征数 $p = 1$ 时（只有一列数据），`np.linalg.eigh` 求协方差矩阵和 `np.linalg.svd` 直接分解数据矩阵，得到的特征值和主成分是否一致？推广到 $p > n$（特征数 > 样本数）的情况，为什么 SVD 更稳定？

8. **t-SNE 的随机性：** 对同样的数据，连续两次运行 `TSNE(random_state=None)` 会得到不同的低维表示。为什么？这对使用 t-SNE 做数据探索的影响是什么？有哪些策略可以应对这种随机性？

**综合题（9-10）**

9. **从投影方差推导 PCA：** 给定中心化后的数据矩阵 $\mathbf{X}$（$n \times p$）和一个单位方向向量 $\mathbf{w}$（$p \times 1$），投影后的数据为 $\mathbf{z} = \mathbf{X}\mathbf{w}$。写出 $\text{Var}(\mathbf{z})$ 的表达式，利用拉格朗日乘子法在约束 $\mathbf{w}^{\mathsf{T}}\mathbf{w} = 1$ 下最大化方差，证明最优的 $\mathbf{w}$ 满足 $\mathbf{C}\mathbf{w} = \lambda\mathbf{w}$——即 $\mathbf{w}$ 是协方差矩阵 $\mathbf{C}$ 的特征向量。

10. **实战分析：** 用 `load_digits` 数据集（64 维手写数字特征），完成以下实验：
    - (a) 用 PCA 降到 2 维 → 画散点图，观察不同数字是否可分
    - (b) 用 PCA 保留 95% 方差，记录所需的主成分数 K
    - (c) 在原始 64 维和降维后的 K 维上分别训练 LogisticRegression，对比 5-fold CV 准确率
    - (d) 用 t-SNE (perplexity=30) 降到 2 维画图，和 (a) 的 PCA 结果对比——哪一种对数字 4 和 9 的区分更清晰？为什么？

---

#### 11b. 解答

<details>
<summary><b>点击展开完整解答</b></summary>

**第 1 题：**

(a) 均值：$\bar{x} = (0+2+4)/3 = 2$，$\bar{y} = (0+1+2)/3 = 1$。

(b) 中心化：$(0,0)-(2,1)=(-2,-1)$，$(2,1)-(2,1)=(0,0)$，$(4,2)-(2,1)=(2,1)$。

(c) 协方差：

$$\sum x_i^2 = 4+0+4 = 8, \; C_{11} = 8/2 = 4.0$$
$$\sum y_i^2 = 1+0+1 = 2, \; C_{22} = 2/2 = 1.0$$
$$\sum x_i y_i = 2+0+2 = 4, \; C_{12} = 4/2 = 2.0$$

$$\mathbf{C} = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$$

(d) $\det(\mathbf{C} - \lambda\mathbf{I}) = (4-\lambda)(1-\lambda) - 4 = \lambda^2 - 5\lambda = 0$

$\lambda_1 = 5, \lambda_2 = 0$。与手算章节的主例相同——因为这三个点也在直线 $y = 0.5x$ 上（完全共线）。

特征向量：$\mathbf{v}_1 = [2, 1]/\sqrt{5}$（沿 $y=0.5x$ 方向），$\mathbf{v}_2 = [-1, 2]/\sqrt{5}$。

(e) 投影到 1D：

$z_1 = (-2,-1) \cdot [2,1]/\sqrt{5} = (-4-1)/\sqrt{5} = -\sqrt{5}$
$z_2 = 0$
$z_3 = (2,1) \cdot [2,1]/\sqrt{5} = (4+1)/\sqrt{5} = \sqrt{5}$

(f) 重建（加回均值）：

$z_1$: $(-\sqrt{5})(2,1)/\sqrt{5} + (2,1) = (-2,-1) + (2,1) = (0,0)$ ✓
$z_2$: $(0,0) + (2,1) = (2,1)$ ✓
$z_3$: $(\sqrt{5})(2,1)/\sqrt{5} + (2,1) = (2,1)+(2,1) = (4,2)$ ✓

**你注意到什么？** 这三个点也完美共线（在 $y = 0.5x$ 上），所以 $\lambda_2 = 0$，PC1 捕获 100% 方差，重建完全精确。只要数据落在一条直线上，PCA 总能完美压缩为 1D。

---

**第 2 题：**

原始方差：$\text{Var}(x) = C_{11} = 2.5$，$\text{Var}(y) = C_{22} = 10.0$。总和 = $2.5 + 10.0 = 12.5$。

特征值之和：$\lambda_1 + \lambda_2 = 12.5 + 0 = 12.5$。**相等。**

证明：$\sum \text{Var}(\text{特征}_j) = \text{trace}(\mathbf{C}) = \sum \lambda_i$。这是线性代数的基本定理——矩阵的迹 = 特征值之和。PCA 不消灭方差，只是**重新分配**它：把方差从多个维度"汇聚"到少数几个方向上。

---

**第 3 题：**

总方差 = $50 + 30 + 20 = 100$。

降到 2 维：保留 $50 + 30 = 80$，丢失 $20/100 = 20\%$。

降到 1 维：保留 $50$，丢失 $50/100 = 50\%$。

换句话说：PC1 解释了 50% 的方差，PC2 解释了 30%，PC3 解释了 20%。保留前两个意味着保留了 80% 的信息。

---

**第 4 题：**

如果取 $\mathbf{v}_1 = (-1, -2)$（反方向），投影结果：

$z_1 = (-2,-4) \cdot (-1,-2) = 2+8 = 10$（原为 -10）
$z_2 = 5$（原为 -5），$z_3 = 0$，$z_4 = -5$（原为 +5），$z_5 = -10$（原为 +10）。

所有 $z$ 值**符号反转**，绝对值不变。重建时：
$z_1' \cdot \mathbf{v}_1' + \text{mean} = 10 \cdot (-1,-2) + (3,6) = (-10,-20) + (3,6) = (-7,-14)$

不对……让我重新算。$\mathbf{v}_1' = (-1, -2)$，$z$ 也变了，重建 = $z \times (\mathbf{v}_1' / \sqrt{5})。$

$z_1 = (-2,-4)·(-1,-2) = 2+8 = 10$（原来是 -10）
重建（中心化）：$(10)(-1,-2)/\sqrt{5} = (-10/\sqrt{5}, -20/\sqrt{5})$ 

不对——如果取 $\mathbf{v}_1' = (-1, -2)/\sqrt{5}$ (已归一化)：
$z_1 = (-2,-4)·(-1,-2)/\sqrt{5} = 10/\sqrt{5} = 2\sqrt{5}$
重建 = $2\sqrt{5} \cdot (-1,-2)/\sqrt{5} = (-2,-4)$
加均值 $(3,6) = (1,2)$ — **完全一致。**

关键在于：特征向量符号翻转 × 投影值符号同步翻转 = 重建结果不变（负负得正）。对可视化来说，整个散点图沿原点翻转；对下游建模来说，**完全没有影响**——因为模型只关心相对位置。

---

**第 5 题：**

未标准化时，PCA 会被量纲大的特征"劫持"。身高方差 ≈ 20² = 400，收入方差 ≈ 15² = 225。协方差矩阵中身高对角元素是 $400/(n-1)$，收入是 $225/(n-1)$。第一主成分会严重偏向身高方向——因为它"看起来"方差更大。

标准化后每个特征都是均值为 0、方差为 1 的分布——所有特征站在同一起跑线上，PCA 才真正寻找**相关性结构**中的主方向，而非因量纲差异而被误导。

**数学解释：** 协方差矩阵的元素 $\text{Cov}(x_i, x_j)$ 依赖于原始特征的尺度——把身高从厘米换成米，$\text{Cov}$ 会缩小 10⁴ 倍。而相关矩阵（标准化后的协方差矩阵）的元素 $\rho_{ij}$ 是无量纲的。PCA 对量纲敏感 → 必须标准化。

---

**第 6 题：**

原始数据（完全共线）：$y = 2x$，协方差 = $\begin{bmatrix} 2.5 & 5.0 \\ 5.0 & 10.0 \end{bmatrix}$，$\lambda_2 = 0$。

加噪声 $\mathcal{N}(0, 0.3^2)$ 后的理论变化（以单次模拟为例）：
- $y$ 坐标的微小变化导致 $\sum y_i^2$ 略增 → $C_{22}$ 略增
- $\sum x_i y_i$ 也略变 → $C_{12}$ 略变
- 结果 $\lambda_2$ 从一个"纯 0"变成一个小的正数（≈ 噪声方差规模）
- PC1 方向从精确的 $(1,2)$ 偏移一点

**PCA 的"自动去噪"机制：** 数据的主要结构（$y = 2x$）被分配到大特征值 $\lambda_1$；噪声被分配到小特征值 $\lambda_2$。当你丢弃第二主成分（只保留 PC1），你恰好丢弃了这些噪声——而保留了数据的主要信号方向。这就是 PCA 去噪的原理。

```python
# 实验验证
np.random.seed(42)
X_noisy = X.copy()
X_noisy[:, 1] += np.random.randn(5) * 0.3
print("加噪后:\n", X_noisy)

pca_n = PCA_FromScratch(n_components=2).fit(X_noisy)
print(f"特征值: {pca_n.explained_variance_}")
print(f"PC1 比例: {pca_n.explained_variance_ratio_[0]:.3f}")
print(f"PC1 方向: {pca_n.components_[:, 0]}")
print("(接近 [0.447, 0.894] = [1,2]/√5)")
```

---

**第 7 题：**

当 $p=1$ 时（单列数据），协方差矩阵是 $1 \times 1$ 的标量（= 方差），操作退化为按比例缩放。`eigh` 和 `svd` 结果完全一致。

当 $p > n$（特征数 > 样本数）时：
- 协方差路线需要构造 $p \times p$ 的矩阵 → 内存爆炸 + 求逆不稳定
- SVD 只需要分解 $n \times p$ 的数据矩阵 → 更经济
- 协方差矩阵 $\mathbf{X}^{\mathsf{T}}\mathbf{X}$ 的秩 ≤ $\min(n, p) = n < p$ → 必然奇异 → 特征值中包含大量"接近 0 的噪声"
- SVD 天然处理低秩矩阵，数值更稳定

这就是为什么 `sklearn.decomposition.PCA` 默认 `svd_solver='auto'`——自动选择最适合数据形状的 SVD 实现。

---

**第 8 题：**

t-SNE 的优化过程从一个随机初始化开始，使用随机梯度下降。不同的随机种子 → 不同的初始化 → 收敛到不同的局部极小值 → 每次结果不同。

**影响：** 你不能直接对比两次 t-SNE 结果的坐标（比如"A 点离 B 点更近"未必在两次运行中都成立）。只有**相对聚类结构**是稳定的——同一类点趋向于聚在一起，但聚簇的形状、旋转、间距都会变化。

**应对策略：**
1. 固定 `random_state` 获得可复现的结果（便于论文、报告）
2. 多次运行取平均——观察哪些聚类模式是稳定的
3. 配合 PCA 结果使用——PCA 的位置关系是确定的，用于锚定理解
4. 不要过度解读聚类间的距离——t-SNE 只保留邻居关系，类间距离没有定量含义

---

**第 9 题：**

**推导：** 目标是最大化投影方差：

$$\max_{\mathbf{w}} \text{Var}(\mathbf{X}\mathbf{w}) = \max_{\mathbf{w}} \frac{1}{n-1} \mathbf{w}^{\mathsf{T}} \mathbf{X}^{\mathsf{T}} \mathbf{X} \mathbf{w} = \max_{\mathbf{w}} \mathbf{w}^{\mathsf{T}} \mathbf{C} \mathbf{w}$$

约束条件：$\mathbf{w}^{\mathsf{T}}\mathbf{w} = 1$（单位向量）。

构造拉格朗日函数：$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^{\mathsf{T}} \mathbf{C} \mathbf{w} - \lambda(\mathbf{w}^{\mathsf{T}}\mathbf{w} - 1)$

对 $\mathbf{w}$ 求梯度（利用 $\frac{\partial}{\partial\mathbf{w}} \mathbf{w}^{\mathsf{T}}\mathbf{C}\mathbf{w} = 2\mathbf{C}\mathbf{w}$）：

$$\nabla_{\mathbf{w}} \mathcal{L} = 2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0 \implies \mathbf{C}\mathbf{w} = \lambda\mathbf{w}$$

**证毕。** 最优 $\mathbf{w}$ 就是协方差矩阵的特征向量，对应的 $\lambda$ 就是该方向上的投影方差。

---

**第 10 题：**

```python
from sklearn.datasets import load_digits

digits = load_digits()
X_d, y_d = digits.data, digits.target

# (a) PCA 2D
pca2 = PCA(n_components=2, random_state=42).fit(X_d)
X_pca2 = pca2.transform(X_d)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(X_pca2[:, 0], X_pca2[:, 1], c=y_d, cmap='tab10', s=8)
axes[0, 0].set_title('(a) PCA 2D — 数字 4 和 9 有大量重叠')
axes[0, 0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
axes[0, 0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')

# (b) 95% 方差
pca_full = PCA(random_state=42).fit(X_d)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
K_95 = np.argmax(cumsum >= 0.95) + 1
axes[0, 1].plot(range(1, len(cumsum)+1), cumsum, 'b-o', ms=3)
axes[0, 1].axhline(y=0.95, color='r', ls='--')
axes[0, 1].axvline(x=K_95, color='g', ls='--', label=f'K={K_95}')
axes[0, 1].set_xlabel('主成分数'); axes[0, 1].set_ylabel('累计方差比')
axes[0, 1].set_title(f'(b) 保留 95% 方差需 K={K_95}')
axes[0, 1].legend()

# (c) 模型对比
pca_k = PCA(n_components=K_95, random_state=42)
X_pca_k = pca_k.fit_transform(X_d)

lr_full = LogisticRegression(max_iter=5000)
cv_full = cross_val_score(lr_full, X_d, y_d, cv=5).mean()

lr_pca = LogisticRegression(max_iter=5000)
cv_pca = cross_val_score(lr_pca, X_pca_k, y_d, cv=5).mean()

# (d) t-SNE
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_d)
axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_d, cmap='tab10', s=8)
axes[1, 0].set_title('(d) t-SNE — 数字 4 和 9 分离更清晰')
axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

# 文字总结
axes[1, 1].axis('off')
summary = (
    f"结果总结:\n\n"
    f"95% 方差: K = {K_95} (原 {X_d.shape[1]} 维)\n"
    f"压缩比: {K_95}/{X_d.shape[1]} = {K_95/X_d.shape[1]:.1%}\n\n"
    f"CV 准确率:\n"
    f"  64 维 (原始):  {cv_full:.4f}\n"
    f"  {K_95} 维 (PCA):  {cv_pca:.4f}\n\n"
    f"t-SNE vs PCA:\n"
    f"  t-SNE 对 4/7/9 的区分更好\n"
    f"  因为 t-SNE 保留局部邻域结构\n"
    f"  而 PCA 只关注全局方差方向"
)
axes[1, 1].text(0, 0.5, summary, transform=axes[1, 1].transAxes,
                fontsize=11, va='center', family='monospace')

plt.tight_layout(); plt.show()
print(f"64 维 CV: {cv_full:.4f}, {K_95} 维 PCA CV: {cv_pca:.4f}")
```

**(d) 分析：** t-SNE 对数字 4 和 9 的区分更清晰，因为这两个数字的**全局**形状相似（顶部都有圈），PCA 的线性投影无法区分它们——它们的像素亮度在全局方差方向上差不多。但 t-SNE 关注**局部**结构：4 的右下角和 9 的右下角在像素空间里的邻域模式不同，t-SNE 捕捉到了这个细分差异，将它们分成了更清晰的簇。

</details>

---

### 12. 本章小结

你从一个"100 列 Excel 无法可视化"的困境出发，完整走过了降维的学习路径：

```
维度灾难（100 维 → 看不见形状）
    ↓
PCA 的直觉（蚂蚁爬棍子 → 找方差最大的方向）
    ↓
手算（5 个点 × 6 步，从减法到特征分解到投影 → 还原，每步写下来）
    ↓
Python 从零实现（PCA_FromScratch：fit → transform → inverse_transform）
    ↓
数学本质（瑞利商 → 协方差矩阵 → 特征值 = 方差；PCA = SVD 等价性）
    ↓
选择 K（碎石图 / 95% 阈值 / 下游任务驱动）
    ↓
理论与实践的分岔路（t-SNE 可视化 ≠ PCA 降维；LDA / TruncatedSVD / Autoencoder）
    ↓
sklearn 实战（Pipeline + cross_val_score → 验证降维是否真的帮助了模型）
    ↓
10 道思考题（从手算到拉格朗日推导 → 从噪声实验到 t-SNE 随机性）
```

**PCA 是你进入"高维思维"的第一扇门。** 它强迫你重新思考什么是"信息"——信息不在单个特征里，而在特征之间的**相关性**里。扔掉 90 个维度保留 5 个，不是因为那 90 个"没用"，而是因为它们说的和那 5 个"是一回事"。你从"每个数都重要"的思维跳跃到了"结构性信息胜于个体数据"的思维。

**如果整个教程你只记住一句话：** 特征值大小 = 该方向包含的信息量。扔掉小特征值的方向 = 去噪。保留大特征值的方向 = 压缩。PCA 的核心就是对这个简单思想的数学实现。

**理解了 PCA，你就理解了为什么数据可以压缩、为什么深度学习需要 Embedding 层、为什么 Transformer 要做 Attention 矩阵的低秩近似。** 它们都从同一个洞见出发：数据的真实维度远低于表面维度。

#### 关键公式速查

| 公式 | 用途 |
|------|------|
| $\mathbf{C} = \frac{1}{n-1} \mathbf{X}^{\mathsf{T}}\mathbf{X}$ | 协方差矩阵（中心化后） |
| $\mathbf{C}\mathbf{v}_i = \lambda_i\mathbf{v}_i$ | 特征分解 → 主成分方向 |
| $\mathbf{Z} = \mathbf{X} \mathbf{V}_K$ | 降维：$n \times p \to n \times K$ |
| $\hat{\mathbf{X}} = \mathbf{Z} \mathbf{V}_K^{\mathsf{T}}$ | 重建（需加回均值） |
| $\text{EV}_i = \frac{\lambda_i}{\sum_j \lambda_j}$ | 解释方差比 |
| $\|\mathbf{X} - \hat{\mathbf{X}}\|_F^2 = \sum_{i=K+1}^p \lambda_i$ | 重建误差 = 被丢弃特征值之和 |
| $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\mathsf{T}}$ | PCA 的 SVD 等价形式 ($\mathbf{V}$ 的列 = 主成分) |

#### 从 PCA 出发你能去哪

- **去噪 Autoencoder** ($\rightarrow$ [神经网络](./15-neural-networks-intro.md))：PCA 的线性投影 → Encoder-Decoder 的非线性映射。瓶颈层 = 降维后的代码。
- **LDA** ($\rightarrow$ [SVM](./08-svm.md) 附近)：把标签信息注入降维过程，寻找"最有判别力"而非"最大方差"的方向。
- **NMF (非负矩阵分解)**：当数据全是非负的（如图像像素、文本计数），PCA 的负数主成分毫无意义 → NMF 保证主成分也是非负的，每个成分可解释为"这部分"。
- **Kernel PCA**：数据在原始空间不可线性分离 → 通过核函数映射到高维 → 在高维做 PCA → 实现非线性降维。
- **UMAP**：t-SNE 的进化版——更快、更好地保留全局结构，同时维持局部邻域。正逐渐取代 t-SNE 成为默认选择。

---

下一步：[神经网络入门](./15-neural-networks-intro.md) — 从 PCA 的线性变换跳到神经网络的多层非线性变换，以及 Autoencoder 作为 PCA 的非线性推广。

---

> **"降维不是扔掉数据——是把数据用更少的数字重新说一遍。"** 你手算的那 5 个点，原来用 10 个数描述，PCA 后用 5 个数就够了——而且随时可以近乎无损地还原回来。这就是数学的力量。
