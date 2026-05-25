## 聚类算法：发现数据中的隐藏结构

> **核心问题**：如果数据没有标签，我们还能从中学到什么？

前面章节的每一个算法——线性回归、逻辑回归、SVM、决策树——都依赖于标注数据。你需要告诉模型"这是猫"、"那是狗"，它才能学会区分。但现实中大量数据根本没有标签：用户行为日志、基因序列、天文观测、传感器读数……聚类正是回答这个问题的武器——在没有标签的情况下，让数据自己"说话"，发现其中隐藏的分组结构。

### 0. 本章导览

聚类是无监督学习的核心。本章不仅讲三个经典算法"是什么"，更重要的是带你**亲手算一遍**——K-Means 的质心怎么移动、层次聚类的树状图怎么画。算法的灵魂不在公式里，在铅笔划过草稿纸的每一步里。

读完本章你将能够：

1. 手算 K-Means 的完整 3 次迭代——6 个二维点、2 个簇、从初始质心到收敛的每一步
2. 手算凝聚层次聚类（5 个点），画出树状图，在任意高度截断得到 K 个簇
3. 从零实现 K-Means（Lloyd's 算法），理解广播运算的向量化技巧
4. 用肘部法则、轮廓系数、Gap 统计量选择最佳 K 值
5. 区分 Single / Complete / Average / Ward 四种链接准则的几何含义
6. 理解 DBSCAN 为什么能发现任意形状的簇——以及它为什么在高维数据上失效
7. 独立完成一个客户分群项目：标准化 → 选 K → 聚类 → 特征画像 → 业务解读
8. 答完 10 道思考题，覆盖理论盲区、实践陷阱和跨算法对比

> 本章目标 1200+ 行。**第 3.3 节和第 5.3 节的手算部分请务必拿纸笔跟着算——这是真正"拥有"这些算法的最短路径。**

前置章节：无硬性要求，但建议先了解[欧氏距离等基本概念](./07-knn.md)
下一章：[降维](./14-dimensionality-reduction.md)

---

### 1. 什么是聚类？无监督学习的核心

**直觉理解：** 你走进一间教室，学生三三两两聚在一起。你不认识任何人，但一眼就能看出教室里有几个"小圈子"——不用知道名字、爱好、成绩，仅观察空间分布就自然识别出分组。这就是聚类的直觉本质：利用数据点在特征空间中的**密度和距离**，发现自然形成的群体结构。

继续教育学籍类比：监督学习的分类好比按花名册点名分班——你知道每个班的名单，只是决定新生去哪个班。聚类是连花名册都没有——你只能根据学生们的自然聚集来"发明"班级。分类预测的是**已知类别**，聚类发现的是**未知结构**。

| | 监督学习（分类） | 无监督学习（聚类） |
|---|---|---|
| 训练数据 | 有标签 | 无标签 |
| 目标 | 学习决策边界 | 发现数据分组 |
| 输出 | 类别标签 | 簇编号 |
| 评估 | 准确率、F1 等 | 轮廓系数、惯性等 |
| 典型例子 | 垃圾邮件检测 | 用户分群、异常检测 |

**形式化定义：** 给定无标签数据集 $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$，其中 $\mathbf{x}_i \in \mathbb{R}^d$，聚类的目标是找到一个划分，使得**簇内相似度高、簇间相似度低**。"相似度"由距离度量（欧氏距离、余弦相似度等）来定义。

| 场景 | 聚类做什么 | 价值 |
|------|-----------|------|
| 客户分群 | 按消费行为将客户分成几个群体 | 精准营销、个性化推荐 |
| 图像压缩 | 将像素颜色聚成 K 个代表色 | 减少存储空间（K-Means 的经典用途） |
| 异常检测 | 不属于任何簇的点就是异常 | 欺诈检测、设备故障预警 |
| 文档聚类 | 按主题将文章自动分组 | 新闻聚合、搜索优化 |
| 基因分析 | 发现具有相似表达模式的基因群 | 疾病机理研究 |

---

### 2. 聚类算法全景图

在深入每个算法之前，先建立一个俯瞰视角——三大聚类方法的哲学差异：

```
聚类三大范式：

  K-Means               层次聚类              DBSCAN
  "找中心"              "建树"               "找密集区"
     │                     │                     │
  ┌──┴──┐              ┌──┴──┐              ┌──┴──┐
  │ 点→质心 │          │ 点→点→簇 │          │ 密度→连通 │
  └─────┘              └───────┘              └──────┘
     │                     │                     │
  需要预设K             不需要预设K           不需要预设K
  球形假设              各种链接准则           任意形状
  迭代优化              贪心合并              单次扫描
  可处理大数据          中小数据              中等数据
```

同一份数据，三种算法可能给出三种不同的分组方案——不是因为哪个"错了"，而是它们定义了三种不同的"相似"。这个理解比记住公式重要得多。

---

### 3. K-Means 聚类

#### 3.1 直觉理解

想象你要在城市里开 K 个配送中心。你会把它们放在哪里？很自然地，你会把每个中心放在一组居民的"重心"位置，使得所有居民到最近中心的总距离最小。这就是 K-Means 的核心思想。

具体过程像一场"抢椅子"游戏：

```
(1) 随机放 K 把椅子（质心初始化）
    ↓
(2) 每个人站到离自己最近的椅子旁边（分配）
    ↓
(3) 每把椅子移到围过来的这群人的中心（更新）
    ↓
(4) 重复 (2)(3) 直到椅子不再移动（收敛）
```

#### 2.2 形式化定义：Lloyd's 算法

K-Means 的优化目标是最小化**簇内平方和**（Within-Cluster Sum of Squares, WCSS），也称**惯性（inertia）**：

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

其中 $C_k$ 是第 $k$ 个簇，$\boldsymbol{\mu}_k$ 是该簇的质心（簇内所有点的均值）。

算法步骤：

1. **初始化**：随机选择 K 个点作为初始质心 $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$
2. **分配（Assignment）**：将每个点分配给最近的质心：
   $$C(\mathbf{x}_i) = \mathop{\arg\min}_{k \in \{1,\ldots,K\}} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|$$
3. **更新（Update）**：每个质心移到其簇内所有点的均值：
   $$\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$$
4. **重复**步骤 2-3，直到质心不再变化或达到最大迭代次数

收敛性保证：每次迭代的 WCSS 不会增加（分配步骤让每个点去最近的质心、更新步骤让质心成为簇内均值——这两个操作各自都不会增大 WCSS）。因此算法一定收敛到**局部最优**（不一定是全局最优——取决于初始化）。

#### 2.3 手算 K-Means：6 个二维点，3 次迭代，从零走到收敛

> **请拿出纸笔，跟着算。这是本章最重要的内容。**

**数据集：**

我们使用 6 个明显分成两簇的二维点：

| 点 | x | y |
|----|---|---|
| P1 | 1 | 1 |
| P2 | 2 | 1 |
| P3 | 1 | 2 |
| P4 | 9 | 9 |
| P5 | 10| 9 |
| P6 | 9 | 10|

看一眼坐标：P1-P3 挤在左下角 (1~2, 1~2)，P4-P6 挤在右上角 (9~10, 9~10)。肉眼直接告诉你 K=2。我们用 K-Means 来验证——看算法是否也能"看出来"。

**初始化：** K=2，随机选两个点作为初始质心。假设选中 P1(1,1) 和 P4(9,9)。

```
初始质心: C₁⁰ = (1, 1), C₂⁰ = (9, 9)
```

---

##### 第 1 次迭代

**分配步骤：** 计算每个点到两个质心的平方距离（为方便比较，直接用平方距离——大小关系和开方后一致）。

首先计算每个点到 C₁⁰(1, 1) 的平方距离：

$$\begin{aligned}
d(\text{P1}, C₁⁰) &= (1-1)^2 + (1-1)^2 = 0 \\
d(\text{P2}, C₁⁰) &= (2-1)^2 + (1-1)^2 = 1 \\
d(\text{P3}, C₁⁰) &= (1-1)^2 + (2-1)^2 = 1 \\
d(\text{P4}, C₁⁰) &= (9-1)^2 + (9-1)^2 = 64 + 64 = 128 \\
d(\text{P5}, C₁⁰) &= (10-1)^2 + (9-1)^2 = 81 + 64 = 145 \\
d(\text{P6}, C₁⁰) &= (9-1)^2 + (10-1)^2 = 64 + 81 = 145
\end{aligned}$$

再计算每个点到 C₂⁰(9, 9) 的平方距离：

$$\begin{aligned}
d(\text{P1}, C₂⁰) &= (1-9)^2 + (1-9)^2 = 64 + 64 = 128 \\
d(\text{P2}, C₂⁰) &= (2-9)^2 + (1-9)^2 = 49 + 64 = 113 \\
d(\text{P3}, C₂⁰) &= (1-9)^2 + (2-9)^2 = 64 + 49 = 113 \\
d(\text{P4}, C₂⁰) &= (9-9)^2 + (9-9)^2 = 0 \\
d(\text{P5}, C₂⁰) &= (10-9)^2 + (9-9)^2 = 1 + 0 = 1 \\
d(\text{P6}, C₂⁰) &= (9-9)^2 + (10-9)^2 = 0 + 1 = 1
\end{aligned}$$

汇总对比：

| 点 | 到 C₁⁰ | 到 C₂⁰ | 分配结果 |
|----|---------|---------|---------|
| P1 | **0** | 128 | → C₁ |
| P2 | **1** | 113 | → C₁ |
| P3 | **1** | 113 | → C₁ |
| P4 | 128 | **0** | → C₂ |
| P5 | 145 | **1** | → C₂ |
| P6 | 145 | **1** | → C₂ |

分配结果：$C_1 = \{\text{P1, P2, P3}\}$，$C_2 = \{\text{P4, P5, P6}\}$

**更新步骤：** 计算每个簇的新质心（均值）。

$$\begin{aligned}
\boldsymbol{\mu}_1^1 = C₁¹ &= \left(\frac{1+2+1}{3}, \frac{1+1+2}{3}\right) = \left(\frac{4}{3}, \frac{4}{3}\right) = (1.333, 1.333) \\[8pt]
\boldsymbol{\mu}_2^1 = C₂¹ &= \left(\frac{9+10+9}{3}, \frac{9+9+10}{3}\right) = \left(\frac{28}{3}, \frac{28}{3}\right) = (9.333, 9.333)
\end{aligned}$$

质心从 (1,1) 和 (9,9) 分别"缩"到了各自簇的中心位置 (1.333, 1.333) 和 (9.333, 9.333)。

---

##### 第 2 次迭代

**分配步骤：** 用新质心 C₁¹(1.333, 1.333) 和 C₂¹(9.333, 9.333) 重新分配。

计算每个点到 C₁¹ 的平方距离：

$$\begin{aligned}
d(\text{P1}, C₁¹) &= \left(1-\tfrac{4}{3}\right)^2 + \left(1-\tfrac{4}{3}\right)^2 = \left(-\tfrac{1}{3}\right)^2 + \left(-\tfrac{1}{3}\right)^2 = \tfrac{1}{9} + \tfrac{1}{9} = \tfrac{2}{9} \approx 0.222 \\
d(\text{P2}, C₁¹) &= \left(2-\tfrac{4}{3}\right)^2 + \left(1-\tfrac{4}{3}\right)^2 = \left(\tfrac{2}{3}\right)^2 + \left(-\tfrac{1}{3}\right)^2 = \tfrac{4}{9} + \tfrac{1}{9} = \tfrac{5}{9} \approx 0.556 \\
d(\text{P3}, C₁¹) &= \left(1-\tfrac{4}{3}\right)^2 + \left(2-\tfrac{4}{3}\right)^2 = \left(-\tfrac{1}{3}\right)^2 + \left(\tfrac{2}{3}\right)^2 = \tfrac{1}{9} + \tfrac{4}{9} = \tfrac{5}{9} \approx 0.556 \\
d(\text{P4}, C₁¹) &= \left(9-\tfrac{4}{3}\right)^2 + \left(9-\tfrac{4}{3}\right)^2 = \left(\tfrac{23}{3}\right)^2 + \left(\tfrac{23}{3}\right)^2 = \tfrac{529}{9} + \tfrac{529}{9} = \tfrac{1058}{9} \approx 117.6 \\
d(\text{P5}, C₁¹) &= \left(10-\tfrac{4}{3}\right)^2 + \left(9-\tfrac{4}{3}\right)^2 = \left(\tfrac{26}{3}\right)^2 + \left(\tfrac{23}{3}\right)^2 = \tfrac{676}{9} + \tfrac{529}{9} = \tfrac{1205}{9} \approx 133.9 \\
d(\text{P6}, C₁¹) &= \left(9-\tfrac{4}{3}\right)^2 + \left(10-\tfrac{4}{3}\right)^2 = \left(\tfrac{23}{3}\right)^2 + \left(\tfrac{26}{3}\right)^2 = \tfrac{529}{9} + \tfrac{676}{9} = \tfrac{1205}{9} \approx 133.9
\end{aligned}$$

计算每个点到 C₂¹ 的平方距离（由对称性，P4-P6 到 C₂¹ 的距离和 P1-P3 到 C₁¹ 对称）：

$$\begin{aligned}
d(\text{P1}, C₂¹) &\approx 133.9 \\
d(\text{P2}, C₂¹) &\approx 117.6 \\
d(\text{P3}, C₂¹) &\approx 117.6 \\
d(\text{P4}, C₂¹) &= \tfrac{2}{9} \approx 0.222 \\
d(\text{P5}, C₂¹) &= \tfrac{5}{9} \approx 0.556 \\
d(\text{P6}, C₂¹) &= \tfrac{5}{9} \approx 0.556
\end{aligned}$$

分配结果与第 1 次迭代**完全相同**：P1-P3 → C₁，P4-P6 → C₂。

**更新步骤：** 重新计算质心。既然分配没变，新质心也不变：

$$\boldsymbol{\mu}_1^2 = \left(\tfrac{4}{3}, \tfrac{4}{3}\right), \quad \boldsymbol{\mu}_2^2 = \left(\tfrac{28}{3}, \tfrac{28}{3}\right)$$

---

##### 第 3 次迭代

**分配步骤：** 质心和第 2 次迭代完全一样，分配结果自然也完全一样——**没有任何点的归属发生变化**。

**更新步骤：** 质心不变。

**收敛！** 算法在第 2 次迭代结束时就已收敛（质心位移为 0）。第 3 次迭代只是确认——实际实现中通过检查 `shift < tol` 提前终止。

---

##### 收敛后的指标

最终质心：

$$C₁^{\text{final}} = \left(\frac{4}{3}, \frac{4}{3}\right), \quad C₂^{\text{final}} = \left(\frac{28}{3}, \frac{28}{3}\right)$$

最终惯性（WCSS）——所有点到各自质心的平方距离之和：

$$\begin{aligned}
\text{WCSS} &= \sum_{i \in C_1} \|\mathbf{x}_i - \boldsymbol{\mu}_1\|^2 + \sum_{j \in C_2} \|\mathbf{x}_j - \boldsymbol{\mu}_2\|^2 \\[4pt]
&= \left(\tfrac{2}{9} + \tfrac{5}{9} + \tfrac{5}{9}\right) + \left(\tfrac{2}{9} + \tfrac{5}{9} + \tfrac{5}{9}\right) \\[4pt]
&= \frac{12}{9} + \frac{12}{9} = \frac{24}{9} = \frac{8}{3} \approx 2.667
\end{aligned}$$

**关键直觉：**

1. **收敛速度极快**——这个小例子 2 次迭代就停了。在真实大数据上，K-Means 通常 10-30 次迭代。
2. **初始质心选得好**——P1 和 P4 天然在各自簇内。如果初始选了 P1 和 P2（都在左下角），算法可能收敛到一个很差的局部最优（把一个自然簇强行劈成两个）。
3. **WCSS 单调递减**——分配步骤让每个点去离它最近的质心（减小或保持 WCSS），更新步骤让质心成为均值（最小化簇内平方和）。两个操作都保证不增大 WCSS，所以算法必然收敛。
4. **平方距离代替欧氏距离**——在分配步骤，$\|a-b\|^2$ 和 $\|a-b\|$ 的大小排序完全一致（平方函数单调递增），省掉开方操作。

**迭代可视化（ASCII）：**

```
初始:  C₁⁰(1,1)    C₂⁰(9,9)
        │  │  │      │  │  │
        P1 P2 P3     P4 P5 P6

迭代1: 分配 → 更新
        C₁¹(1.33,1.33)   C₂¹(9.33,9.33)
        │  │  │           │  │  │
        P1 P2 P3          P4 P5 P6

迭代2: 分配不变 → 更新不变 → 收敛！
        C₁^(1.33,1.33)   C₂^(9.33,9.33)
        │  │  │           │  │  │
        P1 P2 P3          P4 P5 P6

迭代3: 验证收敛——质心位移 = 0，算法终止。
```

如果你把这个例子真正跟着算了一遍，恭喜——你已经比 90% 只会 `from sklearn.cluster import KMeans` 的人更懂 K-Means。

#### 2.4 从零实现 K-Means

```python
import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    """从零实现的 K-Means（Lloyd's 算法）"""

    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 random_state: int | None = None, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # 1. 随机选择 K 个样本作为初始质心
        init_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[init_indices].copy().astype(float)

        for i in range(self.max_iter):
            # 2. 分配步骤：计算每个点到所有质心的距离，分配最近质心
            # X: (n, d), centroids: (K, d) → distances: (n, K)
            # 广播技巧：X[:, np.newaxis, :] 变成 (n, 1, d)
            # centroids[np.newaxis, :, :] 变成 (1, K, d)
            # 相减后广播 → (n, K, d)，再沿 axis=2 求和 → (n, K)
            distances = ((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. 更新步骤：重新计算每个簇的质心
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0
                else X[rng.choice(n_samples)]  # 空簇处理：随机重选一个点
                for k in range(self.n_clusters)
            ])

            # 4. 检查收敛：任意质心的最大位移小于阈值
            shift = np.sqrt(((new_centroids - self.centroids_) ** 2).sum(axis=1)).max()
            self.centroids_ = new_centroids

            if shift < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids_[labels]) ** 2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = ((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1)


# ============================================================
# 演示：手算例子的验证 + 可视化
# ============================================================
import matplotlib.pyplot as plt

# 手算所用的 6 个点
X_hand = np.array([
    [1, 1], [2, 1], [1, 2],   # 左下簇
    [9, 9], [10, 9], [9, 10],  # 右上簇
])

km = KMeans(n_clusters=2, random_state=42)
km.fit(X_hand)

print(f"最终质心:\n{km.centroids_}")
print(f"期望质心: (1.333, 1.333) 和 (9.333, 9.333)")
print(f"惯性: {km.inertia_:.3f}  (期望 2.667)")
print(f"迭代次数: {km.n_iter_}")
print(f"标签: {km.labels_}")

# 生成 blobs 数据做完整演示
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60,
                       random_state=0, n_features=2)

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis',
                alpha=0.6, edgecolors='none')
axes[0].scatter(kmeans.centroids_[:, 0], kmeans.centroids_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=1.5)
axes[0].set_title(f'K-Means 聚类结果\n惯性={kmeans.inertia_:.1f}, '
                  f'迭代次数={kmeans.n_iter_}')
axes[0].set_xlabel('特征 1'); axes[0].set_ylabel('特征 2')

axes[1].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                alpha=0.6, edgecolors='none')
axes[1].set_title('真实分组（标签仅用于对比）')
axes[1].set_xlabel('特征 1'); axes[1].set_ylabel('特征 2')

plt.tight_layout(); plt.show()
```

**代码要点：**

- **广播距离计算**：`X[:, np.newaxis, :] - centroids[np.newaxis, :, :]` 一行完成所有 `n × K` 个距离的计算，比双重 for 循环快几十倍。
- **空簇处理**：如果某个簇在分配步骤中没有获得任何点（特别是在高 K 或数据稀疏时），简单地随机重选一个数据点作为该簇的质心。
- **收敛判据**：`shift = max(质心位移)` < `1e-4`。这个阈值对大多数数据足够——位移小于万分之一意味着质心基本稳定。

#### 2.5 K-Means 的局限性与 K-Means++

标准 K-Means 有一个致命弱点——**对初始质心敏感**。如果你随机选的初始质心恰好在同一个簇里，算法就可能把一个自然簇强行撕成两半。

**K-Means++ 初始化的原理：**

1. 随机选择第一个质心
2. 对每个点 $\mathbf{x}_i$，计算它到已选质心的最短距离 $D(\mathbf{x}_i)$
3. 以概率 $\frac{D(\mathbf{x}_i)^2}{\sum_j D(\mathbf{x}_j)^2}$ 选择下一个质心——离已选质心越远的点被选中的概率越大
4. 重复 2-3 直到选出 K 个质心

这个机制的直觉：**故意让质心分散**。理论保证：K-Means++ 的期望 WCSS 不超过最优解的 $O(\log K)$ 倍。

```python
from sklearn.cluster import KMeans as SklearnKMeans

# 生成对初始化敏感的数据
X_sensitive, _ = make_blobs(n_samples=500, centers=4,
                            cluster_std=1.5, random_state=170)
X_sensitive = np.vstack([X_sensitive,
                         np.random.RandomState(42).multivariate_normal(
                             [-4, -6], [[8.0, 2.0], [2.0, 1.0]], 100)])

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 随机初始化：两次运行结果可能不同
for i, seed in enumerate([0, 42]):
    km = KMeans(n_clusters=5, random_state=seed)
    km.fit(X_sensitive)
    axes[0][i].scatter(X_sensitive[:, 0], X_sensitive[:, 1],
                       c=km.labels_, cmap='tab10', alpha=0.6, s=10)
    axes[0][i].scatter(km.centroids_[:, 0], km.centroids_[:, 1],
                       c='red', marker='X', s=100, edgecolors='black')
    axes[0][i].set_title(f'随机初始化 (seed={seed})\n惯性={km.inertia_:.0f}')

# K-Means++：两次一致
for i, seed in enumerate([0, 42]):
    km_pp = SklearnKMeans(n_clusters=5, init='k-means++', n_init='auto',
                           random_state=seed)
    km_pp.fit(X_sensitive)
    axes[1][i].scatter(X_sensitive[:, 0], X_sensitive[:, 1],
                       c=km_pp.labels_, cmap='tab10', alpha=0.6, s=10)
    axes[1][i].scatter(km_pp.cluster_centers_[:, 0],
                       km_pp.cluster_centers_[:, 1],
                       c='red', marker='X', s=100, edgecolors='black')
    axes[1][i].set_title(f'K-Means++ (seed={seed})\n惯性={km_pp.inertia_:.0f}')

plt.suptitle('随机初始化 vs K-Means++：稳定性对比', fontsize=14)
plt.tight_layout(); plt.show()
```

**K-Means 的四大局限速查：**

| 局限 | 表现 | 解决方案 |
|------|------|---------|
| 预设 K 值 | 必须事先知道簇数 | 肘部法则、轮廓系数、Gap 统计量 |
| 球形假设 | 只能发现球形/凸形簇 | DBSCAN、谱聚类 |
| 对尺度敏感 | 范围大的特征主导聚类 | 标准化数据（StandardScaler） |
| 对离群点敏感 | 离群点会拉偏质心 | 数据预处理、使用 K-Medoids |

**应用连接：** K-Means 的 $O(n \cdot K \cdot d \cdot t)$ 复杂度使其能处理百万级数据。sklearn 默认 `n_init=10`（跑 10 次取惯性最小的一次），结合 K-Means++ 初始化，实践中几乎不会掉进很差的局部最优。

#### 3.6 大规模数据：Mini-Batch K-Means

标准 K-Means 每次迭代需要扫描全部数据来更新质心。当数据量达到千万级别时，这变得不切实际——单次迭代就要等待数分钟。

**Mini-Batch K-Means** 是 K-Means 的随机版本：每次迭代不使用全部数据，而是从数据中随机抽取一个小批量（batch，如 1000 个样本），仅用这批数据来更新质心。虽然收敛路径比标准 K-Means 更"嘈杂"，但每次迭代速度快了几个数量级，总训练时间通常远低于标准版本。

```python
from sklearn.cluster import MiniBatchKMeans

# 1000万样本的模拟数据——标准 K-Means 跑完要几分钟
# Mini-Batch K-Means 十几秒搞定，惯性差距通常 < 5%
X_large, _ = make_blobs(n_samples=100000, centers=8, random_state=0)

# 标准 vs Mini-Batch 对比
km_std = SklearnKMeans(n_clusters=8, n_init='auto', random_state=0)
km_mb = MiniBatchKMeans(n_clusters=8, batch_size=1024, random_state=0)

import time
t0 = time.time(); km_std.fit(X_large); t1 = time.time()
t2 = time.time(); km_mb.fit(X_large); t3 = time.time()

print(f"标准 K-Means:  {t1-t0:.2f}s, 惯性={km_std.inertia_:.0f}")
print(f"Mini-Batch:    {t3-t2:.2f}s, 惯性={km_mb.inertia_:.0f}")
print(f"惯性差异:      {(km_mb.inertia_ - km_std.inertia_) / km_std.inertia_ * 100:.1f}%")
```

> **选择建议：** 数据 < 10万样本 → 标准 K-Means。数据 > 10万 → Mini-Batch。两者惯性差距通常在 3-5% 以内，训练时间可能差 10 倍以上。

---

### 4. 如何选择 K 值？

**直觉理解：** K 增大 → 簇变多 → 每个簇更紧凑 → WCSS 一定下降。但下降的"边际收益"递减——你要找那个"再增加 K 收益不再显著"的拐点。

```python
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
K_range = range(1, 11)

for k in K_range:
    km = SklearnKMeans(n_clusters=k, init='k-means++', n_init='auto',
                        random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    if k >= 2:
        silhouettes.append(silhouette_score(X, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：肘部法则
axes[0].plot(K_range, inertias, 'bo-', markersize=6)
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='肘部 (K=4)')
axes[0].set_xlabel('K 值'); axes[0].set_ylabel('惯性 (WCSS)')
axes[0].set_title('肘部法则 (Elbow Method)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 右：轮廓系数
axes[1].plot(range(2, 11), silhouettes, 'go-', markersize=6)
axes[1].axvline(x=4, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('K 值'); axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数 (Silhouette Score)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
```

**三种方法的对比：**

- **肘部法则（Elbow Method）**：画惯性-K 曲线，找"弯曲"最明显的点。简单直观但主观——有时曲线平滑、找不到明显肘部。
- **轮廓系数（Silhouette Score）**：对每个样本计算 $s = \frac{b - a}{\max(a, b)}$，其中 $a$ 是样本到同簇其他点的平均距离（凝聚度），$b$ 是样本到最近其他簇的平均距离（分离度）。$s \in [-1, 1]$，接近 1 表示好聚类，接近 0 表示点在簇边界，负值表示点可能分错了簇。
- **Gap 统计量**：将实际数据的 $\log(\text{WCSS})$ 与均匀随机分布做对比，选 Gap 最大的 K。最严谨，但计算开销最大。

> **一句话：** 三种方法交叉验证，结合业务知识拍板。如果市场部门说"我们只能运营 5 个客户群"，那 K=5 就是最优——比任何统计量都有说服力。

---

### 5. 层次聚类 (Hierarchical Clustering)

#### 4.1 直觉理解

K-Means 像一个"分地盘"的过程——中心先定好，再把点分配过去。层次聚类则像一棵树的生长——从底往上，初始每个点自己是一个小簇，然后不断合并"最相似"的两个簇，直到所有点合成一个大簇。整个过程就像生物分类学：物种 → 属 → 科 → 目 → 纲 → 门 → 界。

#### 4.2 形式化定义：凝聚型层次聚类

1. 初始：$n$ 个簇，每个点自成一个簇
2. 找到距离最近的两个簇 $C_a$ 和 $C_b$，合并成一个大簇
3. 重复步骤 2，直到只剩 1 个簇
4. 在**树状图（Dendrogram）**上选择合适的高度截断，得到最终聚类

关键在于**簇间距离怎么定义**——即链接准则（Linkage Criteria）：

| 链接方法 | 簇间距离 | 特点 |
|----------|---------|------|
| **Single（单链）** | $\min_{a \in C_A, b \in C_B} d(a, b)$ | 能发现非球形/细长簇，但对噪声敏感（链式效应：一个点就能把两个大簇连起来） |
| **Complete（全链）** | $\max_{a \in C_A, b \in C_B} d(a, b)$ | 产生更紧凑的簇，倾向等直径球形簇 |
| **Average（均链）** | $\frac{1}{|C_A||C_B|}\sum_{a \in C_A}\sum_{b \in C_B} d(a, b)$ | 折中方案，鲁棒性好 |
| **Ward** | $\Delta \text{WCSS}$（合并后 WCSS 增加量） | 最小化每次合并的 WCSS 增量，倾向等大小球形簇 |

#### 4.3 手算层次聚类 & 画树状图：5 个点，从单点到一棵树

> **请再次拿出纸笔。跟着算完，你就能教别人了。**

**数据集：** 5 个一维点（一维便于手算，原理在二维和高维完全一样）：

$$A = 2,\quad B = 5,\quad C = 9,\quad D = 12,\quad E = 18$$

使用 **Single Linkage（单链）**——两个簇之间的距离 = 簇内所有点对之间的最小距离。

---

##### 步骤 0：初始化 —— 5 个独立簇

$$\{A\},\; \{B\},\; \{C\},\; \{D\},\; \{E\}$$

计算所有点对距离（一维用绝对值）：

| 点对 | A(2) | B(5) | C(9) | D(12) | E(18) |
|------|------|------|------|-------|-------|
| A(2) | — | 3 | 7 | 10 | 16 |
| B(5) | | — | 4 | 7 | 13 |
| C(9) | | | — | 3 | 9 |
| D(12)| | | | — | 6 |

最小距离 = **1**（不存在），实际最小 = **3**。出现了两个距离 3：AB = 3，CD = 3。我们先合并 AB。

##### 步骤 1：合并 A 和 B（距离 = 3）

新簇 $\{A, B\}$。现在 4 个簇：$\{A,B\}, \{C\}, \{D\}, \{E\}$

计算 $\{A,B\}$ 到其他簇的单链距离（取最小值）：

$$\begin{aligned}
d(\{A,B\}, C) &= \min(d(A,C), d(B,C)) = \min(7, 4) = 4 \\
d(\{A,B\}, D) &= \min(d(A,D), d(B,D)) = \min(10, 7) = 7 \\
d(\{A,B\}, E) &= \min(d(A,E), d(B,E)) = \min(16, 13) = 13
\end{aligned}$$

当前所有簇间最小距离 = **3**（C 和 D 之间）。

##### 步骤 2：合并 C 和 D（距离 = 3）

新簇 $\{C, D\}$。现在 3 个簇：$\{A,B\}, \{C,D\}, \{E\}$

计算 $\{C,D\}$ 到其他簇的单链距离：

$$\begin{aligned}
d(\{C,D\}, \{A,B\}) &= \min(d(C,A), d(C,B), d(D,A), d(D,B)) \\
                     &= \min(7, 4, 10, 7) = 4 \\
d(\{C,D\}, E) &= \min(d(C,E), d(D,E)) = \min(9, 6) = 6
\end{aligned}$$

当前最小 = **4**（$\{A,B\}$ 和 $\{C,D\}$ 之间）。

##### 步骤 3：合并 $\{A,B\}$ 和 $\{C,D\}$（距离 = 4）

新簇 $\{A, B, C, D\}$。现在 2 个簇：$\{A,B,C,D\}, \{E\}$

距离：

$$d(\{A,B,C,D\}, E) = \min(d(A,E), d(B,E), d(C,E), d(D,E)) = \min(16, 13, 9, 6) = 6$$

##### 步骤 4：合并 $\{A,B,C,D\}$ 和 $\{E\}$（距离 = 6）

只剩 1 个簇 $\{A,B,C,D,E\}$——算法终止。

---

##### 画树状图（Dendrogram）

合并记录：

| 合并步 | 合并对象 | 合并距离 |
|--------|---------|---------|
| 1 | A + B | 3 |
| 2 | C + D | 3 |
| 3 | {A,B} + {C,D} | 4 |
| 4 | {A,B,C,D} + E | 6 |

树状图：

```
距离 ↑
  7 |
    |
  6 |                                    ┌───────────────┐
    |                                    │               │
  5 |                                    │               │
    |                                    │               │
  4 |          ┌───────────────┐         │               │
    |          │               │         │               │
  3 |    ┌─────┤         ┌─────┤         │               │
    |    │     │         │     │         │               │
  2 |    │     │         │     │         │               │
    |    │     │         │     │         │               │
  1 |    │     │         │     │         │               │
    |    │     │         │     │         │               │
  0 └────┴─────┴─────────┴─────┴─────────┴───────────────┴──→
        A     B         C     D         E
       (2)   (5)       (9)   (12)      (18)
```

**如何在树状图上选择 K：**

横切一刀，数被切到的垂直线数 = 簇数。

- 在距离 **3.5** 处横切 → 切到 3 条垂直线 → **K=3**：{A,B}、{C,D}、{E}
- 在距离 **5** 处横切 → 切到 2 条垂直线 → **K=2**：{A,B,C,D}、{E}
- 在距离 **7** 处横切 → 切到 1 条垂直线 → **K=1**：全部

这个灵活性是层次聚类相对于 K-Means 的巨大优势——你不需要预先决定 K，可以画完树状图后再选。

**关键直觉：**

1. **合并距离突然跳跃**——从距离 4 跳到距离 6（最后一次合并），说明要把一个已经很不一样的大簇硬和 E 合并。通常在跳跃处截断：K=2 或 K=3。
2. **Single Linkage 的链式效应**——如果在两个簇之间有一个噪声点，Single Linkage 可能通过它把两个本该分开的簇"串"起来（就像锁链的环）。本例数据干净，没有这个问题。
3. **一维和多维**——在一维上距离就是绝对值差；在二维及以上用欧氏距离。合并逻辑完全相同。
4. **树状图 = 聚类历史**——每一步都永久记录在树中。你可以回顾任意时刻的聚类状态——这是 K-Means 做不到的（它只给你最终结果，中间过程不可见）。

**合并过程的方块图视角：**

```
步骤   距离    合并操作           当前簇状态
─────────────────────────────────────────────────
  0     —      初始状态          [A] [B] [C] [D] [E]
  1     3      A+B → {A,B}      [{A,B}] [C] [D] [E]     → 4簇
  2     3      C+D → {C,D}      [{A,B}] [{C,D}] [E]     → 3簇
  3     4     {A,B}+{C,D}       [{A,B,C,D}] [E]         → 2簇
  4     6     +E 合并全部       [{A,B,C,D,E}]           → 1簇
─────────────────────────────────────────────────
从 K=5 到 K=1，每一步都有明确的几何解释：
  在高度 3 截断 → 自然分出 {A,B} 和 {C,D} 两对最近邻居
  在高度 5 截断 → {A,B,C,D} 是一组，E 独自一组
```

如果把这个例子也真正跟着算了一遍，你现在已经掌握了两种完全不同范式的聚类算法——从原理到实现。

#### 4.4 Python 实现

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 将我们手算的 5 个点变成二维（y 坐标全部为 0）
X_hc = np.array([[2, 0], [5, 0], [9, 0], [12, 0], [18, 0]])

# 计算链接矩阵：每行 [cluster_a, cluster_b, distance, num_points]
Z_single = linkage(X_hc, method='single')
print("链接矩阵 (single linkage):")
print(Z_single)

# 不同截断方式得到不同 K
for t in [3.5, 5.0, 7.0]:
    labels = fcluster(Z_single, t=t, criterion='distance')
    n_clusters = len(np.unique(labels))
    print(f"在距离 {t} 处截断 → K={n_clusters}, 标签: {labels}")

# 也可直接指定 K
labels_k3 = fcluster(Z_single, t=3, criterion='maxclust')
print(f"直接指定 K=3 → 标签: {labels_k3}")

# 可视化：对比四种链接方法
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, method in zip(axes.flat, ['single', 'complete', 'average', 'ward']):
    Z = linkage(X_hc, method=method)
    dendrogram(Z, ax=ax, labels=['A', 'B', 'C', 'D', 'E'],
               leaf_font_size=12)
    ax.set_title(f'{method.capitalize()} Linkage')
    ax.set_ylabel('距离')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5,
               label='截断距离=5')
    ax.legend(fontsize=8)

plt.suptitle('四种链接准则的树状图对比', fontsize=14)
plt.tight_layout(); plt.show()
```

**应用连接：** 树状图是向非技术人员解释聚类结果的最佳可视化工具——你不需要解释"质心"或"惯性"，只需一句话："越早合并的越相似"。它特别适合需要**多粒度聚类**的场景，如生物分类学、文档主题体系等。代价是 $O(n^3)$ 的计算复杂度（可优化到 $O(n^2)$），不适合超大规模数据。

---

### 6. DBSCAN：基于密度的聚类

**直觉理解：** K-Means 假设簇是球形的。但真实世界的数据很少这么规整——两个同心圆、月牙形、螺旋形，K-Means 永远无法正确分开它们。DBSCAN 换了一个视角：**不关心形状，只关心密度**。一个簇就是一大团挤在一起的点；簇之间由稀疏地带自然分隔。不属于任何密集区域的点就是**噪声**。

**形式化定义：** DBSCAN 基于两个参数：

- **eps（ε）**：邻域半径
- **min_samples（minPts）**：成为核心点所需的最少邻域点数

三种点类型：

- **核心点**：$|N_\varepsilon(\mathbf{x})| \geq \text{min\_samples}$ —— 周围足够密集
- **边界点**：自身不是核心点，但在某个核心点的 eps-邻域内
- **噪声点**：既不是核心点也不是边界点 → 标签为 -1

簇的定义：从核心点出发，通过**密度可达**关系不断扩展形成的最大连通区域。

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles

n_samples = 300
X_moons, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=0)
X_circles, _ = make_circles(n_samples=n_samples, noise=0.05,
                             factor=0.5, random_state=0)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for row, (X_data, title_prefix) in enumerate([
    (X_moons, '双月牙形'),
    (X_circles, '同心圆形')
]):
    # K-Means
    km = SklearnKMeans(n_clusters=2, init='k-means++', n_init='auto',
                        random_state=0)
    axes[row][0].scatter(X_data[:, 0], X_data[:, 1],
                         c=km.fit_predict(X_data), cmap='tab10', s=15)
    axes[row][0].set_title(f'K-Means\n{title_prefix}')
    axes[row][0].set_xlabel('x'); axes[row][0].set_ylabel('y')

    # DBSCAN
    db = DBSCAN(eps=0.15 if row == 0 else 0.2, min_samples=5)
    db_labels = db.fit_predict(X_data)
    n_noise = (db_labels == -1).sum()
    n_c = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    axes[row][1].scatter(X_data[:, 0], X_data[:, 1],
                         c=db_labels, cmap='tab10', s=15)
    axes[row][1].set_title(f'DBSCAN\n{title_prefix} '
                           f'(簇数={n_c}, 噪声={n_noise})')
    axes[row][1].set_xlabel('x'); axes[row][1].set_ylabel('y')

    # eps 灵敏度
    for j, eps_val in enumerate(
        [0.10, 0.15, 0.25] if row == 0 else [0.10, 0.20, 0.30]
    ):
        db2 = DBSCAN(eps=eps_val, min_samples=5).fit(X_data)
        n_c2 = len(set(db2.labels_)) - (1 if -1 in db2.labels_ else 0)
        axes[row][2].scatter(X_data[:, 0], X_data[:, 1],
                             c=db2.labels_, cmap='tab10', s=3, alpha=0.7)
        axes[row][2].set_title(f'DBSCAN eps 灵敏度 ({title_prefix})')
        axes[row][2].text(0.98, 0.95 - j * 0.08,
                          f'eps={eps_val} → {n_c2}簇',
                          transform=axes[row][2].transAxes, fontsize=8,
                          ha='right', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat',
                                    alpha=0.5))

plt.tight_layout(); plt.show()
```

**DBSCAN 的三个杀手级优势：**

1. **自动发现任意形状的簇**——不需要球形假设
2. **自动检测噪声**——K-Means 和层次聚类做不到这一点
3. **不需要预先指定 K**——你只需要给 eps 和 min_samples

**DBSCAN 的一个微型手算示例：**

给定 7 个一维点，eps=1.5，min_samples=3：

```
点:  A=1, B=2, C=2.5, D=4.5, E=8, F=9, G=14
位置: 1   2   2.5   4.5   8   9   14
```

审查每个点的 eps-邻域（eps=1.5 内的点数）：

- A(1)：域内 {A, B, C} = 3 点 ≥ 3 → **核心点** ✓
- B(2)：域内 {A, B, C} = 3 点 ≥ 3 → **核心点** ✓
- C(2.5)：域内 {A, B, C, D} = 4 点 ≥ 3 → **核心点** ✓
- D(4.5)：域内 {C, D} = 2 点 < 3 → 不是核心点。但 D 在 C 的邻域内 → **边界点**
- E(8)：域内 {E, F} = 2 点 < 3。不在任何核心点邻域内 → **噪声点** ✗
- F(9)：域内 {E, F} = 2 点 < 3 → **噪声点** ✗
- G(14)：域内 {G} = 1 点 → **噪声点** ✗

结果：
- **簇 1**：{A, B, C, D}（核心点 A, B, C 互相密度可达，D 是边界点）
- **噪声**：{E, F, G}

关键观察：E 和 F 距离只有 1 ≤ eps，但它们的 eps-邻域点数不足 min_samples（3）——即使它们彼此靠近，也都被标为噪声。DBSCAN 的"密度"条件比"距离"条件更严格。

如果把 min_samples 降到 2，E 和 F 都变成核心点，形成第二个簇 {E, F}。把 eps 增大到 2，D 也变成核心点，整个簇变成一个。这就是 DBSCAN 对参数敏感的根源。

**参数选择指南：**

- **min_samples**：经验法则取 `2 * 维度`（二维取 4，三维取 6）。更大的值对噪声更鲁棒，但可能丢失小簇。如果你预期数据有很多小但密集的簇，减小 min_samples；如果数据噪声大，增大 min_samples。
- **eps**：画 k-距离图（对每个点计算到第 k 近邻的距离，排序后画图，找拐点）。可以用 `sklearn.neighbors.NearestNeighbors` 辅助。

```python
from sklearn.neighbors import NearestNeighbors

# k-距离图辅助选 eps
k = 5  # 一般取 min_samples 的值
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, _ = nbrs.kneighbors(X)
k_dist = np.sort(distances[:, -1])  # 每个点到第 k 近邻的距离，排序

plt.plot(k_dist)
plt.xlabel('点索引（按距离排序）'); plt.ylabel(f'到第{k}近邻的距离')
plt.title(f'k-距离图 (k={k})——找拐点作为 eps')
plt.grid(True, alpha=0.3)
plt.show()
# eps 取拐点处的 y 值
```

**两个关键局限：**

1. **对参数敏感**——eps 差 0.05 可能让结果从"完美"变成"全是噪声"。经验法则：先用 **k-距离图**（对每个点画出到第 k 近邻的距离，排序，找拐点）估计 eps。
2. **维度灾难**——在高维空间（如 100 维）中，所有点之间的欧氏距离趋于均等，"密度"的概念失效。DBSCAN 通常只在 2-20 维的空间中有效。

---

### 7. 软聚类：高斯混合模型（GMM）简介

K-Means 做了一个**硬分配**——每个点只能属于一个簇。但真实世界中的很多数据是模糊的：一个客户可能"60% 像学生群体、40% 像年轻白领群体"。

**高斯混合模型（Gaussian Mixture Model, GMM）** 是 K-Means 的概率升级版：

- K-Means：点 → 最近的质心（硬分配）
- GMM：点 → 属于每个簇的概率（软分配）

**直觉理解：** 假设数据由 K 个高斯分布（正态分布）混合生成。每个高斯有自己的均值 $\boldsymbol{\mu}_k$ 和协方差矩阵 $\boldsymbol{\Sigma}_k$。GMM 的目标是估计这些参数，使得数据被生成的概率（似然）最大。

GMM 和 K-Means 的关系：

| | K-Means | GMM |
|---|---|---|
| 分配方式 | 硬（0 或 1） | 软（概率 [0, 1]） |
| 簇形状 | 球形（各向同性） | 椭圆形（$\boldsymbol{\Sigma}_k$ 决定方向和拉伸） |
| 优化算法 | Lloyd's 算法 | EM（期望最大化）算法 |
| 可解释性 | 高 | 中 |
| 计算开销 | 低 | 中高 |

GMM 的一个关键优势：它能自然地处理**重叠簇**和**不同大小/形状的簇**。但代价是——你需要选择协方差类型（full/tied/diag/spherical），而且 EM 算法可能收敛到比 K-Means 更差的局部最优。

> **一句话判断：** 如果你的数据看起来像"几个大小不同的重叠椭圆"，用 GMM。如果像"几个分离良好的球形团"，K-Means 就够了。

```python
from sklearn.mixture import GaussianMixture

# 生成有重叠的两簇数据
X_overlap = np.vstack([
    np.random.multivariate_normal([0, 0], [[2, 0.5], [0.5, 1]], 200),
    np.random.multivariate_normal([3, 2], [[1, -0.3], [-0.3, 2]], 200),
])

# GMM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(X_overlap)
gmm_probs = gmm.predict_proba(X_overlap)  # (n, 2) 的软分配概率矩阵

# K-Means 对照
km_labels = SklearnKMeans(n_clusters=2, n_init='auto',
                           random_state=42).fit_predict(X_overlap)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# K-Means：硬边界
axes[0].scatter(X_overlap[:, 0], X_overlap[:, 1], c=km_labels, cmap='coolwarm', s=15)
axes[0].set_title('K-Means（硬分配）\n每个点只有一个颜色')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

# GMM：用概率着色——越"不确定"的点颜色越接近白色
uncertainty = 1 - gmm_probs.max(axis=1)  # 最大概率的补数 → 不确定度
axes[1].scatter(X_overlap[:, 0], X_overlap[:, 1],
                c=gmm_labels, cmap='coolwarm', alpha=1 - uncertainty, s=15)
axes[1].set_title('GMM（软分配）\n颜色越浅 = 归属越不确定')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

plt.tight_layout(); plt.show()

# 查看边界附近的点的概率分布
boundary_idx = np.argsort(uncertainty)[-5:]  # 最不确定的5个点
print("边界点的归属概率（格式：[属于簇0的概率, 属于簇1的概率]）：")
for i in boundary_idx:
    print(f"  点{i:3d}: [{gmm_probs[i, 0]:.3f}, {gmm_probs[i, 1]:.3f}]")
```

**应用连接：** GMM 在图像分割（前景/背景的概率化分割，每像素输出"属于前景的概率"而非黑白二值）、语音识别（GMM-HMM 是 1990-2010 年间的工业标准）、异常检测（低概率密度区域的点 = 异常）中有广泛应用。EM 算法的收敛速度和对初始化的敏感性使其在工程中不如 K-Means 即插即用。学习建议：先把 K-Means 学透，再过渡到 GMM 会轻松得多——K-Means 本质上是 GMM 在"所有簇协方差相同且趋于 0"这个极限下的特例。

---


聚类评估分两套体系：

| 类型 | 依赖标签？ | 用途 | 例子 |
|------|-----------|------|------|
| **内部指标** | 否 | 衡量簇的紧凑度和分离度 | 轮廓系数、DB 指数、CH 指数 |
| **外部指标** | 是 | 和已知标签对比（仅用于验证，不用于训练） | ARI、NMI、同质性 |

```python
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score,
                              normalized_mutual_info_score)

X_eval, y_eval = make_blobs(n_samples=500, centers=4, random_state=0)

kmeans_labels = SklearnKMeans(n_clusters=4, n_init='auto',
                               random_state=0).fit_predict(X_eval)
hc_labels = fcluster(linkage(X_eval, method='ward'), t=4,
                      criterion='maxclust') - 1
dbscan_labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X_eval)

results = []
for name, labels in [('K-Means', kmeans_labels),
                     ('Hierarchical', hc_labels),
                     ('DBSCAN', dbscan_labels)]:
    sil = silhouette_score(X_eval, labels)
    db_idx = davies_bouldin_score(X_eval, labels)
    ch = calinski_harabasz_score(X_eval, labels)
    ari = adjusted_rand_score(y_eval, labels)
    nmi = normalized_mutual_info_score(y_eval, labels)
    results.append([name, sil, db_idx, ch, ari, nmi])

header = f"{'算法':<14s} {'轮廓↑':>7s} {'DB↓':>7s} {'CH↑':>9s} {'ARI↑':>6s} {'NMI↑':>6s}"
print(header)
print('-' * len(header))
for row in results:
    print(f"{row[0]:<14s} {row[1]:7.3f} {row[2]:7.3f} "
          f"{row[3]:9.1f} {row[4]:6.3f} {row[5]:6.3f}")
```

**内部指标速查：**

| 指标 | 测量什么 | 方向 |
|------|---------|------|
| 轮廓系数 (Silhouette) | 紧凑度 vs 分离度，值域 [-1, 1] | ↑ 越大越好 |
| Davies-Bouldin 指数 | 簇内散度 vs 簇间距离比值均值 | ↓ 越小越好 |
| Calinski-Harabasz 指数 | 簇间方差 / 簇内方差 | ↑ 越大越好 |

> **最大陷阱：** 高轮廓系数 ≠ 好的业务聚类。一个统计上紧凑的方案可能把"高价值客户"和"流失客户"混在一起。最佳实践：先用内部指标筛选候选方案，再用**领域知识**定性验证每个簇的业务含义。

---

### 8. 聚类评估：没有标签怎么打分？

**直觉理解：** 监督学习有标签可以算准确率，无监督学习没有标签怎么办？答案是两套指标：**内部指标**只看数据本身的紧凑度和分离度（模型自己证明自己分得好），**外部指标**需要真实标签（仅用于验证，不能用于训练）。

| 类型 | 依赖标签？ | 用途 | 例子 |
|------|-----------|------|------|
| **内部指标** | 否 | 衡量簇的紧凑度和分离度 | 轮廓系数、DB 指数、CH 指数 |
| **外部指标** | 是 | 和已知标签对比（仅用于评估，不参与训练） | ARI、NMI、同质性 |

**形式化定义——三大内部指标：**

**1. 轮廓系数（Silhouette Score）：**

对每个样本单独计算 $s_i$：

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

- $a_i$：样本 $i$ 到同簇其他样本的**平均距离**（凝聚度——越小越紧凑）
- $b_i$：样本 $i$ 到最近的其他簇中所有样本的**平均距离**（分离度——越大越分离）

整体轮廓系数 = 所有样本 $s_i$ 的均值。值域 [-1, 1]：
- $s \approx 1$：样本紧贴邻居 + 远离其他簇 → 优秀聚类
- $s \approx 0$：样本在簇边界上，到自家和隔壁距离差不多
- $s < 0$：样本离隔壁比离自家更近 → 大概率分错了簇

**2. Davies-Bouldin 指数：**

$$\text{DB} = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)}$$

其中 $\sigma_i$ 是簇 $i$ 内样本到质心的平均距离（散度），$d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)$ 是两簇质心距离。DB 衡量的是"最相似的簇对"有多相似——**越小越好**。0 是最优值（所有簇完全不重叠）。

**3. Calinski-Harabasz 指数（方差比准则）：**

$$\text{CH} = \frac{\text{tr}(B_K)}{\text{tr}(W_K)} \cdot \frac{n - K}{K - 1}$$

其中 $B_K$ 是簇间散度矩阵，$W_K$ 是簇内散度矩阵。本质上就是 **簇间方差 / 簇内方差**——和我们熟悉的 F 统计量同构。**越大越好**。

```python
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score,
                              normalized_mutual_info_score)

# 用一个已知标签的数据集评估各种指标
X_eval, y_eval = make_blobs(n_samples=500, centers=4, random_state=0)

kmeans_labels = SklearnKMeans(n_clusters=4, n_init='auto',
                               random_state=0).fit_predict(X_eval)
hc_labels = fcluster(linkage(X_eval, method='ward'), t=4,
                      criterion='maxclust') - 1
dbscan_labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X_eval)

results = []
for name, labels in [('K-Means', kmeans_labels),
                     ('Hierarchical', hc_labels),
                     ('DBSCAN', dbscan_labels)]:
    sil = silhouette_score(X_eval, labels)
    db_idx = davies_bouldin_score(X_eval, labels)
    ch = calinski_harabasz_score(X_eval, labels)
    ari = adjusted_rand_score(y_eval, labels)
    nmi = normalized_mutual_info_score(y_eval, labels)
    results.append([name, sil, db_idx, ch, ari, nmi])

header = f"{'算法':<14s} {'轮廓↑':>7s} {'DB↓':>7s} {'CH↑':>9s} {'ARI↑':>6s} {'NMI↑':>6s}"
print(header)
print('-' * len(header))
for row in results:
    print(f"{row[0]:<14s} {row[1]:7.3f} {row[2]:7.3f} "
          f"{row[3]:9.1f} {row[4]:6.3f} {row[5]:6.3f}")
```

**内部指标速查：**

| 指标 | 测量什么 | 方向 | 最佳值 |
|------|---------|------|--------|
| 轮廓系数 (Silhouette) | 紧凑度 vs 分离度的个体级评估 | ↑ | 1 |
| Davies-Bouldin 指数 | 簇间相似度（越低越好） | ↓ | 0 |
| Calinski-Harabasz 指数 | 簇间方差 / 簇内方差 | ↑ | +∞ |

> **最大陷阱：** 高轮廓系数 ≠ 好的业务聚类。一个统计上紧凑的方案可能把"高价值客户"和"流失客户"混在一起——内部指标只关心几何结构，不关心业务含义。最佳实践：先用内部指标筛选 2-3 个候选方案，再用**领域知识**定性验证每个簇是否有合理的业务解释。

---

### 9. sklearn 实战：客户分群全流程

用一个包含年龄、年收入、消费评分的商场客户数据集，完整体验聚类分析的闭环：

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 生成模拟客户数据
np.random.seed(42)
n_customers = 200

age = np.concatenate([
    np.random.normal(25, 5, 50),    # 年轻群体
    np.random.normal(40, 8, 60),    # 中年群体
    np.random.normal(55, 6, 50),    # 较年长群体
    np.random.normal(30, 4, 40),    # 混合群体
])

income = np.concatenate([
    np.random.normal(8, 2, 50),     # 收入较低
    np.random.normal(25, 8, 60),    # 中高收入
    np.random.normal(15, 5, 50),    # 中等收入
    np.random.normal(45, 10, 40),   # 高收入
])

score = np.concatenate([
    np.random.normal(70, 10, 50),   # 高消费
    np.random.normal(40, 12, 60),   # 中低消费
    np.random.normal(50, 8, 50),    # 中等消费
    np.random.normal(80, 6, 40),    # 高消费
])

df = pd.DataFrame({'Age': age, 'Income': income, 'SpendingScore': score})
df = df.clip(lower=0)

# 标准化——聚类前必须做
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 肘部法则选 K
inertias = []
for k in range(1, 11):
    km = SklearnKMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 肘部法则
axes[0][0].plot(range(1, 11), inertias, 'bo-')
axes[0][0].axvline(x=4, color='red', linestyle='--', alpha=0.5)
axes[0][0].set_title('肘部法则 → 建议 K=4')
axes[0][0].set_xlabel('K'); axes[0][0].set_ylabel('惯性')
axes[0][0].grid(True, alpha=0.3)

# 2. 三种算法对比
models = {
    'K-Means': SklearnKMeans(n_clusters=4, n_init='auto', random_state=42),
    'Hierarchical': fcluster(linkage(X_scaled, method='ward'),
                              t=4, criterion='maxclust') - 1,
    'DBSCAN': DBSCAN(eps=1.2, min_samples=5).fit_predict(X_scaled),
}

for j, (name, labels) in enumerate(models.items()):
    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    axes[0][1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10',
                       alpha=0.6, s=30, label=f'{name} (sil={sil:.2f})')
axes[0][1].set_title('三种聚类算法对比 (PCA 2D 投影)')
axes[0][1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
axes[0][1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
axes[0][1].legend(fontsize=8)

# 3. K-Means 最终结果
final_km = SklearnKMeans(n_clusters=4, n_init='auto', random_state=42)
df['Cluster'] = final_km.fit_predict(X_scaled)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for k in range(4):
    mask = df['Cluster'] == k
    axes[1][0].scatter(df.loc[mask, 'Income'],
                       df.loc[mask, 'SpendingScore'],
                       c=colors[k], label=f'簇{k}', s=40, alpha=0.7)
axes[1][0].set_xlabel('年收入（万）'); axes[1][0].set_ylabel('消费评分')
axes[1][0].set_title('客户分群：收入 vs 消费评分')
axes[1][0].legend(); axes[1][0].grid(True, alpha=0.3)

# 4. 各簇画像
cluster_profiles = df.groupby('Cluster').agg(
    avg_age=('Age', 'mean'),
    avg_income=('Income', 'mean'),
    avg_score=('SpendingScore', 'mean'),
    count=('Cluster', 'count')
).round(1)

axes[1][1].axis('off')
table_text = '各簇客户画像\n' + '=' * 40 + '\n'
profile_names = {
    0: '簇0: 年轻消费型 (低龄 低收 中高消费)',
    1: '簇1: 高价值型 (中高龄 高收 中消费)',
    2: '簇2: 节俭成熟型 (中高龄 中收 低消费)',
    3: '簇3: 优质潜力型 (中龄 高收 高消费)',
}
for k in range(4):
    row = cluster_profiles.loc[k]
    table_text += f'\n{profile_names.get(k, f"簇{k}")}\n'
    table_text += f'  人数: {int(row["count"])} | '
    table_text += f'均龄: {row["avg_age"]:.0f}岁 | '
    table_text += f'均收: {row["avg_income"]:.1f}万 | '
    table_text += f'均消费评分: {row["avg_score"]:.0f}\n'
axes[1][1].text(0.1, 0.9, table_text, transform=axes[1][1].transAxes,
                fontsize=9, fontfamily='monospace', verticalalignment='top')

plt.suptitle('商场客户分群 — 聚类分析实战', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

print(f'\n最终轮廓系数: {silhouette_score(X_scaled, df["Cluster"]):.3f}')
print(f'Davies-Bouldin 指数: {davies_bouldin_score(X_scaled, df["Cluster"]):.3f}')
```

**应用连接：** 将客户分成四个可解释的群体后，市场部门可以对每个群体制定策略——"高价值型"推高端产品、"节俭成熟型"推性价比组合、"年轻消费型"推社交活动、"优质潜力型"推交叉销售。精准投放的 ROI 远高于"群发所有人同样的广告"。

---

### 10. 算法对比速查

| 算法 | 需预设 K | 球形假设 | 抗噪声 | 适用规模 | 最佳场景 |
|------|---------|---------|--------|---------|---------|
| **K-Means** | 是 | 是 | 弱 | 大（百万级） | 球形簇、图像压缩、快速探索 |
| **层次聚类** | 否（可后选） | 取决于链接方法 | 弱 | 中小（万级） | 需要层级结构、生物分类、汇报可视化 |
| **DBSCAN** | 否 | 否 | 强（内置噪声检测） | 中（十万级） | 任意形状、含噪声、空间地理数据 |

> **一句话建议：** 不知道数据长什么样？先跑 K-Means 快速摸底，再用 DBSCAN 捕获非球形结构，最后用层次聚类的树状图向团队汇报——三种算法互补，没有一个能包打天下。

---

### 11. 聚类算法选择实战决策树

面对一个新数据集，按以下顺序问自己：

```
START
  │
  ├─ 我知道要分几组吗？
  │   ├─ 是 → K-Means（快速、可解释、工程成熟）
  │   └─ 否 ↓
  │
  ├─ 数据有噪声点需要自动剔除吗？
  │   ├─ 是 → DBSCAN
  │   └─ 否 ↓
  │
  ├─ 需要多层级的分组结构（如"大类→中类→小类"）吗？
  │   ├─ 是 → 层次聚类（树状图是产品经理也能看懂的聚类结果）
  │   └─ 否 ↓
  │
  ├─ 数据量多大？
  │   ├─ > 10万样本 → K-Means（唯一能跑的选项）
  │   ├─ 1万~10万 → K-Means 或 DBSCAN（如果特征 < 20维）
  │   └─ < 1万   → 三种都可以，优先层次聚类（可视化价值高）
  │
  └─ 终极建议：三种都跑一遍，轮廓系数 + 业务直觉共同决定
```

**场景 → 算法 映射表：**

| 实际场景 | 推荐算法 | 原因 |
|---------|---------|------|
| 商场客户 200万条交易记录分群 | K-Means | 数据量太大，只有 K-Means 能跑 |
| 地图上找"城市热点商圈"（不规则形状） | DBSCAN | 商圈是任意形状的密度聚集区 |
| 基因表达谱 500 个样本、20000 个基因 | 先用 PCA 降维到 20 维，再 K-Means | 高维数据聚类前必须先降维 |
| 文档自动分类（20 个新闻组） | 层次聚类（Cosine + Average） | 文档关系天然是层级的 |
| 工业传感器异常检测（99% 正常 + 1% 异常） | DBSCAN 或 Isolation Forest | 异常检测不是标准聚类问题，但 DBSCAN 的噪声标签可用 |

---

### 12. 常见误区与最佳实践

#### 误区 1：聚类前不做标准化

这是新手在 Kaggle 上最常见的翻车姿势。K-Means 的距离计算对特征尺度极度敏感——一个取值范围 [0, 100000] 的"年收入"会完全碾压取值范围 [0, 1] 的"性别"。正确的做法：

```python
from sklearn.preprocessing import StandardScaler
# 永远在 fit 之前做
X_scaled = StandardScaler().fit_transform(X)
```

> 唯一的例外：所有特征已经是同单位、同量纲（如图像像素值 [0, 255]）。

#### 误区 2：只看轮廓系数，不检查簇的实际含义

轮廓系数 0.55 的方案可能把"忠实客户"和"即将流失的客户"混在一起。数值好看 ≠ 业务合理。每次聚类后必须做**特征画像**（每个簇在各特征上的均值/分布），确保每个簇有可解释的差异化特征。

#### 误区 3：对高维数据直接跑 DBSCAN

100 维的数据中，"密度"概念已经崩塌——所有点对距离趋于均等。标准流程是：

```
高维数据 → PCA/UMAP 降至 5~20 维 → DBSCAN
```

如果必须保留原始维度，考虑换用谱聚类（Spectral Clustering）。

#### 误区 4：用聚类做分类

聚类用于发现**未知结构**，不能直接替代分类。有人会用"聚类后给每个簇打标签 → 新数据用最近质心预测"的方式做分类——这本质上是 K-Means 分类器，效果通常不如专门的分类算法（SVM、随机森林等）。聚类 + 人工标注 → 训练分类器 是合理的流程，但直接用聚类做分类是在滥用工具。

#### 误区 5：K 值越大越好

惯性（WCSS）随 K 单调递减，K=N 时 WCSS=0。但 K=N 意味着你把每个数据点当成一个独立的簇——没有发现任何结构，纯属过拟合。聚类追求的是**用最少的簇解释最多的结构**——肘部法则、轮廓系数都在帮你找这个平衡点。

#### 最佳实践清单

| # | 实践 | 为什么 |
|---|------|--------|
| 1 | 聚类前先可视化（散点图矩阵、PCA 2D 投影） | 了解数据的几何结构：球形？带状？含噪声？ |
| 2 | 永远做 StandardScaler | 消除特征量纲差异 |
| 3 | 至少比较 K-Means + DBSCAN 两种算法 | 球形和非球形假设会产生不同视角 |
| 4 | 不只依赖一个评估指标 | 肘部法则 + 轮廓系数 + 业务直觉 三角验证 |
| 5 | 对聚类结果做特征画像 | 验证每个簇是否有合理的业务解释 |
| 6 | 为业务方画树状图或散点图 | 聚类算法对非技术人员是"黑箱"，可视化是你的翻译官 |
| 7 | 小数据（<1000）上用手算验证 | 确认你的理解和大规模跑出来的结果一致 |

---

### 13. 思考题

以下 10 道题由浅入深，覆盖手算验证、直觉理解、实践陷阱和跨算法对比。**建议每题先自己思考 2 分钟再对答案。**

---

**Q1：在手算 K-Means 的例子中，如果初始质心选的是 P1(1,1) 和 P2(2,1)（都在左下角的簇内），算法会收敛到什么结果？WCSS 会是多少？和课文中用 P1, P4 初始化的结果相比如何？**

<details>
<summary>点击查看解答</summary>

初始 C₁=(1,1)，C₂=(2,1)。第 1 次迭代分配：

到 C₁(1,1) 的距离平方：P1=0, P2=1, P3=1, P4=128, P5=145, P6=145
到 C₂(2,1) 的距离平方：P1=1, P2=0, P3=2, P4=(9-2)²+(9-1)²=49+64=113, P5=(10-2)²+(9-1)²=64+64=128, P6=(9-2)²+(10-1)²=49+81=130

比较后：
- P1 → C₁（0 < 1）
- P2 → C₂（1 > 0）
- P3 → C₁（1 < 2）
- P4-P6 全 → C₂（都离 C₂ 更近）

C₁ = {P1, P3}，C₂ = {P2, P4, P5, P6}

更新质心：
- C₁' = ((1+1)/2, (1+2)/2) = (1, 1.5)
- C₂' = ((2+9+10+9)/4, (1+9+9+10)/4) = (30/4, 29/4) = (7.5, 7.25)

接下来 C₁(1, 1.5) 和 C₂(7.5, 7.25) 分别位于原两个真实簇中。经过几轮迭代，P2 最终也会回到 C₁——因为 (2,1) 离 (1, 1.5) 比离 (7.5, 7.25) 近得多。

最终：C₁ = {P1, P2, P3}，C₂ = {P4, P5, P6}，和课文中一样。但达到这个结果需要更多次迭代（因为初始质心不理想），且中间有个 P2 "摇摆"的阶段。WCSS 最终也相同（8/3），但这个初始化路径更曲折——K-Means++ 的价值就是避免这种不必要的周折。

**关键教训：** 不同的初始化可以收敛到同一个最优解（好的情况）或不同的局部最优解（坏的情况）。这就是为什么 sklearn 默认 `n_init=10`——跑 10 次选惯性最小的那次。
</details>

---

**Q2：肘部法则的"肘部"如果不够明显（曲线平滑下降），你还有哪些方法来确定 K？请至少说出两种。**

<details>
<summary>点击查看解答</summary>

1. **轮廓系数（Silhouette Score）**：定量指标，求使轮廓系数最大的 K。不需要肉眼判断"拐点"。

2. **Gap 统计量**：将实际 WCSS 和均匀随机分布的 WCSS 做对比。当实际数据的聚类结构超出随机波动最显著时，Gap 最大。比肘部法则严谨得多。

3. **业务约束**：如果业务方明确说"我们只能运营 N 个客户群"，那 K=N 就是硬约束。统计上稍差一点无关紧要。

4. **稳定性分析**：对数据做多次 Bootstrap 采样，每次跑 K-Means，看不同 K 下聚类结果的稳定性（如 ARI 的方差）。K 选在稳定性最高的位置。

5. **先降维再可视化**：用 PCA 或 t-SNE 把数据降到 2-3 维后直接看图——如果肉眼能看到明显的分离，那就是合理的 K。
</details>

---

**Q3：在手算层次聚类的例子中，如果改用 Complete Linkage（全链）而不是 Single Linkage，树状图会有什么不同？在距离 5 处截断，K 值是否相同？**

<details>
<summary>点击查看解答</summary>

**Complete Linkage** 的簇间距离 = 所有点对距离的**最大值**。

重算合并过程：

步骤 0：{A}, {B}, {C}, {D}, {E}

最小点对距离 = 3（AB 或 CD）。先合并 AB（距离=3）。

- d({A,B}, C) = max(d(A,C), d(B,C)) = max(7, 4) = 7
- d({A,B}, D) = max(10, 7) = 10
- d({A,B}, E) = max(16, 13) = 16

最小簇间距离 = min(7, 10, 16, 3, 9, 6) = 3 → 合并 CD（距离=3）。

合并 CD 后：
- d({C,D}, {A,B}) = max(7, 4, 10, 7) = 10
- d({C,D}, E) = max(9, 6) = 9

最小 = 9 → 合并 {C,D} 和 E（距离=9）。

- d({C,D,E}, {A,B}) = max(16, 13, 7, 4) = 16

最后合并 {A,B} 和 {C,D,E}（距离=16）。

合并顺序：AB(3), CD(3), {C,D}+E(9), {A,B}+{C,D,E}(16)

**和 Single Linkage 的关键差异：**

| | Single | Complete |
|---|---|---|
| 合并 {A,B} 和 {C,D} | 距离 4（很近） | 距离 10（较远） |
| E 的归属 | 和 {A,B,C,D} 合并，距离 6 | 先和 {C,D} 合并，距离 9 |
| 最终合并距离 | 6 | 16 |

在距离 5 处截断：
- **Single**: {A,B,C,D}, {E} → K=2
- **Complete**: {A,B}, {C,D}, {E} → K=3

完全不同！这说明链接准则的选择强烈影响聚类结果——Complete Linkage 更"保守"（要求簇内所有点都互相近），所以倾向于产生更多的小簇。

**延伸：如果改用 Average Linkage？**

Average 取所有点对的平均距离：
- 步骤 0-1 相同：合并 AB(3), CD(3)
- d({A,B}, {C,D})_avg = mean(7,4,10,7) = 7.0
- d({A,B}, E)_avg = mean(16,13) = 14.5
- d({C,D}, E)_avg = mean(9,6) = 7.5
- 最小 = 7.0 → 合并 {A,B}+{C,D}（距离=7.0）
- 最后合并 {A,B,C,D}+E（距离=7.5）

Average 的结果介于 Single 和 Complete 之间——合并顺序略有不同（E 最后才加入），距离跨度更平滑。这就是为什么 Average 在实践中是"最安全"的默认选择——它不像 Single 那么激进（易受链式效应），也不像 Complete 那么保守（可能过度分割）。
</details>

---

**Q4：DBSCAN 中，`min_samples=1` 和 `min_samples=N`（N 为样本总数）分别会发生什么？**

<details>
<summary>点击查看解答</summary>

**min_samples = 1：**
每个点自己就是核心点（因为每个点在自己的 eps-邻域中至少有自己这 1 个点）。所有点在 eps 内有邻居的都会被连成一片——只要数据中任意两点之间存在一条"跳板路径"（每跳 ≤ eps），整个数据集就会变成**一个簇**。纯噪声点（完全孤立的点，eps 邻域只有自己）才会被标为 -1。这基本废掉了 DBSCAN 的密度区分能力。

**min_samples = N：**
没有任何点能成为核心点（因为 eps-邻域内的点数不可能达到 N，除非 eps 无限大使得所有点在彼此的邻域内）。结果：**所有点都是噪声（标签全部 = -1）**。算法退化成一个什么也不做的检测器。

**实践意义：**
- min_samples 太小 → 噪声被当作簇（过分割）
- min_samples 太大 → 簇被当作噪声（欠分割）
- 经验法则：二维数据取 min_samples ≥ 4，高维数据取 min_samples ≥ 2*d（d 为维度）。但最终还是要根据数据的噪声容忍度来调。
</details>

---

**Q5：为什么在聚类之前几乎总是需要做标准化（StandardScaler）？给出一个具体的反例说明不标准化的后果。**

<details>
<summary>点击查看解答</summary>

K-Means 和层次聚类都依赖距离度量。如果特征的尺度差异巨大，范围大的特征会**完全主导**距离计算。

**反例：** 假设你在聚类客户，有两个特征：
- 年龄（Age）：取值范围 [0, 100]
- 年收入（Income）：取值范围 [0, 10000000]（单位：元）

一个 25 岁年收入 50 万的客户和一个 30 岁年收入 48 万的客户，欧氏距离为：
$$d = \sqrt{(25-30)^2 + (500000-480000)^2} \approx \sqrt{25 + 4 \times 10^8} \approx 20000$$

两点在年龄上只差 5 岁，收入上差 2 万，但距离几乎完全由收入差距决定——年龄的 5 个单位差异被 20000 单位的距离淹没了。聚类结果将完全按收入划分，年龄信息被浪费。

标准化后，Age 和 Income 都变成均值 0、标准差 1 的分布，两个特征在距离计算中权重相等——这才是我们想要的行为。
</details>

---

**Q6：轮廓系数接近 0、接近 1、为负值，分别意味着什么？请用几何直觉解释，不列公式。**

<details>
<summary>点击查看解答</summary>

把每个点想象成一个人站在自己簇的"小区"里：

- **s ≈ 1**：这个人紧挨着自己的邻居，离最近的其他小区非常远。他对自己"属于这个小区"非常确信。对应：紧凑簇、大间距。

- **s ≈ 0**：这个人站在两个小区的交界处——他到自家邻居的距离和到隔壁小区的距离差不多。他"属于哪边"很模糊。对应：点在簇边界上，或两簇之间有重叠。

- **s < 0**：这个人离隔壁小区的邻居比自己小区的更近——他被"分错小区"了。对应：该样本应该属于另一个簇，或者它本身就是个离群点。

直观理解：轮廓系数本质上是在问每个点——"你确定自己站对地方了吗？"
</details>

---

**Q7：为什么 DBSCAN 在 100 维数据上几乎无法工作？这和"维度灾难"有什么关系？**

<details>
<summary>点击查看解答</summary>

这是维度灾难（curse of dimensionality）在密度聚类中的经典表现形式。两个原因：

**1. 距离趋于均等：** 在高维空间中，任意两点间的欧氏距离差异趋于消失。所有点对的距离都接近同一个常数（数据分布半径）。这意味着 DBSCAN 的 eps 变得极难设置——eps 稍微大一点，几乎所有点都在彼此的邻域内（一个簇）；eps 稍微小一点，几乎没有点相连（全是噪声）。没有"恰好"的 eps。

**2. 密度概念失效：** DBSCAN 依赖"eps-邻域内点数 ≥ min_samples"来定义核心点。在高维空间，数据极度稀疏（体积随维度指数增长），要想 eps 球包含足够的点，eps 必须非常大——大到"邻域"已经没有局部意义了。

**经验法则：** DBSCAN 在 2-20 维效果最好。超过 20 维，建议：
1. 先用 PCA 或 UMAP 降维再聚类
2. 换用谱聚类（Spectral Clustering）或基于子空间的聚类方法
3. 如果必须用 DBSCAN，尝试用余弦距离代替欧氏距离（余弦距离对高维向量的"方向"敏感，不受幅度影响）

**一个直观的数值验证：** 在 2 维空间中，单位正方形内随机两点距离 < 0.1 的概率约为 3%；在 100 维空间中，这个概率接近于 0（需要用 Monte Carlo 积分才能找到非零的数字）。密度聚类的整个理论框架都建立在"局部密度有定义"的前提上——高维空间恰好把这个前提粉碎了。
</details>

---

**Q8：K-Means 的 WCSS 随着 K 增大一定单调递减。这是否意味着 WCSS 越低聚类越好？为什么不能直接用 WCSS 最小值来选择 K？**

<details>
<summary>点击查看解答</summary>

WCSS 随 K 增加必然递减——极端情况：K=N（每个点自成一个簇），WCSS = 0。但这个"完美"解毫无意义——你只是把每个点都标成了独立的群，没有发现任何分组结构。

这本质上是**过拟合**：你用更多参数（更多质心）来"更好地拟合数据"，但模型泛化能力反而下降。K=1 是欠拟合（一个簇解释一切），K=N 是过拟合（每个点都是自己的簇）。

这就是为什么肘部法则找的是**WCSS 下降速率明显放缓的拐点**，而不是 WCSS 的最小值。拐点前的每个新增 K 都显著改善了聚类质量（把真正需要分开的簇分开了），拐点后的新增 K 只是把已有的大簇"切得更碎"——边际收益递减。
</details>

---

**Q9：在客户分群实战中，你发现 K=3 的轮廓系数（0.42）高于 K=4 的轮廓系数（0.38）。但业务部门说他们需要 4 个用户群来对应四季营销活动。你会选 K=3 还是 K=4？为什么？**

<details>
<summary>点击查看解答</summary>

**选 K=4。** 理由：

1. **聚类是手段，业务是目的。** 轮廓系数 0.42 和 0.38 的差异在实际业务中几乎可以忽略——两者都表明聚类结构合理（>0.25 通常就算可接受了）。但 K=3 vs K=4 对营销策略的影响是结构性的——少一个群意味着有一类客户被混入其他群中，策略无法差异化。

2. **统计最优 ≠ 业务最优。** 统计指标不知道你们的营销团队有 4 个季度的预算和人力配置。如果强行选 K=3，其中 1 个季度需要"借用"另一个群的策略——执行层面可能完全走不通。

3. **可以验证 K=4 的业务可解释性：** 检查 4 个簇是否有各自清晰的业务画像。如果第 4 个簇人数极少（<5%）或和其他簇高度重合（轮廓系数接近 0），才有理由和业务部门讨论是否合并。但如果 4 个簇都有明确的差异化特征——义无反顾选 K=4。

**核心原则：** 算法建议，业务决策。数据科学家的价值不在于算出"最优的统计数字"，而在于把统计结论翻译成业务语言，让业务方做出知情的权衡。
</details>

---

**Q10：请不查资料说出：K-Means 的"均值"（Mean）、层次聚类的"链接准则"（Linkage）、DBSCAN 的"密度"（Density）——三者分别定义了什么是"相似"。这个结构揭示了什么设计哲学？**

<details>
<summary>点击查看解答</summary>

三种算法用三套完全不同的"语言"定义相似：

- **K-Means = 距离到中心的远近**（点-点 → 点-质心）。相似 = 你离质心近，你们就是一类。几何上要求簇是凸的。

- **层次聚类 = 簇间的最值/均值距离**（簇-簇）。相似 = 两个簇按照某种规则（最近点、最远点、平均距离）靠得近，就合并。可以处理任意簇间关系。

- **DBSCAN = 局部密度连通性**（密度可达）。相似 = 你周围足够挤，而且能通过挤的区域走到我这儿——我们就是一家人。完全不需要"中心"或"簇间距"的概念。

**设计哲学：** 每一种聚类算法的核心不是它的优化目标函数，而是它对**"什么是两个样本属于同一群"这个问题的定义**。选算法本质上是在选一种"相似的定义"——这和 KNN 中选择距离度量是同构的思想。

这也解释了为什么没有"最好的聚类算法"：你的数据中，分组是中心的（K-Means 好）、层次的（层次聚类好）、还是密度的（DBSCAN 好）——取决于数据的**几何结构**和你的**业务定义**。理解这三种定义，你就有了选择武器的基础框架。

**进阶思考：** 谱聚类（Spectral Clustering）定义了第四种"相似"——通过图拉普拉斯矩阵的特征向量来发现簇，本质上是在图的谱域中做 K-Means。它能处理 K-Means 和 DBSCAN 都搞不定的"环形中有环形"的结构。如果你对这类问题感兴趣，降维章节中的特征分解会为你打下数学基础。

这个问题的答案实际上给了你一个**算法选择元框架**：打开 sklearn 的 clustering 页面，你能看到 10+ 种算法。不要被数量吓到——它们只是在用不同的"相似定义"回答同一个问题。K-Means 看距离到中心、层次聚类看簇间距离、DBSCAN 看密度连通——理解这三个就够了，其他都是这三个原型的变体或优化。

> **记住这个"聚类三定律"：**
> 1. 质心派：两点相似 = 它们靠近同一个中心（K-Means 及其衍生）
> 2. 层级派：两点相似 = 它们在树中早早就被合并（层次聚类）
> 3. 密度派：两点相似 = 它们可以通过稠密区域一步步走到彼此（DBSCAN）
>
> 你遇到任何聚类问题，第一件事不是写代码，而是坐下来想：**"我的数据中，'同一组'到底意味着什么？"** 答案会自然告诉你该用哪个算法。
</details>

---

### 14. 本章总结：你应该带走的 7 个核心认知

聚类不是一门精确科学，而是一种数据探索工具。就像你用显微镜看细胞——不同倍率（K 值）、不同染色方法（算法）、不同观察角度（距离度量）会给你不同的图像。一个优秀的聚类分析者不是"选出正确算法"的人，而是"能用多种视角看清数据结构"的人。

**七个核心认知：**

1. **聚类 = 无监督发现结构**。和分类不同，没有人告诉你"正确答案"——你自己是裁判。
2. **K-Means = 最快、最简单、最常用**。球形数据 + 标准化 + K-Means++ 初始化 = 80% 问题的第一把利器。
3. **层次聚类 = 最可解释**。树状图是向任何人解释聚类结果的通用语言。不需要讲数学，只需要说"越早合并越相似"。
4. **DBSCAN = 任意形状 + 自动去噪**。你的数据里有噪声、有不规则形状？这是你的武器。但记住维度诅咒。
5. **标准化不是可选项，是必选项**。给 K-Means 未标准化的数据 = 给医生一个刻度错乱的血压计。
6. **指标建议方向，业务拍板决策**。轮廓系数告诉你聚类是否紧凑，业务知识告诉你聚类是否有用。两者缺一不可。
7. **手算 = 真正的理解**。本章两份手算材料（K-Means 3 次迭代 + 层次聚类树状图）如果都亲自算了一遍，你已经比只看书不动笔的人领先了一大步。

**算法选择的终极备忘单：**

| 你的数据长这样 | 用这个 |
|--------------|--------|
| 大致球形的几个团，知道要分几组 | K-Means + StandardScaler + K-Means++ |
| 不知道分几组，想看数据的"家庭树" | 层次聚类（Ward + 树状图截断） |
| 有噪声点 + 形状不规则 + 不知道分几组 | DBSCAN（先画 k-距离图定 eps） |
| 大小差异巨大 + 想要软分配（概率） | GMM |
| 样本超过 100 万 | Mini-Batch K-Means |
| 不知道数据长什么样 | 三种都跑一遍，轮廓系数 + 散点图 + 业务直觉共同决策 |

> **"记住三句话就够了。"** 聚类不是魔法——它只是一种把相似的东西放在一起的方法。K-Means 告诉你"物以类聚"（簇以质心分），层次聚类告诉你"关系有远近"（树状图就是亲疏关系的全景图），DBSCAN 告诉你"挤在一起的就是一家"（不管什么形状）。三种算法三个哲学视角，对应三套数据语言。花时间理解它们对"同一组"的定义差异，比死记任何公式都重要。

- 维度超过 50？先降维再聚类；不知道选什么降维方法？PCA 是永远的安全选择。
- 聚类结果和你的直觉不一致？不要急着换算法——先检查数据是否标准化了、特征是否合理、K 是否真的选对了。
- 最后一句话：**跑聚类之前，先用散点图看一眼你的数据。** 散点图 + 领域经验 > 任何自动化选 K 的工具。

---

聚类只是无监督学习的冰山一角。我们已经学会了把数据分成几个有意义的群——但如果数据的维度太高（几十、几百、几千维），聚类之前常见的做法是先**降维**。下一章：[降维](./14-dimensionality-reduction.md) — PCA、t-SNE 以及"为什么 2D 可视化上的簇看起来那么好，但实际聚类效果很差"。
