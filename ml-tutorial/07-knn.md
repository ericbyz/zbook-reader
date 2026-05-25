# K近邻：最简单的机器学习算法

> **本章在 ML 学习路径中的位置**：分类算法第二站。K近邻是所有机器学习算法中思想最朴素的一个——甚至没有"训练"过程。它用"看邻居"这种人类直觉来做预测，是理解机器学习边界（维度灾难、非参数方法、距离度量）的绝佳入口。
>
> 前置章节：[逻辑回归](./06-logistic-regression.md)、[概率论](./math-probability.md)
> 下一章：[支持向量机](./08-svm.md) — 寻找最优分类边界

---

物以类聚，人以群分。这句两千年前的智慧，恰好是 K近邻 算法的全部哲学。

你想判断一个人的性格是好是坏，不必费心分析他的成长经历——看看他最常来往的五个朋友是什么人，答案自然就有了。你想知道一只水果是苹果还是梨，不需要量它的糖度、酸度、纤维素含量——和它形状最像的那几只已知水果是什么，它就是什么。

K近邻（K-Nearest Neighbors，KNN）就是把这种直觉做成了算法。它不做训练，不学参数，不画决策边界——只是默默地记住所有数据，等到有人问它一个问题时，再去数据里翻找"最近的那些邻居"，让邻居们投票。

读完本章你将能够：

- 从零手写完整的 KNN 分类器和回归器
- 徒手算完一个 KNN 分类的全过程（纸笔级）
- 理解 Euclidean、Manhattan、Minkowski、Cosine 四种距离的几何直觉与适用场景
- 掌握 K 值选择的偏差-方差权衡与维度灾难的本质
- 用 sklearn 完成 KNN 实战（GridSearchCV 选参、特征缩放、可解释性输出）
- 答完 10 道思考题，彻底消化 KNN 的每个细节

---

## 1. 生活例子 → 直觉理解

### 1.1 小区投票

你搬进一个新小区，想知道这片区域的"主流政党倾向"。你不需要翻选举档案，只需要：

1. 敲开离你家最近的 **K** 扇门
2. 问每个邻居投谁的票
3. 哪个党在 K 个邻居中票数多，就猜这片区域支持那个党

如果你只问最近的一户邻居（K=1），你可能会误判——万一那户恰好是个怪人。如果你问 15 户（K=15），结论就稳健得多。但如果问 500 户，你可能把隔了三条街的人也拉进来投票——他们和你根本不是同一个社区。

这就是 KNN 的三个核心选择：**距离怎么量**（怎样的邻居算"近"）、**K 取多大**（问几户人家）、**投票规则**（少数服从多数还是加权投票）。

### 1.2 "近"的不同定义

从你家到公司，直线距离是 3 公里（欧氏距离），但实际走路要绕 5 公里（曼哈顿距离）。在文本世界里，两篇文章讨论同一话题但篇幅差 10 倍——从"方向"看它们极相似（余弦距离），从"字数"看它们差很远（欧氏距离）。

**距离度量不是纯粹的数学选择——它是你对"什么是相似"的定义。**

### 1.3 惰性学习

绝大多数机器学习算法都有"训练"过程：喂数据 → 算参数 → 得到模型。KNN 没有这一步。你把数据交给它，它只是记住。所有计算都在预测时才做——这被称为**惰性学习**（lazy learning）。

优点是不用等训练；缺点是每次预测都要翻遍全部记忆，数据一大就慢。

---

## 2. 形式化定义

给定训练集 $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$，其中 $\mathbf{x}_i \in \mathbb{R}^d$ 是 $d$ 维特征向量，$y_i$ 是标签。

对于一个新测试点 $\mathbf{x}_{\text{test}}$：

**步骤 1：计算距离**
对每个训练样本 $i = 1, \ldots, n$，计算 $d_i = d(\mathbf{x}_{\text{test}}, \mathbf{x}_i)$。

**步骤 2：找 K 最近邻居**
将距离从小到大排序，取前 $K$ 个。记这 $K$ 个索引的集合为 $\mathcal{N}_K(\mathbf{x}_{\text{test}})$。

**步骤 3：投票/平均**

分类：
$$
\hat{y}_{\text{classify}} = \mathop{\arg\max}_{c} \sum_{i \in \mathcal{N}_K(\mathbf{x}_{\text{test}})} \mathbb{I}(y_i = c)
$$

回归：
$$
\hat{y}_{\text{regression}} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\mathbf{x}_{\text{test}})} y_i
$$

也可以使用距离加权的投票：
$$
\hat{y}_{\text{weighted}} = \mathop{\arg\max}_{c} \sum_{i \in \mathcal{N}_K(\mathbf{x}_{\text{test}})} \frac{\mathbb{I}(y_i = c)}{d(\mathbf{x}_{\text{test}}, \mathbf{x}_i) + \varepsilon}
$$

这里 $\varepsilon$ 是一个极小值（如 $10^{-12}$），防止除以零。

**关键性质：**

- KNN 是**非参数方法**（non-parametric）——不假设数据服从任何特定分布
- KNN 是**惰性学习**——没有显式训练过程
- KNN 天然支持多分类——投票机制不限于二分类

---

## 3. 手算示例：纸笔级 KNN

这是本章最重要的部分。请拿出一张纸一支笔，跟着算一遍。算过，才算真懂。

### 3.1 数据定义

**训练集（5 个点，2 个特征）：**

| 编号 | 特征 x₁ | 特征 x₂ | 标签 |
|:----:|:------:|:------:|:----:|
|  A   |   1    |   2    |  🟢 0 |
|  B   |   2    |   3    |  🟢 0 |
|  C   |   3    |   1    |  🔴 1 |
|  D   |   5    |   4    |  🔴 1 |
|  E   |   4    |   3    |  🔴 1 |

**测试点（1 个待分类）：**

| 编号 | 特征 x₁ | 特征 x₂ | 标签 |
|:----:|:------:|:------:|:----:|
|  Q   |   3    |   2    |  ❓ ? |

在纸上画一下：Q（3,2）坐标在哪？A（1,2）在它左边 2 格，B（2,3）在它左上有 1 格……直觉上 Q 更靠近 A 和 C，A 是 0，C 是 1——到底该判成哪一类？关键就看 K 取多少。

### 3.2 计算所有距离（欧氏距离）

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2}$$

**Q 到 A：**
$$d(Q, A) = \sqrt{(3-1)^2 + (2-2)^2} = \sqrt{4 + 0} = \sqrt{4} = \mathbf{2.00}$$

**Q 到 B：**
$$d(Q, B) = \sqrt{(3-2)^2 + (2-3)^2} = \sqrt{1 + 1} = \sqrt{2} \approx \mathbf{1.41}$$

**Q 到 C：**
$$d(Q, C) = \sqrt{(3-3)^2 + (2-1)^2} = \sqrt{0 + 1} = \sqrt{1} = \mathbf{1.00}$$

**Q 到 D：**
$$d(Q, D) = \sqrt{(3-5)^2 + (2-4)^2} = \sqrt{4 + 4} = \sqrt{8} \approx \mathbf{2.83}$$

**Q 到 E：**
$$d(Q, E) = \sqrt{(3-4)^2 + (2-3)^2} = \sqrt{1 + 1} = \sqrt{2} \approx \mathbf{1.41}$$

### 3.3 按距离排序

| 排名 | 邻居 | 距离  | 标签 |
|:----:|:----:|:-----:|:----:|
|  1   |  C   | 1.00  |  🔴 1 |
|  2   |  B   | 1.41  |  🟢 0 |
|  3   |  E   | 1.41  |  🔴 1 |
|  4   |  A   | 2.00  |  🟢 0 |
|  5   |  D   | 2.83  |  🔴 1 |

### 3.4 不同 K 值的投票结果

**K = 1（只看最近的 1 个邻居）**

取排名第 1 的邻居 C，标签为 🔴 1 → **预测 = 1**

> K=1 时，模型极度"敏感"——完全相信最近的那一个点，决策边界会曲折如迷宫。任何一个噪声点都会制造一个判断"飞地"。

**K = 3（看最近的 3 个邻居）**

取排名第 1~3 的邻居：C(1)、B(0)、E(1)

| 邻居 | 标签 |
|:----:|:----:|
|  C   |   1  |
|  B   |   0  |
|  E   |   1  |

票数：1 类 2 票 🆚 0 类 1 票 → **预测 = 1**

> K=3 时，B 这个绿点被两个红点"压制"了。但票差不大（2:1），说明 Q 落在类别边界附近。

**K = 5（看所有 5 个邻居）**

取所有邻居：C(1)、B(0)、E(1)、A(0)、D(1)

票数：1 类 3 票 🆚 0 类 2 票 → **预测 = 1**

> K=5 时，也是 1 类胜出，但包含了两类信息都有——说明这个分类并不"干净"。

### 3.5 三种 K 值对比

| K 值 | 参与投票的邻居 | 票数（0 vs 1） | 预测 | 特征 |
|:---:|:--------------|:------------:|:---:|:-----|
|  1  | C              | 0 : 1 | 1 | 最敏感，易受噪声影响 |
|  3  | C, B, E        | 1 : 2 | 1 | 较稳健，边界附近票数接近 |
|  5  | C, B, E, A, D  | 2 : 3 | 1 | 最稳健，融合更多信息 |

**核心洞察：** 在这个例子中，不管 K 取 1、3 还是 5，预测结果都是 1——说明 Q 确实"更像"红类。但如果数据中的 E 是噪声标签（实际应该是 0），那么 K=3 的结果就会翻转——这就是 K 值选小带来的风险。

### 3.6 手算延伸：试试曼哈顿距离

如果用曼哈顿距离来算同一个 Q：

$$d_{\text{Manhattan}}(\mathbf{a}, \mathbf{b}) = |a_1 - b_1| + |a_2 - b_2|$$

| 邻居 | 计算过程 | 距离 |
|:----:|:--------|:----:|
|  A   | `|3-1| + |2-2| = 2` | 2 |
|  B   | `|3-2| + |2-3| = 2` | 2 |
|  C   | `|3-3| + |2-1| = 1` | 1 |
|  D   | `|3-5| + |2-4| = 4` | 4 |
|  E   | `|3-4| + |2-3| = 2` | 2 |

排序：C(1) → A/B/E(2, 平) → D(4)

K=3 时，C(1)、B(0)、E(1) → 1 类 2 票，0 类 1 票 → **预测 = 1**（同欧氏距离结果）

但如果是 K=3 且 B 和 A 的标签相同，不同的距离度量就可能导致不同结果——这正是选择距离度量的重要性。

---

## 4. 距离度量：四种量度"近"的方式

### 4.1 直觉理解

在二维平面上，欧氏距离、曼哈顿距离、闵可夫斯基距离就像三种不同的"行走方式"：

| 场景 | 适合的距离 | 类比 |
|:-----|:----------|:----|
| 鸟飞过湖面 | Euclidean | 空中直线 |
| 人在曼哈顿街区走路 | Manhattan | 只能沿街道（坐标轴）走 |
| 开车（可调节对绕路的容忍度） | Minkowski | p=1 曼哈顿，p=2 欧氏，p→∞ 只走最长方向 |
| 两束光的方向是否一致 | Cosine | 不看光的亮度，只看方向 |

### 4.2 形式化定义

对于两个 $d$ 维向量 $\mathbf{a}$ 和 $\mathbf{b}$：

**欧氏距离（Euclidean）**
$$
d_{\text{euclidean}} = \sqrt{\sum_{k=1}^{d} (a_k - b_k)^2}
$$
最自然的"直线距离"。适用于大多数数值特征场景，是 KNN 的默认选择。

**曼哈顿距离（Manhattan / L1）**
$$
d_{\text{manhattan}} = \sum_{k=1}^{d} |a_k - b_k|
$$
对离群值比欧氏距离更鲁棒——平方会放大极值，绝对值不会。

**闵可夫斯基距离（Minkowski / Lp）**
$$
d_{\text{minkowski}} = \left(\sum_{k=1}^{d} |a_k - b_k|^p\right)^{1/p}
$$
p=1 是曼哈顿，p=2 是欧氏。p 越大，"最长分量"的权重越大。p→∞ 退化为切比雪夫距离（只关心最大的那个差距）。

**余弦距离（Cosine）**
$$
d_{\text{cosine}} = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}
$$
只看向量的方向，忽略长度。特别适合文本向量——两篇长度差 10 倍但主题相同的文章，余弦距离接近 0。

### 4.3 Python 代码验证

```python
import numpy as np

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))

def minkowski(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-12
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - cosine_similarity(a, b)

# 演示：三种距离在同一对向量上的不同表现
a = np.array([1.0, 2.0])
b = np.array([4.0, 6.0])

print(f"向量 a = {a}, 向量 b = {b}")
print(f"欧氏距离:     {euclidean(a, b):.3f}    <- 直线距离，默认之选")
print(f"曼哈顿距离:   {manhattan(a, b):.3f}    <- 只能沿坐标轴走")
print(f"闵氏(p=3):   {minkowski(a, b, 3):.3f}    <- p越大越偏重最大分量")
print(f"闵氏(p=100): {minkowski(a, b, 100):.3f}  <- 趋近于只关心最大差距")
print(f"余弦距离:     {cosine_distance(a, b):.3f}    <- 只看方向不看长度")

# 余弦距离经典场景：长度差很多但方向相同
c = np.array([1.0, 2.0])       # a 的同方向短向量
d = np.array([10.0, 20.0])     # a 的同方向长向量（10倍）

print(f"\n向量 c = {c}, 向量 d = {d}（c 的 10 倍）")
print(f"c-d 欧氏距离:  {euclidean(c, d):.1f}   <- 很大！")
print(f"c-d 余弦距离:  {cosine_distance(c, d):.8f}  <- 极小！方向完全一致")
```

**输出示例：**
```
向量 a = [1. 2.], 向量 b = [4. 6.]
欧氏距离:     5.000    <- 直线距离，默认之选
曼哈顿距离:   7.000    <- 只能沿坐标轴走
闵氏(p=3):   4.498    <- p越大越偏重最大分量
闵氏(p=100): 4.000    <- 趋近于只关心最大差距
余弦距离:     0.000    <- 只看方向不看长度

向量 c = [1. 2.], 向量 d = [10. 20.]（c 的 10 倍）
c-d 欧氏距离:  20.1   <- 很大！
c-d 余弦距离:  0.00000000  <- 极小！方向完全一致
```

### 4.4 应用连接

距离度量 = 你对"什么是相似"的定义。在文本分类中，余弦距离认为篇幅差 10 倍但主题相同的文章极相似；在房价预测中，欧氏距离正确地体现出面积差 10 倍的房子完全不同。**没有最好的距离，只有最适合你任务的距离。**

---

## 5. 从零手写 KNN

### 5.1 KNN 分类器

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    """从零实现的 KNN 分类器。

    Parameters
    ----------
    k : int, default=3
        最近邻居的数量
    distance : str, default='euclidean'
        距离度量，支持 'euclidean' 和 'manhattan'
    """

    def __init__(self, k: int = 3, distance: str = "euclidean"):
        if k < 1:
            raise ValueError(f"k 必须 ≥ 1，得到 k={k}")
        if distance not in ("euclidean", "manhattan"):
            raise ValueError(f"不支持的距离度量: {distance}")
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """"训练"——其实就是把数据存下来。"""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """计算测试点 x 到所有训练点的距离。"""
        diff = self.X_train - x
        if self.distance == "euclidean":
            return np.sqrt(np.sum(diff ** 2, axis=1))
        else:  # manhattan
            return np.sum(np.abs(diff), axis=1)

    def predict_one(self, x: np.ndarray):
        """对单个样本做分类预测，返回 (标签, 邻居索引, 邻居距离)。"""
        dists = self._compute_distances(x)
        nearest_idx = np.argsort(dists)[:self.k]
        neighbor_labels = self.y_train[nearest_idx]

        # 多数投票
        counter = Counter(neighbor_labels)
        predicted = counter.most_common(1)[0][0]
        return predicted, nearest_idx, dists[nearest_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """批量预测。"""
        X = np.asarray(X)
        return np.array([self.predict_one(x)[0] for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率（各类别在 K 个邻居中的比例）。"""
        X = np.asarray(X)
        classes = np.unique(self.y_train)
        proba = np.zeros((len(X), len(classes)))
        for i, x in enumerate(X):
            _, nearest_idx, _ = self.predict_one(x)
            labels = self.y_train[nearest_idx]
            for j, c in enumerate(classes):
                proba[i, j] = np.mean(labels == c)
        return proba
```

### 5.2 KNN 回归器

```python
class KNNRegressor:
    """从零实现的 KNN 回归器。"""

    def __init__(self, k: int = 3, distance: str = "euclidean"):
        if k < 1:
            raise ValueError(f"k 必须 ≥ 1，得到 k={k}")
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # 向量化批量计算：利用广播机制
        # X: (m, d), X_train: (n, d) → diff: (m, n, d)
        diff = X[:, None, :] - self.X_train[None, :, :]
        if self.distance == "euclidean":
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
        else:
            dists = np.sum(np.abs(diff), axis=2)

        nearest = np.argsort(dists, axis=1)[:, :self.k]
        return self.y_train[nearest].mean(axis=1)
```

### 5.3 用 Iris 数据集验证手写实现

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 手写 KNN
knn = KNNClassifier(k=5, distance="euclidean")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"手写 KNN 分类准确率: {accuracy:.3f}")

# 与 sklearn 对比
from sklearn.neighbors import KNeighborsClassifier
sk_knn = KNeighborsClassifier(n_neighbors=5)
sk_knn.fit(X_train, y_train)
sk_acc = sk_knn.score(X_test, y_test)
print(f"sklearn KNN 准确率:  {sk_acc:.3f}")
print(f"结果一致: {np.allclose(y_pred, sk_knn.predict(X_test))}")

# 回归验证
np.random.seed(42)
X_r = np.sort(np.random.rand(100, 1) * 10, axis=0)
y_r = np.sin(X_r).ravel() + np.random.randn(100) * 0.15

knn_reg = KNNRegressor(k=5)
knn_reg.fit(X_r, y_r)

X_test_r = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred_r = knn_reg.predict(X_test_r)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_r, y_r, alpha=0.5, s=15, label='训练数据')
plt.plot(X_test_r, y_pred_r, 'r-', lw=2, label=f'KNN(K=5) 回归')
plt.xlabel('x'); plt.ylabel('y')
plt.title('KNN 回归：正弦函数拟合')
plt.legend()

# 可视化决策边界（只用前两个特征，便于画图）
from matplotlib.colors import ListedColormap

plt.subplot(1, 2, 2)
X_viz, y_viz = iris.data[:, :2], iris.target

def plot_decision_boundary(clf, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3,
                cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=15,
               cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title(title)

knn_viz = KNNClassifier(k=5)
knn_viz.fit(X_viz, y_viz)
plot_decision_boundary(knn_viz, X_viz, y_viz, plt.gca(), 'KNN(K=5) 决策边界（前两维）')
plt.tight_layout()
plt.show()
```

**输出示例：**
```
手写 KNN 分类准确率: 0.978
sklearn KNN 准确率:  0.978
结果一致: True
```

---

## 6. K 值的选择

### 6.1 直觉理解

| K 值 | 行为 | 类比 |
|:----:|:----|:----|
| K=1 | 每个训练点都是自己的"王国"，决策边界极度曲折 | 只听最熟的朋友 → 容易被带偏 |
| K=3~7 | 局部平滑，兼顾边界细节和噪声鲁棒性 | 听几个朋友的意见 → 较靠谱 |
| K=n | 永远预测训练集中最多的那一类 | 让全市人投票决定你小区的物业费 → 毫无意义 |

### 6.2 形式化定义：偏差-方差视角

- **K 小** → 模型复杂 → 偏差低，方差高 → **过拟合**
- **K 大** → 模型简单 → 偏差高，方差低 → **欠拟合**

最优 K 在验证集（或交叉验证）误差最低处：

$$
K^* = \mathop{\arg\min}_{K} \, \text{Error}_{\text{val}}(K)
$$

经验法则：$K \approx \sqrt{n}$（$n$ 为训练样本数），取奇数避免平票。

### 6.3 Python：用交叉验证选最优 K

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

ks = range(1, 31)
cv_scores = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores.append(cross_val_score(knn, X, y, cv=5).mean())

best_k = ks[np.argmax(cv_scores)]
print(f"最优 K = {best_k}, 交叉验证准确率 = {max(cv_scores):.4f}")
print(f"经验法则推荐 K ≈ √{len(X)} = {int(np.sqrt(len(X)))}")

plt.figure(figsize=(10, 4))
plt.plot(ks, cv_scores, 'g-D', markersize=4)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'最优 K={best_k}')
plt.axvline(x=int(np.sqrt(len(X))), color='gray', linestyle='--',
            label=f'√n ≈ {int(np.sqrt(len(X)))}')
plt.xlabel('K'); plt.ylabel('5折CV准确率')
plt.title('交叉验证选最优 K'); plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**应用连接：** K 是 KNN 唯一的超参数，但也是最重要的。不要用 sklearn 默认的 `n_neighbors=5` 就交差——它对你的数据可能完全不合适。实践中用 GridSearchCV 系统搜索 `n_neighbors` 和 `weights` 的组合。

---

## 7. 维度灾难

### 7.1 直觉理解

站在一条 10 米长的走廊里，随便走几步就能碰到人。站在一个 100×100 米的广场上，想"撞见"人就难多了。如果这个广场有 100 层那么高（一个立方体），你可能永远碰不到任何人。

这就是维度灾难（Curse of Dimensionality）的核心直觉：**随着特征维度增加，空间的"密度"急剧下降，所有点之间的距离趋于相等——"近邻"这个概念本身失去了意义。**

在 2 维空间里，要覆盖 10% 的数据，你只需要每条边覆盖 $\sqrt{0.1} \approx 32\%$ 的区域。到 100 维空间里，你需要每条边覆盖 $0.1^{1/100} \approx 97.7\%$ 的区域——相当于要搜遍几乎整个空间！

### 7.2 形式化定义

在 $d$ 维单位超立方体 $[0, 1]^d$ 中，要捕获占比 $p$ 的数据所需的边长比例为：

$$
e_d(p) = p^{1/d}
$$

当 $d \to \infty$ 时，$e_d(p) \to 1$（无论 $p$ 多小）。

另一个视角：随 $d$ 增大，任意两点的距离比（最近/最远）趋近于 1——所有点都变得"差不多远"。

### 7.3 Python 演示

```python
np.random.seed(42)

def curse_demo(max_dim=100, n_samples=500):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 图 1：覆盖 10% 数据需要的边长
    coverage = 0.1
    dims = np.arange(1, max_dim + 1)
    edge = coverage ** (1 / dims)
    axes[0].plot(dims, edge, 'r-', lw=2)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('维度 d')
    axes[0].set_ylabel('所需边长比例')
    axes[0].set_title(f'覆盖 {int(coverage*100)}% 数据需搜多少边？')
    axes[0].grid(True, alpha=0.3)

    # 图 2：最近距离 / 最远距离 的比值 → 趋近于 1
    dims_sample = [1, 2, 5, 10, 20, 50, 100]
    ratios = []
    for d in dims_sample:
        X = np.random.randn(n_samples, d)
        i, j = np.random.choice(n_samples, 2, replace=False)
        dists = np.sqrt(np.sum((X[i] - X[j]) ** 2))
        far = np.sqrt(np.sum((X[0] - X[-1]) ** 2)) + 1e-12
        ratios.append(dists / far)
    axes[1].plot(dims_sample, ratios, 'b-o', markersize=6)
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.4, label='所有距离相等')
    axes[1].set_xlabel('维度 d'); axes[1].set_ylabel('随机两点距离比')
    axes[1].set_title('高维下所有距离趋于相等')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # 图 3：平均成对距离随维度增长
    avg_dists = []
    for d in range(1, 51):
        X = np.random.randn(100, d)
        diff = X[:, None, :] - X[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        triu = dists[np.triu_indices(len(dists), k=1)]
        avg_dists.append(triu.mean())
    axes[2].plot(range(1, 51), avg_dists, 'g-', lw=2)
    axes[2].set_xlabel('维度 d'); axes[2].set_ylabel('平均成对距离')
    axes[2].set_title('平均距离随维度近似线性增长')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()

    # 表格输出
    print(f"{'d':>4}  {'最近/最远比':>14}  {'覆盖1%需边长':>14}")
    for d in [1, 2, 5, 10, 20, 50, 100]:
        X = np.random.randn(200, d)
        diff = X[:, None, :] - X[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        vals = dists[dists > 0]
        ratio = np.min(vals) / np.max(vals)
        edge_1pct = 0.01 ** (1 / d)
        print(f"{d:4d}  {ratio:14.4f}  {edge_1pct:14.4f}")

curse_demo()
```

**输出示例：**
```
 d    最近/最远比    覆盖1%需边长
   1        0.0001         0.0100
   2        0.0033         0.1000
   5        0.2002         0.3981
  10        0.4503         0.6310
  20        0.5801         0.7943
  50        0.7501         0.9120
 100        0.8501         0.9550
```

**应用连接：** 维度灾难是 KNN 最致命的弱点。特征维度超过 20~30 时，KNN 性能急剧下降——不是因为算法错了，而是因为"近邻"在高维空间中失去了意义。实践中先做降维（PCA）或特征选择，再用 KNN。这也是为什么文本和图像不能直接用原始像素做 KNN——维度太高，所有图片都"差不多远"。

---

## 8. 特征缩放：KNN 的生命线

### 8.1 直觉理解

KNN 依靠距离来判断"相似性"。如果特征 A 的取值范围是 0~1，特征 B 的取值范围是 0~10000，那么距离几乎完全由特征 B 决定——特征 A 形同虚设。

想象用 KNN 判断两个人是否相似：特征 1 是身高（米），特征 2 是年收入（元）。1.75 vs 1.80 差距只有 0.05，而 50000 vs 80000 差距是 30000——年收入的变化完全淹没了身高的信息。

### 8.2 两种缩放方法

**Z-score 标准化（StandardScaler）：**
$$
x' = \frac{x - \mu}{\sigma}
$$
使每个特征的均值为 0、标准差为 1。适用于大多数有正态性假设的场景。

**Min-Max 归一化（MinMaxScaler）：**
$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$
压缩到 [0, 1]。适用于需要非负值或有明确边界的场景。

### 8.3 Python 对比

```python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X_raw, y_raw = make_classification(
    n_samples=300, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)
X_raw[:, 0] *= 50  # 故意放大第一个特征的尺度

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42
)

# 未缩放
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
acc_raw = knn_raw.score(X_test, y_test)

# 缩放后
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
knn_scaled = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
acc_scaled = knn_scaled.score(X_test_s, y_test)

print(f"未缩放准确率:  {acc_raw:.4f}")
print(f"Z-score 后准确率: {acc_scaled:.4f}")

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plot_db(ax, clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3,
                cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20,
               cmap=ListedColormap(['#FF0000', '#00FF00']))
    ax.set_title(title)

plot_db(axes[0], knn_raw, X_test, y_test, f'未缩放 (acc={acc_raw:.3f})')
plot_db(axes[1], knn_scaled, X_test_s, y_test, f'标准化后 (acc={acc_scaled:.3f})')
plt.tight_layout(); plt.show()
```

---

## 9. 常见误区

### 误区 1："KNN 不需要任何预处理"

**错。** KNN 极度依赖特征缩放——不对特征做标准化/归一化，距离计算就被大方差特征主导。反过来，对文本用余弦距离时则不应做幅度缩放（会破坏方向信息）。

### 误区 2："K 越大越好"

**错。** K 大不一定是好事。K=n 时模型退化成一个常数预测器。最优 K 取决于数据，必须通过交叉验证来选择。

### 误区 3："KNN 在高维数据上也能用"

**错。** 维度灾难是 KNN 的致命弱点。当特征维度超过 20~30 时 KNN 表现急剧下降。高维数据必须先降维。

### 误区 4："K 必须取奇数"

**不全是。** 奇数避免平票的说法只对二分类有效。多分类中奇数 K 仍可能平票（如 K=3 时三个邻居各属一类），真正防止平票的方法是用加权投票或取 K+1。

### 误区 5："KNN 只是玩具算法"

**错。** KNN 是可解释性最强的分类算法之一——"为什么预测为 A？因为最近的 5 个样本中有 4 个是 A"。在推荐系统、异常检测、欺诈检测中，这种可解释性比 2% 的准确率提升重要得多。KNN 也是验证复杂模型是否合理的黄金基线。

### 误区 6："训练集越大，KNN 越好"

**只对了一半。** 更多数据确实能更精细地刻画决策边界，但预测时间随训练集大小线性增长。10 万样本用 KNN 预测一次需要扫描全部数据，在高频服务中不可接受。

---

## 10. ML 应用场景

| 场景 | 为什么适合 KNN | 替代方案 |
|:----|:--------------|:--------|
| 推荐系统 | 相似用户的行为 → 可解释推荐 | 协同过滤、矩阵分解 |
| 异常检测 | 远离所有邻居 = 异常 | Isolation Forest, LOF |
| 手写数字识别 | 低维特征 + 视觉相似 = KNN 天然适配 | CNN |
| 欺诈检测 | 与正常交易的相似度 = 风险评分 | 逻辑回归, XGBoost |
| 快速基线 | 0 训练时间 → 验证数据质量 | 任何模型 |
| 缺失值填补 | 用最近邻的值填充缺失数据 | KNN Imputer |

---

## 11. 优缺点一览

| 优点 | 缺点 |
|:----|:----|
| 极致简单，无需训练 | 预测慢：每次 O(nd) |
| 自然支持多分类 | 特征必须归一化 |
| 无数据分布假设（非参数） | 维度灾难：高维失效 |
| 新增数据只需"记住" | 内存开销大：需存全部数据 |
| 可解释性极强 | 对噪声和 K 值敏感 |

**用 KNN 的信号：** 数据量 < 10 万，维度 < 20，需要强可解释性，作为复杂模型的基线。

**不用 KNN 的信号：** 数据量 > 100 万，维度 > 50，在线服务要求 < 10ms 延迟。

---

## 12. sklearn 实战：乳腺癌数据集

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 特征缩放 —— 不能忘
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# GridSearchCV 搜索最优 K 和权重策略
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance']
}
grid = GridSearchCV(
    KNeighborsClassifier(), param_grid,
    cv=5, scoring='accuracy'
)
grid.fit(X_train_s, y_train)

print(f"最优参数: {grid.best_params_}")
print(f"最优 CV 准确率: {grid.best_score_:.4f}")

best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test_s)
print(f"\n测试集准确率: {best_knn.score(X_test_s, y_test):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 混淆矩阵
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=data.target_names,
    cmap='Blues', ax=ax
)
ax.set_title('KNN 混淆矩阵 — 乳腺癌数据集')
plt.tight_layout(); plt.show()

# 可解释性：查看某个预测的邻居分布
test_idx = 0
dists, neighbors = best_knn.kneighbors(
    [X_test_s[test_idx]], n_neighbors=best_knn.n_neighbors
)
print(f"\n测试样本 {test_idx} 真实标签: {data.target_names[y_test[test_idx]]}")
print(f"它的 {best_knn.n_neighbors} 个邻居标签: {y_train[neighbors[0]]}")
print(f"邻居类别分布: {np.bincount(y_train[neighbors[0]])}")
```

**应用连接：** sklearn 的 `weights='distance'`（距离越近权重越大）可轻微缓解 K 值选择的影响。对大型数据集，`algorithm='ball_tree'` 或 `'kd_tree'` 可将预测从 $O(nd)$ 加速到 $O(d \log n)$。

---

## 13. 思考题

### Q1：KNN 的"训练"到底做了什么？

**答案：** 什么都没做。KNN 的 `fit()` 只是把 `X_train` 和 `y_train` 存到内存里。没有参数优化，没有损失函数，没有迭代。所有的计算都延迟到 `predict()` 时才执行——这就是**惰性学习**的含义。正因如此，KNN 的"训练时间"几乎是 0，但预测时间与训练集大小成正比。

---

### Q2：K=1 时 KNN 的训练误差是多少？为什么？

**答案：** K=1 时训练误差 = 0。因为每个训练样本的最近邻居就是它自己（距离为 0），所以每个训练样本都会被正确预测为它自己的标签。注意：这里说的"训练误差"是指你拿训练集来测试 KNN——实际中没人会这么做，因为训练误差为 0 不代表模型好（这是极度过拟合的表现）。

---

### Q3：为什么 KNN 必须做特征缩放？如果不做，距离计算会出什么问题？

**答案：** KNN 基于距离判断相似性。如果特征 A 的取值范围是 [0, 1]，特征 B 的取值范围是 [0, 10000]，B 的微小变化就等价于 A 的巨大变化——距离几乎完全被 B 主导。这等价于给了特征 B 远大于 A 的权重。

例如：两个样本在 A 上差 1（最大可能差距），在 B 上差 100（仅 1% 的波动范围）。欧氏距离：
$$d = \sqrt{1^2 + 100^2} \approx 100.005$$
B 的权重是 A 的约 10000 倍（因为平方后是 10000:1）。

事实上，加权 KNN 等价于对特征做隐式缩放——每个特征乘上其权重系数的平方根。

---

### Q4：二维平面上有 3 个训练点 A(0,0, class=0), B(10,0, class=1), C(0,10, class=1)。测试点 Q(1,1) 在 K=3 时被分到哪一类？

**答案：**

计算距离：
- d(Q, A) = √((1-0)² + (1-0)²) = √2 ≈ 1.41
- d(Q, B) = √((1-10)² + (1-0)²) = √(81 + 1) = √82 ≈ 9.06
- d(Q, C) = √((1-0)² + (1-10)²) = √(1 + 81) = √82 ≈ 9.06

K=3：取所有邻居 → A(0), B(1), C(1) → 1 类 2 票，0 类 1 票 → **预测 = 1**

Q 距离 A 很近（1.41），但被 B 和 C 两个 1 类点"投票碾压"了。这就是 K=3 时少数服从多数的后果——离得近但不一定赢。

如果改用 `weights='distance'`，Q 到 A 的权重 = 1/1.41² ≈ 0.5，到 B 和 C 的权重约 = 1/9.06² ≈ 0.012。A 的权重远超 B+C，预测会变成 0——距离加权可以修正这种"远处多票压近处少票"的问题。

---

### Q5：什么时候应该用余弦距离而不是欧氏距离？举出具体例子。

**答案：** 当"向量的方向"比"向量的长度"更重要时，用余弦距离。

**具体例子：**
- **文本分类**：两篇关于"机器学习"的文章，一篇 500 字，一篇 5000 字。它们的词频向量方向几乎一致（出现相同的关键词），但欧氏距离会因为长度差异而很大。余弦距离正确地识别出它们主题相同。
- **用户行为分析**：用户 A 浏览了 3 个商品，用户 B 浏览了 30 个商品，但两人的兴趣分布比例相似。
- **图像色调分析**：两张照片一张偏暗一张偏亮，但色调组成比例相同。

**反过来**，当幅度本身就有含义时（如房价、身高、收入），用欧氏距离。200 平米的房子和 20 平米的房子，应该被识别为"不相似"——幅度差异本身就是信号。

---

### Q6：一个数据集的训练样本 n=10000，特征维度 d=500。训练一个 KNN 分类器会有什么问题？应该如何改进？

**答案：** 两个严重问题：

**问题 1：维度灾难。** d=500 时，任意两点距离趋于相等，"近邻"概念失去意义。覆盖 1% 的数据需要每条边搜 $0.01^{1/500} \approx 99.1\%$ 的范围。

**问题 2：预测速度。** 每次预测要计算 10000 个距离，每个距离涉及 500 维向量的运算 → O(10000 × 500) = 每次预测 500 万次乘加。在高频服务中不可接受。加上内存要存 10000×500 的矩阵。

**改进方案：**
1. 先用 PCA 降到 10~30 维（保留大部分方差的同时消除维度灾难）
2. 或用特征选择（基于方差、互信息等）挑出最 informative 的维度
3. 用 Ball-Tree / KD-Tree 加速最近邻搜索
4. 考虑换模型：高维数据上 SVM、随机森林、梯度提升树通常远超 KNN

---

### Q7：KNN 分类器和 KNN 回归器在预测逻辑上有什么区别？代码上哪里不同？

**答案：**

| | 分类 | 回归 |
|:---|:----|:----|
| 邻居标签使用方式 | 多数投票 | 取平均 |
| 输出类型 | 离散类别 | 连续值 |
| 平票处理 | 需要策略（降 K、加权、随机选） | 不存在平票 |
| 可输出概率 | 是（各类别在 K 邻居中的占比） | 否 |

**代码核心区别：**

```python
# 分类 —— 多数投票
neighbor_labels = y_train[nearest_indices]
predicted = Counter(neighbor_labels).most_common(1)[0][0]

# 回归 —— 取均值
neighbor_values = y_train[nearest_indices]
predicted = neighbor_values.mean()
```

---

### Q8：不用 sklearn，只用 numpy，如何加速 KNN 的预测——避免对每个测试样本逐一循环？

**答案：** 利用广播（broadcasting）一次性计算所有测试样本到所有训练样本的距离矩阵：

```python
def predict_vectorized(X_test, X_train, y_train, k):
    # X_test:  (m, d)
    # X_train: (n, d)
    # 广播: (m, 1, d) - (1, n, d) → (m, n, d)
    diff = X_test[:, None, :] - X_train[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (m, n)

    nearest = np.argsort(dists, axis=1)[:, :k]   # (m, k)
    # 多数投票
    neighbor_labels = y_train[nearest]            # (m, k)
    # 使用 bincount 的批量版本
    n_classes = len(np.unique(y_train))
    votes = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_classes), axis=1,
        arr=neighbor_labels
    )
    return votes.argmax(axis=1)
```

关键点：`X_test[:, None, :]` 将 (m,d) 变成 (m,1,d)，`X_train[None, :, :]` 将 (n,d) 变成 (1,n,d)。广播相减产生 (m,n,d) 的张量，一次循环都不需要。代价是内存——m×n×d 的矩阵可能很大。

---

### Q9：逻辑回归和 KNN 的决策边界有什么本质区别？从几何上解释。

**答案：**

| | 逻辑回归 | KNN |
|:---|:-------|:---|
| 边界形状 | 直线（或超平面） | 由数据点"拼"出来的分段折线 |
| 全局/局部 | 全局模型：一条边界管整个空间 | 局部模型：每个区域的边界由局部邻居决定 |
| 可解释性 | 权重 = 方向 + 重要性 | "最近的邻居是谁" |
| 对新区域的推广 | 好（线性外推合理） | 差（以外推无依据） |

**几何上：** 逻辑回归先学出 $w^T x + b = 0$ 这条直线，所有点都按同一规则判断符号。KNN 在空间中没有"学出"任何东西——它只是记住每个点的位置，测试时在局部"画"出以邻居为投票人的区域。K=1 时，KNN 的决策边界就是训练点的 Voronoi 图。

---

### Q10：KNN 是参数模型还是非参数模型？K 算"参数"吗？

**答案：** KNN 是**非参数模型**，但 K 是**超参数**。

在统计学和机器学习中，"参数模型"指的是模型的参数数量是固定的、不随训练数据增长而变化的——如线性回归的 $w$ 和 $b$。KNN 没有这样一个固定的参数集合——它的"知识库"就是整个训练集，数据越多，存储量越大。所以它是非参数模型（尽管名字叫 K**参数**）。

K 是**超参数**（hyperparameter）——它控制模型行为，但不是从数据中学习得到的。超参数需要通过交叉验证等外部方法来选择，而不是通过梯度下降来优化。

类比：贝叶斯非参数模型（如高斯过程）也是非参数——随着数据增多，"参数"（实际上是基函数的数量）自动增长。KNN 也是如此：数据越多，"决策面"越精细，这不是因为优化了什么参数，而是因为记住了更多样本位置。

---

## 14. 小结

KNN 的简单不是它的弱点——正是这份简单，让它成为最诚实的机器学习算法。它不假装数据服从某种分布，不构建复杂隐空间，不优化损失函数。它只是忠实地告诉你：**"在你的数据里，和它最像的是这几个。"**

本章你学会了：

- 从直觉到形式化定义的完整推导
- 用手算走通 K=1/3/5 的投票全过程
- 四种距离度量的几何意义和适用场景
- 从零手写 KNN 分类器和回归器
- K 值选择的偏差-方差权衡与交叉验证
- 维度灾难的本质和中招条件
- 特征缩放的成败关键
- 6 个常见误区和 10 道思考题的精解

当你面临一个新问题时，在掏出深度神经网络之前，不妨先跑一下 KNN。如果连"最像它的邻居是什么"都回答不了，更复杂的模型大概率也只是在错得优雅。而当"相似性"本身就是任务的核心时——推荐、异常检测、范例检索——KNN 往往是最难被打败的基线。

下一步：[支持向量机](./08-svm.md) — 寻找最优分类边界
