# 机器学习教程 — 从数学基础到实战

## 概述

本教程是一套**从零开始**的机器学习系统学习资料，覆盖从数学前置知识到主流算法的完整学习路径。教程面向零基础学习者，不需要任何机器学习或高等数学的先修知识——我们会在需要的时候详细讲解每一个数学概念。

## 目标读者

- 具备基本 Python 编程能力的学习者（会写函数、会用 numpy/pandas 更好但不强制）
- 对机器学习感兴趣但被数学门槛劝退的初学者
- 希望系统梳理 ML 数学基础的有经验开发者
- 想要理解和复现经典的 ML 算法而非仅仅调包的数据科学爱好者

## 前置要求

| 要求 | 说明 |
|------|------|
| Python 基础 | 变量、函数、循环、列表/字典等基本数据结构 |
| 高中数学 | 基础的函数概念、坐标系，其他数学知识会在教程中从零讲解 |
| 学习态度 | 耐心动手——每个概念都配有 Python 代码示例，建议亲自运行并修改 |

> **不需要**：线性代数、微积分、概率论的先修课程。这些都会在「数学基础」部分从零讲起。

## 内容地图

本教程按学习顺序组织，建议按序阅读。每个部分独立成章，也支持按需跳转。

```
机器学习教程 (ml-tutorial)
│
├── 📖 README.md                    ← 你在这里
├── 🗺️ learning-path.md             ← 学习路线图
│
├── 📐 第一部分：数学基础 (Math Foundations)
│   ├── math-linear-algebra.md      线性代数：向量、矩阵、特征值、SVD
│   ├── math-calculus.md            微积分：导数、梯度、链式法则
│   ├── math-probability.md         概率论：随机变量、分布、贝叶斯定理
│   ├── math-statistics.md          统计学：估计、检验、置信区间
│   ├── math-optimization.md        最优化：梯度下降、凸优化
│   └── math-information-theory.md  信息论：熵、交叉熵、KL散度
│
├── 🧠 第二部分：机器学习基础 (ML Fundamentals)
│   ├── 01-introduction.md          什么是机器学习？
│   ├── 02-bias-variance.md         偏差-方差权衡
│   └── 03-data-preprocessing.md    数据预处理
│
├── 📈 第三部分：监督学习 — 回归
│   ├── 04-linear-regression.md     线性回归
│   └── 05-regularized-regression.md 正则化回归（岭回归、Lasso）
│
├── 🏷️ 第四部分：监督学习 — 分类
│   ├── 06-logistic-regression.md   逻辑回归
│   ├── 07-knn.md                   K近邻
│   ├── 08-svm.md                   支持向量机
│   ├── 09-decision-trees.md        决策树与随机森林
│   ├── 10-naive-bayes.md           朴素贝叶斯
│   └── 11-classification-metrics.md 分类评估指标
│
├── 🌲 第五部分：集成学习
│   └── 12-ensemble-methods.md      Bagging、Boosting、Stacking
│
├── 🔍 第六部分：无监督学习
│   ├── 13-clustering.md            聚类算法
│   └── 14-dimensionality-reduction.md 降维
│
└── 🧬 第七部分：神经网络与实战
    ├── 15-neural-networks-intro.md 神经网络入门
    ├── 16-cnn-rnn-basics.md        CNN与RNN基础
    ├── 17-model-evaluation.md      模型评估与调参
    ├── 18-feature-engineering.md   特征工程
    ├── 19-ml-pipeline.md           端到端ML流水线
    └── 20-ml-best-practices.md     ML最佳实践
```

## 教程特色

### 1. 数学先行，从零讲起
每个数学概念都会先**建立直观理解**（几何意义、生活类比），再给出**形式化定义**，最后用 **Python 代码**验证。不会出现「这里要用到矩阵求导，请自行查阅」这种情况。

### 2. 代码伴随
每个概念都配有可运行的 Python 代码。代码以 `numpy`/`scipy` 为主，算法实现尽量从零手写，帮助理解底层原理，之后再介绍 `scikit-learn` 等库的用法。

### 3. 概念连接
每个数学概念都会明确标注**这个数学知识在哪里用到**（类似「这个概念 → 用于哪种算法 → 解决什么问题」的链条），让你知道为什么学、学了有什么用。

### 4. 渐进式结构
遵循「直觉 → 数学 → 代码 → 应用」的四步讲解模式。

## 如何使用本教程

1. **从学习路线图开始**：阅读 [learning-path.md](./learning-path.md) 了解推荐的学习路径
2. **打好数学基础**：按顺序学习「数学基础」部分的六个章节
3. **动手实践**：每个章节的代码示例都建议亲手运行、修改参数并观察结果
4. **按需跳转**：如果已有某部分基础，可以直接跳到感兴趣的算法章节

## 环境准备

```bash
# 推荐使用 conda 创建独立环境
conda create -n ml-tutorial python=3.10
conda activate ml-tutorial

# 安装核心依赖
pip install numpy scipy matplotlib scikit-learn pandas jupyter
```

## 快速示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成模拟数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.randn(100) * 2

# 训练一个简单的线性回归模型
model = LinearRegression()
model.fit(X, y)

print(f"斜率: {model.coef_[0]:.2f}")
print(f"截距: {model.intercept_:.2f}")
print(f"R² 分数: {model.score(X, y):.3f}")

# 可视化
plt.scatter(X, y, alpha=0.5, label='数据点')
plt.plot(X, model.predict(X), 'r-', label='拟合线')
plt.legend()
plt.title('你的第一个机器学习模型：线性回归')
plt.show()
```

**输出示例：**
```
斜率: 2.03
截距: 0.89
R² 分数: 0.895
```

---

## 推荐学习顺序

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 阶段 1 | 数学基础（6章） | 2-3 周 |
| 阶段 2 | ML 基础 + 数据预处理 | 1 周 |
| 阶段 3 | 回归算法 | 1 周 |
| 阶段 4 | 分类算法 | 2 周 |
| 阶段 5 | 集成学习 + 无监督学习 | 1-2 周 |
| 阶段 6 | 神经网络 + 实战 | 2-3 周 |

---

**准备好了吗？让我们从 [学习路线图](./learning-path.md) 开始，或者直接进入 [数学基础 — 线性代数](./math-linear-algebra.md)！**
