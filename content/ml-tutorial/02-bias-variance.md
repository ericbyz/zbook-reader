# 02 — 偏差与方差：机器学习最核心的权衡

> **核心问题**：模型犯错只有两种原因——要么它"笨"（偏差高），要么它"太敏感"（方差高）。理解了这一点，你就理解了诊断和改进模型的根本方法。

---

## 1. 生活例子：射箭类比

### 1.1 场景描述

想象你是一个弓箭手，固定站在距靶心 10 米的位置，每次连射 5 支箭。你的目标是所有箭都命中靶心。但现实中，有很多因素会导致偏离：

- 你的瞄准镜可能**没校准**——即使你手很稳，平均落点也偏离靶心
- 你的手可能**在抖**——瞄准镜没问题，但每支箭都偏一点，方向还各不相同

这两个问题对应了机器学习中的两个核心概念：

```
   低偏差 + 低方差              低偏差 + 高方差
   ┌─────────────────┐         ┌─────────────────┐
   │     ·  · ·      │         │  ·     ·    ·   │
   │    ·  *  ·      │         │     · * ·       │
   │     ·  · ·      │         │  ·   ·   · ·    │
   └─────────────────┘         └─────────────────┘
   箭密集，都在靶心附近           箭分散在靶心周围
   → 又准又稳                    → 平均准，但每箭飘忽

   高偏差 + 低方差              高偏差 + 高方差
   ┌─────────────────┐         ┌─────────────────┐
   │     ···          │         │ ·               │
   │     ···      *   │         │·        ·    *  │
   │     ···          │         │       ·  ·      │
   └─────────────────┘         └─────────────────┘
   箭密集，但全偏在角落           箭又散又偏
   → 稳定地偏                    → 全都不行
```

### 1.2 翻译成机器学习

| 射箭概念 | 机器学习含义 | 关键指标 |
|---------|------------|---------|
| 你 = 模型 | 从训练数据中学到的预测函数 | |
| 靶心 = 真实标签 | 每个数据点正确的输出值 | $y$ |
| 每组箭 = 一个训练集 | 由于随机采样，每次得到稍有不同的训练数据 | $D_1, D_2, \ldots$ |
| 平均落点离靶心的距离 = **偏差** | 模型预测的期望值与真实值之间的差距 | $\text{Bias}[\hat{f}] = E[\hat{f}] - f$ |
| 箭之间的分散程度 = **方差** | 不同训练集产生的模型预测的差异程度 | $\text{Var}[\hat{f}] = E[(\hat{f} - E[\hat{f}])^2]$ |
| 靶心本身的模糊 = **噪声** | 数据本身的随机误差，任何模型都无法消除 | $\sigma^2 = \text{Var}(\epsilon)$ |

### 1.3 更生活的例子：认人

换一个你每天都会遇到的场景。你是一个老师，要从一个班里挑出所有"爱打篮球的学生"。你通过观察得出一些规则，带去下一个班使用。

**场景 A：高偏差（欠拟合）**
你的规则只有一条——"挑男生"。
- 在你的训练班上，大部分打篮球的确实是男生，你觉得规则够用了
- 去了新班级，你漏掉所有打篮球的女生，还错误地挑出了不打篮球的男生
- 结果：**系统性地犯错**。换个班也错得一样。偏差高——你的规则太简单，根本抓不住"爱打篮球"的本质特征

**场景 B：高方差（过拟合）**
你把训练班上每个爱打篮球的学生的细节全记了下来——姓名、座位号、发型、穿什么颜色的鞋、那天吃了什么早餐。
- 在你的训练班上，因为你是对着名单念的，**100% 正确**
- 去了新班级，名单上的学生一个都不在，发型不同了、座位不同了、早餐也不同了——你的规则全部失效
- 结果：**对训练集过度敏感**。规则里塞满了无关信息和噪声。方差高

**场景 C：刚好**
你观察后发现，爱打篮球的学生通常：课间手上拿着篮球、运动鞋磨得比较厉害、桌上有运动手环。这些是**真正有区分力**的特征。
- 在你的训练班上表现不错（但不是 100%）
- 去了新班级同样管用，因为这些特征是普适的
- 结果：偏差低（抓住了本质）+ 方差低（不受无关因素干扰）

---

## 2. 直观理解：偏差-方差如何随模型复杂度变化

用一句话概括：
- **模型太简单** → 连训练数据的规律都学不全 → **高偏差、低方差**（欠拟合）
- **模型太复杂** → 连训练数据的噪声都记住了 → **低偏差、高方差**（过拟合）
- **最佳点** → 偏差和方差的平衡 → **总误差最小**

直观的"跷跷板"关系：

```
误差 ↑
     |
     |   \                             总误差 (测试)
     |    \                          ╱
     |     \                       ╱
     |      \                    ╱
     |       \__________________╱  ← 最佳模型复杂度
     |       ╲                ╱╱
     |   训练误差 ╲          ╱╱
     |           ╲        ╱╱
     |            ╲     ╱╱
     |             ╲  ╱╱
     |              ╲╱╱
     |_____________╱╱_________________→ 模型复杂度
          低       中          高
         欠拟合   最佳         过拟合
```

**注解**：
- 训练误差随模型复杂度增加而单调下降（模型越复杂，记忆力越强）
- 测试误差呈 **U 形**：先降（偏差减小主导）→ 最低点 → 后升（方差增大主导）
- 最佳点不是偏差² = 方差的点，而是**偏差² + 方差取最小值的点**

---

## 3. 形式化定义与 MSE 分解推导

### 3.1 设定

在一个监督学习任务中：
- $x \in \mathbb{R}^d$：输入数据点
- $y \in \mathbb{R}$：$x$ 的真实标签（回归任务）
- $f(x)$：真实但未知的函数。通常假设 $y = f(x) + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 是噪声
- $\hat{f}(x)$：从训练集 $D$ 上学到的模型对 $x$ 的预测——它是一个随机变量（依赖于随机采样的 $D$）
- $E_D[\cdot]$：对所有可能的训练集取期望

**关键认知**：$\hat{f}$ 是一个随机变量。给你不同的训练数据，你会学到不同的 $\hat{f}$。偏差和方差衡量的是**这个随机变量的两个统计性质**。

### 3.2 推导

对于一个固定的测试点 $x$，模型的期望预测误差（MSE）定义为：

$$E_D[(y - \hat{f}(x))^2]$$

其中 $y = f(x) + \epsilon$，且 $\epsilon$ 独立于 $\hat{f}(x)$。

展开：

$$
\begin{aligned}
E[(y - \hat{f})^2] &= E[(f + \epsilon - \hat{f})^2] \\
&= E[(f - \hat{f})^2] + E[\epsilon^2] + 2E[(f - \hat{f})\epsilon]
\end{aligned}
$$

由于 $E[\epsilon] = 0$ 且 $\epsilon$ 与 $\hat{f}$ 独立，交叉项 $E[(f - \hat{f})\epsilon] = E[f - \hat{f}] \cdot E[\epsilon] = 0$。所以：

$$E[(y - \hat{f})^2] = E[(f - \hat{f})^2] + \sigma^2$$

现在聚焦 $E[(f - \hat{f})^2]$，加一项减一项 $E[\hat{f}]$（经典技巧）：

$$
\begin{aligned}
E[(f - \hat{f})^2] &= E[(f - E[\hat{f}] + E[\hat{f}] - \hat{f})^2] \\
&= E[(f - E[\hat{f}])^2 + (E[\hat{f}] - \hat{f})^2 + 2(f - E[\hat{f}])(E[\hat{f}] - \hat{f})]
\end{aligned}
$$

拆开三项取期望：
- 第一项：$(f - E[\hat{f}])^2$ 相对于 $D$ 是常数 → $E[(f - E[\hat{f}])^2] = (f - E[\hat{f}])^2$
- 第二项：$E[(E[\hat{f}] - \hat{f})^2] = \text{Var}[\hat{f}]$（方差的定义）
- 第三项交叉项：$2(f - E[\hat{f}]) \cdot E[E[\hat{f}] - \hat{f}] = 2(f - E[\hat{f}]) \cdot 0 = 0$

（因为 $E[E[\hat{f}] - \hat{f}] = E[\hat{f}] - E[\hat{f}] = 0$）

最终：

$$\boxed{E[(y - \hat{f}(x))^2] = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2[\hat{f}(x)]} + \underbrace{E[(\hat{f}(x) - E[\hat{f}(x)])^2]}_{\text{Variance}[\hat{f}(x)]} + \underbrace{\sigma^2}_{\text{Irreducible Error}}}$$

### 3.3 逐项解释

| 项 | 数学定义 | 含义 | 能否减小 | 减小方法 | 典型量级 |
|----|---------|------|---------|---------|---------|
| $\text{Bias}^2$ | $(E[\hat{f}] - f)^2$ | 平均预测与真实函数的平方差距 | 可以 | 更复杂模型 | 随模型复杂度↓ |
| $\text{Variance}$ | $E[(\hat{f} - E[\hat{f}])^2]$ | 预测在期望周围的波动程度 | 可以 | 更多数据、正则化 | 随模型复杂度↑ |
| $\sigma^2$ | $\text{Var}(\epsilon)$ | 数据内在的随机噪声 | **不能** | 需要更好的数据 | 取决于问题本身 |

这就是机器学习最根本的误差分解。任何降低总误差的努力，都在偏差²和方差之间做取舍——噪声是无论如何也绕不开的底线。

---

## 4. 手算示例：在小数据集上亲算偏差-方差分解

只看公式太抽象。让我们用一组**极小**的数据，一步步手算，感受偏差²和方差的数值意义。

### 4.1 设定

真实函数：$f(x) = 2x$。噪声：$\epsilon \sim \mathcal{N}(0, 1)$，所以 $\sigma^2 = 1$。

我们收集了两个训练集（每个只有 3 个点），并对每个训练集做最小二乘线性回归（过原点）$\hat{f}(x) = wx$。

| 训练集 $D_1$ | 训练集 $D_2$ |
|--------------|-------------|
| $(1, 3.2)$ | $(1, 1.8)$ |
| $(2, 3.8)$ | $(2, 4.5)$ |
| $(3, 6.1)$ | $(3, 5.7)$ |

**目标**：在测试点 $x = 2$ 处，计算 $\text{Bias}^2$、$\text{Variance}$ 和总误差，验证分解公式。

### 4.2 拟合模型

对于 $\hat{f}(x) = wx$，最小化 $\frac{1}{3}\sum_{i=1}^{3}(y_i - wx_i)^2$。

对 $w$ 求导并设为零：
$$\frac{d}{dw} \sum_{i=1}^{3}(y_i - wx_i)^2 = -2\sum_{i=1}^{3} x_i(y_i - wx_i) = 0$$

$$\sum_{i=1}^{3} x_i y_i = w \sum_{i=1}^{3} x_i^2$$

$$w = \frac{\sum x_i y_i}{\sum x_i^2}$$

**训练集 $D_1$：**

| $i$ | $x_i$ | $y_i$ | $x_i y_i$ | $x_i^2$ |
|-----|-------|-------|-----------|--------|
| 1 | 1 | 3.2 | 3.2 | 1 |
| 2 | 2 | 3.8 | 7.6 | 4 |
| 3 | 3 | 6.1 | 18.3 | 9 |
| **和** | | | **29.1** | **14** |

$$w_1 = \frac{29.1}{14} \approx 2.0786, \quad \hat{f}_1(x) = 2.0786x$$

验证：$\hat{f}_1(1) = 2.08, \hat{f}_1(2) = 4.16, \hat{f}_1(3) = 6.24$

**训练集 $D_2$：**

| $i$ | $x_i$ | $y_i$ | $x_i y_i$ | $x_i^2$ |
|-----|-------|-------|-----------|--------|
| 1 | 1 | 1.8 | 1.8 | 1 |
| 2 | 2 | 4.5 | 9.0 | 4 |
| 3 | 3 | 5.7 | 17.1 | 9 |
| **和** | | | **27.9** | **14** |

$$w_2 = \frac{27.9}{14} \approx 1.9929, \quad \hat{f}_2(x) = 1.9929x$$

### 4.3 计算偏差²和方差

在测试点 $x = 2$，真实值 $f(2) = 4$。

**期望预测**：

$$E[\hat{f}(2)] = \frac{\hat{f}_1(2) + \hat{f}_2(2)}{2} = \frac{2.0786 \times 2 + 1.9929 \times 2}{2} = \frac{4.1571 + 3.9857}{2} = 4.0714$$

**偏差²**：

$$\text{Bias}^2 = (E[\hat{f}(2)] - f(2))^2 = (4.0714 - 4)^2 = 0.0714^2 \approx \mathbf{0.00510}$$

**方差**：

$$\text{Var} = \frac{(\hat{f}_1(2) - E[\hat{f}(2)])^2 + (\hat{f}_2(2) - E[\hat{f}(2)])^2}{2}$$

$$= \frac{(4.1571 - 4.0714)^2 + (3.9857 - 4.0714)^2}{2}$$

$$= \frac{0.0857^2 + (-0.0857)^2}{2} = \frac{0.00735 + 0.00735}{2} = \mathbf{0.00735}$$

### 4.4 验证分解

理论预测的总误差：
$$\text{Total} = \text{Bias}^2 + \text{Var} + \sigma^2 = 0.00510 + 0.00735 + 1 = \mathbf{1.01245}$$

直接计算验证（对每个训练集，$y$ 来自 $\mathcal{N}(f(2), 1) = \mathcal{N}(4, 1)$）：

$$E_{(y|D_1)}[(y - \hat{f}_1(2))^2] = E[(4 + \epsilon - 4.1571)^2] = E[(-0.1571 + \epsilon)^2] = 0.1571^2 + E[\epsilon^2] = 0.02468 + 1 = 1.02468$$

$$E_{(y|D_2)}[(y - \hat{f}_2(2))^2] = E[(4 + \epsilon - 3.9857)^2] = 0.0143^2 + 1 = 0.00021 + 1 = 1.00021$$

取期望（对两个训练集平均）：
$$\frac{1.02468 + 1.00021}{2} = \mathbf{1.01244} \;\checkmark$$

与分解公式结果一致！偏差² ($0.00510$) + 方差 ($0.00735$) + 噪声 ($1$) = $1.01244$。

### 4.5 核心洞察

这个手算例子揭示了几个重要事实：

1. **噪声是误差的主体**（占 98.8%）。这意味着在这个问题设定下，无论你模型多好，MSE 最低也就是 1 左右——噪声是真正的"硬地板"。
2. **偏差² 和方差量级接近**（$0.0051$ vs $0.00735$），在这个简单线性模型中，两者对总误差的贡献都不大。
3. **如果你只用了一个训练集**（比方 $D_1$），你永远不知道方差的存在——因为方差是**跨训练集的统计量**。这就是为什么交叉验证如此重要。
4. **即使只有两个训练集，分解也近似成立**（只要把期望理解为样本均值）。真实实验中，$D_1, D_2, \ldots, D_{100}$ 越多近似越准。

---

## 5. Python 实验

### 5.1 多项式回归的过拟合演示

目标：直观感受不同复杂度模型在同一个数据上的表现——以及偏差²和方差如何随复杂度变化。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

np.random.seed(42)

# -------- 生成数据 --------
n_points = 60
X = np.linspace(0, 1, n_points).reshape(-1, 1)
true_f = lambda x: np.sin(2 * np.pi * x.ravel())
y = true_f(X) + np.random.normal(0, 0.3, n_points)

# -------- 4 种复杂度，每种采样 100 次拟合 --------
degrees = [1, 3, 9, 20]
n_repeats = 100
n_subset = 30  # 每次从 60 个点中随机取 30 个

all_preds = np.zeros((len(degrees), n_repeats, n_points))

for deg_idx, deg in enumerate(degrees):
    for rep in range(n_repeats):
        idx = np.random.choice(n_points, size=n_subset, replace=False)
        X_sub, y_sub = X[idx], y[idx]
        model = Pipeline([
            ('poly', PolynomialFeatures(deg)),
            ('lr', LinearRegression())
        ])
        model.fit(X_sub, y_sub)
        all_preds[deg_idx, rep] = model.predict(X)

# -------- 绘图 --------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
titles = [
    'deg=1 (高偏差, 低方差)',
    'deg=3 (适中)',
    'deg=9 (高方差, 过拟合)',
    'deg=20 (严重过拟合)'
]

for deg_idx in range(len(degrees)):
    # 上排：展示 20 条采样拟合曲线
    ax1 = axes[0, deg_idx]
    for i in range(20):
        ax1.plot(X.ravel(), all_preds[deg_idx, i], alpha=0.35, lw=0.5)
    ax1.scatter(X.ravel(), y, alpha=0.4, s=10, c='black')
    ax1.plot(X.ravel(), true_f(X), 'g--', lw=1.5, label='真实函数 f(x)')
    ax1.set_title(titles[deg_idx], fontsize=11)
    ax1.set_ylim(-1.8, 1.8)
    if deg_idx == 0:
        ax1.legend(fontsize=7)

    # 下排：偏差² 和方差曲线
    ax2 = axes[1, deg_idx]
    mean_pred = all_preds[deg_idx].mean(axis=0)
    variance = all_preds[deg_idx].var(axis=0)
    bias_sq = (mean_pred - true_f(X).ravel()) ** 2
    ax2.plot(X.ravel(), bias_sq, color='red', lw=1.5, label='偏差²')
    ax2.plot(X.ravel(), variance, color='blue', lw=1.5, label='方差')
    avg_bias = bias_sq.mean()
    avg_var = variance.mean()
    ax2.set_title(f'Avg Bias²={avg_bias:.3f}, Avg Var={avg_var:.3f}', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 1.2)

plt.suptitle('多项式回归的偏差-方差分解：100 次重复采样的结果', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('bias_variance_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("关键观察：")
print("deg=1: 20 条拟合曲线几乎重合（低方差），但都偏离真实函数（高偏差²）")
print("deg=3: 曲线在真实函数周围小范围波动——最佳平衡")
print("deg=9: 曲线大幅摆动——低偏差²，极高的方差")
print("deg=20: 曲线变得更剧烈——方差爆炸，完全过拟合")
print("=" * 60)
```

### 5.2 经典 U 形曲线

这个实验验证：**训练误差随复杂度单调下降，测试误差呈 U 形**。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(42)
n_total = 200
X = np.linspace(-3, 3, n_total).reshape(-1, 1)
y = 2 * np.sin(1.5 * X.ravel()) + np.random.normal(0, 0.3, n_total)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

max_degree = 15
train_errors, test_errors = [], []

for degree in range(1, max_degree + 1):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    train_errors.append(np.mean((model.predict(X_train) - y_train) ** 2))
    test_errors.append(np.mean((model.predict(X_test) - y_test) ** 2))

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(range(1, max_degree + 1), train_errors, 'o-',
         label='训练误差', color='#2196F3')
ax1.plot(range(1, max_degree + 1), test_errors, 's-',
         label='测试误差', color='#F44336')
best_deg = np.argmin(test_errors) + 1
ax1.axvline(x=best_deg, linestyle='--', color='green',
            label=f'最佳 deg={best_deg}')
ax1.axvspan(1, 3, alpha=0.1, color='orange', label='欠拟合区')
ax1.axvspan(10, 15, alpha=0.1, color='purple', label='过拟合区')
ax1.set_xlabel('多项式次数（模型复杂度）')
ax1.set_ylabel('MSE')
ax1.set_title('训练误差 vs 测试误差')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 右边：展示 3 个典型度数的拟合效果
for idx, deg in enumerate([1, best_deg, 14]):
    model = Pipeline([
        ('poly', PolynomialFeatures(deg)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    colors = ['orange', 'green', 'purple']
    labels = [f'deg={deg} (欠拟合)', f'deg={deg} (最佳)', f'deg={deg} (过拟合)']
    ax2.plot(X_plot, y_plot, color=colors[idx], lw=1.5, label=labels[idx])

ax2.scatter(X_train, y_train, alpha=0.3, s=8, c='blue', label='训练数据')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('不同复杂度模型的拟合效果')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle('偏差-方差权衡：经典 U 形曲线', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('overfit_underfit_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"最佳多项式次数: {best_deg}，对应测试 MSE: {min(test_errors):.4f}")
print("观察：训练误差单调下降但测试误差先降后升——经典的 U 形")
```

### 5.3 学习曲线诊断

学习曲线是你在实际项目中**最常用的诊断工具**。它展示：当训练样本数从少到多，训练误差和验证误差如何变化。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(42)
n_total = 400
X = np.linspace(0, 10, n_total).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.25, n_total)

def plot_learning_curve(degree, ax, title):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('lr', LinearRegression())
    ])
    train_sizes = np.linspace(0.1, 1.0, 12)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=42
    )
    train_mse = -train_scores.mean(axis=1)
    val_mse = -val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    ax.plot(train_sizes_abs, train_mse, 'o-', color='#2196F3',
            label='训练误差')
    ax.fill_between(train_sizes_abs,
                    train_mse - train_std, train_mse + train_std,
                    alpha=0.15, color='#2196F3')
    ax.plot(train_sizes_abs, val_mse, 'o-', color='#F44336',
            label='交叉验证误差')
    ax.fill_between(train_sizes_abs,
                    val_mse - val_std, val_mse + val_std,
                    alpha=0.15, color='#F44336')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('训练样本数')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plot_learning_curve(1, axes[0], '欠拟合 (deg=1)')
plot_learning_curve(3, axes[1], '适中 (deg=3)')
plot_learning_curve(15, axes[2], '过拟合 (deg=15)')

plt.suptitle('学习曲线诊断：三种典型模型状态', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
┌──────────────────────┬──────────────────┬────────────────────┐
│ 学习曲线形态          │ 诊断             │ 对策                │
├──────────────────────┼──────────────────┼────────────────────┤
│ 两条线都高, 靠近      │ 高偏差 (欠拟合)   │ 增加模型复杂度       │
│ 训练低, 验证高,       │                  │                    │
│ 有宽的固定间隙        │ 高方差 (过拟合)   │ 加数据 / 正则化      │
│ 两条线都低, 靠近      │ 刚好             │ 保持                │
│ 验证线还在下降趋势     │ 加数据有帮助      │ 收集更多数据         │
└──────────────────────┴──────────────────┴────────────────────┘
""")
```

### 5.4 双重下降现象

经典理论认为测试误差是 U 形。但现代研究发现：当模型参数数量**超过**训练样本数之后，测试误差可能出现**第二次下降**。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

n_train = 15  # 故意用小样本
X_train = np.linspace(0, 1, n_train).reshape(-1, 1)
true_f = lambda x: np.sin(4 * np.pi * x.ravel())
y_train = true_f(X_train) + np.random.normal(0, 0.3, n_train)

X_test = np.linspace(0, 1, 200).reshape(-1, 1)
y_test = true_f(X_test)

max_deg = 40
degrees = np.arange(1, max_deg + 1)
train_errors, test_errors = [], []

for deg in degrees:
    poly = PolynomialFeatures(deg)
    X_tr = poly.fit_transform(X_train)
    X_te = poly.transform(X_test)
    model = Ridge(alpha=1e-10)
    model.fit(X_tr, y_train)
    train_errors.append(np.mean((model.predict(X_tr) - y_train) ** 2))
    test_errors.append(np.mean((model.predict(X_te) - y_test) ** 2))

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.plot(degrees, train_errors, 'o-', label='训练误差',
        color='#2196F3', markersize=4)
ax.plot(degrees, test_errors, 's-', label='测试误差',
        color='#F44336', markersize=4)

interp_threshold = n_train + 1
classical_best = np.argmin(test_errors[:interp_threshold]) + 1

ax.axvline(x=interp_threshold, linestyle='--', color='green',
           label=f'插值阈值 (deg≈{interp_threshold})')
ax.axvline(x=classical_best, linestyle=':', color='orange',
           label=f'经典最佳点 deg={classical_best}')
ax.axvspan(interp_threshold, max_deg, alpha=0.08, color='purple',
           label='过参数化区')

ax.set_xlabel('多项式次数（参数数量）')
ax.set_ylabel('MSE')
ax.set_title(
    f'双重下降 (Double Descent)：n_train={n_train}, 特征数从1到{max_deg}'
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('double_descent.png', dpi=150, bbox_inches='tight')
plt.show()

print("双重下降三个关键区间：")
print(f"1. 经典 U 形区 (deg 1-{interp_threshold}): 测试误差先降后升")
print(f"2. 插值阈值附近 (deg≈{interp_threshold}): 测试误差达到局部极大值")
print(f"3. 过参数化区 (deg>{interp_threshold}): 测试误差再次下降")
print("结论：极大模型 + 隐式正则化 → 能找到泛化良好的解")
```

---

## 6. 正则化直觉：用偏差换方差

### 6.1 核心思想

正则化是对抗过拟合最直接的方法。以 L2 正则化（Ridge）为例：

$$\min_w \;\; \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{f}_w(x_i))^2 + \lambda \|w\|^2$$

- 第一项：拟合项。鼓励 $w$ 尽量匹配训练数据 → 降低偏差
- 第二项：惩罚项。鼓励 $w$ 尽量小 → 降低方差

**$\lambda$ 就是一个"偏差² ↔ 方差"的调节旋钮**。

### 6.2 $\lambda$ 的行为

| $\lambda$ | 模型行为 | 偏差² | 方差 |
|-----------|---------|-------|------|
| $0$ | 无约束，完全拟合训练数据 | 最低 | 最高 |
| 小正值 | 轻微约束，最优泛化通常在此 | 略高 | 略低 |
| 中等 | 明显约束，参数被压缩 | 中等 | 中等 |
| $\infty$ | $w \approx 0$，模型退化到常数预测 | 极高 | 极低 |

### 6.3 为什么压缩参数能降低方差？

直观理解：方差衡量的是"训练数据的一点变化就能引起模型参数的剧烈变化"。

- 无正则化时，参数的某些方向对数据噪声极其敏感，$w$ 可以取任意大的值来"迁就"噪声
- L2 正则化对每个 $w_j$ 施加"回归到 0"的拉力，限制了 $w$ 的自由度
- 结果：模型变"懒"了——不再拼命拟合每一个数据点的噪声。偏差略微增大，但方差大幅下降

用射箭类比说：正则化就像在手上加了一个稳定支架——略微降低了瞄准的精确度（偏差），但大幅减少了手抖（方差）。

---

## 7. 常见误区

### 误区一："训练误差低 = 模型好"

**错误且危险**。训练误差可以降到近乎为零（给够复杂度），但这通常意味着方差爆炸。$R^2_{\text{train}}=0.99$ 但 $R^2_{\text{test}}=0.45$ 的模型几乎没有实用价值。**只看测试集/验证集上的指标**。

### 误区二："加更多数据总是有帮助"

**部分正确**。加数据主要降低**方差**。如果问题本质是高偏差（模型容量不够，连训练集的基本规律都学不会），加数据治标不治本。判断方法：看学习曲线——如果训练误差和验证误差在高位汇合，说明是高偏差，加数据无用；如果两者之间有大的间隙且随数据增加而缩小，加数据有用。

### 误区三："偏差-方差是绝对矛盾的"

**近似正确但有例外**。经典 U 形曲线是"同一模型族、固定数据量、固定优化算法"下的表现。现代深度学习中，增加模型规模有时能**同时降低偏差和方差**（更大的模型有更好的隐式正则化、训练动力学更优）。不要把 U 形当成物理定律。

### 误区四："偏差² + 方差 + 噪声的分解对所有损失函数都成立"

**不精确**。这个漂亮的加法分解严格来说只对**平方误差损失（MSE）**成立。交叉熵损失下的分解更复杂（Domingos, 2000 给出了统一的框架），但核心直觉——模型太简单则系统偏、模型太复杂则不稳定——是普适的。

### 误区五："高方差一定是不好的"

**不一定**。在某些场景中（如对抗训练、不确定性估计），适度的方差反而是需要的——它表示模型"知道自己不知道什么"。追求零方差的模型可能在你没见过的分布转移上完全崩溃。

### 误区六："交叉验证能降低模型方差"

**混淆因果**。交叉验证降低的是**评估的方差**（帮你更准确地估计模型的真实性能），不是**模型的方差**（模型本身的好坏）。一个高方差的模型在 CV 下仍然是高方差的——你只是更准确地知道了它有多差。

---

## 8. ML 应用：诊断与决策框架

### 8.1 四格诊断表

拿到训练集和验证集的误差后，立刻对照下表：

| 训练误差 | 验证误差 | 诊断 | 首要行动 |
|---------|----------|------|---------|
| 高 | 高 | **欠拟合（高偏差）** | 增加模型复杂度、添加特征、减少正则化 |
| 低 | 高 | **过拟合（高方差）** | 更多数据、加强正则化、数据增强 |
| 低 | 低 | **理想** | 检查是否有数据泄露、继续优化 |
| 低 | 更低 | **严重数据泄露** | 检查验证集是否混入了训练集信息 |

### 8.2 基于学习曲线的决策

学习曲线是你判断"下一步该做什么"的最佳指导：

| 观察到的情况 | 含义 | 行动 |
|------------|------|------|
| 训练误差和 CV 误差在高位汇合 | 欠拟合 | 增复杂度 |
| 训练误差和 CV 误差之间有逐渐缩小的宽间隙 | 过拟合 + 加数据有用 | 收集更多数据 |
| 训练误差和 CV 误差之间的间隙保持不变 | 过拟合 + 加数据帮助不大 | 正则化 / 简化模型 |
| CV 误差还在下降（数据量方向） | 模型尚未饱和 | 加数据 |
| 训练误差随样本数增加反而上升 | 模型拟合能力不足 | 增复杂度或换模型 |

### 8.3 解决方案速查

**降低高偏差（欠拟合）：**

| 方法 | 原理 | 优先级 |
|------|------|--------|
| 增加模型复杂度 | 更多参数 = 更大容量 | ★★★★★ |
| 特征工程 | 创造更有表达力的特征 | ★★★★ |
| 减少/移除正则化 | 释放模型自由度 | ★★★ |
| 训练更久（增加 epoch） | 确保充分收敛 | ★★★ |
| 换更强的模型族 | 线性 → 树 → 神经网络 | ★★ |
| 减少数据预处理 | 不过度简化输入 | ★ |

**降低高方差（过拟合）：**

| 方法 | 原理 | 优先级 |
|------|------|--------|
| 增加训练数据 | 更多样本平均化噪声 | ★★★★★ |
| L1/L2 正则化 | 限制参数空间 | ★★★★ |
| 数据增强 | 人造多样性 | ★★★★ |
| 早停 (Early Stopping) | 在验证误差上升前停止 | ★★★ |
| Dropout / BatchNorm | 隐式正则化 | ★★★ |
| 简化模型 | 减少参数数量 | ★★★ |
| 交叉验证 | 更稳健的模型选择 | ★★ |
| 集成方法 (Bagging) | 平均化多个模型的方差 | ★★ |

### 8.4 实践铁律

> **先用一个能过拟合训练集的模型开始，然后加正则化把它拉回来。**

具体操作：
1. 选一个足够大的模型，验证它能在小数据子集 ($\sim 500$ 样本) 上过拟合（训练误差 $\approx 0$）
2. 这证明你的模型容量足够——你知道"天花板"在哪
3. 然后逐层添加约束（正则化、Dropout、数据增强），在验证集上找最优平衡点

这样的好处：你永远知道性能瓶颈不在模型容量，而在泛化控制。

---

## 9. 思考题

### 题目 1 — 射箭类比应用

在射箭类比中，如果一个人的箭**全部均匀散布在整个靶面上**，他的偏差和方差分别如何？在机器学习中，这对应什么场景？

<details>
<summary>解答</summary>

如果箭均匀散布在整个靶面上，可以认为：
- 平均值大致在靶心（因为分布对称）→ **偏差 ≈ 0**
- 箭极度分散 → **方差极大**

这是一种"无偏但极其不稳定"的状态。在机器学习中，这对应一个模型族容量很大但数据太少的情况——模型有能力表达真实函数（无偏），但每次训练都给出不同的、完全靠不住的预测。典型场景：100 维特征、1000 参数、但只有 20 个训练样本。

**生活类比**：一个射击新手虽然瞄准了靶心，但每次扣扳机的瞬间手都在乱抖——箭头分布的中心位置是靶心附近，但单箭几乎不可能命中。
</details>

### 题目 2 — 手算练习

真实函数 $f(x) = 3x + 1$，噪声方差 $\sigma^2 = 0.25$。两个训练集：
- $D_1: \{(1, 4.5), (2, 7.2), (3, 9.8)\}$
- $D_2: \{(1, 3.8), (2, 6.9), (3, 10.5)\}$

对每个训练集用最小二乘拟合 $\hat{f}(x) = wx + b$，在测试点 $x = 2$ 处计算 $\text{Bias}^2$、$\text{Variance}$ 和 $\sigma^2$，验证分解成立。

<details>
<summary>解答</summary>

**$D_1$**：$\bar{x} = 2, \bar{y} = \frac{4.5+7.2+9.8}{3} = 7.167$

$w_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2} = \frac{(-1)(-2.667) + 0(0.033) + 1(2.633)}{2} = \frac{2.667 + 2.633}{2} = 2.65$

$b_1 = \bar{y} - w_1\bar{x} = 7.167 - 2.65 \times 2 = 1.867$
$\hat{f}_1(x) = 2.65x + 1.867$

**$D_2$**：$\bar{x} = 2, \bar{y} = \frac{3.8+6.9+10.5}{3} = 7.067$

$w_2 = \frac{(-1)(-3.267) + 0(-0.167) + 1(3.433)}{2} = 3.35$

$b_2 = 7.067 - 3.35 \times 2 = 0.367$
$\hat{f}_2(x) = 3.35x + 0.367$

**在 $x = 2$**：$f(2) = 3 \times 2 + 1 = 7$

$\hat{f}_1(2) = 2.65 \times 2 + 1.867 = 7.167$
$\hat{f}_2(2) = 3.35 \times 2 + 0.367 = 7.067$
$E[\hat{f}(2)] = \frac{7.167 + 7.067}{2} = 7.117$

$\text{Bias}^2 = (7.117 - 7)^2 = 0.0137$
$\text{Variance} = \frac{(7.167 - 7.117)^2 + (7.067 - 7.117)^2}{2} = 0.0025$

理论总误差：$0.0137 + 0.0025 + 0.25 = 0.2662$

直接计算：
- $D_1$：$E[(y - 7.167)^2] = (7.167 - 7)^2 + 0.25 = 0.0279 + 0.25 = 0.2779$
- $D_2$：$E[(y - 7.067)^2] = (7.067 - 7)^2 + 0.25 = 0.0045 + 0.25 = 0.2545$

平均：$\frac{0.2779 + 0.2545}{2} = 0.2662$ ✓ 一致。

**洞察**：$\text{Bias}^2 = 0.0137 > \text{Variance} = 0.0025$，模型主要是"偏"不是"抖"。噪声（$0.25$）是误差的压倒性主成分（$94\%$）。
</details>

### 题目 3 — 学习曲线判读一

你训练了一个文本分类模型。学习曲线如下：
- 训练误差：始终在 $0.03$ 附近，略微下降
- 验证误差：在 $0.25$ 附近，基本水平
- 两者之间始终有约 $0.22$ 的间隙

诊断是什么？你应该尝试哪些改进？哪些改进不会有帮助？

<details>
<summary>解答</summary>

**诊断：高方差（过拟合）**——训练误差低、验证误差高，两者有固定的大间隙。

**应该尝试（按优先级）**：
1. 数据增强（文本回译、同义词替换、随机删除/交换词）
2. 加强正则化（增加 Dropout rate、增大 weight decay）
3. 早停（在验证损失最低点停止训练）
4. 预先收集更多标注数据

**不会有帮助的**：
- 增加模型复杂度（只会让过拟合更严重）
- 减少正则化（反方向操作）
- 更换优化器（问题不在优化而在泛化）

**关键判读**：验证误差线水平 + 间隙不变 → 加数据帮助不大（如果间隙逐渐缩小，加数据才有效）。
</details>

### 题目 4 — 学习曲线判读二

另一个模型的学习曲线：
- 训练误差：$0.45$ 附近
- 验证误差：$0.48$ 附近
- 两者非常接近，几乎贴着

但你发现 baseline（随机猜测）的误差是 $0.50$。诊断是什么？你应该做什么？

<details>
<summary>解答</summary>

**诊断：高偏差（严重欠拟合）**——两条曲线在高位汇合，且比随机猜测好不了多少。

模型几乎没有学到任何有用的规律。你"控制"了方差，但偏差代价太大了。

**应该尝试**：
1. 大幅增加模型复杂度（加层、加参数、换更强架构）
2. 检查特征工程是否有问题（有没有丢失关键特征？）
3. 减少/移除正则化（当前正则化可能太强）
4. 检查数据标签是否正确（标签错误会导致高偏差假象）
5. 检查学习率是否太小或优化是否收敛

**注意**：这种情况加数据几乎是浪费时间——模型连当前数据都学不好，加更多数据也不会让一个简单模型变聪明。
</details>

### 题目 5 — 不可约误差的真面目

在偏差-方差分解中，$\sigma^2$ 被称为"不可约误差"。即使你有 $f(x)$（完美模型），为什么你仍然无法消除 $\sigma^2$？给一个真实世界的例子。

<details>
<summary>解答</summary>

因为 $y = f(x) + \epsilon$，而 $\epsilon$ 与 $x$ 无关——它来自数据生成过程中的内在随机性。即使 $\hat{f} = f$（偏差 = 0，方差 = 0），仍然有：

$$E[(y - f(x))^2] = E[\epsilon^2] = \sigma^2$$

**真实世界例子——预估外卖送达时间**：
- $x$：餐厅位置、骑手位置、交通状况、天气、订单量……
- $f(x)$：在给定条件下送达时间的"平均值"
- $\epsilon$：随机因素——电梯等了 2 分钟、路口被挡了一下、骑手接了个电话……
- $\sigma^2$：这些随机因素导致的时间方差

即使你掌握了 $f(x)$（知道所有条件下送达时间的期望值），某一次具体配送仍然有随机波动。**$\sigma^2$ 不是模型的问题，是现实世界本身就是随机的。**

**另一个例子**：投骰子预测。即使你训练出完美模型预测投掷一个公平骰子的期望值（$3.5$），单次投掷的结果仍然在 $1-6$ 之间波动——这就是 $\sigma^2$。
</details>

### 题目 6 — 交叉验证的作用

从偏差-方差的角度，为什么 k-fold 交叉验证比单次 train-test split 更可靠？CV 降低的是什么？

<details>
<summary>解答</summary>

单次 train-test split 的评估分数是一个随机变量（依赖于恰好哪些样本进了测试集）。其方差为 $v$（评估方差，不是模型方差）。

k 折交叉验证计算 $k$ 次评估并取平均：
$$\text{CV score} = \frac{1}{k}\sum_{i=1}^{k} \text{score}_i$$

如果 $k$ 次评估近似独立：
$$\text{Var}(\text{CV score}) \approx \frac{v}{k}$$

**CV 降低的是评估的方差**——让你更准确地估计模型在总体上的真实性能，而不是恰好碰上一个"好/坏"的 train-test split 就下结论。

**CV 不降低模型方差**。一个高方差的模型在 CV 下仍然是高方差的——CV 只是让你更精确地**发现自己有一个高方差的模型**。

**关键区分**：
- 降低模型方差 → 加数据、正则化、简化模型
- 降低评估方差 → 多次评估取平均（CV）

两者是不同层面的事。
</details>

### 题目 7 — 正则化的偏差²-方差调节

L2 正则化在损失函数中加入 $\lambda\|w\|^2$。

(a) $\lambda = 0$ 时偏差² 和方差如何？为什么？
(b) $\lambda \to \infty$ 时偏差² 和方差如何？为什么？
(c) 画出 $\lambda$ 从 $0$ 增大到 $\infty$ 时，偏差² 和方差的定性变化曲线。最优 $\lambda$ 在什么位置？

<details>
<summary>解答</summary>

**(a) $\lambda = 0$**：无约束，模型完全拟合训练数据。偏差² 最小（模型能表达真实函数），方差最大（对训练数据中每个噪声点都斤斤计较）。

**(b) $\lambda \to \infty$**：$w \to 0$，模型退化到常数预测。方差 → 0（训练数据怎么变，预测几乎不变），偏差² 极大（模型只会说一个数）。

**(c)** 定性曲线：

```
        ↑
    高   │    方差
        │  ╲
        │    ╲
        │      ╲
        │        ╲________
        │
        │         _______
        │       ╱
        │     ╱    偏差²
    低   │   ╱
        └──────────────────→ λ
        0         λ*      ∞
               ←最优→
```

- $\lambda = 0$：方差最大，偏差² 最小
- $\lambda$ 增大：方差单调递减，偏差² 单调递增
- $\lambda \to \infty$：方差→0，偏差²→最大值

最优 $\lambda^*$ 是**总误差 = 偏差² + 方差**取最小值的点（注意不是偏差² = 方差的点！）。

实际操作中用交叉验证搜索 $\lambda^*$：
```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
model.fit(X_train, y_train)
print(f"最优 λ: {model.alpha_}")
```
</details>

### 题目 8 — 集成与方差

随机森林通过训练 $m$ 棵决策树并取平均来做预测。假设每棵树的预测方差为 $\sigma^2$，树之间的相关系数为 $\rho$（$0 \le \rho < 1$）。

(a) 写出 $m$ 棵树平均后的方差公式。
(b) 当 $m \to \infty$ 时方差是多少？这说明什么？
(c) 随机森林的哪些设计直接降低了 $\rho$？

<details>
<summary>解答</summary>

**(a)** 对于 $m$ 个方差相同、两两相关系数为 $\rho$ 的随机变量，平均值的方差为：

$$\text{Var}\left(\frac{1}{m}\sum_{i=1}^{m} \hat{f}_i\right) = \rho\sigma^2 + \frac{1-\rho}{m}\sigma^2$$

推导思路：$\text{Var}(\bar{f}) = \frac{1}{m^2}\left(\sum_i \text{Var}(f_i) + \sum_{i \neq j} \text{Cov}(f_i, f_j)\right)$，其中 $\text{Cov}(f_i, f_j) = \rho\sigma^2$。

**(b)** 当 $m \to \infty$，$\frac{1-\rho}{m} \to 0$，方差 $\to \rho\sigma^2$。

**含义**：集成方法的方差有一条**不可逾越的下界**——树之间的相关性 $\rho$ 决定了方差的最低极限。如果所有树都高度相关（$\rho \approx 1$），即使有 $10000$ 棵树，方差也降不了多少。

**(c)** 随机森林降低 $\rho$ 的两个关键设计：
1. **Bootstrap 采样**：每棵树用不同的数据子集训练 → 树学到的规律略有不同 → $\rho$ 降低
2. **随机特征子集**：每次分裂时只考虑 $\sqrt{d}$ 个特征 → 即使数据相同，分裂路径也可能不同 → $\rho$ 进一步降低

没有这两个随机化，所有决策树会几乎完全一样（$\rho \approx 1$），Bagging 就毫无意义。
</details>

### 题目 9 — 双重下降的直觉

传统理论说测试误差是 U 形，但双重下降发现在过参数化区域误差可能再次下降。为什么过参数化不会导致方差无限大？

<details>
<summary>解答</summary>

当模型参数数量 $p$ **远大于**训练样本数 $n$ 时，欠定方程组 $Xw = y$ 有无穷多组解。但优化算法（如梯度下降从 $w=0$ 初始化）**会隐式地选择其中"最简约"的那一组解**。

在最小二乘下：梯度下降收敛到的是所有可行解中 $\|w\|_2$ 最小的那个——相当于隐式 L2 正则化。

**直觉**：
- $p \approx n$（插值阈值）：只有一个解，这个解必须精确拟合每个训练点（包括噪声），导致方差爆炸
- $p \gg n$：有无穷多解，优化算法选了"最平滑"的那个 → 噪声被隐式地平滑掉了 → 方差反而比 $p \approx n$ 小

**类比**：你有 2 个点，要求用经过它们的直线来拟合 → 只有一条直线。如果你有 3 个点但允许用三次函数 → 有无穷多条三次曲线过这 3 个点 → 你自然选最平滑的那条，它比被迫精确过点的低次插值更不奇怪。
</details>

### 题目 10 — 深度学习中偏差-方差的新理解

经典偏差-方差 U 形曲线来自 $n > p$ 的统计学习场景。深度学习（参数数量通常远超训练样本数）为什么经常能同时获得低偏差和低方差？给出至少两个原因。

<details>
<summary>解答</summary>

经典 U 形的前提是：同一模型族、固定数据量、无隐式正则化。DL 至少在以下维度打破了这些前提：

**1. 隐式正则化（Implicit Regularization）**
SGD 的优化轨迹自身带有隐式偏置：从零初始化、用小批量 SGD 训练的过参数化网络，收敛到的解在泛函空间里通常是最小范数的——相当于免费获得了 L2 正则化。这让一个参数数量巨大的网络实际上只使用了远小于理论"自由度"的有效容量。

**2. 架构先验（Architectural Priors）**
CNN 的平移等变性、Transformer 的自注意力机制都嵌入了强有力的结构先验。这些先验把搜索空间从一个"任意函数"的广袤海洋缩小到了一个"在这个物理世界中有意义"的湖泊——同时降低偏差（更准确的函数族）和有效方差（更小的搜索空间）。

**3. 大量数据的方差稀释**
海量训练数据（百万/亿级）使得方差项被严重压低。经典 U 形是在数据量**固定**的前提下画的，而 DL 实践中数据和模型规模是一起增长的。

**4. 表示能力的非连续跃迁**
神经网络不是"逐渐变复杂"——存在相变现象：当宽度/深度超过某个阈值，网络突然能表示某类函数。在阈值之前偏差高、方差低（模型根本表示不了那个函数）；跨过阈值后偏差跳水，但方差不一定同步上升。

**核心**：DL 的成功不是"绕过了"偏差-方差权衡，而是在更高维度上重新定义了这场博弈。
</details>

---

下一步：[数据预处理](./03-data-preprocessing.md)
