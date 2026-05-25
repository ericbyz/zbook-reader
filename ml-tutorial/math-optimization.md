# 第 2 章：最优化基础 — 机器学习的引擎

> 一切机器学习模型的训练，本质上都是在求解一个最优化问题。理解优化，你就理解了"学习"到底是怎么发生的。

---

## 什么是最优化？

### 生活中处处是最优化

你每天都在无意识中求解优化问题，只是没意识到：

- **导航**：从家到公司有 5 条路。哪条最短？哪条最不堵？你在"最小化通勤时间"。
- **购物**：三个超市里同一款牛奶卖不同价钱。哪家最便宜？你在"最小化花费"。
- **烤蛋糕**：烤箱温度设为多少？太低烤不熟，太高烤糊了。你在寻找"最佳温度"。
- **买股票**：什么时候买入、什么时候卖出才能赚最多？你在"最大化收益"。

这些问题的共同模式是：**在给定的条件下，找到让某个"好坏标准"达到极值的方案**。

把这个直觉翻译成机器学习：模型有一堆参数（比如线性回归的斜率和截距），你有一堆数据。你希望调整这些参数，让模型的预测和真实数据之间的"差距"尽可能小。这个"差距"就是损失函数，而寻找最优参数的过程，就是最优化。

| 生活场景 | 你要调的变量 | 你要优化的目标 |
|----------|-------------|---------------|
| 导航选路 | 走哪条路 | 最小化时间 |
| 烤箱烤蛋糕 | 温度、时间 | 烤出最好的口感 |
| 买股票 | 买入价、卖出时机 | 最大化利润 |
| 训练模型 | 模型参数 | 最小化预测误差 |

整个机器学习，归根结底就是两件事：

1. **定义一个损失函数** — 告诉计算机"什么叫做好"
2. **用优化算法找到使损失最小的参数** — 让计算机自己"学会"

从最简单的线性回归到 GPT-4，核心逻辑完全一样。

---

## 1. 一元函数的最优化 — 从初中数学出发

### 1.1 生活例子 → 直观理解

假设你开了一家奶茶店。每杯奶茶的**定价**是 $x$ 元。定价太低，卖得多但不赚钱；定价太高，没人买。你发现每天的利润可以近似表示为：

$$f(x) = -(x-15)^2 + 500$$

意思是：定价 15 元时利润最高（500 元），偏离 15 元越多利润越低。你的任务很简单：**找到让 $f(x)$ 最大的 $x$**。

画出来是一个倒扣的碗（开口向下的抛物线），最高点一目了然。但问题来了——如果函数很复杂，你没法一眼看出来呢？你需要一个系统的方法。

### 1.2 形式化定义：导数为零

对于光滑函数 $f(x)$，在极值点（最小值或最大值）处，切线是水平的——导数为零：

$$f'(x) = 0$$

但要区分是最大值还是最小值，需要看二阶导数：

- $f''(x) > 0$：开口向上 → **最小值** （碗底）
- $f''(x) < 0$：开口向下 → **最大值** （山顶）
- $f''(x) = 0$：可能是拐点，需要进一步判断

### 1.3 手算示例：求 $f(x) = x^2 - 4x + 5$ 的最小值

这是你中学就见过的最简单形式。让我们一起亲手算一遍，每一步都不能跳：

**第一步：求导数。**

$$f'(x) = 2x - 4$$

**第二步：令导数等于零，解方程。**

$$2x - 4 = 0 \quad \Rightarrow \quad x = 2$$

**第三步：验证是极小值。**

$$f''(x) = 2 > 0 \quad \Rightarrow \quad x=2 \text{ 处是极小值}$$

**第四步：代入原函数，得到极小值。**

$$f(2) = 2^2 - 4 \times 2 + 5 = 4 - 8 + 5 = 1$$

**结论：函数 $f(x)=x^2-4x+5$ 的最小值是 1，在 $x=2$ 处取得。**

> **常见误区**：很多人以为 $f'(x)=0$ 就一定是极值点。但 $f(x)=x^3$ 在 $x=0$ 处 $f'(0)=0$，却既不是最大值也不是最小值——它是一个拐点。$f'(x)=0$ 只是"候选点"，需要二阶导数来确认。

### 1.4 为什么需要数值方法？

上面的例子中，$f'(x)=0$ 是一个简单的一次方程，轻松解出 $x=2$。但现实没那么友好。

考虑这个函数：

$$f(x) = x^4 - 3x^3 + 2x + \sin(x)$$

它的导数是：

$$f'(x) = 4x^3 - 9x^2 + 2 + \cos(x)$$

令 $f'(x)=0$：$4x^3 - 9x^2 + 2 + \cos(x) = 0$

这个方程**没有解析解**——你无法像解 $2x-4=0$ 那样写出一个精确的公式。三角函数的介入让一切变得复杂。

更根本的原因是：在机器学习中，损失函数往往由**几十万条数据**定义，根本没有一个简洁的公式，更谈不上"求导然后令其等于零"。你只能通过**数值方法**——也就是一步一步地试探——来逼近最优解。

这就是梯度下降的用武之地。

### 1.5 Python 验证

用代码验证一下解析解的计算：

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x + 5

def df(x):
    """导数 f'(x) = 2x - 4"""
    return 2*x - 4

x = np.linspace(-1, 5, 200)
y = f(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', linewidth=2, label='$f(x)=x^2-4x+5$')
plt.axvline(x=2, color='r', linestyle='--', alpha=0.7, label='$x=2$ (最小值)')
plt.plot(2, f(2), 'ro', markersize=10)
plt.annotate(f'最小值 (2, {f(2)})', xy=(2, f(2)), xytext=(3.5, 6),
             arrowprops=dict(arrowstyle='->'), fontsize=11)

plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('一元函数的最优化：$f\'(x)=0$ 给出了最小值的位置')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

---

## 2. 梯度下降 — 当解析解不存在时

### 2.1 生活例子：浓雾中的下山者

想象你在大雾弥漫的山里，能见度只有脚下几米。你不知道整座山的形状，也不知道最低点在哪里。你要怎么下山？

唯一的策略是：**看看脚下哪个方向是下坡，往那个方向走一小步；到了新位置，重新判断方向，再走一小步；重复，直到感觉脚下已经是平地。**

在数学上：
- "脚下哪个方向是下坡" = **梯度**（导数），它告诉你函数增长最快的方向
- "往反方向走" = **负梯度方向**，这就是下山的方向
- "走一小步" = **学习率** $\eta$ 控制步长

### 2.2 从数学推导梯度下降

对于一元函数 $f(x)$，在当前位置 $x$ 附近做一阶泰勒展开：

$$f(x + \Delta x) \approx f(x) + f'(x) \cdot \Delta x$$

我们的目标是让 $f(x + \Delta x)$ 变小。也就是说，需要 $f'(x) \cdot \Delta x < 0$（增量项为负）。

怎么选 $\Delta x$ 才能让这个乘积最负？答案是：**让 $\Delta x$ 的方向和 $f'(x)$ 相反**：

$$\Delta x = -\eta \cdot f'(x), \quad \eta > 0$$

代入泰勒展开：

$$f(x + \Delta x) \approx f(x) - \eta \cdot [f'(x)]^2 \leq f(x)$$

因为 $[f'(x)]^2 \geq 0$，$-\eta \cdot [f'(x)]^2 \leq 0$，函数值一定不会增加。

于是得到**梯度下降的更新公式**：

$$\boxed{x_{t+1} = x_t - \eta \cdot f'(x_t)}$$

这就是一切深度学习训练的数学根基。

### 2.3 亲手算一遍：$f(x)=x^2$ 的梯度下降

这是最简单的优化问题：找到 $f(x)=x^2$ 的最小值。显然 $x=0$ 时最小（$f=0$）。但我们假装不知道答案，用梯度下降一步步逼近。

已知：$f(x)=x^2$，$f'(x)=2x$，起始点 $x_0 = 3$，学习率 $\eta = 0.1$。

| 迭代 | $x$ | $f'(x)=2x$ | $\Delta x = -\eta f'(x)$ | 新 $x$ | $f($新$x)$ |
|------|-----|-----------|------------------------|-------|------------|
| 0 | 3.0000 | 6.0000 | -0.6000 | 2.4000 | 5.7600 |
| 1 | 2.4000 | 4.8000 | -0.4800 | 1.9200 | 3.6864 |
| 2 | 1.9200 | 3.8400 | -0.3840 | 1.5360 | 2.3593 |
| 3 | 1.5360 | 3.0720 | -0.3072 | 1.2288 | 1.5100 |
| 4 | 1.2288 | 2.4576 | -0.2458 | 0.9830 | 0.9664 |
| 5 | 0.9830 | 1.9661 | -0.1966 | 0.7864 | 0.6185 |
| 6 | 0.7864 | 1.5729 | -0.1573 | 0.6291 | 0.3958 |
| 7 | 0.6291 | 1.2583 | -0.1258 | 0.5033 | 0.2533 |

8 步之后，$x$ 从 3 降到了约 0.5，函数值从 9 降到了约 0.25。继续迭代，$x$ 会越来越接近 0。

**观察**：每一步的步长都在缩小——因为越靠近最小值，斜率越小，更新量也越小。这是一个微妙的平衡：在远离最优解的地方大步前进，在靠近最优解的地方小心翼翼。梯度下降的"自适应"步长正是来自斜率本身的变化。

### 2.4 学习率的选择 — 最关键的调参

学习率 $\eta$ 决定了每一步迈多大。这是整个优化过程中**最重要的超参数**。

仍然用 $f(x)=x^2$，起始点 $x_0=3$，对比三种学习率：

#### 情况一：学习率太小 — $\eta = 0.01$

| 迭代 | $x$ | $f(x)$ |
|------|-----|--------|
| 0 | 3.000 | 9.000 |
| 1 | 2.940 | 8.644 |
| 5 | 2.714 | 7.367 |
| 10 | 2.455 | 6.025 |
| 20 | 2.008 | 4.033 |
| 50 | 1.051 | 1.105 |

**问题**：走得太慢了。50 步之后还没到最优解的一半。

#### 情况二：学习率刚好 — $\eta = 0.1$

| 迭代 | $x$ | $f(x)$ |
|------|-----|--------|
| 0 | 3.000 | 9.000 |
| 5 | 0.983 | 0.966 |
| 10 | 0.322 | 0.104 |
| 20 | 0.035 | 0.001 |

**表现**：稳定、高效地收敛。每一步都在进步。

#### 情况三：学习率太大 — $\eta = 1.1$

| 迭代 | $x$ | $f(x)$ |
|------|-----|--------|
| 0 | 3.000 | 9.000 |
| 1 | -3.600 | 12.960 |
| 2 | 4.320 | 18.662 |
| 3 | -5.184 | 26.874 |
| 4 | 6.221 | 38.699 |

**灾难**：$x$ 在正负之间越跳越远，函数值越来越大——**发散**了！

#### 直观总结

```
学习率太小：    ●→●→●→●→●→●→●→●→●→●→●→●   （蜗牛爬，半天不到）
学习率刚好：    ●→●→●→●→○               （稳定高效到终点）
学习率太大：    ●→←●→←●→←→...↗           （震荡发散，越走越远）
```

**经验法则**：学习率通常从 $0.1$、$0.01$、$0.001$ 这三个数量级开始尝试。对于 $f(x)=x^2$，可以推导出收敛条件是 $\eta < 1$（见思考题 4）。

> **常见误区**：很多人认为"学习率越大，收敛越快"。前半句对——在初期大学习率确实快。但超过某个临界值后，不仅不会更快，反而会震荡甚至发散。存在一个"甜蜜点"：刚好比临界值小一点的时候收敛最快。

### 2.5 Python：一元梯度下降的完整实现

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_1d(df, x0, lr, n_iters):
    """一元梯度下降，返回所有历史位置"""
    path = [x0]
    x = x0
    for _ in range(n_iters):
        x = x - lr * df(x)
        path.append(x)
    return path

def f(x): return x**2
def df(x): return 2*x

# 对比三种学习率
x0 = 3.0
results = {}
for lr, label in [(0.01, '太小 η=0.01'), (0.1, '合适 η=0.1'), (1.01, '太大 η=1.01')]:
    results[label] = gradient_descent_1d(df, x0, lr, 20)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x_vals = np.linspace(-4, 4, 200)
axes[0].plot(x_vals, f(x_vals), 'k-', linewidth=1.5, label='$f(x)=x^2$')
axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)
colors = ['blue', 'green', 'red']
for (label, path), c in zip(results.items(), colors):
    axes[0].plot(path, f(np.array(path)), '-o', color=c, markersize=3,
                 label=label, alpha=0.8)
axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
axes[0].set_title('梯度下降轨迹'); axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

for (label, path), c in zip(results.items(), colors):
    losses = f(np.array(path))
    axes[1].plot(losses, '-o', color=c, markersize=3, label=label)
axes[1].set_yscale('log')
axes[1].set_xlabel('迭代次数'); axes[1].set_ylabel('Loss (log scale)')
axes[1].set_title('损失下降曲线（对数坐标）')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 3. 多元梯度下降 — 从一根轴到高维空间

### 3.1 直观理解

生活中很少有只依赖一个因素的决定。比如选房子，你要同时考虑面积和通勤时间。目标函数变成了 $f(\text{面积}, \text{通勤时间})$，有两个变量。

在数学上，多元函数的"坡度"不再是一个数，而是一个**向量**——梯度。梯度的每个分量告诉你：**在这一维上，往哪个方向走函数增长最快**。梯度的反方向，就是下山的方向。

### 3.2 形式化定义

对于二元函数 $f(x, y)$，梯度是一个向量：

$$\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)$$

更新公式与一元完全一样，只是把标量换成向量：

$$\boxed{\begin{pmatrix} x_{t+1} \\ y_{t+1} \end{pmatrix} = \begin{pmatrix} x_t \\ y_t \end{pmatrix} - \eta \cdot \nabla f(x_t, y_t)}$$

### 3.3 亲手算一遍：$f(x, y) = x^2 + 2y^2$

这个函数像一个被压扁的碗——$y$ 方向比 $x$ 方向更陡（系数 2 vs 1）。

已知：$f(x,y) = x^2 + 2y^2$，$\nabla f = (2x, 4y)$，起始点 $(3, 3)$，$\eta = 0.1$。

| 迭代 | $x$ | $y$ | $\nabla f$ | $\Delta$ | $f(x,y)$ |
|------|-----|-----|-----------|---------|----------|
| 0 | 3.000 | 3.000 | (6.00, 12.00) | (-0.60, -1.20) | 27.00 |
| 1 | 2.400 | 1.800 | (4.80, 7.20) | (-0.48, -0.72) | 12.24 |
| 2 | 1.920 | 1.080 | (3.84, 4.32) | (-0.38, -0.43) | 6.02 |
| 3 | 1.536 | 0.648 | (3.07, 2.59) | (-0.31, -0.26) | 3.20 |
| 4 | 1.229 | 0.389 | (2.46, 1.56) | (-0.25, -0.16) | 1.81 |
| 5 | 0.983 | 0.233 | (1.97, 0.93) | (-0.20, -0.09) | 1.08 |

**关键观察**：$y$ 下降得比 $x$ 快得多。因为 $y$ 方向的梯度系数是 4，而 $x$ 方向只有 2。这说明梯度下降**自动**在每个方向上分配不同的步长——梯度大的方向走大步，梯度小的方向走小步。

但同时这也暴露了一个问题：如果某个方向梯度太大，用固定学习率就可能过度震荡（就像学习率太大的情况）。这是后续动量法和自适应方法的出发点。

### 3.4 Python 验证

```python
import numpy as np
import matplotlib.pyplot as plt

def f_2d(x, y):
    return x**2 + 2 * y**2

def grad_2d(x, y):
    return np.array([2*x, 4*y])

def gd_2d(grad_fn, x0, lr, n_iters):
    path = [np.array(x0, dtype=float)]
    for _ in range(n_iters):
        w = path[-1]
        path.append(w - lr * grad_fn(*w))
    return np.array(path)

path = gd_2d(grad_2d, [3.0, 3.0], lr=0.1, n_iters=30)

x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
Z = f_2d(X, Y)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].contour(X, Y, Z, levels=25, cmap='viridis', alpha=0.6)
axes[0].plot(0, 0, 'r*', markersize=14)
axes[0].plot(path[:, 0], path[:, 1], '-o', color='blue',
             markersize=2, linewidth=0.8)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title('梯度下降在等高线上：y 方向收敛更快')
axes[0].grid(True, alpha=0.3)

losses = [f_2d(*p) for p in path]
axes[1].semilogy(losses, 'b-o', markersize=3)
axes[1].set_xlabel('迭代次数'); axes[1].set_ylabel('Loss (log scale)')
axes[1].set_title('损失随迭代下降')
axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. 批量 / 随机 / 小批量 — 大数据下的梯度计算策略

### 4.1 直观理解：食堂打饭的比喻

假设你要估算全校学生的平均身高。有三种策略：

- **BGD（批量梯度下降）**：测量**每一个**学生的身高，算精确平均值。太慢了，全校两万人，等你量完天都黑了。
- **SGD（随机梯度下降）**：**随机抓一个**路过的学生量身高，用一个人的身高"代表"全校平均。很快，但误差很大——万一抓到的是篮球队员呢？
- **Mini-batch GD（小批量）**：每次抓 **32 个**学生，算这 32 人的平均身高。既不太慢，也不太吵。这就是深度学习的标准做法。

在数学上，三种方法的区别在于**用多少数据来估算梯度**：

| 方法 | 每步数据量 | 梯度准确性 | 每步速度 | 噪声水平 |
|------|-----------|-----------|---------|---------|
| BGD | 全量 $n$ | 准确 | 慢 $O(n)$ | 无 |
| SGD | 1 条 | 很差 | 快 $O(1)$ | 极大 |
| Mini-batch | $b$ 条（如 32） | 较好 | 快 $O(b)$ | 适中 |

### 4.2 手算示例：一个微型回归问题

数据：$(1, 2), (2, 4), (3, 6)$ —— 完美落在直线 $y=2x$ 上。
模型：$\hat{y} = wx$ （无截距），损失用 MSE。
目标：用梯度下降找到 $w=2$。

损失函数：$J(w) = \frac{1}{3}[(2 - w)^2 + (4 - 2w)^2 + (6 - 3w)^2]$
梯度：$J'(w) = -\frac{2}{3}[(2 - w) + 2(4 - 2w) + 3(6 - 3w)]$

初始 $w_0 = 0$，$\eta = 0.1$。

#### BGD：一次用全部 3 条数据

第一步梯度：$J'(0) = -\frac{2}{3}[2 + 8 + 18] = -\frac{2}{3} \cdot 28 = -18.667$
$w_1 = 0 - 0.1 \times (-18.667) = 1.867$

一步就从 0 跳到了 1.867，非常接近真实值 2。

#### SGD：一次只用 1 条数据，逐条遍历

- 样本 $(1, 2)$：梯度 $= -2 \times 1 \times (2 - 0) = -4$，$w = 0 - 0.1 \times (-4) = 0.40$
- 样本 $(2, 4)$：梯度 $= -2 \times 2 \times (4 - 2 \times 0.40) = -4 \times 3.2 = -12.8$，$w = 0.40 - 0.1 \times (-12.8) = 1.68$
- 样本 $(3, 6)$：梯度 $= -2 \times 3 \times (6 - 3 \times 1.68) = -6 \times 0.96 = -5.76$，$w = 1.68 - 0.1 \times (-5.76) = 2.256$

一轮过后 $w=2.256$，已经超过了真实值 2——这是 SGD 的噪声在作怪。但平均来看方向是对的。

#### Mini-batch：一次用 2 条数据

- Batch $\{(1,2), (2,4)\}$：梯度 $= -\frac{2}{2}[1\times2 + 2\times4] = -[2+8] = -10$，$w = 0 - 0.1 \times (-10) = 1.0$
- Batch $\{(3,6), (1,2)\}$：梯度 $= -\frac{2}{2}[3\times(6-3\times1) + 1\times(2-1\times1)] = -[9+1] = -10$，$w = 1.0 - 0.1 \times (-10) = 2.0$

两轮就到了 $w=2.0$！在这个完美的数据上，Mini-batch 展现了惊人的效率。

### 4.3 SGD 的噪声是礼物还是诅咒？

SGD 的梯度很"吵"——每次更新方向波动很大。这听起来是个缺点，但在深度学习中反而是优势：

1. **噪声帮助逃脱鞍点和差的局部最优**。精算的 BGD 容易被困住，而 SGD 摇摇晃晃反而可能"弹"出去。
2. **噪声自带正则化效果**。每次梯度都略有偏差，相当于隐式地给模型加了扰动，减少了过拟合。
3. **在线学习**。SGD 可以一条一条地处理数据，适用于流式场景（比如用户点击数据实时到来）。

**常见误区**：SGD 收敛到最优解的路上会很"颠簸"，但配合**学习率衰减**（越往后学习率越小），这种颠簸会逐渐减弱，最终稳定在最优解附近。

### 4.4 Python 实现对比

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 2000
X = np.random.randn(n, 1)
true_w, true_b = 3.0, 2.0
y = true_w * X.squeeze() + true_b + np.random.randn(n) * 0.5

def compute_grad(w, b, X_batch, y_batch):
    n = len(X_batch)
    pred = w * X_batch.squeeze() + b
    dw = -2/n * np.sum((y_batch - pred) * X_batch.squeeze())
    db = -2/n * np.sum(y_batch - pred)
    return dw, db

def mse(w, b): return np.mean((y - (w * X.squeeze() + b))**2)

def bgd(epochs=50, lr=0.01):
    w, b, losses = 0.0, 0.0, []
    for _ in range(epochs):
        dw, db = compute_grad(w, b, X, y)
        w -= lr * dw; b -= lr * db
        losses.append(mse(w, b))
    return w, b, losses

def sgd(epochs=5, lr=0.001):
    w, b, losses = 0.0, 0.0, []
    for _ in range(epochs):
        for i in range(n):
            dw, db = compute_grad(w, b, X[i:i+1], y[i:i+1])
            w -= lr * dw; b -= lr * db
            losses.append(mse(w, b))
    return w, b, losses

def mini_batch(epochs=15, batch_size=32, lr=0.05):
    w, b, losses = 0.0, 0.0, []
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            bi = idx[start:start+batch_size]
            dw, db = compute_grad(w, b, X[bi], y[bi])
            w -= lr * dw; b -= lr * db
        losses.append(mse(w, b))
    return w, b, losses

w_b, b_b, loss_b = bgd()
w_m, b_m, loss_m = mini_batch()
w_s, b_s, loss_s = sgd()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(loss_b, label='BGD')
axes[0].plot(loss_m, label='Mini-batch (32)')
sgd_sampled = loss_s[::n][:len(loss_b)]
axes[0].plot(range(1, len(sgd_sampled)+1), sgd_sampled, label='SGD (per epoch)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
axes[0].legend(); axes[0].set_title('损失下降曲线')
axes[0].grid(True, alpha=0.3)

bars = axes[1].bar(['BGD', 'Mini-batch(32)', 'SGD'], [n, 32, 1],
                   color=['steelblue', 'coral', 'seagreen'])
axes[1].set_ylabel('每步使用的样本数')
axes[1].set_title('计算量对比')
plt.tight_layout(); plt.show()

print(f"真实: w={true_w}, b={true_b}")
print(f"BGD:  w={w_b:.4f}, b={b_b:.4f}")
print(f"Mini: w={w_m:.4f}, b={b_m:.4f}")
print(f"SGD:  w={w_s:.4f}, b={b_s:.4f}")
```

---

## 5. 动量法 — 给梯度下降装上"惯性"

### 5.1 直观理解：滚下山坡的球

纯梯度下降像一个**没重量的乒乓球**：每一步完全由当前坡度决定，到了平坦区域就停下来，遇到陡峭峡谷就来回震荡。

动量法像一个**有重量的保龄球**：它不仅有当前推力（梯度），还有之前积累的**速度**。遇到小坑，凭惯性直接碾过去；遇到震荡，速度的指数平均会抵消掉来回的噪音。

核心思想是**指数加权移动平均（EMA）**：不只看当前的梯度，而是看"最近一段时间"梯度的加权平均，越近的梯度权重越大。

### 5.2 形式化定义

$$\begin{aligned}
\mathbf{v}_t &= \beta \mathbf{v}_{t-1} + (1-\beta) \nabla f(\mathbf{w}_t) \quad &\text{（速度 = 动量 × 旧速度 + 当前梯度）} \\
\mathbf{w}_{t+1} &= \mathbf{w}_t - \eta \mathbf{v}_t \quad &\text{（位置 = 旧位置 - 学习率 × 速度）}
\end{aligned}$$

$\beta$ 通常取 0.9，意味着旧速度保留 90%，新梯度只占 10%。这相当于对过去约 $\frac{1}{1-\beta}=10$ 步的梯度做加权平均。

### 5.3 手算对比：动量的威力

用一个极端不对称的函数：$f(x, y) = 0.05x^2 + 5y^2$。$y$ 方向的梯度系数是 100（$10y$），$x$ 方向只有 0.1（$0.1x$）。这是一个狭长峡谷——$y$ 方向极其陡峭但谷底窄，$x$ 方向平缓但路程长。

起始点 $(5, 1)$，$\eta = 0.01$，$\beta = 0.9$。

**纯梯度下降**（$\eta = 0.01$，必须很小否则 $y$ 方向发散）：

| 迭代 | $x$ | $y$ | $f(x,y)$ |
|------|-----|-----|----------|
| 0 | 5.000 | 1.000 | 6.250 |
| 1 | 4.998 | 0.900 | 5.299 |
| 2 | 4.995 | 0.810 | 4.528 |
| 5 | 4.990 | 0.590 | 2.991 |
| 10 | 4.975 | 0.349 | 1.846 |

$x$ 方向进展极为缓慢——因为 $x$ 的梯度只有 0.25（$0.1 \times 5$），每步只移动 0.0025。

**动量法**（$\eta = 0.01$，$\beta = 0.9$）：

| 迭代 | $x$ | $y$ | $v_x$ | $v_y$ | $f(x,y)$ |
|------|-----|-----|-------|-------|----------|
| 0 | 5.000 | 1.000 | 0.00 | 0.00 | 6.250 |
| 1 | 5.000 | 0.990 | 0.05 | 1.00 | 6.150 |
| 2 | 4.998 | 0.882 | 0.10 | 1.89 | 5.139 |
| 5 | 4.984 | 0.393 | 0.28 | 3.36 | 2.014 |
| 10 | 4.902 | 0.072 | 0.54 | 1.45 | 1.227 |

动量的 $x$ 方向速度在累积增长——第 10 步时 $x$ 方向每步已能移动约 0.005，是纯 GD 的两倍。而 $y$ 方向的速度得到平滑，减少了震荡。

**核心结论**：动量法有两个好处——在平缓方向**加速**，在陡峭震荡方向**平滑**。

> **常见误区**：很多人认为动量的 $\beta=0.9$ 意味着"只记住最近 10 步的梯度"。更准确的理解是：$\beta=0.9$ 时，第 $t-10$ 步的梯度权重衰减到 $(0.9)^{10} \approx 0.35$——还剩 35%，并没有完全遗忘。动量的"记忆"是平滑衰减的，而不是一个硬截断窗口。这就像你回忆一周前的事情：不是"忘记"，而是"模糊"。

### 5.3.1 为什么动量能加速？（更深入的分析）

假设每个方向的梯度大致恒定。在平坦的 $x$ 方向，梯度恒为 $g_x$：

纯梯度下降：$x$ 每步移动 $\eta g_x$（恒定）。
动量法：$v_x$ 会从 0 逐渐积累到 $\frac{1-\beta}{1-\beta}g_x = g_x$（极限速度），$x$ 方向的移动从 $\eta g_x / 10$ 加速到 $\eta g_x$。

$v_x$ 的积累过程（每一步的速度）：
- 第 1 步：$v_x = 0.1 g_x$，步长 $= 0.1 \eta g_x$
- 第 2 步：$v_x = 0.9(0.1 g_x) + 0.1 g_x = 0.19 g_x$，步长 $= 0.19 \eta g_x$
- 第 5 步：$v_x \approx 0.41 g_x$，步长 $= 0.41 \eta g_x$
- 第 10 步：$v_x \approx 0.65 g_x$，步长 $= 0.65 \eta g_x$
- 第 30 步：$v_x \approx 0.96 g_x$，步长 $= 0.96 \eta g_x$

动量法的步长从 0.1 加速到接近 1.0——**10 倍的速度提升**！而在陡峭震荡的 $y$ 方向，相邻两步的梯度方向相反（一正一负），$v_y$ 会被 EMA 大幅抵消，步长反而被抑制。

这就是动量法的魔力：自动在**一致梯度方向加速**，在**震荡梯度方向减速**。

### 5.4 Python：动量法 vs 纯梯度下降

```python
import numpy as np
import matplotlib.pyplot as plt

def f_valley(x, y):
    return 0.05 * x**2 + 5 * y**2

def grad_valley(x, y):
    return np.array([0.1 * x, 10 * y])

def plain_gd(grad_fn, x0, lr, steps):
    path = [np.array(x0, dtype=float)]
    for _ in range(steps):
        path.append(path[-1] - lr * grad_fn(*path[-1]))
    return np.array(path)

def momentum_gd(grad_fn, x0, lr, beta, steps):
    path = [np.array(x0, dtype=float)]
    v = np.zeros(2)
    for _ in range(steps):
        g = grad_fn(*path[-1])
        v = beta * v + (1 - beta) * g
        path.append(path[-1] - lr * v)
    return np.array(path)

x0 = [5.0, 1.0]
path_gd = plain_gd(grad_valley, x0, lr=0.005, steps=80)
path_mom = momentum_gd(grad_valley, x0, lr=0.005, beta=0.9, steps=80)

x = np.linspace(-6, 6, 200)
y = np.linspace(-1.2, 1.2, 200)
X, Y = np.meshgrid(x, y)
Z = f_valley(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax in axes[:2]:
    ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
    ax.plot(0, 0, 'r*', markersize=14)
axes[0].plot(path_gd[:, 0], path_gd[:, 1], '-o', color='blue',
             markersize=2, linewidth=0.8)
axes[0].set_title('纯梯度下降（x 方向进展缓慢）')
axes[1].plot(path_mom[:, 0], path_mom[:, 1], '-o', color='orange',
             markersize=2, linewidth=0.8)
axes[1].set_title('动量法（x 方向加速，y 方向平滑）')
for ax in axes[:2]:
    ax.set_xlabel('x'); ax.set_ylabel('y')

loss_gd = [f_valley(*p) for p in path_gd]
loss_mom = [f_valley(*p) for p in path_mom]
axes[2].semilogy(loss_gd, 'b-', label='纯梯度下降', linewidth=1.5)
axes[2].semilogy(loss_mom, color='orange', label='动量法 β=0.9', linewidth=1.5)
axes[2].set_xlabel('迭代步数'); axes[2].set_ylabel('Loss (log scale)')
axes[2].set_title('损失下降对比'); axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 6. 自适应学习率 — 为每个参数定制步长

### 6.1 直观理解

到目前为止，所有参数共用一个学习率。但回顾那个狭长峡谷的例子：$y$ 方向梯度假大（需要小步），$x$ 方向梯度小（需要大步）。用一个学习率必然顾此失彼。

自适应方法的思路很简单：**根据每个参数的历史梯度信息，自动调整它的学习率**：
- 某个参数历史梯度一直很大 → 说明这里很陡 → 学习率调小，防止震荡
- 某个参数历史梯度一直很小 → 说明这里很平 → 学习率调大，加速前进

### 6.2 三大自适应方法

#### AdaGrad：累积历史梯度平方

$$s_t = s_{t-1} + (\nabla f_t)^2, \quad w_{t+1} = w_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot \nabla f_t$$

分母 $\sqrt{s_t}$ 越来越大，学习率单调递减。适用于稀疏特征（NLP 词向量），但深度学习训练中后期学习率会降到零，无法继续学习。

#### RMSProp：指数滑动平均替代累加

$$s_t = \beta s_{t-1} + (1-\beta)(\nabla f_t)^2, \quad w_{t+1} = w_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot \nabla f_t$$

用 EMA 替代 AdaGrad 的单调累加，学习率不会衰减到零。$\beta$ 通常取 0.99。

#### Adam：动量 + RMSProp + 偏差校正

Adam 是 Momentum 和 RMSProp 的融合体，也是目前**使用最广泛的优化器**：

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla f_t \quad &\text{（一阶矩 — 动量）} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(\nabla f_t)^2 \quad &\text{（二阶矩 — 自适应）} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad &\text{（偏差校正）} \\
w_{t+1} &= w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}$$

**偏差校正**是 Adam 的一个精妙设计。在训练刚开始时，$m_0=0$ 和 $v_0=0$，前几步的 EMA 严重偏向 0。除以 $1-\beta^t$ 可以放大它们，使初始几步也有合理的更新量。随着 $t$ 增大，校正因子趋近于 1，不再起作用。

### 6.3 优化器对比总结

| 优化器 | 核心思想 | 超参数 | 适用场景 |
|--------|---------|--------|---------|
| SGD | 朴素梯度下降 | `lr` | 需要精细调参的基准实验 |
| SGD+Momentum | 惯性平滑 | `lr`, `β=0.9` | CV 任务、大批量训练 |
| AdaGrad | 累积梯度平方 | `lr` | 稀疏特征（NLP 词嵌入） |
| RMSProp | 滑动平均自适应 | `lr`, `β=0.99` | RNN / 序列模型 |
| Adam | 动量 + 自适应 + 偏差校正 | `lr=1e-3`, `β₁=0.9`, `β₂=0.999` | **深度学习默认首选** |
| AdamW | Adam + 解耦权重衰减 | 同上 + `weight_decay` | Transformer / LLM |

> **常见误区：Adam 总是最好的**。Adam 确实开箱即用且鲁棒，但在某些场景（特别是计算机视觉的图片分类任务）中，精调过的 SGD+Momentum 往往能达到更好的泛化精度。这是因为 Adam 的自适应学习率有时会导致模型"偷懒"——迅速找到一条看似不错的捷径，但这条捷径在测试集上的泛化效果不如 SGD 步步为营走出来的路径。许多 ImageNet 冠军方案仍然使用 SGD+Momentum。

### 6.4 Python：从零实现 Adam 并对比

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 0.5 * x**2 + 8 * y**2

def grad_f(x, y):
    return np.array([x, 16*y])

class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = None, None, 0

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def optimize_sgd(x0, lr, steps):
    path = [np.array(x0, dtype=float)]
    for _ in range(steps):
        path.append(path[-1] - lr * grad_f(*path[-1]))
    return np.array(path)

def optimize_adam(x0, lr, steps):
    path = [np.array(x0, dtype=float)]
    adam = Adam(lr=lr)
    for _ in range(steps):
        path.append(adam.step(path[-1], grad_f(*path[-1])))
    return np.array(path)

x0 = [3.0, 2.0]
path_sgd = optimize_sgd(x0, lr=0.02, steps=100)
path_adam = optimize_adam(x0, lr=0.3, steps=100)

x = np.linspace(-4, 4, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax in axes[:2]:
    ax.contour(X, Y, Z, levels=25, cmap='viridis', alpha=0.5)
    ax.plot(0, 0, 'r*', markersize=14)
axes[0].plot(path_sgd[:, 0], path_sgd[:, 1], '-o', color='blue',
             markersize=2, linewidth=0.8)
axes[0].set_title('SGD（lr=0.02）')
axes[1].plot(path_adam[:, 0], path_adam[:, 1], '-o', color='green',
             markersize=2, linewidth=0.8)
axes[1].set_title('Adam（lr=0.3）')
for ax in axes[:2]:
    ax.set_xlabel('x'); ax.set_ylabel('y')

loss_sgd = [f(*p) for p in path_sgd]
loss_adam = [f(*p) for p in path_adam]
axes[2].semilogy(loss_sgd, 'b-', label='SGD', linewidth=1.5)
axes[2].semilogy(loss_adam, 'g-', label='Adam', linewidth=1.5)
axes[2].set_xlabel('迭代步数'); axes[2].set_ylabel('Loss (log scale)')
axes[2].set_title('Adam 显著加速收敛'); axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 6.5 牛顿法 — 二阶信息的威力（选修）

### 直观理解

梯度下降只用了一阶信息（坡度）：站在当前位置，看看哪个方向下坡最陡，往那个方向走一步。

牛顿法更进一步，同时用了**二阶信息（曲率）**：不仅知道哪里是下坡，还知道坡道在"变陡"还是"变平"。用这个额外信息，牛顿法能直接推断出谷底的大致位置，一步跨过去，而不是一小步一小步地挪。

用比喻来说：
- **梯度下降**是一个近视的人下山——只能看到脚下两米，老老实实一步一步走
- **牛顿法**是戴上眼镜的人——看清远处谷底的位置，大步流星直接走过去

数学上，梯度下降用**直线**（一阶泰勒展开）近似原函数，然后沿直线往下走；牛顿法用**抛物线**（二阶泰勒展开）近似原函数，然后直接跳到抛物线的顶点。抛物线比直线"拟合"得更好，所以牛顿法的每一步质量更高。

### 形式化定义

对 $f(x)$ 在当前位置 $x$ 做**二阶**泰勒展开：

$$f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)(\Delta x)^2$$

对 $\Delta x$ 求导并令其为零（找到这个抛物线的顶点）：

$$\frac{d}{d(\Delta x)}\left[f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)(\Delta x)^2\right] = f'(x) + f''(x)\Delta x = 0$$

$$\Delta x = -\frac{f'(x)}{f''(x)}$$

更新公式：

$$\boxed{x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}}$$

对比梯度下降的 $x_{t+1} = x_t - \eta f'(x_t)$：牛顿法用 $\frac{1}{f''(x)}$ **自动决定**了步长——曲率大的地方（$f''$ 大，函数弯得厉害）步长自动缩小，曲率小的地方步长自动放大。不需要手动调学习率！

对于多元函数，推广为：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{H}_t^{-1} \nabla f(\mathbf{w}_t)$$

其中 $\mathbf{H}$ 是 Hessian 矩阵（二阶偏导数组成的 $d \times d$ 矩阵）。

### 手算示例：$f(x) = x^2 - 4x + 5$

$f'(x) = 2x - 4$，$f''(x) = 2$

从 $x_0 = 10$ 出发：

$$x_1 = 10 - \frac{2 \times 10 - 4}{2} = 10 - \frac{16}{2} = 10 - 8 = 2$$

解析最小值：$f'(x)=0 \Rightarrow x=2$。**一步到位！** 无论从哪个点出发，牛顿法对二次函数都是一步到最优解。这就是牛顿法的理论魅力——**二次收敛速度**（在最优解附近，每一步的有效数字位数翻倍）。

### 为什么不用于深度学习？

理论虽美，但在深度学习中你几乎看不到纯牛顿法：

1. **Hessian 太大**：100 万参数的模型，Hessian 是 $10^6 \times 10^6$ 的矩阵，存储需要 **8 TB** 内存
2. **求逆太慢**：$\mathbf{H}^{-1}$ 的计算量是 $O(d^3)$，$10^6$ 参数需要 $10^{18}$ 次运算，完全不可承受
3. **Hessian 可能不正定**：在鞍点附近，$f''(x) \leq 0$，牛顿步可能朝错误方向走

### XGBoost 的天才设计

XGBoost 和 LightGBM 把牛顿法的思想用到了极致：在每棵决策树的节点分裂时，对损失函数做二阶泰勒展开来计算最佳分裂增益。因为每棵树上只需优化少数几个分裂参数，Hessian 计算成本完全可控。这是二阶优化在工业界最高调的成功应用。

**拟牛顿法**（BFGS、L-BFGS）则通过只存储 Hessian 的低秩近似来规避存储问题，是牛顿法在中等规模问题上最实用的变体。

### Python 验证

```python
import numpy as np

def f(x): return x**2 - 4*x + 5
def df(x): return 2*x - 4
def ddf(x): return 2

def newton_1d(x0, n_iters=3):
    path = [x0]
    x = x0
    for _ in range(n_iters):
        x = x - df(x) / ddf(x)
        path.append(x)
    return path

def gd_1d(x0, lr=0.1, n_iters=30):
    path = [x0]
    x = x0
    for _ in range(n_iters):
        x = x - lr * df(x)
        path.append(x)
    return path

path_newton = newton_1d(10.0)
path_gd = gd_1d(10.0, lr=0.1, n_iters=30)

print(f"牛顿法: x0=10 → x1={path_newton[1]:.4f} → x2={path_newton[2]:.4f}")
print(f"  1步后误差: {abs(path_newton[1]-2):.2e}")
print(f"梯度下降: 30步后 x={path_gd[-1]:.4f}, 误差={abs(path_gd[-1]-2):.4f}")
```

输出：
```
牛顿法: x0=10 → x1=2.0000 → x2=2.0000
  1步后误差: 0.00e+00
梯度下降: 30步后 x=2.0048, 误差=0.0048
```

---

## 6.6 鞍点 — 深度学习的隐形杀手

### 直观理解

在非凸优化中，比局部最优更常见也更危险的是**鞍点**。想象一个马鞍的形状：前后方向是上坡，左右方向是下坡。站在鞍点上，你在某些方向看到的是"最低点"（往下走能更低），在另一些方向看到的却是"最高点"（往下走其实是往上）。

对于梯度下降来说，鞍点是致命的——梯度在鞍点处**恰好为零**，算法以为"到地方了"，停下了。但实际上你停在了一个马鞍的中间，离真正的谷底还很远。

在高维空间中，鞍点远比局部最优常见。直觉：一个 $d$ 维空间中的随机临界点，要成为局部最小值需要在**每一个方向**上都是上坡——概率是 $\frac{1}{2^d}$，随维度指数衰减。所以高维空间中绝大多数停下来的地方是**鞍点**，而非局部最优。

### 梯度下降在鞍点附近的行为

在鞍点处 $\nabla f = 0$，纯梯度下降会停在原地。但好消息是：
- **SGD 的噪声**会随机地推你一把，让你从鞍点上滑出去
- **动量**的惯性让你即使梯度为零也能继续滑行
- **Hessian 的负特征值方向**（即使梯度为零）是潜在的下坡方向

这解释了为什么 SGD + Momentum 在深度学习中如此成功——它们的随机性和惯性天然适合逃离鞍点。

### Python：可视化鞍点

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2  # 经典鞍点函数

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.set_title('$f(x,y)=x^2-y^2$ 鞍点')

ax2 = fig.add_subplot(132)
ax2.contourf(X, Y, Z, levels=30, cmap='coolwarm')
ax2.plot(0, 0, 'k*', markersize=14, label='鞍点 (0,0)')
# 画几个方向
for angle in [0, np.pi/2]:
    ax2.arrow(0, 0, 1.5*np.cos(angle), 1.5*np.sin(angle),
              head_width=0.1, color='green' if angle==0 else 'red',
              label='下坡' if angle==0 else '上坡')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('等高线：x 方向下坡，y 方向上坡')
ax2.legend()

ax3 = fig.add_subplot(133)
x_slice = np.linspace(-2, 2, 200)
ax3.plot(x_slice, x_slice**2, 'g-', linewidth=2, label='沿 x 轴切面（下坡）')
ax3.plot(x_slice, -x_slice**2, 'r-', linewidth=2, label='沿 y 轴切面（上坡）')
ax3.axvline(0, color='k', linestyle=':', alpha=0.5)
ax3.set_xlabel('坐标'); ax3.set_ylabel('f')
ax3.set_title('不同方向的切面')
ax3.legend(); ax3.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 7. 凸优化 — 优化领域的"免费午餐"

### 7.1 直观理解

把函数图像想象成一张地形图。如果这张图是**凸**的（像一个碗），那你可以闭着眼下山——只要你一直往下走，最终一定会到达最低点。鞍点不存在，局部最优就是全局最优。

反之，如果地形凹凸不平（非凸），你可能走到一个山谷就觉得到了最低点，但翻过前面的山脊还有一个更深的谷——你被困在了局部最优。

凸函数这样"自带全局最优保证"的性质，在数学上被称为"免费午餐"——你不需要任何额外的技巧或运气，严格按照梯度下降就能保证收敛到最优解。

### 7.2 形式化定义

**凸函数**的定义是：函数图像上任意两点之间的连线，始终位于函数图像之上（或重合）。

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y), \quad \forall \lambda \in [0,1]$$

形象地说：在 $f(x)=x^2$ 的图像上任取两点，连接它们的线段一定在图像上方。而 $f(x)=x^4-4x^2+2x$ 则不行——有些地方的线段会穿过图像下方。

凸函数的等价条件（对二次可微函数）：
- 一维：$f''(x) \geq 0$（处处非负）
- 高维：Hessian 矩阵半正定

### 7.3 凸 ML 问题 vs 非凸 ML 问题

| 模型 | 损失函数 | 是否凸 | 为什么 |
|------|---------|--------|--------|
| 线性回归（MSE） | $\frac{1}{n}\|\mathbf{y}-\mathbf{Xw}\|^2$ | ✅ 凸 | MSE 是参数 $\mathbf{w}$ 的二次函数，Hessian $= \frac{2}{n}\mathbf{X}^\top\mathbf{X} \succeq 0$ |
| 逻辑回归（交叉熵） | $-\sum y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)$ | ✅ 凸 | 交叉熵在 logits 上是凸的，线性组合保持凸性 |
| SVM（Hinge Loss） | $\|\mathbf{w}\|^2 + C\sum\max(0,1-y_i\mathbf{w}^\top\mathbf{x}_i)$ | ✅ 凸 | 两项都是凸的：L2 范数 + max(0,·) 的线性组合 |
| 深度神经网络 | 交叉熵 / MSE | ❌ 非凸 | 多层非线性激活打破了凸性，损失曲面上有无数鞍点和局部最优 |

**关键认知**：传统 ML 模型的损失函数是**精心设计成凸的**——这是保证它们能用数学严格证明收敛的根本原因。深度学习的损失是非凸的，但实践中发现：在高维空间中，大多数"看起来是局部最优"的点其实是鞍点（某个方向是下坡，只是在当前维度看不出），而且 SGD 的噪声恰好能帮模型从鞍点滑出去。

### 7.4 Python：可视化凸 vs 非凸

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 300)

f_convex = x**2
f_nonconvex = x**4 - 4*x**2 + 2*x

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(x, f_convex, 'b-', linewidth=2)
pairs = [(-2, 2), (-1, 2.5), (0.5, 2.5)]
for a, b in pairs:
    t = np.linspace(0, 1, 50)
    xs = a + t*(b - a)
    ys = f_convex(a) + t*(f_convex(b) - f_convex(a))
    axes[0].plot(xs, f_convex(xs), '--', color='gray', alpha=0.3)
    axes[0].plot(xs, ys, 'r:', linewidth=2)
axes[0].set_title('凸函数 $f(x)=x^2$\n（连线始终在图像上方）')
axes[0].plot(0, 0, 'g*', markersize=14, label='全局最优 = 局部最优')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(x, f_nonconvex, 'b-', linewidth=2)
local_mins = [(-1.3, f_nonconvex(-1.3)), (1.5, f_nonconvex(1.5))]
global_min = (-1.8, f_nonconvex(-1.8))
for xm, ym in local_mins:
    axes[1].plot(xm, ym, 'ro', markersize=10)
axes[1].plot(*global_min, 'g*', markersize=16, label='全局最优')
axes[1].set_title('非凸函数 $f(x)=x^4-4x^2+2x$\n（多个局部最优 ≠ 全局最优）')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 8. 约束优化与拉格朗日乘子法

### 8.1 生活例子：工厂的生产预算

你是厂长。生产产品 A（产量 $x$）和产品 B（产量 $y$），利润是：

$$f(x, y) = 50x + 80y$$

但你有预算限制：A 的单位成本是 100 元，B 的单位成本是 200 元，总预算 10000 元。所以约束是：

$$100x + 200y \leq 10000 \quad \Rightarrow \quad x + 2y \leq 100$$

你当然想无限生产（利润线向外无限推），但预算墙挡住了你。**无约束最优解根本不存在**——你必须考虑约束。

### 8.2 直观理解：拉格朗日乘子法

拉格朗日乘子法的核心思想极其优雅：**把"硬约束"转化为目标函数中的一个"软惩罚项"**。

构造拉格朗日函数：

$$\mathcal{L}(x, y, \lambda) = 50x + 80y - \lambda(x + 2y - 100), \quad \lambda \geq 0$$

$\lambda$ 是拉格朗日乘子，代表约束的"紧箍咒有多紧"：
- 如果最优解在约束边界**内部**（$x+2y < 100$），那约束没起到限制作用，$\lambda = 0$
- 如果最优解在约束边界**上**（$x+2y = 100$），那约束"绑紧"了，$\lambda > 0$

这就是 KKT 条件中的**互补松弛**：$\lambda \cdot (x+2y-100) = 0$。$\lambda$ 和约束"间隙"至少有一个为零。

### 8.3 形式化定义

对于标准约束问题 $\min f(\mathbf{w})$ s.t. $g(\mathbf{w}) \leq 0$，定义拉格朗日函数：

$$\mathcal{L}(\mathbf{w}, \lambda) = f(\mathbf{w}) + \lambda g(\mathbf{w}), \quad \lambda \geq 0$$

最优解满足 KKT 条件：
1. **驻点**：$\nabla f + \lambda \nabla g = 0$
2. **原始可行**：$g(\mathbf{w}) \leq 0$
3. **对偶可行**：$\lambda \geq 0$
4. **互补松弛**：$\lambda \cdot g(\mathbf{w}) = 0$

### 8.4 手算示例：$\min x^2 + y^2$ s.t. $x + y = 2$

**问题**：在直线 $x+y=2$ 上找一个点，使它到原点的距离平方最小。

**第一步**：构造拉格朗日函数。

$$\mathcal{L}(x, y, \lambda) = x^2 + y^2 + \lambda(x + y - 2)$$

**第二步**：分别对 $x$、$y$、$\lambda$ 求偏导，令其为零。

$$\begin{aligned}
\frac{\partial\mathcal{L}}{\partial x} &= 2x + \lambda = 0 \quad \Rightarrow \quad x = -\frac{\lambda}{2} \\
\frac{\partial\mathcal{L}}{\partial y} &= 2y + \lambda = 0 \quad \Rightarrow \quad y = -\frac{\lambda}{2} \\
\frac{\partial\mathcal{L}}{\partial \lambda} &= x + y - 2 = 0
\end{aligned}$$

**第三步**：联立求解。

由前两式得 $x = y = -\lambda/2$。代入第三式：

$$-\frac{\lambda}{2} - \frac{\lambda}{2} = 2 \quad \Rightarrow \quad -\lambda = 2 \quad \Rightarrow \quad \lambda = -2$$

注意：对于等式约束，$\lambda$ 可为任意实数，不需要 $\lambda \geq 0$。

$$x = y = -\frac{-2}{2} = 1$$

**结论**：最优解是 $(1, 1)$，此时 $x^2 + y^2 = 2$。

**几何意义**：以原点为圆心的圆，不断放大半径，与直线 $x+y=2$ 的第一个触碰点就是 $(1, 1)$。$\lambda = -2$ 代表在这一点处，目标函数梯度 $(2, 2)$ 与约束梯度 $(1, 1)$ 方向相反且大小为 2 倍。

> **常见误区**：拉格朗日乘子 $\lambda$ 本身也有经济含义——它代表"约束放松一个单位，目标函数能改善多少"。在上面例子中，如果约束从 2 变为 3（放松 1），最优解从 $(1,1)$ 变为 $(1.5,1.5)$，目标值从 2 变为 4.5——变化了 2.5，接近 $|\lambda|=2$ 的估计。

### 8.5 Python 验证

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# min x^2 + y^2  s.t. x + y = 2
def objective(p):
    return p[0]**2 + p[1]**2

constraints = [{'type': 'eq', 'fun': lambda p: p[0] + p[1] - 2}]
result = minimize(objective, [0, 0], constraints=constraints)

print(f"约束最优: ({result.x[0]:.4f}, {result.x[1]:.4f}), f={result.fun:.4f}")
print(f"无约束最优: (0, 0), f=0 (但不满足约束)")

x = np.linspace(-1, 3, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(X, Y, Z, levels=25, cmap='viridis', alpha=0.5)
ax.plot(x, 2 - x, 'r-', linewidth=3, label='约束: x+y=2')
ax.plot(0, 0, 'b*', markersize=14, label='无约束最优 (0,0)')
ax.plot(1, 1, 'go', markersize=12, label='约束最优 (1,1)')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(); ax.set_title('约束优化：最优解是等高线与约束线的切点')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 9. 正则化 — 约束优化的机器学习版本

### 9.1 直观理解

正则化的本质是：**给模型参数加上约束，防止它们"放飞自我"**。

在机器学习中，"放飞自我"意味着过拟合——模型记住了训练数据的每一个噪声点，但丧失了泛化能力。正则化通过限制参数的大小来防止这种情况：
- **L1 正则化**：约束参数的绝对值之和 → 产生稀疏解（很多参数恰好为 0）→ 自动做了特征选择
- **L2 正则化**：约束参数的平方和 → 让所有参数都小但不为零 → 防止过拟合

### 9.2 几何解释：为什么 L1 产生稀疏而 L2 不产生

将损失函数的等高线（椭圆）和约束区域画在一起：

- L1 的约束区域是**菱形**（有四个尖角在坐标轴上）。当椭圆向外扩张时，最先碰到菱形的概率最大的地方是**尖角**——这时某个参数恰好为 0。
- L2 的约束区域是**圆形**（处处光滑）。椭圆和圆的切点几乎不可能恰好落在坐标轴上——所以所有参数都不为 0，但整体被"收缩"了。

这就是**尖角效应**：有尖角的约束区域把最优解推到坐标轴上，产生零参数。

### 9.3 Python：可视化 L1 vs L2

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

mean = np.array([2.5, 1.5])
cov = np.array([[1.0, 0.7], [0.7, 0.6]])

for ax, name, ctype in zip(axes,
    ['L1 (Lasso) → 稀疏解', 'L2 (Ridge) → 收缩解'],
    ['l1', 'l2']):
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
    ax.set_xlabel('$w_1$'); ax.set_ylabel('$w_2$')
    ax.set_title(name)

    for level in [0.5, 1.0, 1.5, 2.0, 3.0]:
        eigvals, eigvecs = np.linalg.eigh(cov)
        theta = np.linspace(0, 2*np.pi, 200)
        ell = np.column_stack([np.cos(theta), np.sin(theta)])
        ell = ell @ np.diag(np.sqrt(eigvals) * level) @ eigvecs.T
        ell += mean
        ax.plot(ell[:, 0], ell[:, 1], '--', color='steelblue', alpha=0.3)

    theta = np.linspace(0, 2*np.pi, 200)
    if ctype == 'l1':
        R = 2.2
        diamond = np.array([[R,0],[0,R],[-R,0],[0,-R],[R,0]])
        ax.fill(diamond[:,0], diamond[:,1], alpha=0.2, color='coral')
        ax.plot(diamond[:,0], diamond[:,1], 'r-', linewidth=2)
        ax.plot(R, 0, 'ro', markersize=12, label='Lasso 解: $(w_1,0)$')
    else:
        R = 1.5
        ax.fill(R*np.cos(theta), R*np.sin(theta), alpha=0.2, color='coral')
        ax.plot(R*np.cos(theta), R*np.sin(theta), 'r-', linewidth=2)
        ax.plot(1.0, 1.1, 'ro', markersize=12, label='Ridge 解')

    ax.plot(mean[0], mean[1], 'b*', markersize=14, label='无正则化解')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
```

### 9.4 正则化对比

| 正则化 | 约束形式 | 等价无约束 | 效果 | 场景 |
|--------|---------|-----------|------|------|
| L1 (Lasso) | $\sum |w_i| \leq t$ | $\min f + \lambda \sum |w_i|$ | 稀疏、特征选择 | 高维数据 |
| L2 (Ridge) | $\sum w_i^2 \leq t$ | $\min f + \lambda \sum w_i^2$ | 收缩、防过拟合 | 共线性 |
| ElasticNet | 两者组合 | $\min f + \lambda_1\sum |w_i| + \lambda_2\sum w_i^2$ | 稀疏+分组 | 高维+相关特征 |

---

## 9.5 实用技巧：学习率调度与梯度裁剪

理解了各种优化算法的原理后，以下几个实用技巧能让你在实际项目中少踩坑。

### 学习率调度（Learning Rate Scheduling）

固定学习率在整个训练过程中不变，但更聪明的做法是**让学习率随着训练推进而衰减**——训练初期大步前行，训练后期精细微调。

常见的调度策略：

| 策略 | 公式 | 适用场景 |
|------|------|---------|
| 阶梯衰减 | 每 $k$ 个 epoch 将 lr 乘以 0.1 | 简单直接，CV 任务常用 |
| 指数衰减 | $\eta_t = \eta_0 \cdot \gamma^t$ | 快速收敛场景 |
| 余弦退火 | $\eta_t = \eta_{\min} + \frac{\eta_0-\eta_{\min}}{2}(1+\cos(\frac{t\pi}{T}))$ | 震荡探索，现代 DL 推荐 |
| 预热（Warmup） | 前 $k$ 步从 0 线性增长到 $\eta_0$ | Transformer 训练标配 |

**为什么需要 Warmup？** 训练刚开始时，模型参数是随机初始化的，梯度方向极其不稳定。如果直接用大学习率，模型可能被"炸飞"。先用小学习率"预热"几步，等梯度方向稳定了再全速前进。

```python
import numpy as np
import matplotlib.pyplot as plt

steps = 200
warmup_steps = 30
T = steps
lr0 = 0.1
lr_min = 1e-5

lr_warmup_cosine = []
for t in range(steps):
    if t < warmup_steps:
        lr = lr0 * t / warmup_steps
    else:
        progress = (t - warmup_steps) / (T - warmup_steps)
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * progress))
    lr_warmup_cosine.append(lr)

lr_step = [lr0 * (0.5 ** (t // 60)) for t in range(steps)]
lr_exp = [lr0 * (0.98 ** t) for t in range(steps)]

plt.figure(figsize=(9, 4))
plt.plot(lr_step, label='阶梯衰减 (×0.5 / 60 steps)')
plt.plot(lr_exp, label='指数衰减 (γ=0.98)')
plt.plot(lr_warmup_cosine, label='Warmup + 余弦退火')
plt.xlabel('训练步数'); plt.ylabel('学习率')
plt.title('常见学习率调度策略')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

### 梯度裁剪（Gradient Clipping）

在 RNN 和 Transformer 训练中，偶尔会出现**梯度爆炸**——某些步骤的梯度异常大（比如 1000+），导致参数更新一步飞到天边，损失变成 NaN，训练崩溃。

梯度裁剪的解决方法很直接：**如果梯度的范数超过一个阈值，就把它等比缩放到阈值**。

$$\text{if } \|\mathbf{g}\| > C: \quad \mathbf{g} \leftarrow \frac{C}{\|\mathbf{g}\|} \cdot \mathbf{g}$$

```python
def clip_gradient(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * max_norm / norm
    return grad

g = np.array([50.0, 30.0, 40.0])
print(f"原始梯度范数: {np.linalg.norm(g):.1f}")
g_clipped = clip_gradient(g, max_norm=10.0)
print(f"裁剪后范数: {np.linalg.norm(g_clipped):.1f}")
print(f"裁剪后梯度: {g_clipped}")
```

输出：
```
原始梯度范数: 70.7
裁剪后范数: 10.0
裁剪后梯度: [7.07 4.24 5.66]
```

**常见误区**：梯度裁剪不是万能药。如果裁剪太频繁，说明你的模型架构或学习率设置有问题——梯度裁剪只是防止崩溃的"保险丝"，不是解决梯度不稳定的根本手段。

### 梯度检查（Gradient Checking）

当你手写反向传播或自定义损失函数时，怎么验证梯度计算是正确的？

利用导数的定义：$f'(x) \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$（双边差分，误差 $O(\epsilon^2)$）

```python
def numerical_gradient(f, x, eps=1e-5):
    """用双边差分估算函数 f 在 x 处的梯度"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def analytical_gradient(x):
    return 2 * x  # f(x) = sum(x^2)

x = np.array([1.5, 2.5, 3.5])
num_grad = numerical_gradient(lambda w: np.sum(w**2), x)
ana_grad = analytical_gradient(x)
diff = np.linalg.norm(num_grad - ana_grad) / np.linalg.norm(num_grad + ana_grad)

print(f"数值梯度: {num_grad}")
print(f"解析梯度: {ana_grad}")
print(f"相对误差: {diff:.2e} {'✓ 通过' if diff < 1e-7 else '✗ 失败'}")
```

相对误差小于 $10^{-7}$ 说明梯度计算基本正确。如果误差在 $10^{-3}$ 量级，说明实现有 bug。

---

## 9.6 优化器选择速查

当你面对一个新项目时，按以下决策树选择优化器：

```
你的数据量大吗（>10万样本）？
├── 不大 → 可以用 BGD 或 L-BFGS（收敛最快）
└── 大
    ├── 你的模型是 Transformer / LLM？
    │   └── 是 → AdamW (lr=1e-4, weight_decay=0.01)
    ├── 你的特征很稀疏？（NLP词向量、推荐系统）
    │   └── 是 → Adam 或 AdaGrad（自适应处理稀疏梯度）
    ├── 你在做计算机视觉（分类/检测）？
    │   └── 是 → SGD + Momentum (lr=0.1, momentum=0.9) + 余弦退火
    ├── 你在做 GAN / 强化学习？
    │   └── 是 → Adam (lr=1e-4, β₁=0.5)
    └── 不确定？
        └── Adam (lr=1e-3) — 99% 情况下不会太差
```

**经验法则**：

| 场景 | 推荐优化器 | 标配学习率 | 备注 |
|------|-----------|-----------|------|
| 新手入门 / 快速原型 | Adam | 1e-3 | 开箱即用，几乎不需要调参 |
| CV（分类/检测/分割） | SGD+Momentum | 0.1 → 0.001 | 配合余弦退火，需要耐心调参 |
| NLP（Transformer） | AdamW | 1e-4 | 标配，配合 warmup |
| GAN | Adam | 1e-4, β₁=0.5 | β₁ 降低以减少震荡 |
| 强化学习 | Adam | 1e-4 ~ 3e-4 | PPO 标配 |
| 在线学习 / 流式数据 | SGD | 从 0.01 开始衰减 | 实时更新 |
| 贝叶斯推断 | Adam / SGD | 取决于问题 | 需要配合 MCMC 采样 |

---

## 10. 总结：优化思维是机器学习的元技能

回顾一下你学到的完整知识链：

```
一元极值（导数为零）
    ↓ 遇到复杂函数，无解析解
梯度下降（一阶逼近，迭代下降）
    ↓ 多变量
多元梯度下降（梯度 = 偏导向量）
    ↓ 数据太大，算不动全部梯度
SGD / Mini-batch（用样本估算梯度）
    ↓ 有些方向震荡，有些方向太慢
动量法（指数平均，惯性平滑）
    ↓ 每个参数需要不同的步长
自适应方法（AdaGrad → RMSProp → Adam）
    ↓ 理论保证
凸优化（局部最优 = 全局最优）
    ↓ 加上约束
拉格朗日乘子法 → 正则化（L1/L2）
```

| 机器学习环节 | 优化问题 | 优化方法 |
|-------------|---------|---------|
| 线性回归训练 | 最小化 MSE | 解析解 / 梯度下降 |
| 逻辑回归训练 | 最小化交叉熵 | 梯度下降 |
| SVM 训练 | 最大化间隔（约束优化） | SMO / 对偶梯度下降 |
| 神经网络训练 | 最小化损失 | Adam / AdamW |
| XGBoost 分裂 | 最小化增益损失 | 二阶泰勒展开（牛顿法） |
| 超参数调优 | 最小化验证集损失 | 贝叶斯优化 / 网格搜索 |
| 正则化 | 约束参数范数 | 拉格朗日乘子法（隐式） |
| GAN 训练 | MinMax 博弈 | 交替梯度下降 |

---

## 常见误区总结

本章中贯穿了多个常见误区，这里集中回顾，帮你避免踩坑：

| # | 误区 | 真相 |
|---|------|------|
| 1 | $f'(x)=0$ 就一定是极值点 | $f(x)=x^3$ 在 $x=0$ 处导数为零但是拐点，不是极值。需要二阶导数确认 |
| 2 | 学习率越大，收敛越快 | 超过临界值后会震荡甚至发散。对于 $f(x)=x^2$，$\eta < 1$ 才收敛 |
| 3 | BGD 因为"精确"所以最好 | 在百万级数据上，BGD 一步都算不完。Mini-batch 在效率和准确度间取得平衡 |
| 4 | SGD 的噪声是缺陷 | 噪声恰恰是优势：帮忙逃离鞍点、自带正则化、适合在线学习 |
| 5 | Momentum 的 β=0.9 意味着"只看最近 10 步" | EMA 是平滑衰减而非硬截断。第 10 步的权重仍有 35%，不是 0 |
| 6 | Adam 总是最好的优化器 | CV 任务中精调过的 SGD+Momentum 往往泛化更好。Adam 可能"抄近道"导致泛化略差 |
| 7 | 损失函数降到很低就是成功了 | 过拟合意味着训练损失低但测试集表现差。正则化就是为此设计的 |
| 8 | 凸优化只是理论概念，不实用 | 线性回归、逻辑回归、SVM 全部是凸优化——这是它们能被数学严格保证收敛的原因 |
| 9 | 拉格朗日乘子 λ 只是一个中间变量 | λ 有经济学含义：约束放松一个单位，目标函数改善约 |λ| |
| 10 | 训练时 loss 变成 NaN 一定是代码 bug | 很可能是梯度爆炸——试试梯度裁剪，或降低学习率 |

---

## 思考题

以下 10 道练习题覆盖从基础到进阶的各个层次。建议先尝试独立求解，再对照答案。

---

### 1. 一元函数最小值（简单）

求 $f(x) = x^2 - 6x + 13$ 的最小值，写出完整步骤。

**解**：$f'(x) = 2x - 6 = 0 \Rightarrow x = 3$。$f''(x) = 2 > 0 \Rightarrow$ 极小值。
$f(3) = 9 - 18 + 13 = 4$。

---

### 2. 手算梯度下降（简单）

$f(x) = x^2 - 2x + 3$，起始点 $x_0 = 5$，$\eta = 0.1$。手算前 5 步，填入下表。

**解**：$f'(x) = 2x - 2$。

| 迭代 | $x$ | $f'(x)$ | $\Delta x$ | $f(x)$ |
|------|-----|---------|-----------|--------|
| 0 | 5.00 | 8.00 | -0.80 | 18.00 |
| 1 | 4.20 | 6.40 | -0.64 | 12.24 |
| 2 | 3.56 | 5.12 | -0.51 | 8.55 |
| 3 | 3.05 | 4.10 | -0.41 | 6.20 |
| 4 | 2.64 | 3.28 | -0.33 | 4.67 |
| 5 | 2.31 | 2.62 | -0.26 | 3.71 |

理论最小值在 $x=1$ 处（$f'=0$）。5 步之后 $x=2.31$，正在稳步靠近。

---

### 3. 多元梯度下降（中等）

$f(x, y) = x^2 + y^2$，起始点 $(4, 0)$，$\eta = 0.1$。手算 3 步，追踪 $(x, y)$ 的变化。

**解**：$\nabla f = (2x, 2y)$。

| 迭代 | $(x, y)$ | $\nabla f$ | $(x, y)$ 新 | $f$ |
|------|---------|-----------|------------|-----|
| 0 | (4.00, 0.00) | (8.00, 0.00) | (3.20, 0.00) | 10.24 |
| 1 | (3.20, 0.00) | (6.40, 0.00) | (2.56, 0.00) | 6.55 |
| 2 | (2.56, 0.00) | (5.12, 0.00) | (2.05, 0.00) | 4.20 |

函数是对称的碗，从 $(4,0)$ 出发直接沿 $x$ 轴向原点移动。

---

### 4. 学习率的收敛边界（中等）

对于 $f(x) = x^2$，用梯度下降 $x_{t+1} = x_t - \eta \cdot 2x_t = (1 - 2\eta)x_t$。推导 $\eta$ 在什么范围内算法收敛（即 $x_t \to 0$）。

**解**：$x_t = (1 - 2\eta)^t x_0$。收敛要求 $|1 - 2\eta| < 1$。

解不等式：$-1 < 1 - 2\eta < 1 \Rightarrow -2 < -2\eta < 0 \Rightarrow 0 < \eta < 1$。

$\eta = 0.5$ 时 $x_1 = 0$，一步到位（临界阻尼）。$\eta = 1$ 时 $x_1 = -x_0$，$x_2 = x_0$，永不休止地振荡。$\eta > 1$ 时发散。

---

### 5. BGD vs SGD vs Mini-batch（中等）

数据：$(1, 2), (2, 4), (3, 6), (4, 8)$。模型 $\hat{y} = wx$（无截距）。$w_0 = 0$，$\eta = 0.05$。

(a) 用 BGD 算一步后的 $w$。
(b) 用 SGD 遍历一轮（4 个样本顺序遍历）后的 $w$。
(c) 用 Mini-batch（batch_size=2）遍历一轮（两个 batch）后的 $w$。

**解**：

J(w) = 1/4 Σ(y_i - wx_i)²
∇J = -2/4 Σ x_i(y_i - wx_i) = -0.5 Σ x_i(y_i - wx_i)

**(a) BGD**：
∇J(0) = -0.5[1×2 + 2×4 + 3×6 + 4×8] = -0.5 × 60 = -30
$w_1 = 0 - 0.05 × (-30) = 1.5$

**(b) SGD**：
样本 (1,2)：∇ = -2×1×(2-0) = -4, w = 0 - 0.05×(-4) = 0.2
样本 (2,4)：∇ = -2×2×(4-2×0.2) = -4×3.6 = -14.4, w = 0.2 - 0.05×(-14.4) = 0.92
样本 (3,6)：∇ = -2×3×(6-3×0.92) = -6×3.24 = -19.44, w = 0.92 - 0.05×(-19.44) = 1.892
样本 (4,8)：∇ = -2×4×(8-4×1.892) = -8×0.432 = -3.456, w = 1.892 - 0.05×(-3.456) = 2.0648

**(c) Mini-batch (batch=2)**：
Batch {(1,2),(2,4)}：∇ = -0.5[1×2+2×4] = -5, w = 0 - 0.05×(-5) = 0.25
Batch {(3,6),(4,8)}：∇ = -0.5[3×(6-3×0.25)+4×(8-4×0.25)] = -0.5[3×5.25+4×7] = -0.5×43.75 = -21.875
w = 0.25 - 0.05×(-21.875) = 1.34375

真实 $w = 2$。SGD 噪声最大但最终最近，Mini-batch 次之，BGD 稳步前进。

---

### 6. 动量法的手算（中等）

$f(x, y) = x^2 + y^2$，起始点 $(1, 2)$，$\eta = 0.1$，$\beta = 0.9$。手算动量法的前 3 步，并对比同等条件下的纯梯度下降。

**解**：

纯 GD（$\eta=0.1$）：每步更新 = $-0.1 \times (2x, 2y) = (-0.2x, -0.2y)$。

| 迭代 | $x$ | $y$ | $f$ |
|------|-----|-----|-----|
| 0 | 1.00 | 2.00 | 5.00 |
| 1 | 0.80 | 1.60 | 3.20 |
| 2 | 0.64 | 1.28 | 2.05 |
| 3 | 0.51 | 1.02 | 1.31 |

动量法（$\eta=0.1$，$\beta=0.9$）：$v_{t} = 0.9v_{t-1} + 0.1 \times \nabla f_t$

| 迭代 | $v$ | $(x,y)$ | $f$ |
|------|-----|---------|-----|
| 0 | — | (1.00, 2.00) | 5.00 |
| 1 | (0.2, 0.4) | (0.98, 1.96) | 4.80 |
| 2 | (0.38, 0.76) | (0.94, 1.88) | 4.44 |
| 3 | (0.54, 1.09) | (0.89, 1.77) | 3.92 |

动量法前几步稍慢（因为速度从零开始积累），但一旦建立起速度，后期会反超纯 GD。

---

### 7. 凸函数判断（中等）

判断以下函数是否凸，并说明理由：
(a) $f(x) = e^x$
(b) $f(x) = x^3$
(c) $f(x, y) = x^2 + 3y^2 - 2xy$
(d) $f(x) = -\log(x)$，其中 $x > 0$

**解**：

(a) $f''(x) = e^x > 0$，严格凸。
(b) $f''(x) = 6x$。$x>0$ 时 $f''>0$（凸），$x<0$ 时 $f''<0$（凹）。整体非凸。
(c) 写成二次型：$f = (x - y)^2 + 2y^2 = \begin{pmatrix}x & y\end{pmatrix}\begin{pmatrix}1 & -1 \\ -1 & 3\end{pmatrix}\begin{pmatrix}x \\ y\end{pmatrix}$。特征值：$4 \pm \sqrt{2}$，均 > 0 → Hessian 正定 → 严格凸。
(d) $f''(x) = 1/x^2 > 0$，严格凸。

---

### 8. 拉格朗日乘子法（困难）

求解：$\min \; x^2 + 4y^2$ s.t. $x + 2y = 6$，写出完整推导。

**解**：

**第一步**：$\mathcal{L}(x, y, \lambda) = x^2 + 4y^2 + \lambda(x + 2y - 6)$

**第二步**：求偏导。

$$\begin{aligned}
\partial\mathcal{L}/\partial x &= 2x + \lambda = 0 &\Rightarrow& \; x = -\lambda/2 \\
\partial\mathcal{L}/\partial y &= 8y + 2\lambda = 0 &\Rightarrow& \; y = -\lambda/4 \\
\partial\mathcal{L}/\partial \lambda &= x + 2y - 6 = 0
\end{aligned}$$

**第三步**：代入。$(-\lambda/2) + 2(-\lambda/4) = 6 \Rightarrow -\lambda/2 - \lambda/2 = 6 \Rightarrow -\lambda = 6 \Rightarrow \lambda = -6$

$x = 3$，$y = 1.5$。代入目标函数：$3^2 + 4 \times 1.5^2 = 9 + 9 = 18$。

**几何解释**：椭圆 $x^2+4y^2 = c$ 向外扩张，首次触碰直线 $x+2y=6$ 的点是 $(3, 1.5)$。

---

### 9. 牛顿法推导（困难）

对于 $f(x) = x^2$，证明从任意起始点 $x_0$ 出发，牛顿法恰好 1 步到达最小值点 $x=0$。

**解**：

牛顿法更新：$x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}$

对于 $f(x) = x^2$：$f'(x) = 2x$，$f''(x) = 2$。

$$x_1 = x_0 - \frac{2x_0}{2} = x_0 - x_0 = 0$$

一步到位。原因：$f(x)=x^2$ 本身就是完美的二次函数，牛顿法的二阶泰勒展开就是精确的原函数，直接定位到顶点。

> 这正是牛顿法的核心优势——对于严格二次函数，一次迭代就能到达最优解。但对于复杂的非凸函数，牛顿法可能收敛到鞍点甚至发散。

---

### 10. AdaGrad 的优势场景（困难）

考虑一个稀疏特征问题：参数 $w = (w_1, w_2)$。$w_1$ 对应的特征在 95% 的样本中为 0（稀疏），$w_2$ 对应的特征始终非零（稠密）。

解释为什么 AdaGrad 比纯 SGD 更适合这种场景。

**解**：

在 SGD 中，$w_1$ 和 $w_2$ 使用相同的学习率。但 $w_1$ 只在 5% 的步骤中获得梯度更新，大部分时间的梯度为 0——导致 $w_1$ 的更新极其缓慢。

在 AdaGrad 中，学习率 $= \eta / \sqrt{s_t}$，其中 $s_t$ 累积历史梯度平方：
- $w_1$：$s_{1,t}$ 增长很慢（因为大部分梯度为 0），所以 $w_1$ 的学习率保持较大 → 稀少的梯度每次都能产生显著更新
- $w_2$：$s_{2,t}$ 增长很快（每次都有梯度），所以 $w_2$ 的学习率逐渐缩小 → 稳定细微调整

这就是 AdaGrad 被设计出来的原始动机——处理稀疏特征。在 NLP 中，词向量矩阵的大部分行在每轮训练中只被激活少数几次，AdaGrad 确保这些稀疏特征也能得到充分的训练。

> **注**：Adam 继承了 RMSProp 的滑动平均 $v_t$，同样具有类似的自适应效果，这也是 Adam 在各种场景下表现优异的原因之一。

---

## 下一步

继续学习[信息论基础](./math-information-theory.md)，理解损失函数（交叉熵、KL 散度）背后的信息论原理——为什么分类问题用交叉熵而不是 MSE？"熵"到底衡量了什么？
