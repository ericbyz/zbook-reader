# 数学基础 — 线性代数

> **本章在 ML 学习路径中的位置**：数学基础第一站。线性代数是机器学习的"语言"——数据是矩阵，参数是矩阵，变换也是矩阵。本章从零开始，用生活例子和手算练习帮你建立线性代数直觉。

---

## 0. 为什么学线性代数？——先给你一个"地图"

在开始之前，你可能想问："我一个学机器学习的，为什么要先啃线性代数？"

答案很简单：**机器学习的一切运算，底层都是线性代数。**

- 一张图片在计算机眼里，是 `(高度, 宽度, 颜色通道)` 的三维数字阵列——一个**张量**。
- 训练数据集的 10 万张图片，是 `(100000, 224, 224, 3)` 的四维张量。
- 神经网络的每一层做的计算：`h = σ(Wx + b)`——**矩阵乘法**加**向量加法**。
- 主成分分析（PCA）降维：找到数据方差最大的方向——**特征值分解**。
- 推荐系统中"你可能喜欢的电影"：把评分矩阵分解成低秩近似——**SVD**。
- 损失函数里加 L2 正则化防止过拟合：$\lambda\|w\|_2^2$——**向量范数**。

这一章的目的不是让你成为线性代数专家，而是让你**建立直觉**——看到一个矩阵乘法，脑子里能浮现出几何画面；看到特征值，能想到"哦，这是数据的主要方向"。

好，我们从最基础的概念开始——慢一点，稳一点。

---

## 1. 什么是标量？

### a) 生活例子

你今年 25 岁。今天气温 30°C。一杯咖啡 22 元。

这三个数字有什么共同点？它们都只用一个数字就能描述清楚——"25"、"30"、"22"。不需要方向，不需要多个分量，一个数字就够了。

### b) 直观理解

**标量（Scalar）就是一个单独的数字。** 它在空间里只占据一个点——没有方向，没有"好几个值"，就一个。

标量是构建一切更复杂数学对象的基石。向量是一串标量排成一列。矩阵是一堆标量排列成表格。张量是一大堆标量排列成多维集装箱。

### c) 形式化定义

> **标量**：零维量（0 阶张量），属于某个数域（通常是实数集 $\mathbb{R}$）。

记法：小写字母 $x, y, a, b, \lambda$。

**例子**：$x = 3.14$, $t = 25$, $\lambda = 0.01$（学习率）, $L = 0.342$（损失值）

### d) 手算示例

标量的运算就是你从小学就会的算术：

$$3 + 5 = 8$$
$$2.5 \times 4 = 10$$
$$(-3) \times (-2) = 6$$

不需要手把手教你，但你心里要清楚：**标量是基本单位**，后面所有的运算最终都会分解成标量之间的加减乘除。

### e) Python 验证

```python
import numpy as np

age = 25
temp = 30.5
price = 22.0
loss = 0.342

print(f"年龄: {age}, 数据类型: {type(age)}")
print(f"温度: {temp}, 数据类型: {type(temp)}")
print(f"所有标量: {age}, {temp}, {price}, {loss}")

# numpy 中的标量——0 维数组
s = np.array(3.14)
print(f"numpy 标量: {s}, 形状: {s.shape}, 维度: {s.ndim}")
```

### g) ML 中的应用

| 概念 | 例子 |
|------|------|
| 损失函数值 | 交叉熵损失 = 0.342 |
| 准确率 | 测试集准确率 = 0.953 |
| 学习率 | $\eta = 0.001$ |
| 正则化强度 | $\lambda = 0.01$ |
| 神经元偏置 | $b = 0.5$ |

> 一句话：**标量是"一个数"。它简单，但它是地基。**

---

## 2. 什么是向量？

### a) 生活例子

想象你在一个大商场里，朋友打电话问："你在哪？"

你只说一个数字行不行？"我在 150。"——150 什么？150 号店铺？150 米？根本说不清楚。

你需要说：**"我在 3 楼的 B 区"**。两个数字：`[楼层=3, 区域=B]`。这就是一个向量。

再举个例子——你去超市买菜，列了个购物清单：

| 物品 | 数量 |
|------|------|
| 苹果 | 3 个 |
| 鸡蛋 | 2 盒 |
| 牛奶 | 1 瓶 |
| 面包 | 2 袋 |

这个清单的"数量"部分就是向量：`[3, 2, 1, 2]`。四个数字，描述一种"状态"。

再比如，你用电脑调一个颜色——橙色：`[R=255, G=128, B=0]`。三个数字，定义了一种颜色。

### b) 直观理解

**向量就是一组有顺序的数字**。每个数字叫一个"分量"（component），分量的个数叫"维度"（dimension）。

几何上，向量是从原点出发的一支**箭头**：箭头指向某个方向，箭头的长度代表大小。

```
      y
      ↑
    4 |      ● (3, 4)
      |     ↗
    3 |   ↗
      | ↗
    2 |↗
      |↑___________
    1 ||           |
      ||           |
    0 +--|--|--|--|→ x
      0  1  2  3  4
```

向量 `v = [3, 4]` 就是从原点 `(0,0)` 出发、到达点 `(3,4)` 的一支箭头。它有两个属性：
- **方向**：箭头的指向
- **长度**：箭头有多长（$\sqrt{3^2 + 4^2} = 5$）

### c) 形式化定义

> **向量**（Vector）：一维的数字阵列。$n$ 维向量 $\mathbf{v} \in \mathbb{R}^n$ 包含 $n$ 个实数值分量 $v_1, v_2, \ldots, v_n$。

记法：
- 粗体小写：$\mathbf{v}, \mathbf{x}, \mathbf{w}$
- 或带箭头：$\vec{v}$
- 列向量（默认）：$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$
- 行向量（带转置）：$\mathbf{v}^\top = [v_1, v_2, \ldots, v_n]$

> 本章默认向量为**列向量**。书写时为节省空间常写成 `[v₁, v₂, ..., vₙ]ᵀ`。

### d) 手算示例

描述一个人的身体状况，用一个 3 维向量：

$$\mathbf{p} = \begin{bmatrix} 170 \\ 65 \\ 25 \end{bmatrix} \quad \text{身高(cm), 体重(kg), 年龄(岁)}$$

三个分量各司其职，合在一起完整描述了这个人。

描述风的状况，用一个 2 维向量：

$$\mathbf{w} = \begin{bmatrix} 5.0 \\ 3.0 \end{bmatrix} \quad \text{风速(m/s), 风向(弧度)}$$

### e) Python 验证

```python
import numpy as np

# 购物清单向量
shopping = np.array([3, 2, 1, 2])
print(f"购物清单: {shopping}")
print(f"维度: {shopping.shape[0]}, 总物品数: {shopping.sum()}")

# 位置向量 (x, y)
position = np.array([3.0, 4.0])
print(f"\n位置: {position}")

# RGB 颜色向量
orange = np.array([255, 128, 0])
print(f"\n橙色 RGB: {orange}")

# 风向量
wind = np.array([5.0, 3.0])
print(f"\n风向量: 风速={wind[0]}m/s, 风向={wind[1]}rad")
```

### f) 常见误区

> **误区**："向量就是空间中的箭头。"

不完全是。箭头只是向量的一种**几何表示**。向量本质上是一组有序数字，它可以表示位置、颜色、购物清单、学生的各科成绩——任何需要多个数字来描述的东西。几何箭头是帮助理解的工具，但不是唯一的解释。

> **误区**："行向量和列向量是一样的。"

在纯数学中可能不做区分，但在机器学习的矩阵运算里，**维度必须匹配**。一个 `(n,)` 形状的 numpy 数组既可以当行向量也可以当列向量（broadcasting 会自动处理），但当你明确做 `(1, n)` 和 `(n, 1)` 的矩阵运算时，差别就非常重要了。

### g) ML 中的应用

| 场景 | 向量表示 |
|------|----------|
| 一个样本的特征 | `x = [身高, 体重, 血压, 心率]` |
| 一张图片展平 | `x ∈ ℝ^{784}`（28×28 像素→784 维） |
| 一个词的嵌入 | `word_vec ∈ ℝ^{300}`（Word2Vec） |
| 神经网络一层的偏置 | `b ∈ ℝ^{128}`（128 个神经元的偏置） |
| 梯度 | `∇L ∈ ℝ^{d}`（每个参数的梯度） |

---

## 3. 向量的几何意义

### a) 生活例子

你在一个方格棋盘上，从起点 `(0,0)` 出发。你走三步向东，四步向北，到达 `(3,4)`。

你在纸上画出来：先画 x 轴，再画 y 轴，从原点画一条箭头到 `(3,4)`。这就是向量的几何表示。

### b) 直观理解

在二维平面中，向量有两个核心几何属性：

**① 长度（模长，magnitude）**：箭头的长度——从原点到终点的"直线距离"。

用**勾股定理**计算：向量 `[a, b]` 的长度 = $\sqrt{a^2 + b^2}$

对 `v = [3, 4]`，长度 = $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

**② 方向（direction）**：箭头指向哪里。

通常用与 x 轴的夹角 $\theta$ 表示：$\theta = \arctan(b/a)$

对 `v = [3, 4]`，$\theta = \arctan(4/3) \approx 53.1°$

### c) 形式化定义

对于向量 $\mathbf{v} = [v_1, v_2, \ldots, v_n]^\top \in \mathbb{R}^n$：

- **长度（欧几里得范数，L2 范数）**：$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$
- **方向**：由各分量之间的比例关系决定

### d) 手算示例

**例题**：向量 $\mathbf{v} = [6, 8]^\top$，求它的长度和方向角。

**解**：

长度：$\|\mathbf{v}\| = \sqrt{6^2 + 8^2} = \sqrt{36 + 64} = \sqrt{100} = 10$

方向角：$\theta = \arctan(8/6) \approx 53.13°$

验证：$6^2 + 8^2 = 10^2$ ✓（这是一个 6-8-10 的直角三角形）

**例题**：向量 $\mathbf{u} = [-3, 4]^\top$，求长度和方向。

长度：$\|\mathbf{u}\| = \sqrt{(-3)^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$（长度总是非负）

方向角：$\theta = \arctan(4/(-3)) \approx 126.87°$（在第二象限）

注意：$\arctan$ 只返回 $(-90°, 90°)$，需要根据 $x$ 的正负调整象限。$x=-3<0, y=4>0$，所以实际角度在第二象限，约为 $180° - 53.13° = 126.87°$。

### e) Python 验证

```python
import numpy as np

v = np.array([6, 8])

# 计算长度
length = np.sqrt(v[0]**2 + v[1]**2)    # 手动
length_np = np.linalg.norm(v)           # numpy 内置
print(f"长度(手动): {length}, 长度(np): {length_np}")

# 计算方向角
angle = np.arctan2(v[1], v[0])          # arctan2 自动处理象限
angle_deg = np.degrees(angle)
print(f"方向角: {angle:.4f} rad = {angle_deg:.2f}°")

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, vec, title in [
    (axes[0], np.array([6, 8]), 'v = [6, 8] 长度=10, 角≈53°'),
    (axes[1], np.array([-3, 4]), 'u = [-3, 4] 长度=5, 角≈127°')
]:
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy',
              scale=1, color='royalblue', width=0.015)
    ax.set_xlim(-4, 8); ax.set_ylim(-1, 9)
    ax.axhline(y=0, color='gray', lw=0.5)
    ax.axvline(x=0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

### f) 常见误区

> **误区**："向量的长度就是终点到原点的直线距离，所以向量就是那个点。"

向量不是那个**终点**，而是从原点指向终点的**箭头本身**。点 `(3,4)` 和向量 `[3,4]` 在坐标上看起来一样，但概念不同：向量可以平移——`[3,4]` 不管从哪出发，都表示"向右 3、向上 4"的位移。

> **误区**："长度和方向完全定义了向量。"

在几何意义上是的。但高维向量（比如 100 维的词向量）你画不出来，只能通过长度和与其他向量的角度关系来理解。

### g) ML 中的应用

在 ML 中，向量的长度和方向处处可见：

- **权重向量的长度**：L2 正则化就是惩罚权重向量长度太大
- **向量之间的夹角**：余弦相似度衡量两个样本或词向量有多"方向一致"
- **梯度向量的长度**：梯度爆炸/消失就是梯度向量的长度异常
- **数据标准化**：让每个特征向量的均值为 0、方差为 1，本质是平移和缩放

---

## 4. 向量加法

### a) 生活例子

你在一个方格街区里走：
1. **第一步**：向东走 3 个街区 → 位移向量 `a = [3, 0]`
2. **第二步**：向北走 4 个街区 → 位移向量 `b = [0, 4]`

你**总共移动了多少**？从起点到终点，你相当于先向东 3，再向北 4——净位移是 `[3, 4]`。

这就是向量加法：**把两段位移"合成"成一段总位移**。

### b) 直观理解

向量加法有两种等价的几何理解：

**平行四边形法则**：把两个向量的起点放在一起，以它们为邻边画一个平行四边形，对角线就是 `a + b`。

```
    y
    ↑
  4 |       ● (3,4)  ← 这是 a+b
    |      ↗↑
  3 |    ↗  ↑
    |  ↗    ↑ b = [0,4]
  2 |↗      ↑
    |        ↑
  1 |← a →  ↑
    | [3,0]
  0 +--+--+--+--→ x
    0  1  2  3  4
```

**首尾相接**：把 `b` 的起点放在 `a` 的终点上，从原点指向 `b` 的终点就是结果。

### c) 形式化定义

对于 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$：

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

**就是对每个对应分量分别做加法**。简单到让人怀疑——没错，就这么简单。

### d) 手算示例

**例题**：$\mathbf{a} = [2, 3]^\top$，$\mathbf{b} = [4, 1]^\top$，求 $\mathbf{a} + \mathbf{b}$。

**解**：

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 2+4 \\ 3+1 \end{bmatrix} = \begin{bmatrix} 6 \\ 4 \end{bmatrix}$$

几何上：
- `a` 从 `(0,0)` 指向 `(2,3)`
- `b` 从 `(0,0)` 指向 `(4,1)`
- `a+b` 从 `(0,0)` 指向 `(6,4)`
- 如果先把 `b` 的起点移到 `a` 的终点 `(2,3)`，`b` 指向 `(2+4, 3+1) = (6,4)`——恰好就是 `a+b` 的终点！

**验证平行四边形**：以 `a` 和 `b` 为邻边的平行四边形，第四个顶点在 `(6,4)`。

### e) Python 验证

```python
import numpy as np

a = np.array([2, 3])
b = np.array([4, 1])

c = a + b
print(f"a + b = [{c[0]}, {c[1]}]")

# 手动逐分量验证
print(f"分量1: {a[0]} + {b[0]} = {a[0] + b[0]} (预期 {c[0]})")
print(f"分量2: {a[1]} + {b[1]} = {a[1] + b[1]} (预期 {c[1]})")

# 可视化
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.015, label='a = [2, 3]')
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.015, label='b = [4, 1]')
# 虚线：平移 b 到 a 的终点
ax.quiver(a[0], a[1], b[0], b[1], angles='xy', scale_units='xy', scale=1,
          color='red', alpha=0.3, width=0.01)
ax.quiver(b[0], b[1], a[0], a[1], angles='xy', scale_units='xy', scale=1,
          color='blue', alpha=0.3, width=0.01)
ax.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1,
          color='purple', width=0.02, label='a + b = [6, 4]')
ax.set_xlim(-1, 8); ax.set_ylim(-1, 6)
ax.axhline(y=0, color='gray', lw=0.5)
ax.axvline(x=0, color='gray', lw=0.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('向量加法的平行四边形法则')
plt.show()
```

### f) 常见误区

> **误区**："向量加法就是把长度加起来。"

**完全错误！** 向量加法不是标量加法。`[3, 0] + [0, 4] = [3, 4]`，长度 $\sqrt{25} = 5$，而不是 $3 + 4 = 7$。这一点非常关键——向量加法遵循**平行四边形法则**，不是简单的长度相加。只有当两个向量方向完全相同时，和的长度才等于长度的和（此时 $a$ 和 $b$ 在同一条直线上同向）。

### g) ML 中的应用

| 应用 | 公式 |
|------|------|
| 神经网络加偏置 | $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ |
| 梯度下降更新参数 | $\theta_\text{new} = \theta_\text{old} - \eta \cdot \nabla L$ |
| 残差连接 (ResNet) | $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ |

---

## 5. 标量乘法

### a) 生活例子

你跑步的速度是 5 km/h 向东。如果加速到**两倍**——10 km/h 向东。这个"乘 2"的操作就是标量乘法：把向量的长度拉伸到 2 倍，方向不变。

如果你决定**反方向跑**——5 km/h 向西。这等价于乘以 -1：长度不变，方向翻转 180°。

### b) 直观理解

标量乘法 $c \cdot \mathbf{v}$ 的效果：

| $c$ 的值 | 效果 |
|-----------|------|
| $c > 1$ | 拉伸（变长） |
| $0 < c < 1$ | 压缩（变短） |
| $c = 0$ | 坍缩为零向量 |
| $c = -1$ | 方向反转，长度不变 |
| $c < -1$ | 反转 + 拉伸 |
| $-1 < c < 0$ | 反转 + 压缩 |

几何上：**标量乘以向量，就是沿着原来的方向（或反方向）缩放箭头的长度。**

### c) 形式化定义

对于标量 $c \in \mathbb{R}$ 和向量 $\mathbf{v} \in \mathbb{R}^n$：

$$c \cdot \mathbf{v} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{bmatrix}$$

每个分量都乘以 $c$。

### d) 手算示例

**例题**：$\mathbf{v} = [3, 4]^\top$，分别计算 $2\mathbf{v}$、$0.5\mathbf{v}$、$-\mathbf{v}$ 的长度。

**解**：

原向量长度：$\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = 5$

$2\mathbf{v} = [6, 8]^\top$，长度 $\|2\mathbf{v}\| = \sqrt{36 + 64} = 10 = 2 \times 5$ ✓

$0.5\mathbf{v} = [1.5, 2]^\top$，长度 $\|0.5\mathbf{v}\| = \sqrt{2.25 + 4} = 2.5 = 0.5 \times 5$ ✓

$-\mathbf{v} = [-3, -4]^\top$，长度 $\|-\mathbf{v}\| = \sqrt{9 + 16} = 5$（长度不变，方向反了）✓

**规律**：$\|c \cdot \mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$

### e) Python 验证

```python
import numpy as np

v = np.array([3.0, 4.0])
length_v = np.linalg.norm(v)

for c, desc in [(2, '2v (拉伸)'), (0.5, '0.5v (压缩)'), (-1, '-v (反向)')]:
    result = c * v
    length_result = np.linalg.norm(result)
    expected_length = abs(c) * length_v
    print(f"{desc}: {result}, 长度={length_result:.2f} (预期 {expected_length:.2f})")
```

### f) 常见误区

> **误区**："乘以负数会让向量长度变为负数。"

长度（范数）永远是**非负**的。乘以 -1 改变的是方向（180° 翻转），长度保持不变。

> **误区**："任何一个向量乘以一个标量就可以变成任意向量。"

不。标量乘法只能沿**同一条直线**伸缩。你无法通过乘一个标量把 `[1, 0]` 变成 `[0, 1]`——方向完全不同。这正是为什么我们需要**向量加法 + 标量乘法**的组合（线性组合）来覆盖整个空间。

### g) ML 中的应用

| 应用 | 公式 |
|------|------|
| 学习率缩放梯度 | $\theta \leftarrow \theta - \eta \cdot \nabla L$ |
| 特征缩放 | $\mathbf{x}_\text{scaled} = \frac{\mathbf{x} - \mu}{\sigma}$ |
| Batch Normalization | $\mathbf{y} = \gamma \cdot \hat{\mathbf{x}} + \beta$ |

---

## 6. 点积（内积）

### a) 生活例子

你在推一个箱子。你用了 10 牛顿的力，箱子移动了 5 米。

但是——你推的方向和箱子移动的方向**不一定完全一致**。如果你斜着推（力的方向与移动方向有夹角），只有力的**在移动方向上的分量**在做功。

物理学中：**功 = 力的大小 × 位移的大小 × cos(夹角)**

这就是点积的物理原型：
$$\text{功} = \mathbf{F} \cdot \mathbf{d} = \|\mathbf{F}\| \cdot \|\mathbf{d}\| \cdot \cos\theta$$

### b) 直观理解

点积衡量两个向量**有多"一致"**：

- **方向相同**（夹角 0°）：点积最大（正数），$\cos 0° = 1$，点积 = 长度之积
- **互相垂直**（夹角 90°）：点积为零，$\cos 90° = 0$——"完全无关"
- **方向相反**（夹角 180°）：点积为负数，$\cos 180° = -1$——"完全对立"

另一种理解：**把 a 投影到 b 的方向上，投影长度 × b 的长度 = 点积**。或者反过来，把 b 投影到 a 上也是一样。

### c) 形式化定义

对于 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$，点积有两种等价定义：

**代数定义**（分量相乘再求和）：
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**几何定义**（长度乘夹角余弦）：
$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos\theta$$

其中 $\theta$ 是两向量之间的夹角。

> 这两个定义是**等价的**——这是线性代数中最优美的结论之一。下面我们手算验证。

### d) 手算示例

**例题**：$\mathbf{a} = [2, 0]^\top$（指向正东），$\mathbf{b} = [1, 2]^\top$（指向东北），用两种定义求 $\mathbf{a} \cdot \mathbf{b}$。

**方法一（代数定义）**：

$$\mathbf{a} \cdot \mathbf{b} = 2 \times 1 + 0 \times 2 = 2 + 0 = 2$$

**方法二（几何定义）**：

- $\|\mathbf{a}\| = \sqrt{2^2 + 0^2} = 2$
- $\|\mathbf{b}\| = \sqrt{1^2 + 2^2} = \sqrt{5} \approx 2.236$
- 用代数定义的结果反推夹角：$2 = 2 \times \sqrt{5} \times \cos\theta$
- $\cos\theta = \frac{2}{2\sqrt{5}} = \frac{1}{\sqrt{5}} \approx 0.447$
- $\theta \approx 63.4°$

**验证**：直接用向量坐标算夹角余弦：

$$\cos\theta = \frac{2 \times 1 + 0 \times 2}{\sqrt{4} \cdot \sqrt{5}} = \frac{2}{2\sqrt{5}} = \frac{1}{\sqrt{5}}$$

一致！✓

**例题**：$\mathbf{a} = [3, 1]^\top$，$\mathbf{b} = [-1, 3]^\top$，求点积。

$$\mathbf{a} \cdot \mathbf{b} = 3 \times (-1) + 1 \times 3 = -3 + 3 = 0$$

点积为 0 → 两向量**垂直**（夹角 = 90°）。检查：$\mathbf{a}$ 方向是 `(3,1)`，$\mathbf{b}$ 方向是 `(-1,3)`——把 $\mathbf{a}$ 旋转 90° 逆时针得到 `(-1,3)`，确实垂直！

### e) Python 验证

```python
import numpy as np

a = np.array([2, 0])
b = np.array([1, 2])

# 方法1: 代数定义
dot_algebraic = np.dot(a, b)  # 或 a @ b
# 方法2: 几何定义
dot_geometric = np.linalg.norm(a) * np.linalg.norm(b) * \
                np.cos(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

print(f"代数定义: a·b = {dot_algebraic}")
print(f"几何定义: a·b = {dot_geometric:.4f}")

# 计算夹角
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
print(f"cosθ = {cos_theta:.4f}, θ = {theta:.1f}°")

# 验证垂直关系
c = np.array([3, 1])
d = np.array([-1, 3])
print(f"\n[3,1] · [-1,3] = {np.dot(c, d)} (垂直? {np.dot(c, d) == 0})")
```

### f) 常见误区

> **误区**："点积的正负直接反映夹角是锐角还是钝角。"

这一点**是对的**！但有人误以为点积的大小直接等于夹角——不对，点积还受两个向量长度的影响。两个长向量的点积可能很大，但它们的夹角其实也很大。**要比较方向相似性，应该用余弦相似度**（除以长度之积）。

> **误区**："点积只适用于 2D/3D 向量。"

不。代数定义适用于**任意维度**。1000 维的词向量一样可以做点积。

### g) ML 中的应用

点积是 ML 中最重要的运算之一：

| 应用 | 公式 | 说明 |
|------|------|------|
| 神经元输出 | $y = \mathbf{w} \cdot \mathbf{x} + b$ | 加权求和 |
| 余弦相似度 | $\text{sim} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | 文本检索、推荐系统 |
| 注意力分数 | $\text{score} = \mathbf{Q} \cdot \mathbf{K}^\top$ | Transformer 核心 |
| 全连接层 | $\mathbf{z} = W\mathbf{x}$ | 矩阵的每一行与输入做点积 |
| 卷积运算 | $y_{ij} = \sum \text{kernel} \odot \text{patch}$ | 逐元素乘加（也是点积） |

---

## 7. 向量的长度与范数

### a) 生活例子

你打车从家到公司。司机问你："走哪条路？"

路线 A：直接走大路，8 公里。
路线 B：先向东 3 公里，再向北 4 公里，然后向西 2 公里，再向北 3 公里……绕来绕去，实际路程 12 公里。

两条路的**直线距离**（点到点的最短距离）可能差不多，但**实际行驶距离**天差地别。

这两种"距离"对应两种最常见的向量范数：
- **直线距离**：L2 范数（欧几里得距离）——"鸟飞的距离"
- **街区行驶距离**：L1 范数（曼哈顿距离）——"出租车沿方格路开的距离"

### b) 直观理解

纽约曼哈顿的街道像方格棋盘。你从 `(0, 0)` 走到 `(3, 4)`：

```
      y
      ↑
    4 |   +---+---+---● 终点 (3,4)
      |   |   |   |
    3 |   +---+---+---+
      |   |   |   |
    2 |   +---+---+---+
      |   |   |   |
    1 |   +---+---+---+
      |   |   |   |
    0 ●---+---+---+---→ x
      0   1   2   3   4
```

- **L2 距离**：直接走斜线，$\sqrt{3^2 + 4^2} = 5$ 公里（如果你能穿楼）
- **L1 距离**：只能沿街道走，先向东 3，再向北 4，共 $3 + 4 = 7$ 公里

### c) 形式化定义

> **Lp 范数**（p-范数）：对于 $\mathbf{v} \in \mathbb{R}^n$，
> $$\|\mathbf{v}\|_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}$$

最常用的两种：

**L1 范数（曼哈顿距离）**（$p=1$）：
$$\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|$$

**L2 范数（欧几里得距离）**（$p=2$）：
$$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**L∞ 范数（最大范数）**（$p \to \infty$）：
$$\|\mathbf{v}\|_\infty = \max(|v_1|, |v_2|, \ldots, |v_n|)$$

### d) 手算示例

**例题**：$\mathbf{v} = [3, -4, 0, 5]^\top$，求 L1、L2 和 L∞ 范数。

**L1 范数**：

$$\|\mathbf{v}\|_1 = |3| + |-4| + |0| + |5| = 3 + 4 + 0 + 5 = 12$$

**L2 范数**：

$$\|\mathbf{v}\|_2 = \sqrt{3^2 + (-4)^2 + 0^2 + 5^2} = \sqrt{9 + 16 + 0 + 25} = \sqrt{50} \approx 7.071$$

**L∞ 范数**：

$$\|\mathbf{v}\|_\infty = \max\{3, 4, 0, 5\} = 5$$

**比较**：$12 \geq 7.071 \geq 5$。实际上，对于任意向量，$\|\mathbf{v}\|_1 \geq \|\mathbf{v}\|_2 \geq \|\mathbf{v}\|_\infty$ 恒成立。

### e) Python 验证

```python
import numpy as np

v = np.array([3, -4, 0, 5])

l1 = np.linalg.norm(v, ord=1)
l2 = np.linalg.norm(v, ord=2)
linf = np.linalg.norm(v, ord=np.inf)

print(f"L1 范数: {l1}")
print(f"L2 范数: {l2:.4f}")
print(f"L∞ 范数: {linf}")
print(f"验证 L1 ≥ L2 ≥ L∞: {l1} ≥ {l2:.4f} ≥ {linf} ? {l1 >= l2 >= linf}")

# 手动计算验证
print(f"\n手动 L1: {np.sum(np.abs(v))}")
print(f"手动 L2: {np.sqrt(np.sum(v**2))}")
print(f"手动 L∞: {np.max(np.abs(v))}")
```

### f) 常见误区

> **误区**："范数和距离是一个意思。"

范数是向量的属性（"这个向量有多大"），距离是两个点之间的属性（"这两个点隔多远"）。但它们是相关的——两点之间的距离 = 两点差向量的范数：$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|$。

> **误区**："L1 比 L2 总是小。"

恰恰相反。$\|\mathbf{v}\|_1 \geq \|\mathbf{v}\|_2$ 恒成立。例如 `[3, 4]`：L1=7, L2=5。

### g) ML 中的应用

| 范数类型 | ML 应用 |
|----------|---------|
| **L2 范数** | L2 正则化（Ridge/权重衰减）：$\lambda\|\mathbf{w}\|_2^2$；欧氏距离分类（KNN） |
| **L1 范数** | L1 正则化（Lasso）：$\lambda\|\mathbf{w}\|_1$，产生稀疏解 |
| **L∞ 范数** | 梯度裁剪：限制梯度每个分量不超过阈值 |
| **L2 距离** | K-means 聚类的默认距离度量 |
| **余弦距离** | 文本/图像检索、推荐系统（`1 - 余弦相似度`） |

**L1 vs L2 正则化的核心区别**：

- **L2** 惩罚所有权重的平方和 → 把权重向零"挤压"，但很少精确为零
- **L1** 惩罚所有权重的绝对值之和 → 很多权重被精确"逼到零"，产生**稀疏模型**

> 这个差异源于几何：L2 球是圆的，与损失函数等高线相切点往往不在坐标轴上；L1 "球"是菱形的，角点在坐标轴上，切线更容易落在坐标轴→权重的某些分量恰好为零。

---

## 8. 什么是矩阵？

### a) 生活例子

打开 Excel，你看到的是一张二维表格。每行一条记录，每列一个字段：

| 姓名 | 身高(cm) | 体重(kg) | 年龄 |
|------|----------|----------|------|
| 小明 | 170 | 65 | 23 |
| 小红 | 165 | 55 | 25 |
| 小刚 | 178 | 78 | 22 |

去掉"姓名"列（因为它不是数字），剩下的就是一个 $3 \times 3$ 的**矩阵**：

$$\begin{bmatrix} 170 & 65 & 23 \\ 165 & 55 & 25 \\ 178 & 78 & 22 \end{bmatrix}$$

另外，一张黑白图片也是一个矩阵——每个格子（像素）的值是灰度（0=黑, 255=白）。一张 $28 \times 28$ 像素的手写数字图片，就是一个 $28 \times 28$ 的矩阵。

### b) 直观理解

一个矩阵有**两个视角**：

**视角一：多个列向量横向堆叠**

把矩阵的每一列看成一个向量。一个 $m \times n$ 的矩阵就是 $n$ 个 $m$ 维列向量并排放置。

$$\begin{bmatrix} 2 & 1 & 0 \\ 3 & 4 & 5 \end{bmatrix} = \left[ \mathbf{c}_1 \; \mathbf{c}_2 \; \mathbf{c}_3 \right]$$

其中 $\mathbf{c}_1 = [2, 3]^\top$，$\mathbf{c}_2 = [1, 4]^\top$，$\mathbf{c}_3 = [0, 5]^\top$。

**视角二：一个变换器**

矩阵乘上一个向量，得到一个新的向量。这个操作可以理解成对空间做了某种**变换**（旋转、拉伸、压缩、投影等）。从这个角度说，**矩阵就是线性变换的"化身"**。

### c) 形式化定义

> **矩阵**（Matrix）：$m \times n$ 的矩形数字阵列。$A \in \mathbb{R}^{m \times n}$ 表示矩阵 $A$ 有 $m$ 行、$n$ 列，所有元素都是实数。

记法：大写字母 $A, B, W, X$

$$A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

$a_{ij}$ 表示第 $i$ 行、第 $j$ 列的元素。

矩阵的形状（shape）= $(m, n)$，$m$ 行 $n$ 列。

### d) 手算示例

一个 $2 \times 3$ 的矩阵有 2 行、3 列：

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$

- $a_{11} = 1$（第 1 行第 1 列）
- $a_{23} = 6$（第 2 行第 3 列）
- 形状：$2 \times 3$
- 第 1 列 = $[1, 4]^\top$，第 2 列 = $[2, 5]^\top$，第 3 列 = $[3, 6]^\top$
- 第 1 行 = $[1, 2, 3]$，第 2 行 = $[4, 5, 6]$

### e) Python 验证

```python
import numpy as np

# 创建矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[170, 65, 23],
              [165, 55, 25],
              [178, 78, 22]])

print(f"A 形状: {A.shape}, 数据类型: {A.dtype}")
print(f"A:\n{A}")
print(f"\nB (身高/体重/年龄):\n{B}")
print(f"B[0, 1] = {B[0, 1]} (第1行第2列, 即小明的体重)")

# 行列索引
print(f"\nA 的第1列: {A[:, 0]}")
print(f"A 的第2行: {A[1, :]}")

# 图像矩阵 — 28×28 像素
np.random.seed(42)
image = np.random.randint(0, 256, (28, 28))
print(f"\n图像矩阵形状: {image.shape}")
print(f"左上角 3×3 像素:\n{image[:3, :3]}")
```

### f) 常见误区

> **误区**："行数和列数相等的矩阵才是矩阵。"

不。$m \times n$ 矩阵中 $m$ 可以等于 $n$（方阵），也可以不等（长矩阵或高矩阵）。只有方阵才能做某些特殊运算（如求行列式、求逆），但不等也一样是矩阵。

> **误区**："矩阵就是一堆数字随便摆。"

矩阵每一列的意义必须一致。在 ML 中，通常是**行 = 样本，列 = 特征**。行列约定必须清楚，否则矩阵乘法会出错。

### g) ML 中的应用

| 数据 | 矩阵形状 | 含义 |
|------|----------|------|
| 训练数据 | $(N, D)$ | $N$ 个样本，$D$ 个特征 |
| 权重矩阵 | $(D_{out}, D_{in})$ | 输出维度 × 输入维度 |
| 一张灰度图 | $(H, W)$ | 高 × 宽 |
| 批量 RGB 图 | $(B, C, H, W)$ | 批量 × 通道 × 高 × 宽（PyTorch） |
| 用户-物品评分 | $(U, I)$ | $U$ 个用户 × $I$ 个物品 |

---

## 9. 矩阵乘法 —— 最重要的运算

### a) 生活例子

你经营一家奶茶店，卖三种饮品。今天上午卖了：

|  | 珍珠奶茶 | 柠檬茶 | 奶盖 |
|--|---------|--------|------|
| 杯数 | 20 | 15 | 10 |

每杯的原料用量（以克为单位）：

|  | 珍珠奶茶 | 柠檬茶 | 奶盖 |
|--|---------|--------|------|
| 茶叶 | 5 | 8 | 4 |
| 糖 | 30 | 25 | 20 |
| 奶 | 50 | 0 | 60 |

**问题**：今天上午总共用了多少茶叶、糖和奶？

**手算**：
- 茶叶：$20 \times 5 + 15 \times 8 + 10 \times 4 = 100 + 120 + 40 = 260$ 克
- 糖：$20 \times 30 + 15 \times 25 + 10 \times 20 = 600 + 375 + 200 = 1175$ 克
- 奶：$20 \times 50 + 15 \times 0 + 10 \times 60 = 1000 + 0 + 600 = 1600$ 克

这个计算过程就是**矩阵乘法**——把销量向量乘以原料矩阵的转置，得到原料总用量向量。

### b) 直观理解

矩阵乘法 $C = A \times B$ 有三种等价的直觉理解：

**观点一：行 × 列的点积**

$C$ 的 $(i, j)$ 位置 = $A$ 的第 $i$ **行**与 $B$ 的第 $j$ **列**做点积。

**观点二：$A$ 的行对 $B$ 的列的加权组合**

$C$ 的第 $i$ 行 = $A$ 的第 $i$ 行作为"权重"，与 $B$ 的**每一行**的加权组合。

**观点三：$A$ 的列的组合（最重要的几何视角）**

$C$ 的第 $j$ 列 = $A$ 的**所有列**按 $B$ 的第 $j$ 列给出的权重做线性组合。这揭示了矩阵乘法的几何本质——$B$ 的每一列是在指定"怎么用 $A$ 的列来建筑 $C$ 的列"。

### c) 形式化定义

对于 $A \in \mathbb{R}^{m \times k}$ 和 $B \in \mathbb{R}^{k \times n}$：

$$C = A \times B \in \mathbb{R}^{m \times n}$$

$$c_{ij} = \sum_{r=1}^{k} a_{ir} \cdot b_{rj} = a_{i1}b_{1j} + a_{i2}b_{2j} + \cdots + a_{ik}b_{kj}$$

**维度规则**：$A$ 的列数必须等于 $B$ 的行数（都是 $k$）。结果矩阵的行数 = $A$ 的行数（$m$），列数 = $B$ 的列数（$n$）。中间维度 $k$ 在乘法中被"消耗"。

> 速记口诀：**"内同可乘，外定形状"**——$(m \times \textcolor{red}{k}) \times (\textcolor{red}{k} \times n) = (m \times n)$

### d) 手算示例 —— 完整展开

**例题**：计算 $C = A \times B$，其中

$$A = \begin{bmatrix} \textcolor{blue}{1} & \textcolor{blue}{2} \\ \textcolor{green}{3} & \textcolor{green}{4} \end{bmatrix}_{2 \times 2},\quad
B = \begin{bmatrix} \textcolor{red}{5} & \textcolor{red}{6} \\ \textcolor{red}{7} & \textcolor{red}{8} \end{bmatrix}_{2 \times 2}$$

**维度检查**：$2 \times \textcolor{red}{2}$ 与 $\textcolor{red}{2} \times 2$，内同（2=2），可乘 ✓，结果 $2 \times 2$。

**逐元素计算**：

$$c_{11} = a_{11}b_{11} + a_{12}b_{21} = \textcolor{blue}{1} \times \textcolor{red}{5} + \textcolor{blue}{2} \times \textcolor{red}{7} = 5 + 14 = 19$$

$$c_{12} = a_{11}b_{12} + a_{12}b_{22} = \textcolor{blue}{1} \times \textcolor{red}{6} + \textcolor{blue}{2} \times \textcolor{red}{8} = 6 + 16 = 22$$

$$c_{21} = a_{21}b_{11} + a_{22}b_{21} = \textcolor{green}{3} \times \textcolor{red}{5} + \textcolor{green}{4} \times \textcolor{red}{7} = 15 + 28 = 43$$

$$c_{22} = a_{21}b_{12} + a_{22}b_{22} = \textcolor{green}{3} \times \textcolor{red}{6} + \textcolor{green}{4} \times \textcolor{red}{8} = 18 + 32 = 50$$

$$C = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

**用列组合观点验证**：

$B$ 的第 1 列是 `[5, 7]ᵀ`，$C$ 的第 1 列 = $5 \times A$ 的第 1 列 + $7 \times A$ 的第 2 列：

$$C_{\text{col1}} = 5 \cdot \begin{bmatrix}1\\3\end{bmatrix} + 7 \cdot \begin{bmatrix}2\\4\end{bmatrix} = \begin{bmatrix}5\\15\end{bmatrix} + \begin{bmatrix}14\\28\end{bmatrix} = \begin{bmatrix}19\\43\end{bmatrix}$$ ✓

### e) Python 验证

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

C = A @ B  # Python 3.5+ 矩阵乘法运算符
print(f"A @ B:\n{C}")

# 手动验证每个元素
print(f"\nc11 = 1×5 + 2×7 = {1*5 + 2*7} (预期 19)")
print(f"c12 = 1×6 + 2×8 = {1*6 + 2*8} (预期 22)")
print(f"c21 = 3×5 + 4×7 = {3*5 + 4*7} (预期 43)")
print(f"c22 = 3×6 + 4×8 = {3*6 + 4*8} (预期 50)")

# 列组合观点验证
col1 = B[0, 0] * A[:, 0] + B[1, 0] * A[:, 1]
col2 = B[0, 1] * A[:, 0] + B[1, 1] * A[:, 1]
print(f"\n列组合得到的 C:\n{np.column_stack([col1, col2])}")
print(f"与 A @ B 一致? {np.allclose(C, np.column_stack([col1, col2]))}")

# 维度规则演示：2×3 乘 3×2 → 2×2
A2 = np.array([[1, 2, 3], [4, 5, 6]])   # (2, 3)
B2 = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)
C2 = A2 @ B2  # (2, 2)
print(f"\n(2,3) @ (3,2) → {C2.shape}:\n{C2}")
```

### f) 常见误区

> **误区**："矩阵乘法就是对应位置相乘。"

那是**哈达玛积**（Hadamard product），不是矩阵乘法。矩阵乘法是行与列的点积，形状规则完全不同。

| 运算 | 符号 | Python | 形状 |
|------|------|--------|------|
| 矩阵乘法 | $C = AB$ | `A @ B` | $(m,k) \times (k,n) \to (m,n)$ |
| 哈达玛积 | $C = A \odot B$ | `A * B` | $(m,n) \times (m,n) \to (m,n)$ |

> **误区**："$AB$ 总是等于 $BA$。"

**绝大多数情况下不相等！** 矩阵乘法不满足交换律。上面 `A @ B` = `[[19,22],[43,50]]`，但 `B @ A` = `[[23,34],[31,46]]`——完全不同。更糟的是，$B \times A$ 的维度可能根本不匹配。例如 $(2,3) \times (3,2) = (2,2)$，但 $(3,2) \times (2,3) = (3,3)$，形状都不一样。

### g) ML 中的应用

矩阵乘法是 ML 的灵魂运算：

| 场景 | 公式 | 形状 |
|------|------|------|
| 全连接层前向传播 | $\mathbf{Z} = X W^\top + \mathbf{b}$ | $(N, D) \times (D, D_{out}) \to (N, D_{out})$ |
| 线性回归预测 | $\hat{\mathbf{y}} = X \mathbf{w}$ | $(N, D) \times (D,) \to (N,)$ |
| 协方差矩阵 | $\Sigma = \frac{1}{N}X^\top X$ | $(D, N) \times (N, D) \to (D, D)$ |
| 注意力 QKᵀ | $\text{scores} = Q K^\top$ | $(N, d_k) \times (d_k, N) \to (N, N)$ |

---

## 10. 矩阵转置

### a) 生活例子

你有一张学生成绩表。现在是**行 = 学生，列 = 科目**：

| | 数学 | 语文 | 英语 |
|--|------|------|------|
| 小明 | 85 | 90 | 78 |
| 小红 | 92 | 88 | 95 |

转置就是把行列互换——变成**行 = 科目，列 = 学生**：

| | 小明 | 小红 |
|--|------|------|
| 数学 | 85 | 92 |
| 语文 | 90 | 88 |
| 英语 | 78 | 95 |

### b) 直观理解

转置就是把矩阵沿着**主对角线**（左上→右下）"翻个面"。第一行变成第一列，第二行变成第二列……依此类推。

为什么需要转置？因为**矩阵乘法要求维度匹配**。很多时候，手里矩阵的数据方向正好反了，转置一下维度就对了。

### c) 形式化定义

对于 $A \in \mathbb{R}^{m \times n}$，其转置 $A^\top \in \mathbb{R}^{n \times m}$：

$$(A^\top)_{ij} = A_{ji}$$

即 $A$ 的第 $i$ 行第 $j$ 列 = $A^\top$ 的第 $j$ 行第 $i$ 列。

**重要性质**：
- $(A^\top)^\top = A$
- $(AB)^\top = B^\top A^\top$（注意反序！）
- $(A + B)^\top = A^\top + B^\top$

### d) 手算示例

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}_{2 \times 3}$$

$$A^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}_{3 \times 2}$$

验证 $(AB)^\top = B^\top A^\top$：

设 $A = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$（单位矩阵），$B = \begin{bmatrix}2 & 3 \\ 4 & 5\end{bmatrix}$

左：$(AB)^\top = B^\top = \begin{bmatrix}2 & 4 \\ 3 & 5\end{bmatrix}$

右：$B^\top A^\top = \begin{bmatrix}2 & 4 \\ 3 & 5\end{bmatrix} \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} = \begin{bmatrix}2 & 4 \\ 3 & 5\end{bmatrix}$

一致 ✓。

### e) Python 验证

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"A (2×3):\n{A}")
print(f"Aᵀ (3×2):\n{A.T}")

# 验证 (AB)ᵀ = Bᵀ Aᵀ
A2 = np.array([[1, 2], [3, 4]])
B2 = np.array([[5, 6], [7, 8]])
lhs = (A2 @ B2).T
rhs = B2.T @ A2.T
print(f"\n(AB)ᵀ = BᵀAᵀ ? {np.allclose(lhs, rhs)}")
print(f"(AB)ᵀ:\n{lhs}")
print(f"BᵀAᵀ:\n{rhs}")
```

### f) 常见误区

> **误区**："转置只是好看，不影响数值计算。"

转置直接影响维度匹配。机器学习中大量公式依赖转置来"对齐"维度。例如线性回归的正规方程：$\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$。如果 $X$ 是 $(N, D)$，$X^\top X$ 是 $(D, D)$，$X^\top \mathbf{y}$ 是 $(D,)$——转置就是让维度对上。

### g) ML 中的应用：数据布局约定

在 ML 中，有一个非常重要的**布局约定**：

> **行 = 样本（samples），列 = 特征（features）**

所以数据矩阵 $X$ 通常是 $(N, D)$ 形状。当你需要计算 $D \times D$ 的协方差矩阵时，你需要 $X^\top X$（$(D, N) \times (N, D) \to (D, D)$）。

---

## 11. 单位矩阵与对角矩阵

### 11.1 单位矩阵 —— "乘什么都不变"

### a) 生活例子

在数的乘法中，**1** 很特殊——任何数乘以 1 还是它自己：$5 \times 1 = 5$。

矩阵里有没有这样的"1"？有——**单位矩阵（Identity Matrix）**。

### b) 直观理解

单位矩阵 $I_n$ 是 $n \times n$ 的方阵，主对角线上全是 1，其他地方全是 0。

$$I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

任何矩阵 $A$ 乘以维度匹配的单位矩阵，结果还是 $A$ 自己：
$$A I_n = A, \quad I_m A = A$$

**几何意义**：单位矩阵代表**恒等变换**——输入什么向量，输出完全一样的向量。你可以把它理解成"什么都不做"的变换。它不会缩放（对角元素=1），不会旋转（非对角元素=0，无交互），也不会投影。

### c) 手算示例

$$A = \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix}, \quad I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$A I_2 = \begin{bmatrix} 2 \times 1 + 3 \times 0 & 2 \times 0 + 3 \times 1 \\ 4 \times 1 + 5 \times 0 & 4 \times 0 + 5 \times 1 \end{bmatrix} = \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} = A$$

### 11.2 对角矩阵 —— "各走各的"

### a) 生活例子

你对三个投资渠道分别配置资金：股票 ×2，基金 ×1.5，存款 ×1。这可以用对角矩阵表示：

$$D = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 1.5 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### b) 直观理解

**对角矩阵**是所有非对角元素均为 0 的方阵。它的几何意义是**沿各坐标轴独立地拉伸或压缩**——x 轴缩放 $d_{11}$ 倍，y 轴缩放 $d_{22}$ 倍……各方向互不影响（这就是非对角元素为 0 的几何含义——没有跨维度的"拉扯"）。

将一个向量 $[x, y, z]^\top$ 乘以对角矩阵 $D = \text{diag}(a, b, c)$：
$$D\mathbf{v} = \begin{bmatrix} a \cdot x \\ b \cdot y \\ c \cdot z \end{bmatrix}$$

各分量被独立缩放，互不干扰。

### c) 形式化定义

对角矩阵 $D = \text{diag}(d_1, d_2, \ldots, d_n)$ 满足 $D_{ij} = 0$ 当 $i \neq j$。

记法：$\text{diag}(d_1, d_2, \ldots, d_n)$

### d) 手算示例

$$D = \text{diag}(2, 3) = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$D\mathbf{v} = \begin{bmatrix} 2 \times 1 + 0 \times 1 \\ 0 \times 1 + 3 \times 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

原来在 `(1,1)` 的点，被水平拉伸 2 倍、垂直拉伸 3 倍，到了 `(2,3)`。

### e) Python 验证

```python
import numpy as np

# 单位矩阵
I3 = np.eye(3)
print(f"3×3 单位矩阵:\n{I3}")

A = np.array([[2, 4, 1], [0, 3, 5], [6, 1, 2]])
print(f"\nA @ I:\n{A @ I3}")
print(f"与 A 相等? {np.allclose(A @ I3, A)}")

# 对角矩阵
D = np.diag([2, 3, 0.5])
print(f"\n对角矩阵 D:\n{D}")

v = np.array([1, 1, 1])
print(f"D @ v = {D @ v}")  # [2, 3, 0.5] — 各维度独立缩放
```

### f) 常见误区

> **误区**："对角矩阵的对角元素必须都是正数。"

不。对角元素可以是零、负数和任意实数。如果某个对角元素为零，那个维度就会被"压扁"；如果为负数，该维度的方向会被反转。

> **误区**："单位矩阵和对角矩阵是一样的。"

单位矩阵是特殊的对角矩阵（所有对角元素=1）。对角矩阵可以有任意对角元素。

### g) ML 中的应用

| 矩阵 | ML 应用 |
|------|---------|
| 单位矩阵 $I$ | 岭回归：$X^\top X + \lambda I$；神经网络的恒等初始化 |
| 对角矩阵 $\Sigma$ | SVD 中的奇异值矩阵；PCA 中各主成分的方差矩阵 |
| 对角矩阵 $\Lambda$ | 特征值对角矩阵 |

---

## 12. 逆矩阵 —— "撤销"变换

### a) 生活例子

你把一张照片放大到 2 倍（乘以 2）。怎么恢复原样？缩小到 $\frac{1}{2}$（乘以 $\frac{1}{2}$）。"乘 $\frac{1}{2}$"就是"乘 2"的逆操作。

在矩阵的世界里，如果你用一个矩阵 $A$ 对空间做了某种变换（旋转+拉伸），逆矩阵 $A^{-1}$ 的作用就是**把变换后的结果"变回原来的样子"**。

### b) 直观理解

**逆矩阵 = 变换的撤销按钮（Ctrl+Z）。**

- 如果 $A$ 把空间旋转 30°，$A^{-1}$ 就把空间往回旋转 30°
- 如果 $A$ 把空间拉伸 3 倍，$A^{-1}$ 就把空间压缩回 $\frac{1}{3}$
- 如果 $A$ 同时旋转+拉伸，$A^{-1}$ 就逆向旋转+压缩

$$A \cdot A^{-1} = A^{-1} \cdot A = I$$

先应用 $A$，再应用 $A^{-1}$，等于什么都没做（单位矩阵）。

### c) 形式化定义

对于方阵 $A \in \mathbb{R}^{n \times n}$，如果存在矩阵 $A^{-1}$ 使得：

$$A A^{-1} = A^{-1} A = I_n$$

则称 $A$ **可逆**（invertible）或**非奇异**（non-singular）。

$A$ 可逆的条件（三者等价）：
1. $\det(A) \neq 0$
2. $A$ 满秩，即 $\text{rank}(A) = n$
3. $A$ 的列向量线性无关

### d) 手算示例 —— 求 2×2 矩阵的逆

对于 $2 \times 2$ 矩阵 $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$，如果 $\det(A) = ad - bc \neq 0$：

$$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

**例题**：$A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$，求 $A^{-1}$。

**步骤一**：计算行列式
$$\det(A) = 2 \times 3 - 1 \times 1 = 6 - 1 = 5 \neq 0 \quad \checkmark$$

**步骤二**：套公式
$$A^{-1} = \frac{1}{5} \begin{bmatrix} 3 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{bmatrix}$$

**步骤三**：验证 $A A^{-1} = I$
$$A A^{-1} = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{bmatrix} = \begin{bmatrix} 2 \times 0.6 + 1 \times (-0.2) & 2 \times (-0.2) + 1 \times 0.4 \\ 1 \times 0.6 + 3 \times (-0.2) & 1 \times (-0.2) + 3 \times 0.4 \end{bmatrix}$$

$$= \begin{bmatrix} 1.2 - 0.2 & -0.4 + 0.4 \\ 0.6 - 0.6 & -0.2 + 1.2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I_2 \quad \checkmark$$

### e) Python 验证

```python
import numpy as np

A = np.array([[2, 1],
              [1, 3]])

A_inv = np.linalg.inv(A)
print(f"A:\n{A}")
print(f"A⁻¹:\n{A_inv}")
print(f"\nA @ A⁻¹:\n{A @ A_inv}")
print(f"等于 I ? {np.allclose(A @ A_inv, np.eye(2))}")

# 演示"撤销"变换
x = np.array([3, 4])
x_transformed = A @ x
x_back = A_inv @ x_transformed
print(f"\n原始 x: {x}")
print(f"A @ x: {x_transformed}")
print(f"A⁻¹ @ (A @ x): {x_back}")
print(f"完全还原? {np.allclose(x, x_back)}")

# 不可逆的例子
A_sing = np.array([[1, 2], [2, 4]])  # 第2列 = 2×第1列
print(f"\n奇异矩阵行列式: {np.linalg.det(A_sing):.0f}")
try:
    np.linalg.inv(A_sing)
except np.linalg.LinAlgError as e:
    print(f"求逆失败: {e}")
```

### f) 常见误区

> **误区**："所有矩阵都可以求逆。"

**不！** 只有方阵且行列式非零的矩阵才可逆。非方阵（$m \neq n$）不能求常规逆（但可以求伪逆）。方阵中，行列式为零也不可逆（因为"压扁"了空间，丢失了信息，无法恢复）。

> **误区**："$A^{-1} = \frac{1}{A}$。"

这只对标量成立。矩阵没有除法——$A^{-1}$ 不是一个分数，而是一个全新的矩阵。

### g) ML 中的应用

| 场景 | 用法 |
|------|------|
| 线性回归正规方程 | $\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$ |
| 马氏距离 | $d = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$ |
| 白化变换 | $\mathbf{z} = \Sigma^{-1/2} (\mathbf{x} - \boldsymbol{\mu})$ |
| 岭回归 | $\mathbf{w} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$（$\lambda I$ 保证可逆） |

---

## 13. 行列式 —— 变换的"面积放大率"

### a) 生活例子

你有一块正方形橡皮泥，边长 1 米，面积 1 平方米。你把它横向拉伸到 2 倍长，纵向压缩到一半——变成了一个 $2 \times 0.5 = 1$ 平方米的长方形。面积没变。

现在你再做另一个变形：横向 2 倍，纵向也 2 倍——面积变成了原来的 4 倍。

**行列式（determinant）就是衡量一个线性变换把"面积/体积"放大多少倍的指标。**

### b) 直观理解

**几何核心直觉**：$\det(A)$ = 线性变换 $A$ 对单位正方形/立方体的 **面积/体积放大倍数**。

- $\det(A) = 2$：面积放大 2 倍
- $\det(A) = 0.5$：面积缩小一半
- $\det(A) = 0$：面积被压成 0——空间被"压扁"到更低维度（例如正方形被压成一条线，面积=0）
- $\det(A) < 0$：空间被"翻转"了（镜像），同时面积按绝对值缩放
- $\det(A) = 1$：面积不变（如旋转、剪切变换）

### c) 形式化定义

**2×2 行列式**：

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**几何解释**：由列向量 $\begin{bmatrix}a\\c\end{bmatrix}$ 和 $\begin{bmatrix}b\\d\end{bmatrix}$ 张成的**平行四边形的有向面积**。

**3×3 行列式**（按第一行展开）：

$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = a \cdot \det\begin{bmatrix}e&f\\h&i\end{bmatrix} - b \cdot \det\begin{bmatrix}d&f\\g&i\end{bmatrix} + c \cdot \det\begin{bmatrix}d&e\\g&h\end{bmatrix}$$

### d) 手算示例

**例题 1（缩放）**：$A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$，求 $\det(A)$。

$$\det(A) = 2 \times 3 - 0 \times 0 = 6$$

**解释**：$A$ 把 x 方向拉伸 2 倍，y 方向拉伸 3 倍。单位正方形（$1 \times 1$）被变成 $2 \times 3 = 6$ 的长方形。面积放大了 6 倍 ✓

**例题 2（剪切）**：$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$，求 $\det(A)$。

$$\det(A) = 1 \times 1 - 1 \times 0 = 1$$

**解释**：剪切变换把正方形变成平行四边形，但**面积保持不变**（仍然是 1）。这就是行列式=1 的含义。

**例题 3（奇异）**：$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$，求 $\det(A)$。

$$\det(A) = 1 \times 4 - 2 \times 2 = 4 - 4 = 0$$

**解释**：两列向量 $[1,2]^\top$ 和 $[2,4]^\top$ 在**同一条直线上**（后者是前者的 2 倍），它们张成的"平行四边形"退化成一条线（面积=0）。$A$ 不可逆，因为信息从 2D 压成了 1D，无法恢复。

### e) Python 验证

```python
import numpy as np

# 缩放
A_scale = np.array([[2, 0], [0, 3]])
print(f"缩放矩阵 det = {np.linalg.det(A_scale):.0f} (预期 6)")

# 剪切 — 面积不变
A_shear = np.array([[1, 1], [0, 1]])
print(f"剪切矩阵 det = {np.linalg.det(A_shear):.0f} (预期 1)")

# 奇异 — 退化为线
A_sing = np.array([[1, 2], [2, 4]])
print(f"奇异矩阵 det = {np.linalg.det(A_sing):.0f} (预期 0)")

# 旋转 — 面积不变
theta = np.radians(30)
A_rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
print(f"旋转矩阵 det = {np.linalg.det(A_rot):.4f} (预期 1)")
```

### f) 常见误区

> **误区**："行列式等于零只是因为矩阵里有全零行。"

不。行列式为零的根本原因是**列向量线性相关**——即某些列可以由其他列的线性组合表示。全零行是线性相关的特例，但不是唯一原因。例如 $\begin{bmatrix}1&2\\2&4\end{bmatrix}$ 没有零行，但行列式为零。

> **误区**："行列式只对方阵有意义。"

几何上是的——非方阵的变换会改变空间的维度（例如 3D→2D），这时"体积"概念不适用。但非方阵有自己的"体积"概念（如 Gram 行列式）。

### g) ML 中的应用

| 应用 | 说明 |
|------|------|
| 可逆性判断 | $\det(X^\top X) = 0$ → 正规方程无解，需要正则化 |
| 多元高斯分布 | $p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det(\Sigma)}} \exp(-\frac{1}{2}(\mathbf{x}-\mu)^\top \Sigma^{-1} (\mathbf{x}-\mu))$ |
| PCA 白化 | 协方差矩阵的行列式衡量数据"分散程度" |

---

## 14. 线性方程组 $A\mathbf{x} = \mathbf{b}$ —— 三种视角

### a) 生活例子

你去商店买水果。苹果和橘子的**总价**是 8 元。但你不知道各自买了多少。店主说："你买的苹果和橘子一样多，总价 8 元，苹果 2 元/个，橘子 1 元/个——你自己算吧。"

设苹果数量为 $x$，橘子数量为 $y$：
- 数量关系：$x = y$ → $x - y = 0$
- 价格关系：$2x + 1y = 8$

写成方程组：
$$\begin{cases} x - y = 0 \\ 2x + y = 8 \end{cases}$$

写成矩阵形式：
$$\begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 8 \end{bmatrix}$$

这就是 $A\mathbf{x} = \mathbf{b}$。下面我们用**三种视角**来解这个系统。

### b) 三种视角

### 视角一：行视角（直线的交点）

方程组的每一行是一条直线。求解 = 找两条直线的交点。

- 直线 1：$x - y = 0$ → $y = x$（过原点，斜率 1）
- 直线 2：$2x + y = 8$ → $y = -2x + 8$（过 `(0,8)`，斜率 -2）

两条直线交于一点——那个点的坐标 $(x, y)$ 就是解。代入：$x = -2x + 8$ → $3x = 8$ → $x = 8/3 \approx 2.67$，$y = x = 8/3$。

**行视角的核心**：每个方程是一行，几何上是一条线（2D）或平面（3D），解是这些线/面的交点。

### 视角二：列视角（列向量的线性组合）

把 $A\mathbf{x} = \mathbf{b}$ 写成列向量的线性组合：

$$x \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} + y \cdot \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 8 \end{bmatrix}$$

**问题变成**：用 $A$ 的两列作为"建筑材料"，按 $x$ 和 $y$ 的比例混合——能否"搭出"向量 $\mathbf{b}$？

解出来：$x = 8/3$，$y = 8/3$。意味着需要 $\frac{8}{3}$ 个 $\begin{bmatrix}1\\2\end{bmatrix}$ 和 $\frac{8}{3}$ 个 $\begin{bmatrix}-1\\1\end{bmatrix}$ 才能组合出 $\begin{bmatrix}0\\8\end{bmatrix}$。

**列视角的核心**：$A\mathbf{x} = \mathbf{b}$ 是在问"$\mathbf{b}$ 是否在 $A$ 的列空间里？如果是，组合系数 $\mathbf{x}$ 是多少？"

### 视角三：矩阵视角（用逆矩阵"撤销"变换）

如果 $A$ 可逆：

$$\mathbf{x} = A^{-1}\mathbf{b}$$

$A$ 把未知的 $\mathbf{x}$ 变成了已知的 $\mathbf{b}$。我们反着来——用 $A^{-1}$ 把 $\mathbf{b}$ 变回去，就得到了 $\mathbf{x}$。

这就是变换的"撤销"视角：先被 $A$ 变换成 $\mathbf{b}$，再用 $A^{-1}$ 撤销变换回到 $\mathbf{x}$。

### d) 手算示例 —— 完整求解

**用矩阵视角解上面那个方程组**：

$$A = \begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0 \\ 8 \end{bmatrix}$$

**步骤一**：计算 $\det(A)$
$$\det(A) = 1 \times 1 - (-1) \times 2 = 1 + 2 = 3 \neq 0 \quad \checkmark$$

**步骤二**：求 $A^{-1}$
$$A^{-1} = \frac{1}{3} \begin{bmatrix} 1 & 1 \\ -2 & 1 \end{bmatrix} = \begin{bmatrix} 1/3 & 1/3 \\ -2/3 & 1/3 \end{bmatrix}$$

**步骤三**：$\mathbf{x} = A^{-1}\mathbf{b}$
$$\mathbf{x} = \begin{bmatrix} 1/3 & 1/3 \\ -2/3 & 1/3 \end{bmatrix} \begin{bmatrix} 0 \\ 8 \end{bmatrix} = \begin{bmatrix} 1/3 \times 0 + 1/3 \times 8 \\ -2/3 \times 0 + 1/3 \times 8 \end{bmatrix} = \begin{bmatrix} 8/3 \\ 8/3 \end{bmatrix}$$

**验证**：
$$\begin{bmatrix}1 & -1 \\ 2 & 1\end{bmatrix} \begin{bmatrix}8/3 \\ 8/3\end{bmatrix} = \begin{bmatrix}8/3 - 8/3 \\ 16/3 + 8/3\end{bmatrix} = \begin{bmatrix}0 \\ 24/3\end{bmatrix} = \begin{bmatrix}0 \\ 8\end{bmatrix} = \mathbf{b} \quad \checkmark$$

苹果 = $8/3 \approx 2.67$ 个，橘子也是 $8/3 \approx 2.67$ 个。答案一致。

### 三种视角对比

| 视角 | 怎么说 | 什么时候最有帮助 |
|------|--------|------------------|
| **行视角** | 找平面/直线的交点 | 低维（≤3D），可视化直观 |
| **列视角** | $\mathbf{b}$ 能不能用列向量"搭出来" | 理解秩/线性相关/解的存在性 |
| **矩阵视角** | 用逆矩阵"撤销"变换 | 数值计算、机器学习 |

> **列视角最值得培养**。它直接连接到"向量空间的基"、"秩"、"线性表出"等核心概念。在 ML 中，当你在想"数据的低维表示"时，你用的就是列视角。

### e) Python 验证

```python
import numpy as np

A = np.array([[1, -1],
              [2,  1]])
b = np.array([0, 8])

# 方法1: 求逆
x1 = np.linalg.inv(A) @ b
# 方法2: solve (数值更稳定)
x2 = np.linalg.solve(A, b)

print(f"方法1 (inv): x = {x1}")
print(f"方法2 (solve): x = {x2}")
print(f"验证 A@x = b: {A @ x1}")

# 行视角: 手动代入验证
print(f"\n方程1: {A[0,0]}×{x1[0]:.2f} + ({A[0,1]})×{x1[1]:.2f} = {A[0,0]*x1[0] + A[0,1]*x1[1]:.2f} (应为 {b[0]})")
print(f"方程2: {A[1,0]}×{x1[0]:.2f} + {A[1,1]}×{x1[1]:.2f} = {A[1,0]*x1[0] + A[1,1]*x1[1]:.2f} (应为 {b[1]})")

# 列视角: x1[0]*col0 + x1[1]*col1 = b
print(f"\n列视角: {x1[0]:.4f}×{A[:,0]} + {x1[1]:.4f}×{A[:,1]} = {x1[0]*A[:,0] + x1[1]*A[:,1]}")
```

### f) 常见误区

> **误区**："$A\mathbf{x} = \mathbf{b}$ 总有一个唯一解。"

不一定。有三种情况：
- **唯一解**：$A$ 可逆（$\det \neq 0$），$n$ 个独立方程，$n$ 个未知数
- **无穷多解**：$A$ 不可逆，但 $\mathbf{b}$ 在 $A$ 的列空间里（方程有冗余）
- **无解**：$\mathbf{b}$ 不在 $A$ 的列空间里（在 ML 中这种情况通常求最小二乘解）

### g) ML 中的应用

| 场景 | 说明 |
|------|------|
| 线性回归正规方程 | $(X^\top X)\mathbf{w} = X^\top \mathbf{y}$，直接求解 $\mathbf{w}$ |
| 最小二乘 | $\min\|A\mathbf{x} - \mathbf{b}\|^2_2$ 的解析解：$\mathbf{x} = (A^\top A)^{-1} A^\top \mathbf{b}$ |
| 岭回归 | $(X^\top X + \lambda I)\mathbf{w} = X^\top \mathbf{y}$ |

---

## 15. 秩（Rank）—— 矩阵的"独立信息量"

### a) 生活例子

你在面试候选人，收了 100 份简历，每份简历有 5 个指标（学历、工作经验、项目数、Github Stars、论文数）。

但你仔细一看发现：**论文数总是 = Github Stars ÷ 100**，而且**工作经验 = 学历年限 × 2 + 项目数**。

这意味着，虽然表格有 5 列，但实际上**只有 3 列是独立的**——另外 2 列是能从前 3 列推导出来的。

这个数据矩阵的**秩（Rank）**就是 3。秩 = 矩阵中真正独立的列（或行）的数量。

### b) 直观理解

**秩 = 矩阵所携带的"独立信息的维度数"。**

几何直觉：
- 一个 $3 \times 3$ 满秩矩阵可以把 3D 空间变换到（另一个）3D 空间——维度不变
- 一个秩为 2 的 $3 \times 3$ 矩阵会把 3D 空间"压扁"成一个 2D 平面——一个维度丢失了
- 一个秩为 1 的 $3 \times 3$ 矩阵会把一切压到一条直线上
- 一个秩为 0 的矩阵把一切压到原点（零矩阵）

**秩 = $\min$($\#$独立行, $\#$独立列)。而且行秩 = 列秩，统称秩。**

### c) 形式化定义

> 矩阵 $A \in \mathbb{R}^{m \times n}$ 的**秩** $\text{rank}(A)$ 等于 $A$ 中线性无关的列向量的最大个数，也等于线性无关的行向量的最大个数。

**满秩**：$\text{rank}(A) = \min(m, n)$——行或列全部独立。

**秩亏**：$\text{rank}(A) < \min(m, n)$——存在冗余。

### d) 手算示例

**例题**：求矩阵 $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{bmatrix}$ 的秩。

**观察**：第 2 行 = 第 1 行 × 2，第 3 行 = 第 1 行 × 3。同样，第 2 列 = 第 1 列 × 2，第 3 列 = 第 1 列 × 3。

所有列都"指向同一个方向"（$\begin{bmatrix}1\\2\\3\end{bmatrix}$ 方向），所以只有一个独立方向。

$$\text{rank}(A) = 1$$

**例题**：求矩阵 $B = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 3 \\ 0 & 0 & 0 \end{bmatrix}$ 的秩。

- 第 1 行 = $[1, 0, 2]$，第 2 行 = $[0, 1, 3]$，两者不是倍数关系 → 独立
- 第 3 行 = $[0, 0, 0]$ → 零行不贡献独立信息

前两列 $\begin{bmatrix}1\\0\\0\end{bmatrix}$ 和 $\begin{bmatrix}0\\1\\0\end{bmatrix}$ 也是独立的，第 3 列 = $2 \times$ 第 1 列 + $3 \times$ 第 2 列 → $\text{rank}(B) = 2$。

### e) Python 验证

```python
import numpy as np

# 秩为1的矩阵
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])
print(f"矩阵 A (各行成比例):\n{A}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

# 秩为2的矩阵
B = np.array([[1, 0, 2],
              [0, 1, 3],
              [0, 0, 0]])
print(f"\n矩阵 B:\n{B}")
print(f"rank(B) = {np.linalg.matrix_rank(B)}")

# 满秩矩阵
C = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
print(f"\n矩阵 C (单位矩阵):\n{C}")
print(f"rank(C) = {np.linalg.matrix_rank(C)}")

# ML 场景: 人造冗余特征
np.random.seed(42)
X = np.random.randn(100, 3)
# 添加一个完全冗余的特征: feature4 = 2*feature1 - feature2
X_red = np.column_stack([X, 2 * X[:, 0] - X[:, 1]])
print(f"\n原始特征 rank = {np.linalg.matrix_rank(X)}")
print(f"加入冗余特征后 rank = {np.linalg.matrix_rank(X_red)} (有4列但只有3个独立)")
```

### f) 常见误区

> **误区**："秩等于矩阵的行数或列数。"

不。秩 $\leq \min(m, n)$，可能远小于行数和列数。例如一个 $1000 \times 1000$ 的矩阵，如果秩只有 3，它本质上只包含 3 维信息。

> **误区**："$X^\top X$ 总是满秩的。"

不。如果 $X$ 的特征之间存在线性相关（多重共线性），$X^\top X$ 就会秩亏，此时正规方程无解。这就是为什么需要岭回归（$X^\top X + \lambda I$ 一定满秩）。

### g) ML 中的应用

| 场景 | 秩的含义 |
|------|----------|
| **多重共线性检测** | 特征矩阵 $X$ 的秩 < 特征数 → 存在冗余特征 |
| **正规方程可解性** | $\text{rank}(X^\top X) < n$ → 需要正则化 |
| **低秩近似/模型压缩** | 用低秩矩阵近似权重矩阵，减少参数量 |
| **矩阵补全** | 利用评分矩阵的低秩性预测缺失评分（协同过滤） |

---

## 16. 特征值与特征向量

### a) 生活例子

想象一个排球场上的扣球。球从高空飞向地面，你很难追踪它的精确轨迹。

但如果你找到**球飞行的主轴方向**——在主轴方向上，球的位置只沿着那条线移动，方向不会偏转。这个主方向就是运动方程的**特征向量**，沿着这个方向的移动速度（缩放倍数）就是**特征值**。

### b) 直观理解

**核心直觉**：当你用一个矩阵 $A$ 乘以一大堆向量时，绝大多数向量都会**改变方向**（被旋转）。但有一类特殊的向量，它们不改变方向，只是被**拉伸或压缩**。

这些特殊向量就是 $A$ 的**特征向量**（Eigenvector），拉伸因子就是**特征值**（Eigenvalue）。

$$A \mathbf{v} = \lambda \mathbf{v}$$

- $\mathbf{v}$：特征向量——方向不变的"幸运儿"
- $\lambda$：特征值——$\mathbf{v}$ 被拉伸的倍数

**旋转地球仪的类比**：地球绕地轴自转。地面上任何一个点都在旋转（方向不断变化），但地轴上的两个极点在旋转下方向不变——它们就是旋转矩阵的特征向量，对应的特征值为 1（旋转不改变南北极的位置——只是自转方向保持不变）。

### c) 形式化定义

对于方阵 $A \in \mathbb{R}^{n \times n}$，如果存在非零向量 $\mathbf{v} \in \mathbb{R}^n$ 和标量 $\lambda \in \mathbb{C}$ 满足：

$$A \mathbf{v} = \lambda \mathbf{v}$$

则 $\lambda$ 称为 $A$ 的**特征值**，$\mathbf{v}$ 称为对应的**特征向量**。

**如何找到特征值**？将等式改写：

$$A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

为了使 $\mathbf{v}$ 有非零解，必须让 $(A - \lambda I)$ 不可逆——即行列式为零：

$$\det(A - \lambda I) = 0$$

这个方程叫**特征方程**，解出来的 $\lambda$ 就是特征值。对每个 $\lambda$，代入 $(A - \lambda I)\mathbf{v} = \mathbf{0}$ 求解 $\mathbf{v}$。

### d) 手算示例 —— 2×2 矩阵的特征值和特征向量

**例题**：$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$，求所有特征值和特征向量。

**步骤一**：写特征方程

$$A - \lambda I = \begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}$$

$$\det(A - \lambda I) = (2-\lambda)(2-\lambda) - 1 \times 1 = (2-\lambda)^2 - 1$$

$$= \lambda^2 - 4\lambda + 4 - 1 = \lambda^2 - 4\lambda + 3 = 0$$

**步骤二**：解二次方程

$$\lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3) = 0$$

$$\lambda_1 = 1, \quad \lambda_2 = 3$$

**步骤三**：求 $\lambda_1 = 1$ 对应的特征向量

$$(A - 1 \cdot I)\mathbf{v} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$\begin{cases} v_1 + v_2 = 0 \\ v_1 + v_2 = 0 \end{cases} \quad \Rightarrow \quad v_2 = -v_1$$

取 $v_1 = 1$，则 $\mathbf{v}_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$（或任意倍数都行）

**步骤四**：求 $\lambda_2 = 3$ 对应的特征向量

$$(A - 3 \cdot I)\mathbf{v} = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$\begin{cases} -v_1 + v_2 = 0 \\ v_1 - v_2 = 0 \end{cases} \quad \Rightarrow \quad v_1 = v_2$$

取 $v_1 = 1$，则 $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

**步骤五**：验证

$$A\mathbf{v}_1 = \begin{bmatrix}2&1\\1&2\end{bmatrix}\begin{bmatrix}1\\-1\end{bmatrix} = \begin{bmatrix}2-1\\1-2\end{bmatrix} = \begin{bmatrix}1\\-1\end{bmatrix} = 1 \cdot \mathbf{v}_1 \quad \checkmark$$

$$A\mathbf{v}_2 = \begin{bmatrix}2&1\\1&2\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}2+1\\1+2\end{bmatrix} = \begin{bmatrix}3\\3\end{bmatrix} = 3 \cdot \mathbf{v}_2 \quad \checkmark$$

**几何解释**：$\mathbf{v}_1 = [1,-1]^\top$ 方向（沿对角线右下）被保持但缩放为原来的 1 倍（不变）。$\mathbf{v}_2 = [1,1]^\top$ 方向（沿对角线右上）被拉伸 3 倍。

### e) Python 验证

```python
import numpy as np

A = np.array([[2, 1],
              [1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值  |  特征向量")
print("--------+-------------------")
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    print(f"λ={lam:.0f}    |  v=[{v[0]:.4f}, {v[1]:.4f}]")
    # 验证 Av = λv
    Av = A @ v
    lamv = lam * v
    print(f"        验证: Av={Av}, λv={lamv}, 一致? {np.allclose(Av, lamv)}")
    print()
```

对于对称矩阵 $A$，所有特征向量两两**正交**（互相垂直），所有特征值都是**实数**——这是对称矩阵的优良性质。

### f) 常见误区

> **误区**："特征值和特征向量总是实数。"

不。非对称矩阵的特征值可能是复数。例如旋转 90° 的矩阵 $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$，特征值是 $\pm i$（没有实数特征向量——因为每一个非零向量在旋转 90° 后都会改变方向）。但**对称矩阵的特征值一定是实数**（ML 中最重要的矩阵，如协方差矩阵、Hessian 矩阵，都是实对称的）。

> **误区**："每个 $n \times n$ 矩阵都有 $n$ 个不同的特征向量。"

不一定。有些矩阵可能有**重根**（相同特征值出现多次）而没有足够的线性无关特征向量。

### g) ML 中的应用

| 应用 | 数学原理 |
|------|----------|
| **PCA** | 协方差矩阵的特征向量 = 主成分方向；特征值 = 各方向方差 |
| **谱聚类** | 图拉普拉斯矩阵的小特征值对应的特征向量 = 社区划分 |
| **PageRank** | 网页转移矩阵的最大特征向量（λ=1）= 页面重要性评分 |
| **梯度下降收敛性** | Hessian 的最大特征值决定最大允许的学习率 |
| **数据协方差分析** | 特征值的大小 = 数据在各方向上的"散布"程度 |

---

## 17. 奇异值分解（SVD）

### a) 生活例子

你有一张黑白照片（一个矩阵），想把它压缩。直接扔掉像素会让画质惨不忍睹。但如果你先把这张照片"拆"成三步——**旋转** → **不同比例拉伸** → **再旋转**——然后只保留拉伸最大的几个方向，扔掉拉伸很小的方向，再把这三步组合回去……你就得到了压缩后的照片，人眼几乎看不出差别。

这个拆解过程，就是 SVD。

### b) 直观理解

**SVD 的核心直觉**：

> **任何线性变换 = 旋转 × 拉伸 × 旋转**

用数学写出来：
$$A = U \Sigma V^\top$$

- $V^\top$：先把输入空间做了一个**旋转**（把坐标对齐到数据的"重要方向"）
- $\Sigma$：然后在这些方向上做**独立拉伸**（有的方向拉伸很多，有的很少）
- $U$：最后把结果空间再做一个**旋转**（把"重要方向"转回原来的坐标系）

**SVD 的威力**：它适用于**任意形状**的矩阵（$m \neq n$ 也可以！），而特征分解只能对方阵做。

### c) 形式化定义

对于任意矩阵 $A \in \mathbb{R}^{m \times n}$，存在分解：

$$A = U \Sigma V^\top$$

其中：
- $U \in \mathbb{R}^{m \times m}$：**左奇异向量矩阵**。列是 $AA^\top$ 的特征向量（正交）。
- $\Sigma \in \mathbb{R}^{m \times n}$：**奇异值矩阵**。对角线上的奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$，非对角全为零。$\sigma_i$ 是沿第 $i$ 个方向的拉伸倍数。
- $V \in \mathbb{R}^{n \times n}$：**右奇异向量矩阵**。列是 $A^\top A$ 的特征向量（正交）。

**与特征分解的关系**：
- $A^\top A = V \Sigma^2 V^\top$——$V$ 的列是 $A^\top A$ 的特征向量，奇异值的平方是 $A^\top A$ 的特征值
- $AA^\top = U \Sigma^2 U^\top$——$U$ 的列是 $AA^\top$ 的特征向量

**SVD vs PCA**：PCA 就是对数据中心化后的协方差矩阵做特征分解。但你不需要显式计算协方差矩阵——直接对中心化后的数据矩阵 $X_c$ 做 SVD，$V$ 的列就是主成分方向，$\sigma_i^2 / (N-1)$ 就是各主成分的方差。

### d) 手算示例 — SVD 的低秩近似

假设 $A$ 是一个 $3 \times 3$ 矩阵，做 SVD 后得到：

$$A = \begin{bmatrix} \mathbf{u}_1 & \mathbf{u}_2 & \mathbf{u}_3 \end{bmatrix}
\begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \sigma_3 \end{bmatrix}
\begin{bmatrix} \mathbf{v}_1^\top \\ \mathbf{v}_2^\top \\ \mathbf{v}_3^\top \end{bmatrix}$$

其中 $\sigma_1 = 10, \sigma_2 = 3, \sigma_3 = 0.01$。

展开：
$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^\top + \sigma_3 \mathbf{u}_3 \mathbf{v}_3^\top$$

由于 $\sigma_3 \approx 0$，它对 $A$ 的贡献几乎为零。**低秩近似**就是扔掉这个分量：
$$A \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^\top \quad (\text{秩=2 近似})$$

这就是 SVD 用于数据压缩的核心思想——**只保留奇异值最大的几个"层"**。

### e) Python 验证

```python
import numpy as np

np.random.seed(42)
A = np.random.randn(5, 3) @ np.random.randn(3, 5)  # 秩≤3
print(f"A 形状: {A.shape}")

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"奇异值: {S}")
print(f"U 形状: {U.shape}, Vᵀ 形状: {Vt.shape}")

# 重构验证
Sigma = np.diag(S)
A_recon = U @ Sigma @ Vt
print(f"重构正确? {np.allclose(A, A_recon)}")

# 低秩近似: k=2
k = 2
A_approx_k2 = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error_k2 = np.linalg.norm(A - A_approx_k2, 'fro')
print(f"\n秩-2 近似 Frobenius 误差: {error_k2:.4f}")

# k=1
k = 1
A_approx_k1 = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error_k1 = np.linalg.norm(A - A_approx_k1, 'fro')
print(f"秩-1 近似 Frobenius 误差: {error_k1:.4f}")

# 信息保留比例
print(f"\nσ₁²占比: {S[0]**2 / np.sum(S**2):.1%}")
print(f"(σ₁²+σ₂²)占比: {np.sum(S[:2]**2) / np.sum(S**2):.1%}")
```

### f) 常见误区

> **误区**："SVD 和特征分解是一回事。"

SVD 适用于**任意矩阵**（包括非方阵），特征分解只适用于**方阵**。对于方阵且对称正定，二者密切相关。但 ML 中的数据矩阵 $X$ 通常是 $(N, D)$ 且 $N \neq D$——SVD 是唯一选择。

> **误区**："奇异值和特征值一样。"

对于对称方阵 $A$，奇异值 = $|\lambda_i|$（特征值的绝对值）。对于数据矩阵 $X$（非方阵），奇异值的平方 $\sigma_i^2$ = $X^\top X$ 的特征值。

### g) ML 中的应用

| 应用 | 如何使用 SVD |
|------|-------------|
| **PCA 降维** | 对中心化数据做 SVD，取 $V_k$ 做投影 |
| **图像压缩** | 保留前 $k$ 个奇异值，存储量从 $mn$ 降到 $k(m+n+1)$ |
| **推荐系统** | 评分矩阵的低秩近似 → 预测缺失评分（Netflix Prize 经典方法） |
| **潜在语义分析（LSA）** | 词-文档矩阵的 SVD → 发现潜在语义主题 |
| **模型压缩** | 全连接层权重矩阵分解为两个小矩阵 |
| **伪逆** | $A^+ = V \Sigma^+ U^\top$（ML 中正规方程求解的数值稳定替代方案） |

---

## 18. 向量范数与矩阵范数（续）——正则化

我们在第 7 节已经详细讨论了向量范数（L1、L2、L∞）。现在我们把范数概念扩展到矩阵，并统一讲解**正则化**是如何使用范数的。

### a) 矩阵范数

矩阵也有"大小"的概念——矩阵范数。最常见的有：

| 矩阵范数 | 公式 | 直觉 |
|----------|------|------|
| **Frobenius 范数** | $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ | 把所有元素平方求和再开根——"总能量" |
| **L2 范数（谱范数）** | $\|A\|_2 = \sigma_{\max}(A)$ | 最大奇异值——"最大拉伸倍数" |
| **核范数** | $\|A\|_* = \sum_i \sigma_i$ | 所有奇异值之和——鼓励低秩 |

```python
import numpy as np

A = np.array([[1, -2, 3],
              [-4, 5, -6]])

fro = np.linalg.norm(A, 'fro')
l2 = np.linalg.norm(A, 2)        # 谱范数 = 最大奇异值
nuclear = np.sum(np.linalg.svd(A, compute_uv=False))

print(f"Frobenius 范数: {fro:.4f}")
print(f"L2 范数 (谱范数): {l2:.4f}")
print(f"核范数: {nuclear:.4f}")
```

### b) 正则化——用范数控制模型复杂度

正则化是 ML 中防止过拟合的核心技术。数学上，它就是在损失函数后面加一个**范数惩罚项**：

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{原}} + \lambda \cdot \text{惩罚项}$$

| 正则化 | 惩罚项 | 效果 |
|--------|--------|------|
| **L2（Ridge / 权重衰减）** | $\lambda \|\mathbf{w}\|_2^2$ | 把所有权重向零挤压，但不精确归零 |
| **L1（Lasso）** | $\lambda \|\mathbf{w}\|_1$ | 很多权重被精确逼到零 → **稀疏解** |
| **Elastic Net** | $\lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2$ | L1 + L2 的组合 |
| **核范数** | $\lambda \|W\|_*$ | 鼓励权重矩阵低秩（模型压缩） |

**L1 为什么产生稀疏解？** 回顾第 7 节的讨论：L1 "球"（菱形）的顶点在坐标轴上。优化问题的解往往出现在范数球与损失等高线的切点——L1 的切点更容易恰好落在坐标轴上，意味着某些权重分量精确为零。

**L2 为什么不产生稀疏解？** L2 球是光滑的圆形，切点几乎不会恰好落在坐标轴上，所以权重被压缩但很少精确归零。

### c) 在 ML 中的其他范数应用

| 场景 | 使用方法 |
|------|----------|
| **梯度裁剪** | 如果 $\|\nabla L\|_2 > \text{threshold}$，则 $\nabla L \leftarrow \nabla L \cdot \frac{\text{threshold}}{\|\nabla L\|_2}$ |
| **谱归一化** | 约束权重矩阵的谱范数 ≤ 1，稳定 GAN 训练 |
| **条件数** | $\kappa(A) = \sigma_{\max} / \sigma_{\min}$ 衡量数值稳定性 |

---

## 19. 总结：线性代数在 ML 中的全景图

### 核心概念与 ML 映射

| 线性代数概念 | 一句话直觉 | ML 核心应用 |
|-------------|-----------|------------|
| **标量** | 一个数 | 损失值、准确率、学习率 |
| **向量** | 箭头，有长度和方向 | 样本特征、词嵌入、梯度 |
| **矩阵** | 二维表格 | 数据集 $(N,D)$、权重矩阵 |
| **点积** | 两个向量有多"一致" | 神经元输出、注意力分数、相似度 |
| **矩阵乘法** | 行×列组合线性变换 | 前向传播、协方差矩阵、全连接层 |
| **转置** | 行列互换 | 维度对齐、正规方程 |
| **单位矩阵** | "乘 1"，什么都不变 | 岭回归正则化项 |
| **逆矩阵** | "Ctrl+Z"，撤销变换 | 正规方程求解 |
| **行列式** | 面积/体积放大倍数 | 可逆性判断、高斯分布归一化 |
| **秩** | 独立信息维度数 | 冗余检测、多重共线性、模型压缩 |
| **特征值/特征向量** | 变换下方向不变的向量 | PCA、谱聚类、PageRank |
| **SVD** | 旋转→拉伸→旋转 | PCA 降维、图像压缩、推荐系统 |
| **范数** | 向量的"大小" | L1/L2 正则化、梯度裁剪 |

### 神经网络的一层

以一层全连接网络为例，回顾涉及的线性代数：

$$\mathbf{Z}^{[l]} = W^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

- $W^{[l]}$：**矩阵**，$(n_{out}, n_{in})$，线性变换
- $\mathbf{A}^{[l-1]}$：**向量**（或矩阵，批处理时），输入
- $\mathbf{b}^{[l]}$：**向量**，偏置（向量加法）
- $W^{[l]} \mathbf{A}^{[l-1]}$：**矩阵乘法**，每个输出神经元 = 权重向量与输入的点积
- 激活函数 $\sigma$ 在此基础上施加**非线性**（否则多层网络等价于一层）

整个深层网络，就是一连串的矩阵乘法加向量加法，穿插非线性激活函数。**理解线性代数，就是理解神经网络的"骨架"。**

---

## 思考题

以下练习题覆盖本章核心概念，从易到难排列。建议先独立尝试，再对照解答。

### 基础题

**题 1（向量加法）**：已知 $\mathbf{a} = [2, -1, 3]^\top$，$\mathbf{b} = [-1, 4, 2]^\top$，求 $\mathbf{a} + \mathbf{b}$ 和 $3\mathbf{a} - 2\mathbf{b}$。

<details>
<summary>点击查看解答</summary>

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} 2+(-1) \\ -1+4 \\ 3+2 \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \\ 5 \end{bmatrix}$$

$$3\mathbf{a} - 2\mathbf{b} = 3\begin{bmatrix}2\\-1\\3\end{bmatrix} - 2\begin{bmatrix}-1\\4\\2\end{bmatrix} = \begin{bmatrix}6\\-3\\9\end{bmatrix} - \begin{bmatrix}-2\\8\\4\end{bmatrix} = \begin{bmatrix}8\\-11\\5\end{bmatrix}$$

</details>

**题 2（点积与夹角）**：已知 $\mathbf{u} = [3, 0]^\top$，$\mathbf{v} = [1, 2]^\top$，求：
1. $\mathbf{u} \cdot \mathbf{v}$
2. $\mathbf{u}$ 和 $\mathbf{v}$ 之间的夹角 $\theta$

<details>
<summary>点击查看解答</summary>

1. $\mathbf{u} \cdot \mathbf{v} = 3 \times 1 + 0 \times 2 = 3$

2. 先求各向量的长度：
   $$\|\mathbf{u}\| = \sqrt{3^2 + 0^2} = 3$$
   $$\|\mathbf{v}\| = \sqrt{1^2 + 2^2} = \sqrt{5}$$

   由 $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$：
   $$\cos\theta = \frac{3}{3 \cdot \sqrt{5}} = \frac{1}{\sqrt{5}} \approx 0.4472$$
   $$\theta = \arccos(0.4472) \approx 63.4°$$

</details>

**题 3（矩阵乘法）**：计算 $C = A \times B$，其中 $A = \begin{bmatrix}1 & 0 & 2 \\ -1 & 3 & 1\end{bmatrix}$，$B = \begin{bmatrix}3 & 1 \\ 2 & 1 \\ 1 & 0\end{bmatrix}$。要求写出每个元素的完整计算过程。

<details>
<summary>点击查看解答</summary>

$A$ 是 $2 \times 3$，$B$ 是 $3 \times 2$ → $C$ 是 $2 \times 2$。

$$c_{11} = 1 \times 3 + 0 \times 2 + 2 \times 1 = 3 + 0 + 2 = 5$$
$$c_{12} = 1 \times 1 + 0 \times 1 + 2 \times 0 = 1 + 0 + 0 = 1$$
$$c_{21} = (-1) \times 3 + 3 \times 2 + 1 \times 1 = -3 + 6 + 1 = 4$$
$$c_{22} = (-1) \times 1 + 3 \times 1 + 1 \times 0 = -1 + 3 + 0 = 2$$

$$C = \begin{bmatrix} 5 & 1 \\ 4 & 2 \end{bmatrix}$$

</details>

**题 4（行列式）**：计算下列矩阵的行列式，并解释每个行列式的几何含义。

$$A = \begin{bmatrix}3 & 0 \\ 0 & 2\end{bmatrix}, \quad B = \begin{bmatrix}1 & 2 \\ 2 & 4\end{bmatrix}, \quad C = \begin{bmatrix}0 & 1 \\ -1 & 0\end{bmatrix}$$

<details>
<summary>点击查看解答</summary>

$\det(A) = 3 \times 2 - 0 \times 0 = 6$

→ 几何含义：$A$ 把 x 方向拉伸 3 倍、y 方向拉伸 2 倍，单位正方形面积放大 6 倍。

$\det(B) = 1 \times 4 - 2 \times 2 = 4 - 4 = 0$

→ 几何含义：$B$ 的两列 $[1,2]^\top$ 和 $[2,4]^\top$ 在同一条直线上（后者是前者的 2 倍），它们张成的"平行四边形"退化为一条线，面积为零。$B$ 不可逆——它把 2D 空间压扁成了 1D 直线。

$\det(C) = 0 \times 0 - 1 \times (-1) = 0 + 1 = 1$

→ 几何含义：$C$ 是旋转 90° 的变换。旋转不改变面积，所以行列式为 1。

</details>

### 进阶题

**题 5（解线性方程组）**：用矩阵求逆的方法解方程组

$$\begin{cases} 2x + y = 5 \\ x + 3y = 5 \end{cases}$$

要求：写出完整的 $A^{-1}$ 计算过程，再用 $A^{-1}\mathbf{b}$ 求 $\mathbf{x}$。最后用**列视角**解释解得的结果。

<details>
<summary>点击查看解答</summary>

**矩阵形式**：
$$\begin{bmatrix}2 & 1 \\ 1 & 3\end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}5 \\ 5\end{bmatrix}$$

**求逆**：$\det(A) = 2 \times 3 - 1 \times 1 = 6 - 1 = 5$

$$A^{-1} = \frac{1}{5}\begin{bmatrix}3 & -1 \\ -1 & 2\end{bmatrix} = \begin{bmatrix}0.6 & -0.2 \\ -0.2 & 0.4\end{bmatrix}$$

**求解**：
$$\mathbf{x} = A^{-1}\mathbf{b} = \begin{bmatrix}0.6 & -0.2 \\ -0.2 & 0.4\end{bmatrix} \begin{bmatrix}5 \\ 5\end{bmatrix} = \begin{bmatrix}0.6 \times 5 + (-0.2) \times 5 \\ -0.2 \times 5 + 0.4 \times 5\end{bmatrix} = \begin{bmatrix}3 - 1 \\ -1 + 2\end{bmatrix} = \begin{bmatrix}2 \\ 1\end{bmatrix}$$

得到 $x = 2$, $y = 1$。

**验证**：$2 \times 2 + 1 = 5$ ✓，$2 + 3 \times 1 = 5$ ✓

**列视角解释**：结果 $\mathbf{x} = [2, 1]^\top$ 表示——用 $A$ 的第 1 列 $[2,1]^\top$ 的 2 份 + 第 2 列 $[1,3]^\top$ 的 1 份，恰好能组合出 $\mathbf{b} = [5,5]^\top$：

$$2 \cdot \begin{bmatrix}2\\1\end{bmatrix} + 1 \cdot \begin{bmatrix}1\\3\end{bmatrix} = \begin{bmatrix}4+1\\2+3\end{bmatrix} = \begin{bmatrix}5\\5\end{bmatrix}$$

</details>

**题 6（秩）**：数据科学家小明收集了 200 个样本，每个样本有 5 个特征：$f_1, f_2, f_3, f_4, f_5$。他发现了以下关系：
- $f_4 = 2f_1 + f_3$
- $f_5 = f_4 - 3f_2$

问：这个 $200 \times 5$ 数据矩阵的秩最多是多少？为什么这对机器学习很重要？

<details>
<summary>点击查看解答</summary>

特征之间有两个线性关系：
1. $f_4 = 2f_1 + f_3$
2. $f_5 = f_4 - 3f_2 = (2f_1 + f_3) - 3f_2 = 2f_1 - 3f_2 + f_3$

这意味着 $f_4$ 和 $f_5$ 都可以由 $f_1, f_2, f_3$ 线性表示。所以在 5 个特征中，最多只有 **3 个是线性无关的**——秩 ≤ 3。

**对 ML 的影响**：
1. **多重共线性**：如果用这 5 个特征做线性回归，$X^\top X$ 秩亏（=3 < 5），行列式为零，正则方程无唯一解。
2. **解决方案**：删掉 $f_4$ 和 $f_5$，或用 PCA 降维，或用岭回归（$X^\top X + \lambda I$ 保证满秩）。
3. **模型冗余**：多余的 2 个特征不仅没用，反而增加模型复杂度、浪费计算资源、可能引入噪声。

</details>

**题 7（特征值/特征向量）**：求矩阵 $A = \begin{bmatrix}4 & 2 \\ 2 & 1\end{bmatrix}$ 的所有特征值和特征向量。

<details>
<summary>点击查看解答</summary>

**特征方程**：
$$\det(A - \lambda I) = \det\begin{bmatrix}4-\lambda & 2 \\ 2 & 1-\lambda\end{bmatrix} = (4-\lambda)(1-\lambda) - 4 = 0$$

展开：$(4-\lambda)(1-\lambda) - 4 = 4 - 4\lambda - \lambda + \lambda^2 - 4 = \lambda^2 - 5\lambda = \lambda(\lambda - 5) = 0$

$\lambda_1 = 0$, $\lambda_2 = 5$

**$\lambda_1 = 0$ 对应的特征向量**：
$$(A - 0I)\mathbf{v} = \begin{bmatrix}4 & 2 \\ 2 & 1\end{bmatrix} \begin{bmatrix}v_1 \\ v_2\end{bmatrix} = \begin{bmatrix}0 \\ 0\end{bmatrix}$$
$4v_1 + 2v_2 = 0$ → $v_2 = -2v_1$
取 $\mathbf{v}_1 = \begin{bmatrix}1 \\ -2\end{bmatrix}$

**$\lambda_2 = 5$ 对应的特征向量**：
$$(A - 5I)\mathbf{v} = \begin{bmatrix}-1 & 2 \\ 2 & -4\end{bmatrix} \begin{bmatrix}v_1 \\ v_2\end{bmatrix} = \begin{bmatrix}0 \\ 0\end{bmatrix}$$
$-v_1 + 2v_2 = 0$ → $v_1 = 2v_2$
取 $\mathbf{v}_2 = \begin{bmatrix}2 \\ 1\end{bmatrix}$

**验证**：
- $A\mathbf{v}_1 = \begin{bmatrix}4&2\\2&1\end{bmatrix}\begin{bmatrix}1\\-2\end{bmatrix} = \begin{bmatrix}4-4\\2-2\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix} = 0 \cdot \mathbf{v}_1$ ✓
- $A\mathbf{v}_2 = \begin{bmatrix}4&2\\2&1\end{bmatrix}\begin{bmatrix}2\\1\end{bmatrix} = \begin{bmatrix}8+2\\4+1\end{bmatrix} = \begin{bmatrix}10\\5\end{bmatrix} = 5\begin{bmatrix}2\\1\end{bmatrix} = 5 \cdot \mathbf{v}_2$ ✓

**几何含义**：
- $\lambda_1 = 0$：沿 $\mathbf{v}_1$ 方向的向量被"压扁"为零向量——这个方向的信息彻底丢失。这也是 $\det(A) = 0$ 的原因。
- $\lambda_2 = 5$：沿 $\mathbf{v}_2$ 方向的向量被拉伸 5 倍。
- $A$ 不是满秩（秩=1，两列成比例）→ $\det(A) = 0$ → 存在零特征值。

</details>

### 综合题

**题 8（PCA 与特征值/特征向量的关系）**：PCA 使用协方差矩阵的特征向量作为主成分方向。请回答：

1. 为什么我们要找的是**协方差矩阵**的特征向量，而不是数据矩阵 $X$ 本身的特征向量？
2. 为什么特征值最大的方向对应"最重要的方向"？
3. 如果数据的协方差矩阵的特征值为 $\lambda_1=10, \lambda_2=3, \lambda_3=0.01$，降维到 2D 后，你保留了原始数据多少比例的信息？

<details>
<summary>点击查看解答</summary>

**1. 为什么是协方差矩阵而不是 $X$？**

首先，$X$ 通常是 $N \times D$ 的非方阵——它根本没有特征向量（特征分解只对方阵有意义）。

协方差矩阵 $\Sigma = \frac{1}{N-1}X_c^\top X_c$（$X_c$ 是中心化后的数据）是 $D \times D$ 的**对称方阵**，可以做特征分解。协方差矩阵的元素 $\Sigma_{ij}$ 是特征 $i$ 和特征 $j$ 之间的协方差，它描述了数据在各个方向上的"散布"程度——而这正是 PCA 要捕捉的。

**2. 为什么最大特征值 = 最重要方向？**

特征值 $\lambda_i$ = 数据在特征向量 $\mathbf{v}_i$ 方向上的**方差**。

方差大 → 数据在那一维上散布得开 → 含有更多区分样本的信息 → 重要。

方差极小 → 样本在那一维上几乎挤在一起 → 这维信息量低 → 可以丢弃。

**3. 降维到 2D 后保留的信息比例？**

总方差 = $\lambda_1 + \lambda_2 + \lambda_3 = 10 + 3 + 0.01 = 13.01$

取前 2 个主成分，保留的方差 = $\lambda_1 + \lambda_2 = 10 + 3 = 13$

保留比例 = $\frac{13}{13.01} \approx 99.92\%$

丢了 0.08% 的信息，把维度从 3 降到 2。这在实际中是非常好的降维效果。

</details>

**题 9（SVD 的几何意义）**：解释为什么 SVD 可以被理解成"旋转 → 拉伸 → 旋转"三步操作。用一句话描述 $V^\top$、$\Sigma$ 和 $U$ 在每一步中的作用。

<details>
<summary>点击查看解答</summary>

SVD：$A = U \Sigma V^\top$，作用于一个向量 $\mathbf{x}$ 时：
$$\mathbf{y} = A\mathbf{x} = U \Sigma V^\top \mathbf{x}$$

分三步理解（从右向左读）：

1. **$V^\top \mathbf{x}$（第一步旋转）**：将输入向量 $\mathbf{x}$ 从原始坐标系旋转到一个"最优坐标系"——在这个新坐标系中，$A$ 的变换恰好是沿各坐标轴的独立拉伸（没有旋转分量）。$V$ 的列就是这个新坐标系的基。

2. **$\Sigma (V^\top \mathbf{x})$（拉伸）**：在这个新坐标系中，沿每个轴按对应的奇异值 $\sigma_i$ 独立拉伸。$\sigma_1$ 是最强拉伸方向，$\sigma_r$ 是最弱拉伸方向。

3. **$U (\Sigma V^\top \mathbf{x})$（第二步旋转）**：将拉伸后的向量再旋转一次，从"最优坐标系"转回输出空间的标准坐标系。

**一句话总结**：$V^\top$ 找到"数据最重要的方向"，$\Sigma$ 在这些方向上拉伸，$U$ 把结果转正。

</details>

**题 10（综合）**：你有一个包含 1000 个样本、50 个特征的数据集。你发现 $X^\top X$ 的行列式非常接近于零。请回答：

a) 这意味着什么？
b) 为什么这会影响线性回归？
c) 给出两种不同的解决方法并解释原理。

<details>
<summary>点击查看解答</summary>

**a) 这意味着什么？**

$\det(X^\top X) \approx 0$ 意味着 $X^\top X$ 接近奇异——$X$ 的列（特征）之间存在高度的线性相关性（多重共线性）。从几何上说，数据在 50 维空间中实际上占据了一个远低于 50 维的子空间。

**b) 为什么影响线性回归？**

线性回归的正规方程：$\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$

如果 $X^\top X$ 接近奇异，行列式 ≈ 0：
- $X^\top X$ 不可逆或数值上极不稳定求逆（条件数巨大）
- 解出的 $\mathbf{w}$ 对 $\mathbf{y}$ 的微小变化非常敏感 → 模型不稳定
- 权重的方差极大 → 过拟合风险高

**c) 两种处理方法：**

**方法 1：岭回归（Ridge Regression）**

$$\mathbf{w} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$$

加上 $\lambda I$ 等于在 $X^\top X$ 的对角线上"充值"——无论 $X^\top X$ 原来多接近奇异，$X^\top X + \lambda I$ 一定满秩（因为 $I$ 的对角都是 $\lambda > 0$，把所有维度的"信息量"都垫高了）。

这等价于：在所有特征方向上给协方差矩阵加了 $\lambda$ 的方差——确保每个方向都"有一定信息量"。

**方法 2：PCA 降维**

对 $X$ 做 SVD → $V$ 的列是主成分方向。只保留前 $k$ 个特征值较大的主成分（$k < 50$）。

在第 $k+1$ 到第 50 个方向上，方差极小 → 数据几乎没有区分能力 → 丢弃它们，降低维度 → 剩下的 $k$ 个方向之间没有线性相关性（因为 PCA 各主成分互相垂直）→ $X_{\text{new}}^\top X_{\text{new}}$ 满秩。

**其他方法**：特征选择（手动删冗余特征）、Lasso 正则化（L1 会自动把冗余特征的权重压到零）。

</details>

---

> **本文完整代码可以在安装了 `numpy`、`matplotlib` 和 `scikit-learn` 的 Python 环境中直接运行。建议你亲手执行每一段代码，修改参数，观察输出变化——这是培养线性代数直觉最快的方式。**
>
> **下一步**：继续学习[微积分基础](./math-calculus.md)，理解梯度下降的数学原理。
