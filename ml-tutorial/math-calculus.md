# 微积分 — 变化的数学

> **核心问题**：世界在变，我们怎么描述"变化"本身？

---

## 开篇：什么是变化？

你早上看天气预报，发现气温从 15°C 升到了 22°C。你开车去公司，仪表盘上的速度一会儿 40 km/h，一会儿 60 km/h。你关注的股票，开盘 128 元，收盘 135 元。

这些事情的共同点是什么？——**某个量在变**。

但"变了多少"只是第一步。更有意思的问题是：

> **变得有多快？** 是缓慢上升，还是突然暴涨？

如果这个"快慢"本身也在变呢？——比如你先慢踩油门，然后猛踩，车速的变化率也在变。

微积分，就是为回答这类问题而生的数学语言。它研究两件事：

1. **微分（Differentiation）**：某个量在某一瞬间的变化有多快？
2. **积分（Integration）**：把无数个小变化累加起来，总共变了多少？

本章聚焦于**微分**，因为它是机器学习的灵魂——梯度下降、反向传播、自动求导，全都建立在微分之上。

不用担心你没学过高等数学。我们从最基础的地方开始，一步一步来。

---

## 1. 函数与变化率

### 1.1 生活例子：开车去旅行

你开车从北京去天津，全程 120 公里，花了 2 小时。

> 平均速度 = $\frac{120 \text{ km}}{2 \text{ h}} = 60 \text{ km/h}$

这是一种"变化率"——距离随时间的**平均**变化率。

但如果你看仪表盘，就会发现速度一会儿是 40，一会儿是 80，一会儿是 110。仪表盘显示的是**瞬时速度**——某一瞬间的速度。

问题来了：什么叫"某一瞬间的速度"？

"一瞬间"的时间长度是零。如果时间长度是零，那走过的距离也是零。用 $\frac{0}{0}$ 算速度，没有意义。

这就是微积分要解决的第一个难题：**如何定义"瞬间的变化率"？**

### 1.2 直观理解：平均变化率

先看一个简单例子。假设你记录了每小时的里程表读数：

| 时间（小时） | 累计距离（公里） |
|---|---|
| 0 | 0 |
| 1 | 50 |
| 2 | 120 |
| 3 | 200 |

从第 1 小时到第 2 小时，你开了 70 公里。平均速度 = 70 km/h。

从第 2 小时到第 3 小时，你只开了 80 公里。平均速度 = 80 km/h。

用数学语言：

> 在区间 $[a, b]$ 上的平均变化率 = $\frac{f(b) - f(a)}{b - a}$

$f(b) - f(a)$ 是函数值的变化量，$b - a$ 是自变量的变化量。它们的比值就是**平均变化率**。

在图像上，这就是**割线（secant line）的斜率**——连接曲线上两点的直线的倾斜程度。

### 1.3 形式化定义

对于函数 $y = f(x)$，在区间 $[x_0, x_0 + h]$ 上的平均变化率为：

$$
\frac{\Delta y}{\Delta x} = \frac{f(x_0 + h) - f(x_0)}{h}
$$

其中 $\Delta$（希腊字母 delta）表示"变化量"。$\Delta y$ 是 $y$ 的变化量，$\Delta x$ 是 $x$ 的变化量。

### 1.4 手算示例：$f(x) = x^2$

算算 $f(x) = x^2$ 在 $x = 2$ 附近、区间分别为 $[2, 3]$、$[2, 2.5]$、$[2, 2.1]$、$[2, 2.01]$ 的平均变化率。

| $h$ | $f(2+h)$ | $f(2)$ | $\frac{f(2+h) - f(2)}{h}$ |
|---|---|---|---|
| 1.0 | 9 | 4 | $\frac{9-4}{1} = 5$ |
| 0.5 | 6.25 | 4 | $\frac{6.25-4}{0.5} = 4.5$ |
| 0.1 | 4.41 | 4 | $\frac{4.41-4}{0.1} = 4.1$ |
| 0.01 | 4.0401 | 4 | $\frac{4.0401-4}{0.01} = 4.01$ |

你有没有发现什么？随着 $h$ 越来越小，平均变化率越来越接近 **4**。

这就是我们接下来要讲的核心概念——极限与导数。

### 1.5 Python 验证

```python
import numpy as np

def f(x):
    return x ** 2

def average_rate(f, x0, h):
    """计算 f 在 [x0, x0+h] 上的平均变化率"""
    return (f(x0 + h) - f(x0)) / h

x0 = 2.0
print(f"f(x) = x² 在 x = {x0} 附近的平均变化率：")
print("-" * 55)
print(f"{'h':<15}{'f(x0+h)':<15}{'平均变化率':<15}")
print("-" * 55)
for h in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]:
    rate = average_rate(f, x0, h)
    print(f"{h:<15.4f}{f(x0 + h):<15.6f}{rate:<15.6f}")
```

### 1.6 常见误区

- **误区**：平均变化率就是"中间时刻的速度"。
- **澄清**：不一定。你可能有快有慢，平均只是把总变化量平摊到时间上。
- **误区**：变化率只能是速度。
- **澄清**：任何两个相关联的量都有变化率——温度随时间的变化率、利润随广告投入的变化率、损失函数随参数的变化率。

### 1.7 机器学习连接

训练模型时，损失函数 $L(w)$ 随参数 $w$ 变化。我们想知道："$w$ 改一点点，$L$ 变多少？"——这正是求变化率。如果变化率是正的，说明增大 $w$ 会让损失变大（不好）；如果是负的，说明增大 $w$ 会让损失变小（好事）。这个方向信息就是梯度下降的决策依据。

---

## 2. 极限的直觉

### 2.1 生活例子：芝诺悖论

古希腊哲学家芝诺讲过一个著名悖论：你要从 A 点走到 B 点。首先你必须走完一半的距离，然后走完剩下距离的一半，然后再一半……看起来你永远到不了 B 点。

但现实中你当然能走到。这个悖论的"解决"就需要极限的概念：**无穷多个越来越小的步，加起来可以是有限的距离。**

用数字来说：$\frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{16} + \dots = 1$

每一项都越来越接近零，但加起来却等于 1。这就是"逼近但不等于"的直觉。

### 2.2 直观理解：无限接近但可以不等于

想象你站在一堵墙前面。你每次走剩下距离的一半。你离墙越来越近——1 米、0.5 米、0.25 米……——你**无限接近**墙但（假设你永远走不到）**永远不会真的撞上**。

极限说的就是：你**趋近于**什么值，而不一定**等于**什么值。

关键区别：极限关心的是"趋近行为"，不关心"在那个点上的值"。

比如这个函数：

$$
g(x) = \frac{x^2 - 1}{x - 1}
$$

$x = 1$ 时，分母为零，$g(1)$ 没有定义。但当 $x$ 趋近于 1 时，$g(x)$ 趋近于什么？

我们来算：

| $x$ | 1.1 | 1.01 | 1.001 | 1.0001 | → 1 |
|---|---|---|---|---|---|
| $g(x)$ | 2.1 | 2.01 | 2.001 | 2.0001 | → 2 |

$g(x)$ 趋近于 2，尽管 $g(1)$ 根本没有定义！

代数上：$x^2 - 1 = (x-1)(x+1)$，所以当 $x \neq 1$ 时，$g(x) = \frac{(x-1)(x+1)}{x-1} = x+1$。当 $x \to 1$，$x+1 \to 2$。

### 2.3 形式化定义

我们写作：

$$
\lim_{x \to a} f(x) = L
$$

读作："当 $x$ 趋近于 $a$ 时，$f(x)$ 的极限是 $L$。"

意思是：我们可以让 $f(x)$ 任意接近 $L$，只要让 $x$ 足够接近 $a$（但不等于 $a$）。

**单侧极限**：
- 从右边趋近：$\lim_{x \to a^+} f(x)$（$x > a$）
- 从左边趋近：$\lim_{x \to a^-} f(x)$（$x < a$）
- 两侧极限相等时，极限存在

### 2.4 手算示例

计算 $\lim_{x \to 2} (3x + 1)$。

$x$ 趋近于 2 时，$3x$ 趋近于 6，$3x+1$ 趋近于 7。所以答案是 7。——这个例子虽然简单，但它确立了一个重要原则：**如果函数在 $a$ 点连续，极限就等于函数值**。

再算一个不平凡的：

$$
\lim_{h \to 0} \frac{(2+h)^2 - 4}{h}
$$

展开分子：$(4 + 4h + h^2) - 4 = 4h + h^2 = h(4 + h)$

所以：

$$
\frac{h(4+h)}{h} = 4 + h \quad (\text{当 } h \neq 0)
$$

当 $h \to 0$，$4 + h \to 4$。所以极限是 **4**。

这恰好是上节末尾 $f(x)=x^2$ 在 $x=2$ 处的平均变化率当 $h \to 0$ 时的极限。你已经不知不觉算了一个导数！

### 2.5 Python 验证

```python
import numpy as np

def f(x):
    return (x**2 - 1) / (x - 1)

print("验证 lim(x→1) (x²-1)/(x-1) = 2")
print("-" * 45)
print(f"{'x':<18}{'f(x)':<18}")
print("-" * 45)
for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    x = 1 + delta
    print(f"{x:<18.10f}{f(x):<18.10f}")

# 从左边趋近
print(f"\n从左边趋近：")
for delta in [1e-1, 1e-2, 1e-3]:
    x = 1 - delta
    print(f"x = {x:.6f}, f(x) = {f(x):.6f}")

print(f"\n极限值 ≈ {f(1 + 1e-10):.10f}")
```

### 2.6 常见误区

- **误区**：极限就是"函数在那点的值"。
- **澄清**：极限关心的是"趋近行为"，函数在那点可以没有定义，也可以有不同的值。$f(x)=x$ 在 $x=3$ 的极限是 3，恰好等于 $f(3)$，但这只是巧合（因为 $f(x)=x$ 是连续函数）。
- **误区**：$\frac{0}{0}$ 型极限说明函数有问题。
- **澄清**：$\frac{0}{0}$ 只是说分子和分母都趋近于 0，极限可能是一个有限值（如上面算出的 2），也可能是无穷，也可能不存在。需要具体分析。

### 2.7 机器学习连接

很多机器学习理论问题最终归结为极限问题：学习率趋近于 0 时梯度下降的行为、无穷样本下模型的收敛性、数值优化中的步长选择。理解极限是理解这些理论的基础。

---

## 3. 导数的定义

### 3.1 生活例子：汽车仪表盘

回到开车的问题。仪表盘怎么知道"这一瞬间"的速度？

它不能测"零时间内的距离"，但可以测极小时间内的距离。比如每 0.01 秒记录一次轮胎转了多少圈，然后算：

> 瞬时速度 ≈ $\frac{\text{0.01秒内走过的距离}}{\text{0.01秒}}$

时间间隔越短，近似越准。极限情况下，这就是导数。

### 3.2 直观理解：割线变切线

在函数图像上取两点 $A(x_0, f(x_0))$ 和 $B(x_0+h, f(x_0+h))$。

- 连接 A 和 B 的直线叫**割线**，斜率 = $\frac{f(x_0+h)-f(x_0)}{h}$
- 当 $h \to 0$，B 沿曲线滑向 A，割线旋转，最后变成**切线**
- 切线的斜率就是**导数**

这就是导数的几何意义：**函数图像在某点的切线斜率**。

也是导数的物理意义：**某个量在某一瞬间的变化率**（瞬时速度、瞬时电流、瞬时增长率）。

### 3.3 形式化定义

函数 $f(x)$ 在点 $x_0$ 处的导数定义为：

$$
f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}
$$

如果这个极限存在，我们说 $f$ 在 $x_0$ 处**可导**（differentiable）。

**常见记号**：

| 记号 | 读法 | 提出者 |
|---|---|---|
| $f'(x)$ | f prime of x | Lagrange |
| $\frac{dy}{dx}$ | dy dx | Leibniz |
| $\frac{df}{dx}$ | df dx | Leibniz |
| $\dot{y}$ | y dot | Newton（物理常用）|

Leibniz 记号 $\frac{dy}{dx}$ 特别有用——它提醒我们导数来自两个微小变化量的比值：$dy$ 是 $y$ 的无穷小变化，$dx$ 是 $x$ 的无穷小变化。链式法则在这个记号下几乎是"显而易见"的。

### 3.4 手算示例：从第一性原理推导 $f'(x)$ for $f(x) = x^2$

用极限定义一步步算：

$$
\begin{aligned}
f'(x) &= \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \\[4pt]
&= \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} \\[4pt]
&= \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} \quad \text{← 展开平方} \\[4pt]
&= \lim_{h \to 0} \frac{2xh + h^2}{h} \quad \text{← 消去 } x^2 \\[4pt]
&= \lim_{h \to 0} \frac{h(2x + h)}{h} \quad \text{← 提取 } h \\[4pt]
&= \lim_{h \to 0} (2x + h) \quad \text{← 约去 } h\ (\text{因为 } h \neq 0) \\[4pt]
&= 2x
\end{aligned}
$$

所以 **$f(x) = x^2$ 的导数是 $f'(x) = 2x$**。

在 $x = 3$ 处，$f'(3) = 6$。意思是：在 $x=3$ 这点，$x^2$ 以"每单位 $x$ 变化带来 6 单位 $y$ 变化"的速率增长。

验证：取 $x=3$，$f(3)=9$。若 $x$ 增大 0.01 到 3.01，$f(3.01)=9.0601$。变化 = 0.0601。变化率 ≈ 0.0601 / 0.01 = 6.01 ≈ 6。

### 3.5 Python 验证

```python
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    """用中心差分法近似计算导数 f'(x)"""
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

# 测试 x = 3 (理论值 f'(3) = 6)
x0 = 3.0
approx = numerical_derivative(f, x0)
exact = 2 * x0

print(f"f(x) = x²")
print(f"在 x = {x0} 处的导数：")
print(f"  数值近似: {approx:.10f}")
print(f"  理论精确: {exact}")
print(f"  误差:     {abs(approx - exact):.2e}")

# 展示误差随 h 缩小的二阶收敛
print(f"\n中心差分法的误差随 h 变化：")
print(f"{'h':<12}{'数值导数':<18}{'误差':<14}")
print("-" * 44)
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    approx = (f(x0 + h) - f(x0 - h)) / (2 * h)
    error = abs(approx - exact)
    print(f"{h:<12.0e}{approx:<18.12f}{error:<14.2e}")
```

### 3.6 常见误区

- **误区**：可导就是连续。
- **澄清**：可导必然连续，但连续不一定可导。比如 $f(x) = |x|$ 在 $x=0$ 处连续，但有一个"尖角"，左导数 = -1，右导数 = +1，两者不等，所以在 $x=0$ 处不可导。
- **误区**：导数大 = 函数值大。
- **澄清**：导数大表示**变化快**，不表示值大。$f(x)=1000$ 的导数是 0（不变），而 $f(x)=x$ 在 $x=0$ 处导数 = 1（在增长），但函数值只有 0。

### 3.7 机器学习连接

当你训练一个模型时，你每次用一个 batch 的数据计算损失 $L$ 对参数 $w$ 的导数 $\frac{dL}{dw}$。这个导数的正负号告诉你"增大 $w$ 会让损失变大还是变小"，绝对值大小告诉你"$w$ 对损失的影响有多敏感"。这正是你调整参数的依据。

---

## 4. 基本求导法则的推导

很多人学微积分时直接背一张求导公式表。但如果你不知道这些公式**为什么**是对的，你就无法在面对新的函数形式时自己推导。

下面我们**从极限定义出发**，逐一推导每一条核心法则。

### 4.1 常数法则：$\frac{d}{dx}(c) = 0$

$$
\begin{aligned}
\frac{d}{dx}(c) &= \lim_{h \to 0} \frac{c - c}{h}
= \lim_{h \to 0} \frac{0}{h}
= 0
\end{aligned}
$$

常数不随 $x$ 变化而变化，所以变化率是零。直观：停在路边的车，速度为 0。

### 4.2 幂法则：$\frac{d}{dx}(x^n) = nx^{n-1}$

推导 $n=2$ 我们已经做了（得 $2x$）。推导 $n=3$：

$$
\begin{aligned}
\frac{d}{dx}(x^3) &= \lim_{h \to 0} \frac{(x+h)^3 - x^3}{h} \\[4pt]
&= \lim_{h \to 0} \frac{x^3 + 3x^2h + 3xh^2 + h^3 - x^3}{h} \quad \text{← 二项式展开} \\[4pt]
&= \lim_{h \to 0} \frac{h(3x^2 + 3xh + h^2)}{h} \\[4pt]
&= \lim_{h \to 0} (3x^2 + 3xh + h^2) \\[4pt]
&= 3x^2
\end{aligned}
$$

一般地，对于任意正整数 $n$，二项式定理给出：

$$
(x+h)^n = x^n + nx^{n-1}h + \frac{n(n-1)}{2}x^{n-2}h^2 + \dots + h^n
$$

代入极限定义，所有含 $h^2$ 及更高次的项在 $h \to 0$ 时消失，只剩下 $nx^{n-1}$。

**生活类比**：$x^3$ 是立方体的体积。边长增加一点点，体积的变化主要来自三个面的扩展（$3x^2$），角落的二次项和三次项在微小的边长变化下可以忽略。

### 4.3 常数倍法则：$\frac{d}{dx}[cf(x)] = c \cdot f'(x)$

$$
\begin{aligned}
\frac{d}{dx}[cf(x)] &= \lim_{h \to 0} \frac{cf(x+h) - cf(x)}{h} \\
&= c \cdot \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \\
&= c \cdot f'(x)
\end{aligned}
$$

常数可以"提出去"。直观：如果把函数的纵轴放大 $c$ 倍，斜率也放大 $c$ 倍。

### 4.4 和差法则：$\frac{d}{dx}[f(x) \pm g(x)] = f'(x) \pm g'(x)$

$$
\begin{aligned}
\frac{d}{dx}[f(x) + g(x)] &= \lim_{h \to 0} \frac{[f(x+h)+g(x+h)] - [f(x)+g(x)]}{h} \\
&= \lim_{h \to 0} \frac{f(x+h)-f(x)}{h} + \lim_{h \to 0} \frac{g(x+h)-g(x)}{h} \\
&= f'(x) + g'(x)
\end{aligned}
$$

和的导数 = 导数的和。因为极限运算可以分配到加法上。

直观：如果你同时在往东走（$f$）和往北走（$g$），你的总变化率就是两个变化率的叠加。

### 4.5 乘积法则：$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$

这是第一个不"显然"的法则。我们用一个经典类比来推导。

**矩形面积类比**：把 $A(x) = f(x) \cdot g(x)$ 想象成矩形的面积——长 × 宽。

当 $x$ 增加 $h$：
- 长从 $f$ 变为 $f + \Delta f$（其中 $\Delta f \approx f'(x)h$）
- 宽从 $g$ 变为 $g + \Delta g$（其中 $\Delta g \approx g'(x)h$）

面积的变化 = 新面积 − 原面积：

$$
\Delta A = (f + \Delta f)(g + \Delta g) - fg = f\Delta g + g\Delta f + \Delta f \Delta g
$$

第三项 $\Delta f \Delta g$ 是两个微小量的乘积，是高阶无穷小（当 $h \to 0$ 时它比 $h$ 更快地趋近于零）。所以：

$$
\frac{\Delta A}{h} \approx f\frac{\Delta g}{h} + g\frac{\Delta f}{h}
$$

取极限：

$$
\frac{dA}{dx} = f(x)g'(x) + g(x)f'(x)
$$

**严格推导**（加一项减一项的技巧）：

$$
\begin{aligned}
\frac{d}{dx}[f(x)g(x)] &= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h} \\[4pt]
&= \lim_{h \to 0} \frac{f(x+h)g(x+h) \textcolor{gray}{- f(x)g(x+h) + f(x)g(x+h)} - f(x)g(x)}{h} \\[4pt]
&= \lim_{h \to 0} \left[ \frac{f(x+h) - f(x)}{h} \cdot g(x+h) + f(x) \cdot \frac{g(x+h) - g(x)}{h} \right] \\[4pt]
&= f'(x) \cdot g(x) + f(x) \cdot g'(x)
\end{aligned}
$$

**例**：$f(x) = x^2 \cdot e^x$。不用乘积法则你会疯掉。使用它：

$$
\frac{d}{dx}(x^2 e^x) = 2x \cdot e^x + x^2 \cdot e^x = e^x(2x + x^2)
$$

### 4.6 链式法则：$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$

这是机器学习中**最重要**的求导法则——没有它就没有反向传播。

**生活例子：齿轮比**

你骑变速自行车。脚踏板转一圈 → 链条带动后齿轮转三圈 → 后齿轮带动车轮转。

- 脚踏板到后齿轮的"转速比"：1:3
- 后齿轮到车轮的"转速比"：3:5

那脚踏板到车轮的"转速比" = $(1:3) \times (3:5) = 1:5$

这就是链式法则：**变化的变化，等于变化的乘积**。

**形式化推导**：

设 $y = f(u)$ 且 $u = g(x)$。那么 $y = f(g(x))$。

$$
\begin{aligned}
\frac{dy}{dx} &= \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{h} \\[4pt]
&= \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{g(x+h) - g(x)} \cdot \frac{g(x+h) - g(x)}{h} \\[4pt]
&= f'(g(x)) \cdot g'(x)
\end{aligned}
$$

用 Leibniz 记号最清晰：$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$。

多层嵌套：$\frac{dy}{dx} = \frac{dy}{du_1} \cdot \frac{du_1}{du_2} \cdot \frac{du_2}{du_3} \cdot \frac{du_3}{dx}$

### 4.7 手算示例：用链式法则

求 $\frac{d}{dx} \sin(x^2 + 1)$。

拆解：外层 $f(u) = \sin(u)$，内层 $u = g(x) = x^2 + 1$

- 外层导数：$f'(u) = \cos(u)$
- 内层导数：$g'(x) = 2x$

链式法则：$\frac{d}{dx} \sin(x^2 + 1) = \cos(x^2 + 1) \cdot 2x$

再求 $\frac{d}{dx} e^{\sin(x)}$：

- 外层：$\frac{d}{du} e^u = e^u$
- 内层：$\frac{d}{dx} \sin(x) = \cos(x)$
- 结果：$e^{\sin(x)} \cdot \cos(x)$

### 4.8 Python 验证

```python
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# 验证乘积法则: d/dx [x²·sin(x)] = 2x·sin(x) + x²·cos(x)
def f1(x):
     return x**2 * np.sin(x)

def f1_deriv_exact(x):
    return 2*x * np.sin(x) + x**2 * np.cos(x)

x0 = 1.5
num = numerical_derivative(f1, x0)
exact = f1_deriv_exact(x0)
print(f"乘积法则验证：d/dx[x²·sin(x)] at x={x0}")
print(f"  数值: {num:.10f}")
print(f"  理论: {exact:.10f}")
print(f"  误差: {abs(num - exact):.2e}")

# 验证链式法则: d/dx sin(x²+1) = cos(x²+1)·2x
def f2(x):
    return np.sin(x**2 + 1)

def f2_deriv_exact(x):
    return np.cos(x**2 + 1) * 2 * x

num2 = numerical_derivative(f2, x0)
exact2 = f2_deriv_exact(x0)
print(f"\n链式法则验证：d/dx[sin(x²+1)] at x={x0}")
print(f"  数值: {num2:.10f}")
print(f"  理论: {exact2:.10f}")
print(f"  误差: {abs(num2 - exact2):.2e}")
```

### 4.9 机器学习连接

链式法则在机器学习中有一个专门的名字：**反向传播**。神经网络 $L = \text{Loss}(\sigma_3(W_3 \cdot \sigma_2(W_2 \cdot \sigma_1(W_1 x))))$ 中，要求 $\frac{\partial L}{\partial W_1}$，你从输出层开始，一层一层往输入方向求导，每一步都用链式法则把梯度"传递"回去。

---

## 5. 常见函数的导数

有了基本法则，我们来看几个在机器学习中反复出现的函数，从定义出发推导它们的导数。

### 5.1 指数函数的导数：$\frac{d}{dx} e^x = e^x$

这是数学中最优雅的事实之一：$e^x$ 的导数就是它自己。

推导需要用到自然常数 $e$ 的定义：

$$
e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n \approx 2.71828\dots
$$

等价地：$\lim_{h \to 0} \frac{e^h - 1}{h} = 1$

用这个极限：

$$
\begin{aligned}
\frac{d}{dx} e^x &= \lim_{h \to 0} \frac{e^{x+h} - e^x}{h} \\
&= \lim_{h \to 0} \frac{e^x \cdot e^h - e^x}{h} \\
&= e^x \cdot \lim_{h \to 0} \frac{e^h - 1}{h} \\
&= e^x \cdot 1 = e^x
\end{aligned}
$$

**生活类比**：$e^x$ 就像一个永远在加速的火箭——当前速度等于当前位置。你在哪，速度就是多少。这也是为什么 $e^x$ 增长如此"爆炸"。

### 5.2 自然对数的导数：$\frac{d}{dx} \ln x = \frac{1}{x}$

对数是指数的逆运算：$y = \ln x$ 意味着 $e^y = x$。

用隐函数求导的思想（两边对 $x$ 求导）：

$$
\frac{d}{dx}(e^y) = \frac{d}{dx}(x)
$$

左边用链式法则：$e^y \cdot \frac{dy}{dx} = 1$

所以：$\frac{dy}{dx} = \frac{1}{e^y} = \frac{1}{x}$

**直观**：$\ln x$ 的增长率是 $\frac{1}{x}$——$x$ 越大，增长越慢。$x=1$ 时斜率是 1；$x=100$ 时斜率只有 0.01。这就是对数函数的"减速"效应——也是为什么对数变换常用于缩小大数值的尺度差异。

### 5.3 $\sin x$ 的导数：$\frac{d}{dx} \sin x = \cos x$

推导需要两个关键极限：

$$
\lim_{h \to 0} \frac{\sin h}{h} = 1, \qquad \lim_{h \to 0} \frac{\cos h - 1}{h} = 0
$$

（第一个极限的直观：当角度极小时，$\sin h \approx h$，想象一个极其扁的直角三角形。）

用三角恒等式 $\sin(A+B) = \sin A \cos B + \cos A \sin B$：

$$
\begin{aligned}
\frac{d}{dx} \sin x &= \lim_{h \to 0} \frac{\sin(x+h) - \sin x}{h} \\[4pt]
&= \lim_{h \to 0} \frac{\sin x \cos h + \cos x \sin h - \sin x}{h} \\[4pt]
&= \lim_{h \to 0} \frac{\sin x (\cos h - 1) + \cos x \sin h}{h} \\[4pt]
&= \sin x \cdot \lim_{h \to 0} \frac{\cos h - 1}{h} + \cos x \cdot \lim_{h \to 0} \frac{\sin h}{h} \\[4pt]
&= \sin x \cdot 0 + \cos x \cdot 1 \\
&= \cos x
\end{aligned}
$$

类似地：$\frac{d}{dx} \cos x = -\sin x$

注意这个循环：$\sin \to \cos \to -\sin \to -\cos \to \sin \to \dots$，求导四次回到原点。

### 5.4 手算示例

现在你会求任意多项式和常见函数的组合的导数。试试这些：

1. $\frac{d}{dx} (3x^4 - 2x^2 + 5x - 7) = 12x^3 - 4x + 5$
2. $\frac{d}{dx} (e^x \sin x) = e^x \sin x + e^x \cos x = e^x (\sin x + \cos x)$
3. $\frac{d}{dx} \ln(x^2 + 1) = \frac{1}{x^2+1} \cdot 2x = \frac{2x}{x^2+1}$

### 5.5 Python 验证

```python
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试点
x0 = 1.5

tests = [
    ("eˣ",          lambda x: np.exp(x),     lambda x: np.exp(x)),
    ("ln(x)",       np.log,                  lambda x: 1 / x),
    ("sin(x)",      np.sin,                  np.cos),
    ("cos(x)",      np.cos,                  lambda x: -np.sin(x)),
    ("eˣ·sin(x)",   lambda x: np.exp(x)*np.sin(x),
     lambda x: np.exp(x)*(np.sin(x)+np.cos(x))),
]

print("常见函数导数验证：")
print("-" * 60)
for name, f, f_exact in tests:
    num = numerical_derivative(f, x0)
    exact = f_exact(x0)
    err = abs(num - exact)
    print(f"d/dx[{name}] at x={x0}: 数值={num:10.6f}  理论={exact:10.6f}  误差={err:.2e}")
```

### 5.6 机器学习连接

- $e^x$ 出现在 **softmax** 函数中：$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$，将任意实数向量转换为概率分布
- $\ln x$ 出现在**交叉熵损失**中：$L = -[y\ln\hat{y} + (1-y)\ln(1-\hat{y})]$，对数让大误差承受更大的惩罚
- $\sin x$ 和 $\cos x$ 出现在**位置编码**（Transformer）和**傅里叶特征**中

---

## 6. 偏导数

### 6.1 生活例子：山的曲面

想象一座山的海拔 $z = f(x, y)$。你站在山腰某点：
- 往正东走（只改变 $x$，$y$ 不变）：坡度是 $\frac{\partial f}{\partial x}$
- 往正北走（只改变 $y$，$x$ 不变）：坡度是 $\frac{\partial f}{\partial y}$

**偏导数就是"固定其他所有变量，只看这一个变量的变化率"。**

### 6.2 直观理解

对于单变量函数 $f(x)$，导数告诉你整条曲线在某点的斜率。

对于多变量函数 $f(x, y, z, \dots)$，你在高维空间里。偏导数就像是：
1. 沿 $x$ 轴方向把曲面"切一刀"，看截面曲线的斜率 → $\frac{\partial f}{\partial x}$
2. 沿 $y$ 轴方向"切一刀"，看截面曲线的斜率 → $\frac{\partial f}{\partial y}$

求偏导数时，把其他变量当作常数，用之前学过的所有单变量求导法则。

**记号**：用 $\partial$（圆体 d，读作"partial"）而不是 $d$，提醒我们这是一个多变量函数，"其他变量被冻结了"。

### 6.3 形式化定义

$$
\frac{\partial f}{\partial x}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + h, y_0) - f(x_0, y_0)}{h}
$$

$$
\frac{\partial f}{\partial y}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0, y_0 + h) - f(x_0, y_0)}{h}
$$

### 6.4 手算示例：$f(x, y) = x^2 y^3$

**对 $x$ 求偏导**（$y$ 视为常数）：

把 $y^3$ 当成常数系数，对 $x^2$ 求导得 $2x$：

$$
\frac{\partial f}{\partial x} = 2x \cdot y^3
$$

**对 $y$ 求偏导**（$x$ 视为常数）：

把 $x^2$ 当成常数系数，对 $y^3$ 求导得 $3y^2$：

$$
\frac{\partial f}{\partial y} = x^2 \cdot 3y^2 = 3x^2 y^2
$$

在点 $(x, y) = (2, 1)$ 处：
- $\frac{\partial f}{\partial x}(2, 1) = 2 \cdot 2 \cdot 1^3 = 4$
- $\frac{\partial f}{\partial y}(2, 1) = 3 \cdot 2^2 \cdot 1^2 = 12$

含义：在 $(2, 1)$ 处，沿 $x$ 方向每走一单位，$f$ 增加约 4；沿 $y$ 方向每走一单位，$f$ 增加约 12。$y$ 方向更陡。

### 6.5 Python 验证

```python
import numpy as np

def f(X):
    """f(x, y) = x²·y³"""
    x, y = X[0], X[1]
    return x**2 * y**3

def partial_derivative(f, X, var_idx, h=1e-5):
    """用中心差分近似偏导数"""
    X_plus = X.copy()
    X_minus = X.copy()
    X_plus[var_idx] += h
    X_minus[var_idx] -= h
    return (f(X_plus) - f(X_minus)) / (2 * h)

point = np.array([2.0, 1.0])
x, y = point

df_dx_num = partial_derivative(f, point, 0)
df_dy_num = partial_derivative(f, point, 1)

# 理论值
df_dx_exact = 2 * x * y**3
df_dy_exact = 3 * x**2 * y**2

print(f"f(x,y) = x²·y³ 在点 (2, 1) 处的偏导数：")
print(f"  ∂f/∂x: 数值={df_dx_num:.6f}  理论={df_dx_exact:.6f}")
print(f"  ∂f/∂y: 数值={df_dy_num:.6f}  理论={df_dy_exact:.6f}")
```

### 6.6 常见误区

- **误区**：偏导数 $\frac{\partial f}{\partial x}$ 就是"沿着 x 轴走的导数"。
- **澄清**：对。但要注意，你只能沿着坐标轴方向走。如果你斜着走，那就要用方向导数（见第 8 节）。
- **误区**：偏导数之间互相独立。
- **澄清**：不对。$\frac{\partial f}{\partial x}$ 本身也是 $x$ 和 $y$ 的函数，你可以在某点同时知道所有偏导数，它们共同描述了函数在该点的局部行为。

### 6.7 机器学习连接

损失函数以所有模型参数为自变量：$L(w_1, w_2, \dots, w_n, b)$。要更新第 $i$ 个参数 $w_i$，你需要计算 $\frac{\partial L}{\partial w_i}$——这正是偏导数。每个参数的更新只依赖"它自己的"偏导数，但所有这些偏导数可以一起计算（这正是反向传播做的事情）。

---

## 7. 梯度

### 7.1 生活例子：山顶还是山谷？

回到山的话题。偏导数只告诉你"正东"和"正北"两个方向的坡度。但你可能想往**某个斜方向**走，或者想知道**哪个方向最陡**。

把所有偏导数打包在一起，你就得到了**梯度（gradient）**。

### 7.2 直观理解

梯度是一个向量，每个分量是函数对该变量的偏导数：

$$
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
$$

**梯度的几何意义**（非常重要）：

> 梯度指向函数值**上升最快**的方向，其长度等于该方向上的变化率。

想象你站在半山腰。你拿出指南针和坡度计，测了一圈。梯度箭头指向**最陡的上坡方向**。如果你要下山（机器学习就是要最小化损失），你就往**梯度的反方向**走。

### 7.3 形式化定义

对于 $n$ 元函数 $f(x_1, x_2, \dots, x_n)$：

$$
\nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right]^T
$$

$\nabla$ 读作"nabla"或"del"。梯度是一个**列向量**。

### 7.4 手算示例：$f(x, y) = x^2 + y^2$

Compute the gradient at an arbitrary point $(x, y)$:

$$
\frac{\partial f}{\partial x} = 2x, \qquad \frac{\partial f}{\partial y} = 2y
$$

$$
\nabla f(x, y) = (2x, 2y)
$$

在点 $(1, 2)$ 处：

$$
\nabla f(1, 2) = (2, 4)
$$

这个向量指向 $(2, 4)$ 方向，正好背离原点。这很合理：$f(x, y) = x^2 + y^2$ 是一个碗形，最小值在 $(0, 0)$。从 $(1, 2)$ 出发，指向原点的是 $(-1, -2)$ 方向，而梯度 $(2, 4)$ 正好是它的反方向——梯度指向上升最快（往山上走），背离最小值；负梯度指向下降最快（往山下走）。

### 7.5 Python 验证与可视化

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def f(X):
    x, y = X[0], X[1]
    return x**2 + y**2

def gradient_f(X):
    x, y = X[0], X[1]
    return np.array([2*x, 2*y])

# 在点 (1, 2) 处
point = np.array([1.0, 2.0])
grad = gradient_f(point)
neg_grad = -grad

print(f"f(x,y) = x²+y² 在点 (1, 2)：")
print(f"  梯度 ∇f      = ({grad[0]}, {grad[1]})  ← 上升最快方向")
print(f"  负梯度 -∇f   = ({neg_grad[0]}, {neg_grad[1]})  ← 下降最快方向")
print(f"  指向原点方向  = ({-point[0]}, {-point[1]})")
print(f"  梯度指向背离原点 ✓" if np.all(np.sign(grad) == np.sign(point)) else "  ✗")

# 可视化
fig, ax = plt.subplots(figsize=(8, 7))
x = np.linspace(-4, 4, 150)
y = np.linspace(-4, 4, 150)
X, Y = np.meshgrid(x, y)
Z = f(np.array([X, Y]))

contour = ax.contour(X, Y, Z, levels=25, cmap='viridis', alpha=0.5)
ax.clabel(contour, fontsize=7)

# 在网格上画梯度向量
for x0 in np.arange(-3, 4, 1.5):
    for y0 in np.arange(-3, 4, 1.5):
        g = gradient_f(np.array([x0, y0]))
        g_norm = np.linalg.norm(g)
        if g_norm > 0.01:
            g_dir = g / g_norm * 0.5
            ax.arrow(x0, y0, g_dir[0], g_dir[1],
                     head_width=0.12, head_length=0.15,
                     fc='red', ec='red', alpha=0.7)

ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('f(x,y) = x² + y² 的梯度向量场\n箭头 = 梯度方向（上升最快），指向背离原点')
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.show()
```

### 7.6 机器学习连接

梯度是机器学习优化引擎的核心。**梯度下降**就是在参数空间中，每次沿负梯度方向走一小步：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L(\theta_{\text{old}})
$$

为什么是负梯度？因为我们想最小化损失，而梯度指向增大损失的方向，所以要反着走。这就是从微积分到机器学习的最短路径。

---

## 8. 方向导数

### 8.1 生活例子：想往哪走就往哪走

梯度告诉你最陡的方向，但如果你想验证"往某个任意方向走，损失会变大还是变小"呢？

比如你想知道"沿 45° 对角线方向走，函数变化多少"——偏导数和梯度本身不能直接回答这个问题。你需要**方向导数**。

### 8.2 直观理解

方向导数就是：从当前点出发，沿任意方向 $\mathbf{v}$（单位向量）走一步，函数值的变化率。

它等于**梯度向量在方向 $\mathbf{v}$ 上的投影**：

$$
D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v} = \|\nabla f\| \cdot \|\mathbf{v}\| \cdot \cos\theta
$$

其中 $\theta$ 是 $\mathbf{v}$ 和 $\nabla f$ 之间的夹角。

- 当 $\mathbf{v}$ 与 $\nabla f$ 同向（$\cos\theta = 1$）：方向导数最大 = $\|\nabla f\|$ —— 正是梯度方向！
- 当 $\mathbf{v}$ 与 $\nabla f$ 反向（$\cos\theta = -1$）：方向导数最小 = $-\|\nabla f\|$ —— 正是负梯度方向！
- 当 $\mathbf{v}$ 与 $\nabla f$ 垂直（$\cos\theta = 0$）：方向导数 = 0 —— 沿等高线走，高度不变

### 8.3 形式化定义

对单位向量 $\mathbf{v} = (v_1, v_2, \dots, v_n)$，其中 $\|\mathbf{v}\| = 1$：

$$
D_{\mathbf{v}} f(x_0) = \lim_{h \to 0} \frac{f(x_0 + h\mathbf{v}) - f(x_0)}{h}
$$

可证明：$D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v}$

**这是"负梯度方向下降最快"的数学证明**：在所有单位向量 $\mathbf{v}$ 中，使得 $\nabla f \cdot \mathbf{v}$ 最小的 $\mathbf{v}$ 就是 $\mathbf{v} = -\frac{\nabla f}{\|\nabla f\|}$，此时 $D_{\mathbf{v}} f = -\|\nabla f\|$。

### 8.4 手算示例

$f(x, y) = x^2 + y^2$，在点 $(1, 2)$ 处：
- 梯度：$\nabla f = (2, 4)$

沿方向 $\mathbf{v}_1 = (1, 0)$（正东）：
$D_{\mathbf{v}_1} f = (2, 4) \cdot (1, 0) = 2$ —— 就是 $\frac{\partial f}{\partial x}$

沿方向 $\mathbf{v}_2 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$（45° 东北）：
$D_{\mathbf{v}_2} f = (2, 4) \cdot \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right) = \frac{2+4}{\sqrt{2}} = \frac{6}{\sqrt{2}} \approx 4.24$

沿梯度方向 $\mathbf{v}_3 = \frac{\nabla f}{\|\nabla f\|} = \frac{(2, 4)}{\sqrt{20}} = \left(\frac{2}{\sqrt{20}}, \frac{4}{\sqrt{20}}\right)$：
$D_{\mathbf{v}_3} f = \|\nabla f\| = \sqrt{20} \approx 4.47$ —— 最大值！

沿负梯度方向 $\mathbf{v}_4 = -\mathbf{v}_3$：
$D_{\mathbf{v}_4} f = -\|\nabla f\| \approx -4.47$ —— 最小值！

### 8.5 Python 验证

```python
import numpy as np

def f(X):
    return X[0]**2 + X[1]**2

def grad_f(X):
    return np.array([2*X[0], 2*X[1]])

point = np.array([1.0, 2.0])
g = grad_f(point)

print(f"f(x,y) = x² + y² 在点 (1, 2)")
print(f"梯度 ∇f = ({g[0]}, {g[1]}),  梯度模长 = {np.linalg.norm(g):.4f}")
print()

directions = {
    "正东 (1,0)":        np.array([1.0, 0.0]),
    "正北 (0,1)":        np.array([0.0, 1.0]),
    "东北 45°":          np.array([1.0, 1.0]) / np.sqrt(2),
    "梯度方向":          g / np.linalg.norm(g),
    "负梯度方向":        -g / np.linalg.norm(g),
    "垂直梯度（等高线）":  np.array([-g[1], g[0]]) / np.linalg.norm(g),
}

for name, v in directions.items():
    dd = np.dot(g, v)
    print(f"  D_v f ({name:20s}) = {dd:8.4f}")
```

### 8.6 机器学习连接

方向导数提供了梯度下降的理论基础。它严格证明了"沿负梯度方向走，损失下降最快"。每次你调用 `optimizer.step()`，优化器做的就是：计算梯度 → 取反 → 乘以学习率 → 更新参数。每一步都在走局部最优的下降方向。

---

## 9. 梯度下降的完整推导

### 9.1 问题设定

给定参数 $\theta$ 和损失函数 $L(\theta)$。目标：找到 $\theta^* = \arg\min_{\theta} L(\theta)$。

没有闭式解（或者闭式解太昂贵），所以用迭代方法：从某个初始点开始，每次往"使损失变小的方向"走一步。

### 9.2 泰勒展开的直觉

一阶泰勒展开告诉我们：当 $\Delta\theta$ 很小时，

$$
L(\theta + \Delta\theta) \approx L(\theta) + \nabla L(\theta)^T \cdot \Delta\theta
$$

我们想要 $L(\theta + \Delta\theta) < L(\theta)$，即：

$$
\nabla L(\theta)^T \cdot \Delta\theta < 0
$$

要这个点积最小（最负），$\Delta\theta$ 应该和 $\nabla L$ **方向相反**：

$$
\Delta\theta = -\eta \cdot \nabla L(\theta)
$$

其中 $\eta > 0$ 是学习率（步长）。

代入泰勒展开验证：

$$
L(\theta - \eta \nabla L) \approx L(\theta) - \eta \|\nabla L\|^2 < L(\theta)
$$

只要 $\eta > 0$ 且 $\eta$ 足够小（泰勒展开在远处不准确），损失一定下降。

### 9.3 梯度下降算法

```
初始化: 选择初始参数 θ₀, 学习率 η
重复直到收敛:
    g ← ∇L(θ)        # 计算梯度
    θ ← θ - η · g    # 沿负梯度方向走一步
```

### 9.4 手算示例：最小化 $f(x) = x^2$ 从 $x = 5$ 开始，$\eta = 0.1$

目标函数：$f(x) = x^2$，最小值在 $x = 0$

梯度：$f'(x) = 2x$

更新公式：$x_{\text{new}} = x_{\text{old}} - 0.1 \cdot 2x_{\text{old}} = x_{\text{old}} - 0.2x_{\text{old}} = 0.8x_{\text{old}}$

**一步步追踪：**

| 迭代 $t$ | $x_t$ | 梯度 $f'(x_t)$ | 更新量 $-\eta f'$ | $x_{t+1}$ | $f(x_{t+1})$ |
|---|---|---|---|---|---|
| 0 | 5.0000 | 10.0000 | -1.0000 | 4.0000 | 16.0000 |
| 1 | 4.0000 | 8.0000 | -0.8000 | 3.2000 | 10.2400 |
| 2 | 3.2000 | 6.4000 | -0.6400 | 2.5600 | 6.5536 |
| 3 | 2.5600 | 5.1200 | -0.5120 | 2.0480 | 4.1943 |
| 4 | 2.0480 | 4.0960 | -0.4096 | 1.6384 | 2.6844 |
| 5 | 1.6384 | 3.2768 | -0.3277 | 1.3107 | 1.7180 |

每迭代一步，$x$ 缩到原来的 0.8 倍。$x$ 指数级趋近于 0。5 步后从 5 降到了约 1.31，损失从 25 降到了 1.72。

如果继续：10 步后 $x \approx 0.54$，20 步后 $x \approx 0.058$，50 步后 $x \approx 7 \times 10^{-6}$。

### 9.5 Python 实现

```python
import numpy as np

def f(x):
    return x**2

def grad_f(x):
    return 2 * x

def gradient_descent_1d(f, grad_f, x0, lr=0.1, max_iter=100, tol=1e-8):
    """一维梯度下降，记录完整轨迹"""
    x = x0
    history = [(0, x, f(x))]
    for t in range(1, max_iter + 1):
        g = grad_f(x)
        x_new = x - lr * g
        loss = f(x_new)
        history.append((t, x_new, loss))
        if abs(x_new - x) < tol:
            break
        x = x_new
    return history

# 运行
history = gradient_descent_1d(f, grad_f, x0=5.0, lr=0.1)

print(f"{'迭代':<6}{'x':<14}{'f(x)':<14}{'梯度':<14}")
print("-" * 48)
for t, x, loss in history[:11]:  # 前 10 步 + 初始
    g = grad_f(x) if t == 0 else None
    print(f"{t:<6}{x:<14.6f}{loss:<14.6f}{grad_f(x):<14.6f}")

print(f"\n... 省略中间迭代 ...")
print(f"最终: x = {history[-1][1]:.10f}, f(x) = {history[-1][2]:.2e}")
print(f"总共 {len(history)-1} 步")
```

### 9.6 常见误区

- **误区**：学习率 $\eta$ 越大越好，收敛更快。
- **澄清**：$\eta$ 太大会导致在最优解附近**振荡**甚至**发散**。$f(x) = x^2$ 用 $\eta = 1.0$：$x$ 永远在 5 和 -5 之间跳。用 $\eta = 1.1$：$x$ 爆炸。
- **误区**：梯度下降每次一定让损失变小。
- **澄清**：只有 $\eta$ 足够小时一阶泰勒近似才准确。$\eta$ 太大时，你沿切线方向走了太远，实际函数可能反而上升了。
- **误区**：梯度为零就找到最优解了。
- **澄清**：梯度为零的点也可能是**鞍点**（saddle point）或局部最大值。对于非凸函数，梯度下降可能卡在局部最小值。

### 9.7 机器学习连接

梯度下降和它的变体（SGD、Adam、RMSprop）几乎是所有深度学习模型的训练方法。原理就是我上面手算的那个过程——只是维度从 1 变成了上百万，梯度从手算变成了自动微分计算。

---

## 10. 二阶导数

### 10.1 生活例子：踩油门

开车时：
- **速度**是一阶导数（位置的变化率）
- **加速度**是二阶导数（速度的变化率）

踩油门的力度决定加速度。即使你现在速度不快，如果你猛踩油门，速度会**加速增长**——这加速度就是二阶导数告诉你的信息。

### 10.2 直观理解

一阶导数 $f'(x)$ 告诉你"函数在增长还是在下降"。二阶导数 $f''(x)$ 告诉你"增长本身在加速还是减速"。

- $f''(x) > 0$：曲线**向上弯**（concave up，凸函数）。一阶导数在增大。如果 $f'(x) > 0$，函数在加速增长；如果 $f'(x) < 0$，函数在减速下降。形象：**碗形**
- $f''(x) < 0$：曲线**向下弯**（concave down，凹函数）。一阶导数在减小。形象：**倒碗形**
- $f''(x) = 0$：可能是**拐点**（inflection point），弯曲方向在此改变

**在优化中的含义**：
- $f'(x) = 0$ 且 $f''(x) > 0$ → 局部**最小值**（碗底）
- $f'(x) = 0$ 且 $f''(x) < 0$ → 局部**最大值**（山顶）
- $f'(x) = 0$ 且 $f''(x) = 0$ → 可能是鞍点，需要更高阶信息

### 10.3 形式化定义

$$
f''(x) = \frac{d}{dx} f'(x) = \lim_{h \to 0} \frac{f'(x+h) - f'(x)}{h}
$$

用原函数直接表达：

$$
f''(x) = \lim_{h \to 0} \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}
$$

### 10.4 手算示例

$f(x) = x^3$：

- $f'(x) = 3x^2$（一阶导数）
- $f''(x) = 6x$（二阶导数）

在 $x = 2$：$f''(2) = 12 > 0$，向上弯。
在 $x = -2$：$f''(-2) = -12 < 0$，向下弯。
在 $x = 0$：$f''(0) = 0$，拐点——弯曲方向从下弯变为上弯。

$f(x) = x^4$：

- $f'(x) = 4x^3$
- $f''(x) = 12x^2$

$f'(0) = 0$，$f''(0) = 0$——但 $x=0$ 确实是最小值点！这说明 $f''=0$ 不能排除极值，需要进一步检查。

### 10.5 Python 验证

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def numerical_second_derivative(f, x, h=1e-4):
    """用中心差分近似二阶导数"""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

x = np.linspace(-3, 3, 300)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

functions = [
    (lambda x: x**3,       r'$f(x)=x^3$',       lambda x: 3*x**2,  lambda x: 6*x),
    (lambda x: x**4,       r'$f(x)=x^4$',       lambda x: 4*x**3,  lambda x: 12*x**2),
    (lambda x: np.sin(x),  r'$f(x)=\sin x$',    np.cos,            lambda x: -np.sin(x)),
    (lambda x: -x**2 + 4,  r'$f(x)=-x^2+4$',    lambda x: -2*x,    lambda x: -2*np.ones_like(x)),
]

for i, (f, label, f1, f2) in enumerate(functions):
    ax = axes[i // 2][i % 2]
    ax.plot(x, f(x), 'b-', linewidth=2, label='f(x)')
    ax.plot(x, f1(x), 'r--', linewidth=1.5, label="f'(x)")
    ax.plot(x, f2(x), 'g:', linewidth=1.5, label="f''(x)")
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(label)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 验证数值二阶导数
print("数值二阶导数验证：")
print("-" * 50)
for name, f, x0, f2_exact in [
    ("x³ at x=2", lambda x: x**3, 2.0, 12.0),
    ("sin(x) at x=π/2", np.sin, np.pi/2, -1.0),
    ("eˣ at x=0", np.exp, 0.0, 1.0),
]:
    num = numerical_second_derivative(f, x0)
    print(f"  {name}: 数值={num:10.6f}  理论={f2_exact:10.6f}  误差={abs(num-f2_exact):.2e}")
```

### 10.6 机器学习连接

二阶导数信息可以帮助梯度下降选更好的步长——曲率大的地方应该走小步（避免冲过头），曲率小的地方可以走大步。基于二阶信息的优化方法（牛顿法、L-BFGS）在某些场景下比一阶方法快得多。

---

## 11. Hessian 矩阵

### 11.1 生活例子：多维曲率

一维函数的曲率用二阶导数 $f''(x)$ 描述。多元函数的曲率怎么描述？

对一个 $n$ 元函数，你需要知道：（1）每个变量自己的"弯曲"程度；（2）不同变量之间"交叉弯曲"的相互影响。Hessian 矩阵就是所有这些信息的组织方式。

### 11.2 直观理解

Hessian 矩阵是一个 $n \times n$ 的对称矩阵，其中第 $(i, j)$ 个元素是 $\frac{\partial^2 f}{\partial x_i \partial x_j}$。

- **对角线元素**：$\frac{\partial^2 f}{\partial x_i^2}$ —— 纯曲率，$x_i$ 自己弯曲的程度
- **非对角线元素**：$\frac{\partial^2 f}{\partial x_i \partial x_j}$ —— 交叉曲率，$x_i$ 的变化如何影响 $x_j$ 方向的斜率

当二阶偏导连续时，$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$，Hessian 是对称矩阵。

### 11.3 形式化定义

$$
H_f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[4pt]
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[4pt]
\vdots & \vdots & \ddots & \vdots \\[4pt]
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

Hessian 矩阵是梯度的 Jacobian 矩阵：$H_f = J_{\nabla f}$。

### 11.4 手算示例：$f(x, y) = x^3 + 3x^2 y + y^2$

一阶偏导数：

$$
\frac{\partial f}{\partial x} = 3x^2 + 6xy, \qquad
\frac{\partial f}{\partial y} = 3x^2 + 2y
$$

二阶偏导数（对一阶偏导再求导）：

$$
\frac{\partial^2 f}{\partial x^2} = 6x + 6y, \qquad
\frac{\partial^2 f}{\partial x \partial y} = 6x
$$

$$
\frac{\partial^2 f}{\partial y \partial x} = 6x, \qquad
\frac{\partial^2 f}{\partial y^2} = 2
$$

Hessian 矩阵：

$$
H_f(x, y) = \begin{bmatrix} 6x + 6y & 6x \\ 6x & 2 \end{bmatrix}
$$

在点 $(1, -1)$ 处：

$$
H_f(1, -1) = \begin{bmatrix} 0 & 6 \\ 6 & 2 \end{bmatrix}
$$

特征值？解 $\det(H - \lambda I) = \lambda^2 - 2\lambda - 36 = 0$，得 $\lambda \approx -5.08, 7.08$——特征值一正一负，说明在 $(1, -1)$ 处是**鞍点**。

### 11.5 牛顿法直觉

梯度下降只看一阶信息（梯度）。牛顿法同时用一阶和二阶信息：

$$
\theta_{\text{new}} = \theta_{\text{old}} - H_f^{-1} \cdot \nabla f(\theta_{\text{old}})
$$

$H_f^{-1} \cdot \nabla f$ 不仅给出了方向，还通过 Hessian 的逆矩阵自动调整了每个维度上的步长——曲率大的维度走小步，曲率小的维度走大步。

**代价**：Hessian 矩阵大小是 $n \times n$，对于深度学习模型（$n$ 可能是百万级别），存储和求逆都不现实。这就是为什么实践中往往用一阶方法（SGD、Adam）或 Hessian 的近似（L-BFGS）。

### 11.6 Python 示例

```python
import numpy as np

def f(X):
    x, y = X[0], X[1]
    return x**3 + 3 * x**2 * y + y**2

def hessian_f(X):
    x, y = X[0], X[1]
    return np.array([[6*x + 6*y, 6*x],
                     [6*x,       2  ]])

point = np.array([1.0, -1.0])
H = hessian_f(point)
eigvals = np.linalg.eigvalsh(H)

print(f"f(x,y) = x³ + 3x²y + y²  在点 (1, -1)：")
print(f"Hessian =\n{H}")
print(f"特征值 = {eigvals}")
print(f"一正一负 → 鞍点 ✓" if eigvals[0] < 0 < eigvals[1] else "")

# Hessian 正定性检测
for pt_name, pt in [("(0, 0)", [0.0, 0.0]),
                     ("(1, 1)", [1.0, 1.0]),
                     ("(-1, 2)", [-1.0, 2.0])]:
    H_pt = hessian_f(np.array(pt))
    eigvals_pt = np.linalg.eigvalsh(H_pt)
    is_psd = np.all(eigvals_pt >= 0)
    print(f"  {pt_name}: 特征值={eigvals_pt}, {'正半定（局部凸）' if is_psd else '非正半定'}")
```

### 11.7 机器学习连接

- **凸性判定**：Hessian 矩阵正半定（特征值全部 ≥ 0）等价于函数是凸的
- **牛顿法和拟牛顿法**：利用 Hessian 信息加速收敛。如 scipy.optimize.minimize 的 method='Newton-CG'
- **自适应优化器**：Adam、RMSprop 等通过维护梯度的二阶矩估计（类似 Hessian 对角线的近似）来为每个参数自适应调整学习率

---

## 12. 凸函数

### 12.1 生活例子：碗和山谷

把一个球放进一个碗里。不论你从碗的哪个位置放手，球都会滚到碗底——同一个位置。这就是凸函数的直觉：**只有一个"底"，所有下坡路都通向它**。

对比：一个连绵起伏的山脉有无数个山谷（局部最小值），从不同位置下山可能到达不同的山谷——这是非凸函数。

### 12.2 直观理解

凸函数的定义（一阶条件）：**函数图像在任意切线的上方**。

更直观的说法（Jensen 不等式）：

> 函数在平均值处的值 ≤ 函数值的平均

即：$f\left(\frac{x_1+x_2}{2}\right) \leq \frac{f(x_1)+f(x_2)}{2}$

等价于：连接曲线上任意两点的弦，始终在函数图像的上方。

### 12.3 形式化定义

$f$ 是凸函数当且仅当对所有 $x_1, x_2$ 和 $\lambda \in [0, 1]$：

$$
f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)
$$

**二阶条件**：若 $f$ 二阶可导：
- $f$ 是凸函数 $\iff$ 对所有 $x$，$f''(x) \geq 0$
- 多元情况：$f$ 是凸函数 $\iff$ Hessian 矩阵处处正半定

**至关重要的性质**：对于凸函数，任何局部最小值都是全局最小值。

这就是为什么机器学习喜欢凸损失函数——你不用担心梯度下降卡在某个山谷里，它一定会走到全局最优。

### 12.4 常见凸函数

| 函数 | 凸域 | 在 ML 中的用途 |
|---|---|---|
| $x^2$ | 全部实数 | MSE 损失的基础 |
| $e^x$ | 全部实数 | softmax 的分母 |
| $-\ln x$ | $x > 0$ | 交叉熵损失 |
| $\|x\|$ | 全部实数 | L1 正则化（Lasso） |
| $\max(0, -x)$ | 全部实数 | Hinge loss（SVM）|
| $x \ln x$ | $x > 0$ | KL 散度 |

### 12.5 Python：验证凸性和 Jensen 不等式

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 验证 Jensen 不等式：f(E[X]) ≤ E[f(X)]
np.random.seed(42)

tests = [
    ("x² (凸)",     lambda x: x**2,     "convex"),
    ("eˣ (凸)",     np.exp,             "convex"),
    ("-ln(x) (凸)", lambda x: -np.log(np.maximum(x, 0.01)), "convex"),
    ("sin(x) (非凸)", np.sin,            "non-convex"),
]

print("Jensen 不等式验证：f(E[X]) ≤ E[f(X)] 应成立（凸函数）")
print("-" * 65)
print(f"{'函数':<20}{'f(E[X])':<14}{'E[f(X)]':<14}{'成立?':<10}")
print("-" * 65)

for name, func, expected in tests:
    samples = np.random.exponential(1.0, 1000) + 0.5
    f_of_mean = func(np.mean(samples))
    mean_of_f = np.mean(func(samples))
    holds = f_of_mean <= mean_of_f
    print(f"{name:<20}{f_of_mean:<14.6f}{mean_of_f:<14.6f}{'✓' if holds else '✗':<10}")

# 可视化凸函数 vs 非凸函数
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x = np.linspace(-3, 3, 200)

# 凸函数：弦在图像上方
axes[0].plot(x, x**2, 'b-', linewidth=2)
axes[0].plot([-2, 2], [4, 4], 'r--', linewidth=1.5, label='弦（在图像上方）')
axes[0].set_title('凸函数 f(x)=x²\n弦始终在图像上方 → 局部最小=全局最小')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 凹函数：弦在图像下方
axes[1].plot(x, -x**2 + 4, 'g-', linewidth=2)
axes[1].plot([-2, 2], [0, 0], 'r--', linewidth=1.5, label='弦（在图像下方）')
axes[1].set_title('凹函数 f(x)=-x²+4\n弦始终在图像下方')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 非凸函数：弦可能穿过图像
y_sin = np.sin(x) + 0.3*x**2
axes[2].plot(x, y_sin, 'm-', linewidth=2)
axes[2].plot([-2.5, 1.5],
             [np.sin(-2.5)+0.3*6.25, np.sin(1.5)+0.3*2.25],
             'r--', linewidth=1.5, label='弦（穿过图像）')
axes[2].set_title('非凸函数\n弦可能穿过图像 → 有多个局部最小值')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
```

### 12.6 机器学习连接

| 场景 | 凸性 | 含义 |
|---|---|---|
| 线性回归 + MSE | 凸 | 梯度下降保证找到全局最优 |
| 逻辑回归 + 交叉熵 | 凸 | 同上 |
| SVM 对偶问题 | 凸 | 高效求解（凸二次规划）|
| 神经网络 | 非凸 | 可能陷入局部最小值，但实践中通常能找到不错的解 |
| Lasso (L1 正则化) | 凸但不可导 | 使用次梯度或坐标下降 |

---

## 13. Jacobian 矩阵简介

### 13.1 生活例子：多输入多输出

梯度和 Hessian 都是对标量函数的——输出是一个数。但很多场景下，函数有多个输出。

比如：一个机械臂有 3 个关节角度 $(\theta_1, \theta_2, \theta_3)$ 作为输入，末端执行器的位置是 3 维坐标 $(x, y, z)$。你想知道"旋转第一个关节 1°，末端位置会怎么变"——这需要所有 3×3 = 9 个偏导数。Jacobian 矩阵收集了所有这些信息。

### 13.2 直观理解

对于向量值函数 $F: \mathbb{R}^n \to \mathbb{R}^m$：
- 梯度：$n \to 1$ 的导数 = $1 \times n$ 的行向量（或 $n \times 1$ 的列向量）
- Jacobian：$n \to m$ 的导数 = **$m \times n$ 的矩阵**

Jacobian 的第 $i$ 行是第 $i$ 个分量函数 $f_i$ 的梯度（的转置），第 $j$ 列是所有输出对第 $j$ 个输入变量的偏导数。

### 13.3 形式化定义

$$
J_F = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\[4pt]
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\[4pt]
\vdots & \vdots & \ddots & \vdots \\[4pt]
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

**关键性质**：对于小的 $\Delta x$，$F(x + \Delta x) \approx F(x) + J_F(x) \cdot \Delta x$

Jacobian 是 $F$ 在 $x$ 处的**局部线性近似**。

### 13.4 链式法则的 Jacobian 形式

如果 $h(x) = g(f(x))$，其中 $f: \mathbb{R}^n \to \mathbb{R}^m$，$g: \mathbb{R}^m \to \mathbb{R}^p$，则：

$$
J_h(x) = J_g(f(x)) \cdot J_f(x)
$$

即复合函数的 Jacobian = Jacobian 的乘积（矩阵乘法）。这是反向传播的矩阵形式。

### 13.5 Python 示例

```python
import numpy as np

def F(X):
    """F: R² → R³
    F(x,y) = [x² + y, sin(xy), eˣ]
    """
    x, y = X[0], X[1]
    return np.array([x**2 + y, np.sin(x * y), np.exp(x)])

def jacobian_F(X):
    """解析 Jacobian"""
    x, y = X[0], X[1]
    return np.array([
        [2*x,           1           ],
        [y*np.cos(x*y), x*np.cos(x*y)],
        [np.exp(x),     0           ],
    ])

def numerical_jacobian(F, X, h=1e-5):
    m = len(F(X))
    n = len(X)
    J = np.zeros((m, n))
    for j in range(n):
        Xp = X.copy(); Xp[j] += h
        Xm = X.copy(); Xm[j] -= h
        J[:, j] = (F(Xp) - F(Xm)) / (2 * h)
    return J

point = np.array([1.0, 2.0])
J_exact = jacobian_F(point)
J_num = numerical_jacobian(F, point)

print("Jacobian 矩阵：解析 vs 数值")
print(f"解析:\n{J_exact}")
print(f"\n数值:\n{J_num}")
print(f"\n最大误差: {np.max(np.abs(J_exact - J_num)):.2e}")
```

### 13.6 机器学习连接

- **神经网络层间映射**：每一层 $y = \sigma(Wx + b)$ 是向量值函数。其 Jacobian 描述输出对输入的局部敏感度
- **反向传播**：梯度通过 Jacobian 的转置在层间传播：$\frac{\partial L}{\partial x} = J_f(x)^T \cdot \frac{\partial L}{\partial y}$
- **Gauss-Newton 优化**：依赖 Jacobian 的非线性最小二乘法

---

## 14. 自动微分的直觉

### 14.1 生活例子：拆解复杂问题

你在算 $f(x) = \sin(x^2 + e^x)$ 在 $x = 1$ 处的导数。

你可以手算：先求 $x^2$ 的导数（$2x$），再求 $e^x$ 的导数（$e^x$），然后求 $x^2 + e^x$ 的导数（$2x + e^x$），最后用链式法则得到 $\cos(x^2 + e^x) \cdot (2x + e^x)$。

每一步你都只处理了**一个基本运算**（+、×、sin、exp），然后把它们按链式法则组合起来。自动微分做的就是这件事——不过是自动的、精确的、对任意复杂的函数都适用。

### 14.2 直观理解

**核心思想**：任何可微函数，不管多复杂，都可以分解为基本运算的序列（计算图）。每个基本运算的求导规则是已知的。自动微分系统做的事就是：
1. 记录前向计算过程，构建**计算图**
2. 在图上反向遍历，用链式法则把各节点的局部导数乘起来

**两种模式**：

| 模式 | 计算方向 | 适用场景 |
|---|---|---|
| 前向模式 | 输入 → 输出 | 输入维度小（如 $n \leq 5$） |
| 反向模式 | 输出 → 输入 | 输出维度小、输入维度大（机器学习！） |

在机器学习中，损失函数 $L$ 是标量（输出维度 = 1），参数可能有百万个（输入维度超大）。反向模式只需**一次反向遍历**就能算出所有参数的梯度。这也是为什么它叫"反向传播"。

### 14.3 计算图示例

对于 $y = \sin(x^2 + e^x)$ 在 $x = 1$：

```
前向传播:
x=1 → v1 = x² = 1      (dw1/dx = 2x = 2)
     → v2 = eˣ = e      (dw2/dx = eˣ = e)
     → v3 = v1+v2 = 1+e (∂v3/∂v1 = 1, ∂v3/∂v2 = 1)
     → y = sin(v3)      (dy/dv3 = cos(v3))
```

```
反向传播:
dy/dv3 = cos(1+e)
dy/dv1 = dy/dv3 × ∂v3/∂v1 = cos(1+e) × 1
dy/dv2 = dy/dv3 × ∂v3/∂v2 = cos(1+e) × 1
dy/dx  = dy/dv1 × dw1/dx + dy/dv2 × dw2/dx
       = cos(1+e) × 2 + cos(1+e) × e
       = cos(1+e) × (2 + e)
```

自动微分系统把这个过程自动化了——你只需要定义前向计算，系统自动构建图并计算梯度。

### 14.4 三种计算导数的方法对比

| 方法 | 原理 | 精度 | 速度 | 适用场景 |
|---|---|---|---|---|
| 手动求导 | 推导公式后编码 | 精确（如果没写错） | 快 | 简单函数 |
| 数值微分 | $\frac{f(x+h)-f(x)}{h}$ | 近似（有截断+舍入误差） | 慢（每参数需 2 次前向） | 验证、调试 |
| 符号微分 | 代数操作 | 精确 | 中等 | 表达式膨胀问题 |
| **自动微分** | 计算图 + 链式法则 | **精确**（机器精度） | **快**（一次反向） | **深度学习** |

### 14.5 Python：微型 autograd 演示

```python
import numpy as np

class Value:
    """一个极简的自动微分引擎 — 展示核心原理"""
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def sin(self):
        out = Value(np.sin(self.data), (self,), 'sin')
        def _backward():
            self.grad += np.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        # 拓扑排序
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # 反向传播
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# 测试：f(x) = sin(x² + eˣ) at x = 1
x = Value(1.0)
f = (x ** 2 + x.exp()).sin()
f.backward()

# 理论导数
xv = 1.0
exact = np.cos(xv**2 + np.exp(xv)) * (2*xv + np.exp(xv))

print(f"f(1)  = sin(1² + e¹) = {f.data:.6f}")
print(f"f'(1) = {x.grad:.10f}  (自动微分)")
print(f"f'(1) = {exact:.10f}  (理论值)")
print(f"误差  = {abs(x.grad - exact):.2e}")
```

### 14.6 机器学习连接

你使用 PyTorch 写 `loss.backward()` 时，内部做的就是这件事——不过它的计算图更复杂，支持 GPU，有更多的运算。但原理完全一样：前向构建计算图，反向传播梯度。这就是现代深度学习的引擎。

---

## 15. 总结：从变化到学习

让我们回顾一路走来的旅程：

| 概念 | 核心问题 | 一句话 |
|---|---|---|
| 函数与变化率 | 量怎么变？ | 平均变化率 = 总变化 ÷ 总时间 |
| 极限 | 趋近于什么？ | "无限接近但不等于" |
| 导数 | 瞬间变得多快？ | 切线的斜率 = 瞬时变化率 |
| 求导法则 | 怎么算导数？ | 从定义出发推导，或用法则组合 |
| 偏导数 | 多变量中单个的影响？ | 冻结其他变量，单独对这个求导 |
| 梯度 | 往哪走上升最快？ | 偏导数组成的向量 |
| 方向导数 | 往任意方向走怎么变？ | 梯度在该方向上的投影 |
| 梯度下降 | 怎么找到最小值？ | 沿负梯度方向一步步走 |
| 二阶导数 | 变化本身在加速还是减速？ | 曲率——碗形还是倒碗形 |
| Hessian 矩阵 | 多维曲率怎么描述？ | 所有二阶偏导数的矩阵 |
| 凸函数 | 有没有唯一最优解？ | 碗状——局部最小 = 全局最小 |
| Jacobian 矩阵 | 多输出函数的导数？ | 每个输出对每个输入的偏导数 |
| 自动微分 | 怎么让计算机自动求导？ | 计算图 + 链式法则 + 反向传播 |

**微积分在机器学习中的核心思想，用一句话总结：**

> **用局部线性近似（导数/梯度）来指导迭代搜索（梯度下降），逐步逼近全局最优解——这就是"学习"的数学本质。**

每一次模型参数的更新，都是一次"沿负梯度方向走一步"的物理动作。这背后，是 17 世纪牛顿和莱布尼兹发明微积分时就埋下的种子。

---

## 思考题

### 基础题

**Q1**：求下列函数的导数。
(a) $f(x) = 3x^4 - 5x^2 + 2x - 7$
(b) $g(x) = e^x \cdot \sin x$
(c) $h(x) = \ln(x^2 + 1)$
(d) $k(x) = \frac{x^2 + 1}{x - 1}$

**Q2**：用导数的极限定义（第一性原理）证明 $\frac{d}{dx}\left(\frac{1}{x}\right) = -\frac{1}{x^2}$。

**Q3**：求 $f(x) = x^3 - 3x$ 的临界点（$f'(x) = 0$ 的点），并判断每个点是局部最大值、局部最小值还是拐点。

### 进阶题

**Q4**：设 $f(x, y) = x^2 + xy + y^2$。
(a) 求梯度 $\nabla f(x, y)$
(b) 求 Hessian 矩阵 $H_f(x, y)$
(c) 证明 $f$ 是凸函数
(d) 求 $f$ 的全局最小值

**Q5**：对 $f(x) = (x - 3)^2$，从 $x_0 = 10$ 开始，用梯度下降（学习率 $\eta = 0.2$），手工追踪 5 次迭代，验证 $x$ 是否在接近最小值 $x^* = 3$。

**Q6**：对 $f(x, y) = x^2 + 3y^2$，从 $(x_0, y_0) = (4, 3)$ 开始，用梯度下降（$\eta = 0.1$），手工追踪前 3 步，写出每一步的梯度、新位置和函数值。

### 挑战题

**Q7**：用链式法则推导 $\frac{d}{dx} a^x$ 的公式（$a > 0$）。
提示：$a^x = e^{x \ln a}$。

**Q8**：求函数 $f(x, y) = x^3 + y^3 - 3xy$ 的所有临界点，并对每个点判断它是局部最小值、局部最大值还是鞍点。
提示：解 $\nabla f = 0$，然后用 Hessian 矩阵的特征值分类。

**Q9**：对 $f(x, y) = x^2 + 4y^2$，证明沿负梯度方向的方向导数等于 $-\|\nabla f\|$，沿任意其他方向的方向导数都大于这个值（绝对值更小）。

**Q10**：设线性回归模型 $\hat{y} = wx + b$，损失函数 $L(w, b) = \frac{1}{2}(y_{\text{true}} - (wx + b))^2$（简化：一个数据点）。推导 $\frac{\partial L}{\partial w}$ 和 $\frac{\partial L}{\partial b}$，并写出梯度下降的更新公式。

---

## 思考题解答

### Q1 解答

**(a)** $f(x) = 3x^4 - 5x^2 + 2x - 7$

逐项求导（幂法则 + 和差法则）：
- $\frac{d}{dx}(3x^4) = 3 \cdot 4x^3 = 12x^3$
- $\frac{d}{dx}(-5x^2) = -5 \cdot 2x = -10x$
- $\frac{d}{dx}(2x) = 2$
- $\frac{d}{dx}(-7) = 0$

$$
f'(x) = 12x^3 - 10x + 2
$$

**(b)** $g(x) = e^x \cdot \sin x$

乘积法则：$(uv)' = u'v + uv'$

- $u = e^x$，$u' = e^x$
- $v = \sin x$，$v' = \cos x$

$$
g'(x) = e^x \cdot \sin x + e^x \cdot \cos x = e^x(\sin x + \cos x)
$$

**(c)** $h(x) = \ln(x^2 + 1)$

链式法则：外层 $\ln(u)$ 导数 = $\frac{1}{u}$，内层 $u = x^2 + 1$ 导数 = $2x$

$$
h'(x) = \frac{1}{x^2 + 1} \cdot 2x = \frac{2x}{x^2 + 1}
$$

**(d)** $k(x) = \frac{x^2 + 1}{x - 1}$

商法则：$\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$

- $u = x^2 + 1$，$u' = 2x$
- $v = x - 1$，$v' = 1$

$$
k'(x) = \frac{2x \cdot (x-1) - (x^2+1) \cdot 1}{(x-1)^2}
= \frac{2x^2 - 2x - x^2 - 1}{(x-1)^2}
= \frac{x^2 - 2x - 1}{(x-1)^2}
$$

---

### Q2 解答

用极限定义：

$$
\begin{aligned}
\frac{d}{dx}\left(\frac{1}{x}\right)
&= \lim_{h \to 0} \frac{\frac{1}{x+h} - \frac{1}{x}}{h} \\[6pt]
&= \lim_{h \to 0} \frac{\frac{x - (x+h)}{x(x+h)}}{h} \quad \text{← 通分} \\[6pt]
&= \lim_{h \to 0} \frac{\frac{-h}{x(x+h)}}{h} \\[6pt]
&= \lim_{h \to 0} \frac{-h}{h \cdot x(x+h)} \\[6pt]
&= \lim_{h \to 0} \frac{-1}{x(x+h)} \\[6pt]
&= -\frac{1}{x^2}
\end{aligned}
$$

证毕。注意这和幂法则 $\frac{d}{dx}(x^{-1}) = -1 \cdot x^{-2} = -\frac{1}{x^2}$ 一致。

---

### Q3 解答

$f(x) = x^3 - 3x$

**第一步**：求一阶导数
$f'(x) = 3x^2 - 3 = 3(x^2 - 1) = 3(x-1)(x+1)$

**第二步**：求临界点（$f'(x) = 0$）
$x = 1$ 或 $x = -1$

**第三步**：求二阶导数
$f''(x) = 6x$

**第四步**：二阶梯测试分类

- 在 $x = -1$：$f''(-1) = -6 < 0$ → 局部**最大值**
  $f(-1) = (-1)^3 - 3(-1) = -1 + 3 = 2$
- 在 $x = 1$：$f''(1) = 6 > 0$ → 局部**最小值**
  $f(1) = 1 - 3 = -2$

**结论**：$f$ 在 $x = -1$ 处有局部最大值 $2$，在 $x = 1$ 处有局部最小值 $-2$。注意 $f''(0) = 0$（拐点），但 $f'(0) \neq 0$，所以 $x = 0$ 不是临界点。

---

### Q4 解答

$f(x, y) = x^2 + xy + y^2$

**(a) 梯度**

$$
\frac{\partial f}{\partial x} = 2x + y, \qquad
\frac{\partial f}{\partial y} = x + 2y
$$

$$
\nabla f(x, y) = (2x + y, \; x + 2y)
$$

**(b) Hessian 矩阵**

$$
\frac{\partial^2 f}{\partial x^2} = 2, \quad
\frac{\partial^2 f}{\partial x \partial y} = 1, \quad
\frac{\partial^2 f}{\partial y \partial x} = 1, \quad
\frac{\partial^2 f}{\partial y^2} = 2
$$

$$
H_f = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

**(c) 证 $f$ 是凸函数**

$H_f$ 是常数矩阵。其特征值：$\det(H - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3)$

特征值 $\lambda_1 = 1 > 0$，$\lambda_2 = 3 > 0$。Hessian 处处正定 → $f$ 是（严格）凸函数。

**(d) 全局最小值**

设 $\nabla f = (0, 0)$：
$$
\begin{cases}
2x + y = 0 \\
x + 2y = 0
\end{cases}
$$

从第一式：$y = -2x$。代入第二式：$x + 2(-2x) = x - 4x = -3x = 0$ → $x = 0$，$y = 0$。

由于 $f$ 是凸函数，这个临界点就是全局最小值。

$f(0, 0) = 0$

---

### Q5 解答

$f(x) = (x-3)^2$，$f'(x) = 2(x-3)$

初始：$x_0 = 10$，$\eta = 0.2$

更新公式：$x_{t+1} = x_t - 0.2 \cdot 2(x_t - 3) = x_t - 0.4(x_t - 3)$

| 迭代 $t$ | $x_t$ | $f'(x_t) = 2(x_t-3)$ | $-\eta \cdot f'$ | $x_{t+1}$ | $f(x_{t+1})$ |
|---|---|---|---|---|---|
| 0 | 10.000 | 14.000 | -2.800 | 7.200 | 17.64 |
| 1 | 7.200 | 8.400 | -1.680 | 5.520 | 6.35 |
| 2 | 5.520 | 5.040 | -1.008 | 4.512 | 2.29 |
| 3 | 4.512 | 3.024 | -0.605 | 3.907 | 0.82 |
| 4 | 3.907 | 1.814 | -0.363 | 3.544 | 0.30 |
| 5 | 3.544 | 1.089 | -0.218 | 3.327 | 0.11 |

$x$ 从 10 开始，逐步趋近 3。5 步后 $x \approx 3.33$，$f(x) \approx 0.11$。继续迭代会越来越接近 0。

---

### Q6 解答

$f(x, y) = x^2 + 3y^2$，最小值在 $(0, 0)$

梯度：$\nabla f = (2x, 6y)$

初始：$(x_0, y_0) = (4, 3)$，$\eta = 0.1$

更新公式：$(x_{t+1}, y_{t+1}) = (x_t, y_t) - 0.1 \cdot (2x_t, 6y_t)$

**第 0 步（初始）**：
- 位置：$(4, 3)$
- 梯度：$(8, 18)$
- $f = 4^2 + 3 \cdot 3^2 = 16 + 27 = 43$

**第 1 步**：
- 更新量：$(-0.8, -1.8)$
- 新位置：$(3.2, 1.2)$
- $f = 3.2^2 + 3 \cdot 1.2^2 = 10.24 + 4.32 = 14.56$

**第 2 步**：
- 梯度：$(6.4, 7.2)$
- 更新量：$(-0.64, -0.72)$
- 新位置：$(2.56, 0.48)$
- $f = 2.56^2 + 3 \cdot 0.48^2 = 6.5536 + 0.6912 = 7.2448$

**第 3 步**：
- 梯度：$(5.12, 2.88)$
- 更新量：$(-0.512, -0.288)$
- 新位置：$(2.048, 0.192)$
- $f = 2.048^2 + 3 \cdot 0.192^2 = 4.1943 + 0.1106 = 4.3049$

3 步后损失从 43 降到了 4.30。$y$ 分量衰减更快（因为系数 6 > 2），解释了为什么 $y$ 比 $x$ 更早接近 0。

---

### Q7 解答

利用 $a^x = e^{x \ln a}$：

$$
\begin{aligned}
\frac{d}{dx} a^x &= \frac{d}{dx} e^{x \ln a} \\
&= e^{x \ln a} \cdot \frac{d}{dx}(x \ln a) \quad \text{← 链式法则，外围 } e^u \text{ 导数为 } e^u \\
&= e^{x \ln a} \cdot \ln a \quad \text{← } \ln a \text{ 是常数} \\
&= a^x \cdot \ln a \quad \text{← 代回 } a^x
\end{aligned}
$$

所以：$\frac{d}{dx} a^x = a^x \ln a$

验证：当 $a = e$，$\frac{d}{dx} e^x = e^x \ln e = e^x \cdot 1 = e^x$ ✓

---

### Q8 解答

$f(x, y) = x^3 + y^3 - 3xy$

**第一步**：求梯度

$$
\frac{\partial f}{\partial x} = 3x^2 - 3y, \qquad
\frac{\partial f}{\partial y} = 3y^2 - 3x
$$

**第二步**：求临界点 $\nabla f = 0$

$$
\begin{cases}
3x^2 - 3y = 0 \implies y = x^2 \\
3y^2 - 3x = 0 \implies x = y^2
\end{cases}
$$

代入：$x = (x^2)^2 = x^4$，所以 $x^4 - x = 0$，即 $x(x^3 - 1) = 0$

解得：$x = 0$ 或 $x = 1$

- $x = 0 \implies y = 0^2 = 0$ → 临界点 $(0, 0)$
- $x = 1 \implies y = 1^2 = 1$ → 临界点 $(1, 1)$

**第三步**：Hessian 矩阵

$$
\frac{\partial^2 f}{\partial x^2} = 6x, \quad
\frac{\partial^2 f}{\partial x \partial y} = -3, \quad
\frac{\partial^2 f}{\partial y^2} = 6y
$$

$$
H_f(x, y) = \begin{bmatrix} 6x & -3 \\ -3 & 6y \end{bmatrix}
$$

**第四步**：分类各临界点

在 $(0, 0)$：
$$
H_f(0, 0) = \begin{bmatrix} 0 & -3 \\ -3 & 0 \end{bmatrix}
$$
特征值：$\lambda^2 - 9 = 0$ → $\lambda = \pm 3$（一正一负）→ **鞍点**

在 $(1, 1)$：
$$
H_f(1, 1) = \begin{bmatrix} 6 & -3 \\ -3 & 6 \end{bmatrix}
$$
特征值：$(6-\lambda)^2 - 9 = 0$ → $\lambda^2 - 12\lambda + 27 = 0$ → $(\lambda - 3)(\lambda - 9) = 0$ → $\lambda = 3, 9$

两个特征值都 > 0 → Hessian 正定 → **局部最小值**

函数值：$f(1, 1) = 1 + 1 - 3 = -1$

---

### Q9 解答

$f(x, y) = x^2 + 4y^2$

梯度：$\nabla f = (2x, 4y)$

在任意点 $(x_0, y_0)$，$\nabla f = (2x_0, 4y_0)$。

沿负梯度方向的单位向量：

$$
\mathbf{v}_{\text{neg}} = -\frac{\nabla f}{\|\nabla f\|} = -\frac{(2x_0, 4y_0)}{\sqrt{4x_0^2 + 16y_0^2}}
$$

方向导数：

$$
\begin{aligned}
D_{\mathbf{v}_{\text{neg}}} f
&= \nabla f \cdot \mathbf{v}_{\text{neg}} \\
&= (2x_0, 4y_0) \cdot \left(-\frac{(2x_0, 4y_0)}{\|\nabla f\|}\right) \\
&= -\frac{4x_0^2 + 16y_0^2}{\|\nabla f\|} \\
&= -\frac{\|\nabla f\|^2}{\|\nabla f\|} \\
&= -\|\nabla f\|
\end{aligned}
$$

对于任意其他单位向量 $\mathbf{v}$，有 $D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v} = \|\nabla f\| \cos\theta$，其中 $\theta$ 是夹角。当 $\mathbf{v} \neq \mathbf{v}_{\text{neg}}$ 时，$\cos\theta > -1$，所以：

$$
D_{\mathbf{v}} f = \|\nabla f\| \cos\theta > -\|\nabla f\|
$$

即方向导数的绝对值更小（值更大，即"下降得更少"）。证毕。

---

### Q10 解答

$L(w, b) = \frac{1}{2}(y - (wx + b))^2$，其中 $y = y_{\text{true}}$

（注：$\frac{1}{2}$ 是为了消去求导后的系数 2，是常用技巧。）

令 $e = y - (wx + b)$，则 $L = \frac{1}{2}e^2$

链式法则：$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial w}$

- $\frac{\partial L}{\partial e} = e = y - (wx + b)$
- $\frac{\partial e}{\partial w} = -x$

$$
\frac{\partial L}{\partial w} = -(y - (wx + b)) \cdot x = x(wx + b - y)
$$

类似地：

- $\frac{\partial e}{\partial b} = -1$

$$
\frac{\partial L}{\partial b} = -(y - (wx + b)) = wx + b - y
$$

**梯度下降更新公式**：

$$
\begin{aligned}
w_{\text{new}} &= w_{\text{old}} - \eta \cdot x(w_{\text{old}} \cdot x + b_{\text{old}} - y) \\[4pt]
b_{\text{new}} &= b_{\text{old}} - \eta \cdot (w_{\text{old}} \cdot x + b_{\text{old}} - y)
\end{aligned}
$$

**直观理解**：$wx + b - y$ 是预测误差。如果预测偏大（误差 > 0），就减小 $w$ 和 $b$；如果预测偏小（误差 < 0），就增大它们。$x$ 乘以误差的项反映：输入 $x$ 越大，$w$ 的调整幅度越大（因为 $w$ 的"杠杆效应"更大）。

---

下一步：继续学习[概率论基础](./math-probability.md)，理解机器学习中的不确定性。
