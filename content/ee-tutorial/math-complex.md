# 复数与相量 — 当"虚数"不再是假的

> **核心问题**：交流电为什么需要"虚数"这个听起来很假的东西？

---

## 0. 本章导览

你很可能听过"复数"这个词，也知道它有个部分是"虚的"。光是"虚数"这个名字，就让人觉得这东西不靠谱——数学家用虚构的东西在算什么？

但如果你学过一点电路，就会看到交流电分析里到处都是 $j$（工程中的虚数单位），欧拉公式 $e^{j\theta} = \cos\theta + j\sin\theta$ 更是被挂在各种工程课本的封面上。这些不是装饰——没有复数，整个交流电理论就塌了。

本章带你从零开始，把复数从一个"奇怪的数学工具"变成一个"你脑子里能画出来的东西"。学完本章，你将能够：

1. 在复平面上画出任意复数，理解模和幅角的含义
2. 手算复数的加减乘除，每一步都不出错
3. 在直角坐标和极坐标之间自由转换
4. 用欧拉公式理解"旋转"和"振荡"的关系
5. 理解相量（phasor）就是"冻结的旋转"，为交流电路分析打下基础

> 前置知识：初中代数（解方程、坐标系）。本章约 1800 行，建议分 3-4 次读完。

---

## 1. 复数从哪来

### 1.1 生活例子：地图上的定位

你给朋友发微信："我在万达广场"。

朋友到了万达，问你："具体在哪？"

你说："我在万达的东边 200 米。"

朋友往东走了 200 米，没看到你。再问。

你又说："哦对了，还得往北走 150 米。"

你用了**两个数字**（东 200 米、北 150 米）来定位一个点。这是因为平面是二维的——你需要两个独立的方向才能唯一确定一个位置。

复数就是数学中的"两个数字打包成一个东西"。笛卡尔坐标系用 $(x, y)$，复数用 $x + jy$。本质上是一回事，但复数的写法让你能对这两个数字**像对一个数字一样做运算**。这一点非常关键——后面你会看到，用小括号 $(x, y)$ 做乘除法会很麻烦，但用 $x + jy$ 却顺理成章。

### 1.2 直观理解：方程的"幽灵解"

你一定解过这类方程：

$$
x^2 = 9 \quad \Rightarrow \quad x = 3 \text{ 或 } x = -3
$$

两个解，都在数轴上，合情合理。

那这个呢？

$$
x^2 = 1 \quad \Rightarrow \quad x = 1 \text{ 或 } x = -1
$$

也没问题。

现在试试这个：

$$
x^2 = -1
$$

什么数的平方是负数？$1^2 = 1$，$(-1)^2 = 1$。在实数范围内，**没有任何数的平方是负数**。任何实数的平方都 $\ge 0$。

所以数学家说：好吧，我们**发明**一个数，它的平方等于 -1。把它叫做 **$i$**（imaginary unit，虚数单位）。

工程上不用 $i$，因为 $i$ 已经被电流（current）占用了。我们用 **$j$**。

$$
j^2 = -1
$$

有了 $j$，方程 $x^2 = -1$ 就有解了：$x = j$ 或 $x = -j$。

### 1.3 形式化定义

一个复数（Complex Number）写成：

$$
z = a + jb
$$

其中：

| 符号 | 名称 | 含义 |
|---|---|---|
| $a$ | 实部（Real Part），记作 $\operatorname{Re}(z)$ | 普通的实数 |
| $b$ | 虚部（Imaginary Part），记作 $\operatorname{Im}(z)$ | 也是一个实数，但它乘以 $j$ |
| $j$ | 虚数单位 | $j^2 = -1$ |

几个例子：

- $z_1 = 3 + j4$：实部 3，虚部 4
- $z_2 = -2 + j$：实部 -2，虚部 1（注意 $j$ 就是 $1j$）
- $z_3 = 5$：实部 5，虚部 0。实数就是虚部为零的复数
- $z_4 = j3$：实部 0，虚部 3。这叫做"纯虚数"（Pure Imaginary Number）

### 1.4 手算：认识复数

来写几个简单的复数，熟悉一下记号：

| 复数 | 实部 $\operatorname{Re}$ | 虚部 $\operatorname{Im}$ | 是实数？ | 是纯虚数？ |
|---|---|---|---|---|
| $2 + j3$ | 2 | 3 | 否 | 否 |
| $-1.5 + j0$ | -1.5 | 0 | 是 | 否 |
| $0 + j5$ | 0 | 5 | 否 | 是 |
| $4.2 - j2.7$ | 4.2 | -2.7 | 否 | 否 |
| $j$ | 0 | 1 | 否 | 是 |

注意最后一行的虚部是 1，不是 $j$。虚部**总是实数**，$j$ 只是写在后面表示"这个数乘以了虚数单位"。

### 1.5 常见误区

- **❌ 误区**："虚数是假的，不存在"。
- **✓ 澄清**："虚"只是名字取得不好（是笛卡尔起的，他本人也不太接受这东西）。复数就像二维坐标 $(a, b)$ 一样"真实"。你在纸上画一个点 (3, 4)，它存在吗？当然存在。复数 $3 + j4$ 就是同一个东西，只是写法和运算规则不同。如果一个平面上的点是"真的"，那复数也是真的。

- **❌ 误区**："$j = \sqrt{-1}$"
- **✓ 澄清**：严格来说应该写 $j^2 = -1$。写成 $\sqrt{-1}$ 会让你忍不住用 $\sqrt{a}\sqrt{b} = \sqrt{ab}$，但这个公式对负数不成立（否则会推出 $1 = -1$ 的矛盾）。工程上大家都这么写，但你要知道这只是一种记法，代数运算时要小心。

- **❌ 误区**：虚部就是带 $j$ 的那一部分。
- **✓ 澄清**：虚部是 $b$（一个实数），不是 $jb$。$\operatorname{Im}(3+j4) = 4$，不是 $j4$。写成 `Im(z)` 时总是拿出一个实数。

### 1.6 应用连接：为什么电路需要复数

在后面学交流电时，你会发现电压和电流不再是固定的数字，而是随时间振荡的正弦波。描述一个正弦波需要两个信息：**振幅**（波有多高）和**相位**（波从哪个位置开始）。这两个信息恰好对应复数的**模**和**幅角**。用复数来表示正弦信号，能让"波"的运算变成复数的运算——加法、乘法、甚至用指数形式简化微积分。这就是为什么整个交流电路分析都建立在复数之上。

---

## 2. 复数的几何直觉

### 2.1 生活例子：箭头指路

你小时候可能玩过"寻宝游戏"：先向东走 5 步，再向北走 3 步，就能找到宝藏。你也可以直接指着一个方向说："往那个方向走约 5.8 步"。这两种说法描述的是同一条路径。

复数也一样：$3 + j4$（直角坐标）和"模为 5、角度 53°"（极坐标）说的是同一个东西。前者像"分别往东和往北走多远"，后者像"朝着一个方向走多远"。

### 2.2 直观理解：复平面

把数轴水平放置，再在中间竖着画一条垂直于它的数轴：

- **水平轴（实轴）**：表示实数，也就是 $a$ 的值
- **竖直轴（虚轴）**：表示虚数（$j$ 的倍数），也就是 $b$ 的值

这个平面叫做复平面（Complex Plane）或 Argand 图。

每个复数 $z = a + jb$ 就是复平面上坐标为 $(a, b)$ 的一个点。

反过来，你也可以把它看成从原点 $(0, 0)$ 指向 $(a, b)$ 的一支**箭头**（向量）。箭头的长度就是复数的大小，箭头的方向就是复数的角度。

两种视角（点 vs 箭头）都对——加法的时候用箭头视角方便，乘法的时候用角度和长度方便。

### 2.3 形式化定义：复数运算的几何意义

**加法：向量相加**

两个复数相加，就是它们对应的箭头**首尾相接**，或者用平行四边形法则。

$$
z_1 + z_2 = (a_1 + jb_1) + (a_2 + jb_2) = (a_1 + a_2) + j(b_1 + b_2)
$$

就是实部和实部加，虚部和虚部加——跟二维向量完全一样。

**乘法：旋转 + 缩放**

两个复数相乘，结果是 **模相乘，角度相加**。

这个性质在"第 4 节 极坐标"中会详细展开。现在先记住这个直觉：乘以一个复数，相当于把箭头**拉长或缩短，再旋转一个角度**。乘 $j$ 就是旋转 90°（因为 $j$ 的模是 1，角度是 90°）。

### 2.4 手算：复平面上的点

标出以下复数在复平面上的位置：

| 复数 | 实部 | 虚部 | 在第几象限？ |
|---|---|---|---|
| $2 + j3$ | 2 | 3 | 第一象限（右上） |
| $-1 + j4$ | -1 | 4 | 第二象限（左上） |
| $-3 - j2$ | -3 | -2 | 第三象限（左下） |
| $4 - j1$ | 4 | -1 | 第四象限（右下） |
| $j2$ | 0 | 2 | 虚轴正半轴 |
| $-5$ | -5 | 0 | 实轴负半轴 |

### 2.5 Python 验证：复平面可视化

```python
# 复平面上的点可视化
import numpy as np
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义几个复数
points = [2+3j, -1+4j, -3-2j, 4-1j, 2j, -5+0j]
labels = ['2+j3', '-1+j4', '-3-j2', '4-j1', 'j2', '-5']

fig, ax = plt.subplots(figsize=(8, 8))

# 画坐标轴
ax.axhline(y=0, color='gray', linewidth=0.8)
ax.axvline(x=0, color='gray', linewidth=0.8)

# 画每个复数从原点出发的箭头
for z, label in zip(points, labels):
    ax.arrow(0, 0, z.real, z.imag,
             head_width=0.15, head_length=0.15,
             fc='steelblue', ec='steelblue', alpha=0.7,
             length_includes_head=True)
    ax.scatter(z.real, z.imag, color='darkred', s=50, zorder=5)
    ax.annotate(label, (z.real, z.imag),
                textcoords="offset points", xytext=(8, 8),
                fontsize=11)

# 添加虚线表示实部和虚部
for z in points[:4]:
    ax.plot([z.real, z.real], [0, z.imag], 'gray', linestyle=':', alpha=0.4)
    ax.plot([0, z.real], [z.imag, z.imag], 'gray', linestyle=':', alpha=0.4)

ax.set_xlabel('实轴（Real Axis）', fontsize=13)
ax.set_ylabel('虚轴（Imaginary Axis）', fontsize=13)
ax.set_title('复平面上的点与箭头', fontsize=15)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**预期输出**：一张图，展示 6 个复数在复平面上的箭头。实轴是横轴，虚轴是纵轴。$2+j3$ 在右上，$-3-j2$ 在左下。

### 2.6 常见误区

- **❌ 误区**："复数就是二维向量，没必要新造一个概念"。
- **✓ 澄清**：作为"点"或"箭头"，复数和二维向量确实很像。但复数多了一样东西：**乘法**。二维向量没有自然的乘法（点积和叉积都退化了维度信息），而复数的乘法有明确的几何意义（旋转+缩放）。这个乘法是复数在电路分析中不可替代的原因。

- **❌ 误区**：实轴和虚轴的区别只是"方向不同"。
- **✓ 澄清**：不是方向不同那么简单。$j$ 的关键性质是 $j^2 = -1$，这意味着"乘以 $j$ 两次 = 乘以 $-1$ = 旋转 180°"。所以乘以 $j$ 一次 = 旋转 90°。普通的 $(x,y)$ 坐标系没有这种旋转和平方的对应关系。

### 2.7 应用连接：向量的傅里叶变换

你在信号处理中会学到傅里叶变换——把一个复杂信号分解成多个旋转的复数箭头。每个频率分量对应一个以特定速度旋转的向量，复数的**模**就是该频率成分的强度，**角度**就是该频率成分的相位。用复数，一个箭头就能同时表达强度和相位——实函数需要两个值。

---

## 3. 复数运算

### 3.1 生活例子：拼乐高

复数运算就像拼乐高：你把每个复数拆成实部和虚部两块，然后分别处理同类型的块。加法就是把实部的乐高和实部的乐高拼一起，虚部的乐高和虚部的乐高拼一起。乘法多一步：$j$ 和 $j$ 碰到一起就变成 $-1$（刚好嵌合）。

### 3.2 直观理解

**加法/减法**：跟二维向量的加法和减法完全一样。没有新的东西要学。

**乘法**：唯一的新规则就是 $j^2 = -1$。展开括号后，只要 $j^2$ 出现，就换成 $-1$。其余步骤跟多项式乘法一样。

**除法**：利用"共轭复数"的技巧把分母变成实数。思路跟初中学的"分母有理化"完全一样（比如 $\frac{1}{\sqrt{2}}$ 分子分母同乘 $\sqrt{2}$）。

### 3.3 形式化定义

设 $z_1 = a_1 + j b_1$，$z_2 = a_2 + j b_2$。

**加法**：

$$
z_1 + z_2 = (a_1 + a_2) + j(b_1 + b_2)
$$

**减法**：

$$
z_1 - z_2 = (a_1 - a_2) + j(b_1 - b_2)
$$

**乘法**（展开括号，用 $j^2 = -1$ 化简）：

$$
\begin{aligned}
z_1 \cdot z_2 &= (a_1 + j b_1)(a_2 + j b_2) \\
&= a_1 a_2 + j a_1 b_2 + j a_2 b_1 + j^2 b_1 b_2 \\
&= a_1 a_2 + j(a_1 b_2 + a_2 b_1) - b_1 b_2 \\
&= (a_1 a_2 - b_1 b_2) + j(a_1 b_2 + a_2 b_1)
\end{aligned}
$$

**共轭复数**（Complex Conjugate）：实部不变，虚部变号。

$$
\overline{z} = a - jb \quad \text{（也写作 } z^* \text{）}
$$

**除法**（分子分母同乘分母的共轭）：

$$
\frac{z_1}{z_2} = \frac{a_1 + j b_1}{a_2 + j b_2}
= \frac{(a_1 + j b_1)(a_2 - j b_2)}{(a_2 + j b_2)(a_2 - j b_2)}
= \frac{(a_1 a_2 + b_1 b_2) + j(a_2 b_1 - a_1 b_2)}{a_2^2 + b_2^2}
$$

分母 $a_2^2 + b_2^2$ 是一个正实数（除非 $z_2 = 0$）。这正是我们想要的效果：把分母从复数变成实数。

### 3.4 手算：8 道复数运算

以下所有计算都展示完整步骤。

---

**题 1：加法** — $(3 + j4) + (1 + j2)$

$$
\begin{aligned}
(3 + j4) + (1 + j2) &= (3 + 1) + j(4 + 2) \\
&= 4 + j6
\end{aligned}
$$

实部加实部，虚部加虚部。一目了然。

---

**题 2：减法** — $(5 + j3) - (2 + j7)$

$$
\begin{aligned}
(5 + j3) - (2 + j7) &= (5 - 2) + j(3 - 7) \\
&= 3 - j4
\end{aligned}
$$

---

**题 3：乘法（简单）** — $(2 + j3)(1 + j2)$

$$
\begin{aligned}
(2 + j3)(1 + j2) &= 2 \cdot 1 + 2 \cdot j2 + j3 \cdot 1 + j3 \cdot j2 \\
&= 2 + j4 + j3 + j^2 \cdot 6 \\
&= 2 + j7 + (-1) \cdot 6 \quad \text{←  } j^2 = -1 \\
&= 2 + j7 - 6 \\
&= -4 + j7
\end{aligned}
$$

---

**题 4：乘法（含负数虚部）** — $(1 - j2)(3 + j4)$

$$
\begin{aligned}
(1 - j2)(3 + j4) &= 1 \cdot 3 + 1 \cdot j4 + (-j2) \cdot 3 + (-j2) \cdot j4 \\
&= 3 + j4 - j6 - j^2 \cdot 8 \\
&= 3 + j4 - j6 - (-1) \cdot 8 \quad \text{←  注意：}-j^2 = -(-1) = +1 \\
&= 3 - j2 + 8 \\
&= 11 - j2
\end{aligned}
$$

**关键细节**：$(-j2) \cdot j4 = -j^2 \cdot 8 = -(-1) \cdot 8 = +8$。很多人会在这步出错，把 $-j^2$ 当成 $-1$。

---

**题 5：共轭** — 求 $z = 3 + j4$ 和 $z = -2 - j5$ 的共轭

| 复数 $z$ | 共轭 $\overline{z}$ |
|---|---|
| $3 + j4$ | $3 - j4$ |
| $-2 - j5$ | $-2 + j5$ |

共轭就是在复平面上**关于实轴的镜像**。

---

**题 6：除法（基础）** — $\frac{3 + j4}{1 + j2}$

$$
\begin{aligned}
\frac{3 + j4}{1 + j2}
&= \frac{(3 + j4)(1 - j2)}{(1 + j2)(1 - j2)} \quad \text{← 分子分母同乘 } (1 - j2) \\[6pt]
&= \frac{3 \cdot 1 + 3 \cdot (-j2) + j4 \cdot 1 + j4 \cdot (-j2)}{1^2 + 2^2} \quad \text{← 分母用平方和公式} \\[6pt]
&= \frac{3 - j6 + j4 - j^2 \cdot 8}{1 + 4} \\[6pt]
&= \frac{3 - j2 - (-1) \cdot 8}{5} \quad \text{← } j^2 = -1 \\[6pt]
&= \frac{3 - j2 + 8}{5} \\[6pt]
&= \frac{11 - j2}{5} \\[6pt]
&= 2.2 - j0.4
\end{aligned}
$$

---

**题 7：除法（纯虚数分母）** — $\frac{4 + j6}{j2}$

这道题有捷径：分子分母同时除以 $j$。

$$
\begin{aligned}
\frac{4 + j6}{j2}
&= \frac{4/j + j6/j}{j2/j} \\
&= \frac{-j4 + 6}{2} \quad \text{← } 1/j = -j, \quad j/j = 1 \\
&= \frac{6 - j4}{2} \\
&= 3 - j2
\end{aligned}
$$

用共轭法验证：

$$
\begin{aligned}
\frac{4 + j6}{j2}
&= \frac{(4 + j6)(-j2)}{(j2)(-j2)} \\
&= \frac{-j8 - j^2 12}{-j^2 4} \\
&= \frac{-j8 + 12}{4} \\
&= 3 - j2 \quad \checkmark
\end{aligned}
$$

---

**题 8：综合运算** — $\frac{(1+j)(2-j)}{3+j4}$

分步来：

**第 1 步**：算分子 $(1+j)(2-j)$。

$$
\begin{aligned}
(1+j)(2-j) &= 1 \cdot 2 + 1 \cdot (-j) + j \cdot 2 + j \cdot (-j) \\
&= 2 - j + j2 - j^2 \\
&= 2 + j - (-1) \\
&= 3 + j
\end{aligned}
$$

**第 2 步**：算 $\frac{3 + j}{3 + j4}$。

$$
\begin{aligned}
\frac{3 + j}{3 + j4}
&= \frac{(3 + j)(3 - j4)}{(3 + j4)(3 - j4)} \\
&= \frac{9 - j12 + j3 - j^2 4}{9 + 16} \\
&= \frac{9 - j9 + 4}{25} \\
&= \frac{13 - j9}{25} \\
&= 0.52 - j0.36
\end{aligned}
$$

### 3.5 Python 验证：全部 8 道题

```python
# 复数运算验证
# Python 原生支持复数：用 j 作为虚数单位（注意：写 1j 而不是 j）

print("=" * 55)
print("复数运算验证")
print("=" * 55)

# ---- 题 1: 加法 ----
z1 = 3 + 4j
z2 = 1 + 2j
result_1 = z1 + z2
print(f"\n题 1: ({z1}) + ({z2}) = {result_1}")
print(f"  实部: {result_1.real}, 虚部: {result_1.imag}")
print(f"  预期: 4 + j6")

# ---- 题 2: 减法 ----
z3 = 5 + 3j
z4 = 2 + 7j
result_2 = z3 - z4
print(f"\n题 2: ({z3}) - ({z4}) = {result_2}")
print(f"  预期: 3 - j4")

# ---- 题 3: 乘法(简单) ----
z5 = 2 + 3j
z6 = 1 + 2j
result_3 = z5 * z6
print(f"\n题 3: ({z5}) * ({z6}) = {result_3}")
print(f"  预期: -4 + j7")

# ---- 题 4: 乘法(含负数虚部) ----
z7 = 1 - 2j
z8 = 3 + 4j
result_4 = z7 * z8
print(f"\n题 4: ({z7}) * ({z8}) = {result_4}")
print(f"  预期: 11 - j2")

# ---- 题 5: 共轭 ----
z9 = 3 + 4j
z10 = -2 - 5j
print(f"\n题 5: 共轭")
print(f"  conj({z9}) = {z9.conjugate()}")
print(f"  conj({z10}) = {z10.conjugate()}")

# ---- 题 6: 除法(基础) ----
z11 = 3 + 4j
z12 = 1 + 2j
result_6 = z11 / z12
print(f"\n题 6: ({z11}) / ({z12}) = {result_6}")
print(f"  预期: 2.2 - j0.4")

# ---- 题 7: 除法(纯虚数分母) ----
z13 = 4 + 6j
z14 = 2j
result_7 = z13 / z14
print(f"\n题 7: ({z13}) / ({z14}) = {result_7}")
print(f"  预期: 3 - j2")

# ---- 题 8: 综合运算 ----
z15 = 1 + 1j
z16 = 2 - 1j
z17 = 3 + 4j
result_8 = (z15 * z16) / z17
print(f"\n题 8: (({z15})*({z16})) / ({z17}) = {result_8}")
print(f"  预期: 0.52 - j0.36")

# 验证：直接比较
assert abs((result_1.real - 4) + 1j*(result_1.imag - 6)) < 1e-10, "题1错误"
assert abs((result_3.real - (-4)) + 1j*(result_3.imag - 7)) < 1e-10, "题3错误"
assert abs((result_4.real - 11) + 1j*(result_4.imag - (-2))) < 1e-10, "题4错误"
assert abs((result_6.real - 2.2) + 1j*(result_6.imag - (-0.4))) < 1e-10, "题6错误"
assert abs((result_7.real - 3) + 1j*(result_7.imag - (-2))) < 1e-10, "题7错误"
assert abs((result_8.real - 0.52) + 1j*(result_8.imag - (-0.36))) < 1e-10, "题8错误"

print(f"\n{'='*55}")
print("✓ 全部 8 道题通过验证！")
print(f"{'='*55}")
```

**预期输出**：

```
=======================================================
复数运算验证
=======================================================

题 1: (3+4j) + (1+2j) = (4+6j)
  实部: 4.0, 虚部: 6.0
  预期: 4 + j6

题 2: (5+3j) - (2+7j) = (3-4j)
  预期: 3 - j4

题 3: (2+3j) * (1+2j) = (-4+7j)
  预期: -4 + j7

题 4: (1-2j) * (3+4j) = (11-2j)
  预期: 11 - j2

题 5: 共轭
  conj((3+4j)) = (3-4j)
  conj((-2-5j)) = (-2+5j)

题 6: (3+4j) / (1+2j) = (2.2-0.4j)
  预期: 2.2 - j0.4

题 7: (4+6j) / 2j = (3-2j)
  预期: 3 - j2

题 8: ((1+1j)*(2-1j)) / (3+4j) = (0.52-0.36j)
  预期: 0.52 - j0.36

=======================================================
✓ 全部 8 道题通过验证！
=======================================================
```

### 3.6 常见误区

- **❌ 误区**：$(a+jb)^2 = a^2 + b^2$
- **✓ 澄清**：这是把复数乘法和模的平方搞混了。$(a+jb)^2 = a^2 - b^2 + j2ab$（展开后用 $j^2 = -1$）。而 $|a+jb|^2 = a^2 + b^2$（模的平方，没有 $j$ 项）。两者完全不同。

- **❌ 误区**：除法就是"实部除实部、虚部除虚部"。
- **✓ 澄清**：$\frac{a+jb}{c+jd} \neq \frac{a}{c} + j\frac{b}{d}$。除法要乘共轭，一通运算后实部和虚部都是混在一起的。如果你直接除，结果完全不对。试试 $\frac{1+j}{1+j}$：正确结果是 $1$（任何数除以自己等于 1），但"分别除"法会给你 $\frac{1}{1} + j\frac{1}{1} = 1+j$，错了。

- **❌ 误区**：共轭只是把 $j$ 变成 $-j$，没什么大用。
- **✓ 澄清**：共轭的用处极大。$z \cdot \overline{z} = |z|^2$（模的平方，一个实数），这是复数的"去虚化"工具。后面你会看到，共轭在计算功率、设计滤波器、处理信号时无处不在。

### 3.7 应用连接：阻抗计算

在交流电路中，电阻（Resistance $R$）、感抗（Inductive Reactance $j\omega L$）、容抗（Capacitive Reactance $1/(j\omega C)$）就是实部和虚部的关系。算总阻抗 $Z$ 的过程跟复数除法一模一样——都是实部虚部混在一起算，不能拆开。你在第 3.4 节练的手算，本质上就是在练"电路阻抗计算"。

---

## 4. 极坐标表示

### 4.1 生活例子：导航的两种说法

你开车去一个地方，可以这样描述路线："先向东开 3 公里，再向北开 4 公里"。

也可以说："往东北方向开 5 公里"。

第一种是**直角坐标**（$(3, 4)$ 或 $3 + j4$），第二种是**极坐标**（$5 \angle 53^\circ$）。两者说的是同一段路。

在电路分析中，极坐标好用的原因是：信号的幅度（波的高度）对应极坐标的**模**，信号的相位（波偏移了多少）对应极坐标的**角度**。用极坐标，一个复数就直接告诉你"这个信号有多大、偏了多少"。

### 4.2 直观理解：模和角度

想象你从原点拉一根皮筋到复平面上的点 $(a, b)$。

- **模（Magnitude）**：皮筋的长度。记作 $|z|$ 或 $r$。
- **幅角（Argument / Angle）**：皮筋与正实轴的夹角。记作 $\arg(z)$ 或 $\theta$。

模永远是正实数（除非 $z = 0$，此时模为 0）。幅角通常从正实轴逆时针为正。

一个复数可以用极坐标写成：

$$
z = r \angle \theta
$$

读作"$r$ at angle $\theta$"。

### 4.3 形式化定义：坐标转换

**直角坐标 → 极坐标**：

$$
\begin{aligned}
r &= \sqrt{a^2 + b^2} \quad \text{（勾股定理——皮筋的长度）} \\[4pt]
\theta &= \operatorname{atan2}(b, a) \quad \text{（反正切，自动判象限）}
\end{aligned}
$$

$\operatorname{atan2}(b, a)$ 跟普通的 $\arctan(b/a)$ 的区别在于：$\arctan$ 只能返回 $(-90^\circ, 90^\circ)$ 之间的角度（不知道 $(a,b)$ 在第几象限），而 $\operatorname{atan2}(b, a)$ 根据 $a$ 和 $b$ 的符号给出正确的 $(-180^\circ, 180^\circ]$ 角度。

**极坐标 → 直角坐标**：

$$
\begin{aligned}
a &= r \cos\theta \\
b &= r \sin\theta
\end{aligned}
$$

所以：

$$
z = r(\cos\theta + j\sin\theta)
$$

### 4.4 手算：坐标转换练习

**题 1**：$z = 3 + j4$，转为极坐标。

$$
\begin{aligned}
r &= \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5 \\[4pt]
\theta &= \operatorname{atan2}(4, 3) = \arctan\left(\frac{4}{3}\right) \approx 53.13^\circ
\end{aligned}
$$

因为 (3, 4) 在第一象限，$\arctan(4/3)$ 直接就是最终答案。

结果：$z = 5 \angle 53.13^\circ$

---

**题 2**：$z = -1 + j1$，转为极坐标。

$$
\begin{aligned}
r &= \sqrt{(-1)^2 + 1^2} = \sqrt{1 + 1} = \sqrt{2} \approx 1.414 \\[4pt]
\theta &= \operatorname{atan2}(1, -1)
\end{aligned}
$$

在第二象限（$a < 0, b > 0$），角度是 $180^\circ - 45^\circ = 135^\circ$（或 $\frac{3\pi}{4}$ 弧度）。

结果：$z = \sqrt{2} \angle 135^\circ$ 或 $1.414 \angle 135^\circ$

---

**题 3**：$z = -3 - j4$，转为极坐标。

$$
\begin{aligned}
r &= \sqrt{(-3)^2 + (-4)^2} = \sqrt{9 + 16} = 5 \\[4pt]
\theta &= \operatorname{atan2}(-4, -3)
\end{aligned}
$$

在第三象限（$a < 0, b < 0$）。$\arctan(-4/-3) = \arctan(4/3) \approx 53.13^\circ$，但因为在第三象限，实际角度是 $-180^\circ + 53.13^\circ = -126.87^\circ$（或 $180^\circ + 53.13^\circ = 233.13^\circ$，两种写法都可以。工程上习惯用 $(-180^\circ, 180^\circ]$ 范围）。

结果：$z = 5 \angle -126.87^\circ$ 或 $5 \angle 233.13^\circ$

---

**题 4**：$z = 4 - j3$，转为极坐标。

$$
\begin{aligned}
r &= \sqrt{4^2 + (-3)^2} = \sqrt{16 + 9} = 5 \\[4pt]
\theta &= \operatorname{atan2}(-3, 4) = \arctan(-3/4) \approx -36.87^\circ
\end{aligned}
$$

第四象限，$\arctan(-3/4)$ 直接给出负角度。

结果：$z = 5 \angle -36.87^\circ$

---

**题 5**：$z = 6 \angle 30^\circ$，转为直角坐标。

$$
\begin{aligned}
a &= 6 \cos 30^\circ = 6 \times \frac{\sqrt{3}}{2} = 3\sqrt{3} \approx 5.196 \\
b &= 6 \sin 30^\circ = 6 \times \frac{1}{2} = 3
\end{aligned}
$$

结果：$z = 3\sqrt{3} + j3 \approx 5.196 + j3$

---

**题 6**：$z = 4 \angle 150^\circ$（第二象限），转为直角坐标。

$$
\begin{aligned}
a &= 4 \cos 150^\circ = 4 \times \left(-\frac{\sqrt{3}}{2}\right) = -2\sqrt{3} \approx -3.464 \\
b &= 4 \sin 150^\circ = 4 \times \frac{1}{2} = 2
\end{aligned}
$$

结果：$z = -2\sqrt{3} + j2 \approx -3.464 + j2$

---

**题 7**：$z = 12 \angle -60^\circ$（第四象限），转为直角坐标。

$$
\begin{aligned}
a &= 12 \cos(-60^\circ) = 12 \times \frac{1}{2} = 6 \\
b &= 12 \sin(-60^\circ) = 12 \times \left(-\frac{\sqrt{3}}{2}\right) = -6\sqrt{3} \approx -10.392
\end{aligned}
$$

结果：$z = 6 - j6\sqrt{3} \approx 6 - j10.392$

---

**汇总表**：

| 直角坐标 | 极坐标（模，角度） | 第几象限 |
|---|---|---|
| $3 + j4$ | $5 \angle 53.13^\circ$ | 一 |
| $-1 + j1$ | $\sqrt{2} \angle 135^\circ$ | 二 |
| $-3 - j4$ | $5 \angle -126.87^\circ$ | 三 |
| $4 - j3$ | $5 \angle -36.87^\circ$ | 四 |
| $5.196 + j3$ | $6 \angle 30^\circ$ | 一 |
| $-3.464 + j2$ | $4 \angle 150^\circ$ | 二 |
| $6 - j10.392$ | $12 \angle -60^\circ$ | 四 |

### 4.5 Python 验证：坐标转换

```python
# 直角坐标 ↔ 极坐标转换验证
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def rect_to_polar(z):
    """直角坐标 → 极坐标：返回 (模, 角度_弧度, 角度_度)"""
    r = np.abs(z)
    theta_rad = np.angle(z)  # 返回 (-π, π]
    theta_deg = np.degrees(theta_rad)
    return r, theta_rad, theta_deg

def polar_to_rect(r, theta_deg):
    """极坐标 → 直角坐标：输入 (模, 角度_度)，返回复数"""
    theta_rad = np.radians(theta_deg)
    return r * (np.cos(theta_rad) + 1j * np.sin(theta_rad))

# ====== 验证题 1-4: 直角 → 极坐标 ======
print("直角坐标 → 极坐标 转换验证")
print("-" * 60)
print(f"{'直角坐标':<15}{'模 r':<12}{'角度(度)':<15}{'角度(弧度)':<15}")
print("-" * 60)

test_rect = [
    (3+4j,    "3+j4"),
    (-1+1j,   "-1+j1"),
    (-3-4j,   "-3-j4"),
    (4-3j,    "4-j3"),
]

for z, label in test_rect:
    r, theta_rad, theta_deg = rect_to_polar(z)
    print(f"{label:<15}{r:<12.4f}{theta_deg:<15.2f}{theta_rad:<15.4f}")

# ====== 验证题 5-7: 极坐标 → 直角坐标 ======
print(f"\n极坐标 → 直角坐标 转换验证")
print("-" * 60)
print(f"{'极坐标':<18}{'直角坐标':<18}{'实部':<12}{'虚部':<12}")
print("-" * 60)

test_polar = [
    (6, 30,    "6∠30°"),
    (4, 150,   "4∠150°"),
    (12, -60,  "12∠-60°"),
]

for r, theta_deg, label in test_polar:
    z = polar_to_rect(r, theta_deg)
    print(f"{label:<18}{str(z):<18}{z.real:<12.4f}{z.imag:<12.4f}")

# ====== 可视化：两种坐标的对比 ======
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 准备数据
all_z = [3+4j, -1+1j, -3-4j, 4-3j]
all_labels = ['3+j4', '-1+j1', '-3-j4', '4-j3']

for ax, title in zip(axes, ['直角坐标视角 (a, b)', '极坐标视角 (r, θ)']):
    ax.axhline(y=0, color='gray', linewidth=0.7)
    ax.axvline(x=0, color='gray', linewidth=0.7)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('实轴')
    ax.set_ylabel('虚轴')

for i, (z, label) in enumerate(zip(all_z, all_labels)):
    r, _, theta_deg = rect_to_polar(z)
    color = plt.cm.tab10(i)

    # 左图：直角坐标——画虚线到坐标轴
    axes[0].arrow(0, 0, z.real, z.imag,
                  head_width=0.2, head_length=0.2,
                  fc=color, ec=color, alpha=0.6,
                  length_includes_head=True)
    axes[0].plot([z.real, z.real], [0, z.imag], ':', color=color, alpha=0.3)
    axes[0].plot([0, z.real], [z.imag, z.imag], ':', color=color, alpha=0.3)
    axes[0].scatter(z.real, z.imag, color=color, s=60)
    axes[0].annotate(f'{label}\n实部={z.real}, 虚部={z.imag}',
                     (z.real, z.imag),
                     textcoords="offset points", xytext=(10, 10), fontsize=9)

    # 右图：极坐标——标出模和角度
    axes[1].arrow(0, 0, z.real, z.imag,
                  head_width=0.2, head_length=0.2,
                  fc=color, ec=color, alpha=0.6,
                  length_includes_head=True)
    axes[1].scatter(z.real, z.imag, color=color, s=60)
    axes[1].annotate(f'{label}\nr={r:.2f}, θ={theta_deg:.1f}°',
                     (z.real, z.imag),
                     textcoords="offset points", xytext=(10, 10), fontsize=9)

plt.tight_layout()
plt.show()
```

**预期输出**：两张并排图。左图展示直角坐标表示（实部虚部标注），右图展示极坐标表示（模和角度标注）。同一组点，两种描述方式。

### 4.6 常见误区

- **❌ 误区**：模就是 $\sqrt{a^2 + b^2}$，幅角就是 $\arctan(b/a)$，直接用计算器的 $\tan^{-1}$ 就行。
- **✓ 澄清**：$\arctan(b/a)$ 只对第一、第四象限正确。如果 $z = -1 - j1$，$b/a = (-1)/(-1) = 1$，$\arctan(1) = 45^\circ$，但实际角度是 $-135^\circ$（或 $225^\circ$）。一定要用 $\operatorname{atan2}(b, a)$！

- **❌ 误区**：角度写多少都行，$30^\circ$ 和 $390^\circ$ 是一回事。
- **✓ 澄清**：数学上确实 $30^\circ$ 和 $30^\circ + 360^\circ = 390^\circ$ 指向同一个方向。但工程上通常把幅角取在 $(-180^\circ, 180^\circ]$ 范围内，方便比较。两个信号如果角度差了 $360^\circ$，虽然方向相同，但它们的时间历程差了一个整周期，不能随便互换。

### 4.7 应用连接：相位与阻抗

在交流电路中，电压和电流都是正弦波。正弦波可以写成 $V_m \cos(\omega t + \phi)$，其中 $V_m$ 是振幅，$\phi$ 是相位。极坐标 $V_m \angle \phi$ 正好把这两个信息打包成一个复数。接下来要看的相量（phasor）就是干这个的：把正弦波的振幅和相位存成一个极坐标复数。

---

## 5. 欧拉公式

### 5.1 生活例子：钟表指针

你看钟表时，秒针在转。秒针的尖端在做什么？它在以钟表中心为圆心，按固定速度画圆。

如果你把秒针的投影打到墙上（从侧面看），墙上那个影子的运动是：一会儿在最右边，一会儿到最左边，来回晃——这就是正弦波。

秒针本身 = 旋转的"复数箭头"，墙上的投影 = 正弦波。这两者之间的关系，就是欧拉公式要告诉你的。

### 5.2 直观理解：泰勒展开的直觉

先回顾一下泰勒展开的核心思想：很多复杂的函数可以用多项式来逼近。$\sin x$、$\cos x$、$e^x$ 都有泰勒展开：

$$
\begin{aligned}
e^x &= 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \frac{x^5}{5!} + \dots \\[4pt]
\cos x &= 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \dots \\[4pt]
\sin x &= x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \dots
\end{aligned}
$$

现在把 $e^x$ 中的 $x$ 换成 $j\theta$（记住 $j^2 = -1$，$j^3 = -j$，$j^4 = 1$，然后循环）：

$$
\begin{aligned}
e^{j\theta}
&= 1 + j\theta + \frac{(j\theta)^2}{2!} + \frac{(j\theta)^3}{3!} + \frac{(j\theta)^4}{4!} + \frac{(j\theta)^5}{5!} + \dots \\[4pt]
&= 1 + j\theta + \frac{j^2\theta^2}{2!} + \frac{j^3\theta^3}{3!} + \frac{j^4\theta^4}{4!} + \frac{j^5\theta^5}{5!} + \dots \\[4pt]
&= 1 + j\theta + \frac{(-1)\theta^2}{2!} + \frac{(-j)\theta^3}{3!} + \frac{(1)\theta^4}{4!} + \frac{(j)\theta^5}{5!} + \dots \\[4pt]
&= \left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \dots\right) + j\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \dots\right)
\end{aligned}
$$

括号里的第一项恰好是 $\cos\theta$ 的泰勒展开，第二项恰好是 $\sin\theta$ 的泰勒展开！所以：

$$
e^{j\theta} = \cos\theta + j\sin\theta
$$

这就是**欧拉公式（Euler's Formula）**。它把指数函数和三角函数连了起来。

### 5.3 形式化定义

**欧拉公式**：

$$
\boxed{e^{j\theta} = \cos\theta + j\sin\theta}
$$

由此可以推导出两个非常有用的恒等式：

$$
\cos\theta = \frac{e^{j\theta} + e^{-j\theta}}{2}, \qquad
\sin\theta = \frac{e^{j\theta} - e^{-j\theta}}{j2}
$$

**极坐标与欧拉公式的关系**：

任何一个复数 $z = r(\cos\theta + j\sin\theta)$ 都可以写成：

$$
z = r e^{j\theta}
$$

这比 $r \angle \theta$ 更好用，因为在指数形式下乘除法直接变成了指数的加减——这正是第 3 节中发现但又觉得繁琐的"模相乘、角度相加"规律的自然体现：

$$
z_1 \cdot z_2 = (r_1 e^{j\theta_1})(r_2 e^{j\theta_2}) = r_1 r_2 e^{j(\theta_1 + \theta_2)}
$$

### 5.4 几何意义：单位圆上的旋转

欧拉公式 $e^{j\theta}$ 的几何意义非常清晰：

- $e^{j\theta}$ 在复平面上就是**模为 1、角度为 $\theta$ 的点**。它落在单位圆（半径为 1 的圆）上。
- 当 $\theta$ 从 0 变到 $2\pi$（即 0° 到 360°），$e^{j\theta}$ 沿单位圆走了一整圈。
- 乘以 $e^{j\theta}$ 相当于**不改变长度、只旋转角度** $\theta$。

特别地：

| $\theta$ | $e^{j\theta}$ | 含义 |
|---|---|---|
| $0$ | $e^{j0} = 1$ | 回到实数 1 |
| $\pi/2$ (90°) | $e^{j\pi/2} = j$ | 旋转 90° |
| $\pi$ (180°) | $e^{j\pi} = -1$ | 旋转 180° |
| $3\pi/2$ (270°) | $e^{j3\pi/2} = -j$ | 旋转 270° |
| $2\pi$ (360°) | $e^{j2\pi} = 1$ | 旋转一圈，回到起点 |

当 $\theta = \pi$ 时，$e^{j\pi} = -1$，移项得：

$$
e^{j\pi} + 1 = 0
$$

这个等式被很多人称为"最优美的数学公式"。它只用了一次加法、一次乘法、一次指数，就把数学中五个最重要的常数（$e$、$j$、$\pi$、$1$、$0$）全部串联到了一起。

### 5.5 手算：用欧拉公式化简复数

**题 1**：把 $5 e^{j 53.13^\circ}$ 转回直角坐标。

$$
\begin{aligned}
5 e^{j 53.13^\circ}
&= 5[\cos(53.13^\circ) + j\sin(53.13^\circ)] \\
&= 5[0.6 + j0.8] \\
&= 3 + j4
\end{aligned}
$$

这正是第 4.4 节题 1 的逆过程。

---

**题 2**：计算 $2 e^{j30^\circ} \cdot 3 e^{j45^\circ}$。

$$
\begin{aligned}
2 e^{j30^\circ} \cdot 3 e^{j45^\circ}
&= (2 \cdot 3) \cdot e^{j(30^\circ + 45^\circ)} \\
&= 6 \cdot e^{j75^\circ} \quad \text{← 模相乘，角度相加}
\end{aligned}
$$

---

**题 3**：计算 $\frac{10 e^{j60^\circ}}{2 e^{j15^\circ}}$。

$$
\begin{aligned}
\frac{10 e^{j60^\circ}}{2 e^{j15^\circ}}
&= \frac{10}{2} \cdot e^{j(60^\circ - 15^\circ)} \\
&= 5 \cdot e^{j45^\circ} \quad \text{← 模相除，角度相减}
\end{aligned}
$$

---

**题 4**：化简 $(1 + j)^4$，用直角坐标和极坐标两种方法。

**方法一（直角坐标，硬展开）**：

$$
\begin{aligned}
(1+j)^2 &= 1 + j2 + j^2 = 1 + j2 - 1 = j2 \\
(1+j)^4 &= (j2)^2 = j^2 \cdot 4 = -4
\end{aligned}
$$

**方法二（极坐标，用欧拉公式）**：

先转极坐标：$1+j = \sqrt{2} e^{j45^\circ}$

$$
(1+j)^4 = (\sqrt{2} e^{j45^\circ})^4 = (\sqrt{2})^4 \cdot e^{j \cdot 4 \cdot 45^\circ} = 4 \cdot e^{j180^\circ} = 4 \cdot (-1) = -4
$$

两种方法得到相同答案，但极坐标快得多——尤其是高次幂的时候。

---

**题 5**：计算 $e^{j\pi/6} + e^{j\pi/3}$，结果用直角坐标表示。

$$
\begin{aligned}
e^{j\pi/6} &= \cos\frac{\pi}{6} + j\sin\frac{\pi}{6} = \frac{\sqrt{3}}{2} + j\frac{1}{2} \\[4pt]
e^{j\pi/3} &= \cos\frac{\pi}{3} + j\sin\frac{\pi}{3} = \frac{1}{2} + j\frac{\sqrt{3}}{2} \\[4pt]
e^{j\pi/6} + e^{j\pi/3} &= \left(\frac{\sqrt{3}}{2} + \frac{1}{2}\right) + j\left(\frac{1}{2} + \frac{\sqrt{3}}{2}\right) \\[4pt]
&= \frac{1+\sqrt{3}}{2} + j\frac{1+\sqrt{3}}{2} \\[4pt]
&\approx 1.366 + j1.366
\end{aligned}
$$

### 5.6 Python 验证与可视化

```python
# 欧拉公式的数值验证与旋转动画
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ====== 数值验证：e^(jθ) = cosθ + j sinθ ======
print("欧拉公式数值验证: e^(jθ) = cosθ + j sinθ")
print("-" * 65)
print(f"{'θ (度)':<12}{'e^(jθ) 实部':<18}{'cosθ':<18}{'实部误差':<14}")
print("-" * 65)

test_angles_deg = [0, 30, 45, 60, 90, 120, 150, 180, 270, 360]
for deg in test_angles_deg:
    rad = np.radians(deg)
    ej = np.exp(1j * rad)  # e^(jθ)
    cos = np.cos(rad)
    sin = np.sin(rad)
    error_real = abs(ej.real - cos)
    error_imag = abs(ej.imag - sin)
    print(f"{deg:<12}{ej.real:<18.10f}{cos:<18.10f}{error_real:<14.2e}")
    if deg == 0:
        print(f"  {'':12}{'e^(jθ) 虚部':<18}{'sinθ':<18}{'虚部误差':<14}")
    print(f"  {'':12}{ej.imag:<18.10f}{sin:<18.10f}{error_imag:<14.2e}")

# ====== 验证：e^(jπ) + 1 = 0 ======
result = np.exp(1j * np.pi) + 1
print(f"\n验证: e^(jπ) + 1 = {result.real:.10f} + j{result.imag:.10f}")
print(f"  即: ≈ 0 (误差: {abs(result):.2e})")

# ====== 可视化：旋转的复数箭头 ======
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axhline(y=0, color='gray', linewidth=0.7)
ax.axvline(x=0, color='gray', linewidth=0.7)
ax.grid(True, alpha=0.3)

# 画单位圆
theta_circle = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'gray', alpha=0.3, linewidth=1)

arrow, = ax.plot([], [], 'steelblue', linewidth=2.5,
                 marker='o', markersize=8, markerfacecolor='darkred')
proj_x, = ax.plot([], [], '--', color='green', linewidth=1, alpha=0.5, label='cosθ (实部)')
proj_y, = ax.plot([], [], '--', color='orange', linewidth=1, alpha=0.5, label='sinθ (虚部)')
title_text = ax.set_title('', fontsize=14)
ax.set_xlabel('实轴 (cosθ)', fontsize=12)
ax.set_ylabel('虚轴 (sinθ)', fontsize=12)
ax.legend(loc='upper right')

def update(frame):
    theta = 2 * np.pi * frame / 100  # 360 帧一圈
    tip_x = np.cos(theta)
    tip_y = np.sin(theta)

    arrow.set_data([0, tip_x], [0, tip_y])
    proj_x.set_data([0, tip_x], [0, 0])
    proj_y.set_data([tip_x, tip_x], [0, tip_y])
    title_text.set_text(f'$e^{{j\\theta}}$ 的旋转可视化\nθ = {np.degrees(theta):.0f}°')

    return arrow, proj_x, proj_y, title_text

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.tight_layout()
plt.show()
```

**预期输出**：

1. 控制台打印验证表：所有 $\theta$ 下 $e^{j\theta}$ 的实部虚部与 $\cos\theta$、$\sin\theta$ 完全一致（误差在 $10^{-16}$ 量级）。
2. 验证 $e^{j\pi} + 1 \approx 0$。
3. 一张动画：一个蓝色箭头沿单位圆旋转，绿色虚线表示向实轴的投影（$\cos\theta$），橙色虚线表示向虚轴的投影（$\sin\theta$）。箭头转一圈，两个投影各完成一个周期的正弦/余弦振荡。

### 5.7 常见误区

- **❌ 误区**：欧拉公式就是"凑巧"，$e^{j\theta}$ 和 $\cos\theta + j\sin\theta$ 恰好在数值上相等而已。
- **✓ 澄清**：欧拉公式不是巧合，它是把指数函数**解析延拓**（analytic continuation）到复数域的自然结果。一旦你把 $e^z$ 定义成对所有复数 $z$ 都适用的那个唯一的"好"函数（可导的），$e^{j\theta} = \cos\theta + j\sin\theta$ 就必然成立。泰勒展开只是其中一种"看到"这个必然性的方式。

- **❌ 误区**：$e^{j\theta}$ 就是 $e$ 的 $j\theta$ 次方，跟 $2^3 = 8$ 一样理解就行。
- **✓ 澄清**："指数是虚数"这件事超出了我们日常对"几次方"的直觉。$e^{j\theta}$ 不是"把 $e$ 乘自己 $j\theta$ 次"（这没有意义），而是**定义在单位圆上的旋转运算**。接受 $e^{j\theta} = \cos\theta + j\sin\theta$ 作为定义，然后放心使用。你不需要"想象"虚数次方是什么意思——你要记住的是：**乘以 $e^{j\theta}$ = 旋转 $\theta$ 角度**。

- **❌ 误区**：所有复数都用 $re^{j\theta}$ 写，$a+jb$ 形式过时了。
- **✓ 澄清**：两种形式各有用途。乘除法用极坐标（指数形式）快，加减法用直角坐标快。如果一个电路问题需要把几个阻抗串并联，就要在两种形式之间来回切换。两种形式都要熟。

### 5.8 应用连接：交流信号 = 旋转的复数

正弦信号 $v(t) = V_m \cos(\omega t + \phi)$ 的本质就是 $V_m e^{j(\omega t + \phi)}$ 的**实部**。

当你写下 $V_m e^{j(\omega t + \phi)}$ 时，你实际上在描述一个以角速度 $\omega$（弧度/秒）匀速旋转的箭头：

- 箭头的长度是 $V_m$（振幅）
- 箭头的初始角度是 $\phi$（初相位）
- 箭头按 $\omega$ 的速度逆时针旋转
- 箭头在实轴上的投影就是 $v(t) = V_m \cos(\omega t + \phi)$

这个视角是下一节"相量"的跳板。

---

## 6. 相量预览

### 6.1 生活例子：旋转木马的照片

你在旋转木马上，木马按固定速度转圈。木马在水平方向的位置随时间变化是一个正弦波：$x(t) = R \cos(\omega t + \phi)$。

如果我用相机拍一张快照，木马的**瞬间位置**就被"冻结"了。照片上木马的位置可以用 $R e^{j(\omega t + \phi)}$ 表示（在拍照那一瞬间，$t$ 是固定的）。

如果我只关心木马之间的**相对位置**，我不需要关心 $t$ 具体是几——只需要知道每个木马的**初始角度**和**离中心的距离**。

电路里的相量（Phasor）就是这种"快照思维"。

### 6.2 直观理解：冻结的旋转

假设有个正弦电压信号：

$$
v(t) = V_m \cos(\omega t + \phi)
$$

我们可以把它看作：

$$
v(t) = \operatorname{Re}\left[ V_m e^{j(\omega t + \phi)} \right]
$$

其中 $\operatorname{Re}[\cdot]$ 表示"取实部"。展开括号：

$$
v(t) = \operatorname{Re}\left[ \underbrace{V_m e^{j\phi}}_{\text{相量}} \cdot \underbrace{e^{j\omega t}}_{\text{旋转}} \right]
$$

注意括号里有两部分：

1. **$V_m e^{j\phi}$**：不随时间变化。只包含振幅 $V_m$ 和初相位 $\phi$。这就是**相量**（Phasor），记作 $\dot{V} = V_m \angle \phi$。
2. **$e^{j\omega t}$**：只负责旋转，所有同频率的信号共用同一个旋转因子。

因为所有同频率的正弦信号都以相同的速度 $\omega$ 旋转，所以你可以把 $e^{j\omega t}$ 放到一边不管，只关注相对固定的相量部分。就像拍旋转木马的快照：木马都在以同样的速度转，快照只记录它们**在哪**，不记录它们**在转**。

### 6.3 相量的优势：微积分变成代数

对正弦信号做微积分是一件痛苦的事：

$$
\begin{aligned}
v(t) &= V_m \cos(\omega t + \phi) \\
\frac{dv}{dt} &= -\omega V_m \sin(\omega t + \phi)
\end{aligned}
$$

但用相量表示后，求导变成了一个简单的乘法：

$$
\frac{d}{dt} \left[ V_m e^{j(\omega t + \phi)} \right]
= j\omega \cdot V_m e^{j(\omega t + \phi)}
$$

在相量域里，求导相当于**乘以 $j\omega$**——旋转 90° 并放大 $\omega$ 倍。积分相当于**除以 $j\omega$**。

这个转化威力巨大：它把微分方程变成了代数方程。这就是为什么学电路的学生做了那么多复数运算——不是为了折磨你，而是因为复数能把电容电感的微分关系变成简单的乘除法。

### 6.4 手算：正弦信号 ↔ 相量

| 时域信号 | 相量表示 | 含义 |
|---|---|---|
| $10\cos(\omega t + 30^\circ)$ | $10 \angle 30^\circ$ | 振幅 10，超前 30° |
| $5\cos(\omega t - 45^\circ)$ | $5 \angle -45^\circ$ | 振幅 5，滞后 45° |
| $3\sin(\omega t + 60^\circ)$ | $3 \angle -30^\circ$ | 注意：$\sin\theta = \cos(\theta - 90^\circ)$ |
| $100\cos(\omega t)$ | $100 \angle 0^\circ$ | 振幅 100，无偏移 |

**重点**：相量只存**振幅**和**相位**两个信息。频率 $\omega$ 不在相量里——所有同频率的信号共用一个 $\omega$，分析时不去管它。

### 6.5 Python：旋转矢量投影 = 正弦波

```python
# 旋转矢量投影 = 正弦波 —— 动画演示
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 参数
Vm = 2.0         # 振幅
phi_deg = 30     # 初相位（度）
phi = np.radians(phi_deg)
omega = 2 * np.pi  # 角频率（1 圈/秒 = 2π rad/s）

# 创建画布：左 = 复平面旋转矢量，右 = 投影随时间变化
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

# ---- 左图：复平面上的旋转矢量 ----
ax_left.set_xlim(-Vm*1.2, Vm*1.2)
ax_left.set_ylim(-Vm*1.2, Vm*1.2)
ax_left.set_aspect('equal')
ax_left.axhline(y=0, color='gray', linewidth=0.5)
ax_left.axvline(x=0, color='gray', linewidth=0.5)
ax_left.grid(True, alpha=0.3)
ax_left.set_xlabel('实轴', fontsize=11)
ax_left.set_ylabel('虚轴', fontsize=11)
ax_left.set_title('复平面：旋转矢量', fontsize=13)

# 画参考圆
circle = plt.Circle((0, 0), Vm, fill=False, color='gray', alpha=0.3, linestyle='--')
ax_left.add_patch(circle)

arrow_left, = ax_left.plot([], [], 'steelblue', linewidth=2.5,
                           marker='o', markersize=8, markerfacecolor='darkred')
proj_line, = ax_left.plot([], [], 'green', linewidth=2, alpha=0.5)

# ---- 右图：投影 vs 时间 ----
t_max = 2.0  # 显示 2 秒
ax_right.set_xlim(0, t_max)
ax_right.set_ylim(-Vm*1.3, Vm*1.3)
ax_right.axhline(y=0, color='gray', linewidth=0.5)
ax_right.grid(True, alpha=0.3)
ax_right.set_xlabel('时间 t (s)', fontsize=11)
ax_right.set_ylabel('投影值 (实部)', fontsize=11)
ax_right.set_title('实轴投影 = 正弦波 v(t)', fontsize=13)

# 存储历史轨迹
trail_t = []
trail_v = []
wave_line, = ax_right.plot([], [], 'darkgreen', linewidth=2)
dot_current, = ax_right.plot([], [], 'ro', markersize=8)

def init():
    arrow_left.set_data([], [])
    proj_line.set_data([], [])
    wave_line.set_data([], [])
    dot_current.set_data([], [])
    return arrow_left, proj_line, wave_line, dot_current

def animate(frame):
    t = frame / 50  # 50 帧 = 1 秒
    angle = omega * t + phi

    # 旋转矢量端点
    tip_x = Vm * np.cos(angle)
    tip_y = Vm * np.sin(angle)

    # 左图：更新箭头和向实轴的投影
    arrow_left.set_data([0, tip_x], [0, tip_y])
    proj_line.set_data([tip_x, tip_x], [0, tip_y])

    # 右图：更新波形
    trail_t.append(t)
    trail_v.append(tip_x)  # 实部 = cos 投影
    wave_line.set_data(trail_t, trail_v)
    dot_current.set_data([t], [tip_x])

    # 波形向左滚动
    if t > t_max:
        # 移除旧数据点
        while trail_t and trail_t[0] < t - t_max:
            trail_t.pop(0)
            trail_v.pop(0)

    return arrow_left, proj_line, wave_line, dot_current

ani = FuncAnimation(fig, animate, init_func=init,
                    frames=200, interval=40, blit=True)

plt.tight_layout()
plt.show()
```

**预期输出**：左图显示一个矢量在复平面上逆时针旋转，绿色竖线标出它在实轴上的投影。右图同步显示这个投影值随时间的变化——一个完美的正弦波 $V_m \cos(\omega t + \phi)$。初相位 $\phi = 30^\circ$ 意味着波在 $t=0$ 时不从零开始，而是从 $V_m \cos 30^\circ \approx 1.73$ 开始。

### 6.6 常见误区

- **❌ 误区**：相量是正弦波本身。
- **✓ 澄清**：相量是正弦波的"快照"，它不包含频率信息。**只有同频率的正弦信号**才能用相量比较。如果你有两个频率不同的信号（比如 50 Hz 和 60 Hz），它们的相量不能直接放在一起做加减法——因为它们的旋转速度不同，"快照"会互相错位。

- **❌ 误区**：相量的角度就是正弦波的相位，直接读就行。
- **✓ 澄清**：这取决于你用 $\cos$ 还是 $\sin$ 作为基准。工程上标准是用 $\cos$ 作为参考：$v(t) = V_m \cos(\omega t + \phi)$ 对应相量 $V_m \angle \phi$。如果你拿到的是 $\sin$ 信号 $v(t) = V_m \sin(\omega t + \phi)$，先转成 $\cos$：$V_m \sin(\omega t + \phi) = V_m \cos(\omega t + \phi - 90^\circ)$，相量是 $V_m \angle (\phi - 90^\circ)$。

- **❌ 误区**：相量分析对直流电也有用。
- **✓ 澄清**：相量是专门针对**正弦稳态交流**的工具。直流（DC）没有相位，也没有频率，用不着相量。在直流电路中，复数的虚部永远是零，复数退化为实数，欧姆定律就是简单的 $V = IR$。

### 6.7 应用连接：交流电路分析的全貌

在后续章节中，你会学到：

- **阻抗** $Z = R + jX$ 是电阻（实部）和电抗（虚部）的复合体
- **欧姆定律的交流版本**：$\dot{V} = \dot{I} Z$（全部是相量/复数）
- 电容的阻抗 $Z_C = \frac{1}{j\omega C}$，电感的阻抗 $Z_L = j\omega L$
- 整个交流电路分析就是：把所有量换成相量，然后像算直流电路一样算复数

你现在学的复数四则运算和极坐标转换，就是交流电路分析的"九九乘法表"。练熟它们，后面学阻抗、功率、三相电时会轻松很多。

---

## 7. 完整实战：用复数表示和运算一个简单交流电路

这一节我们用本章所有知识做一个综合练习：用复数（相量）分析一个最简单的 RC 串联电路。

### 题目

一个正弦电压源 $v_s(t) = 10 \cos(1000 t) \text{ V}$ 串联一个电阻 $R = 100\ \Omega$ 和一个电容 $C = 10\ \mu\text{F}$。求电容两端的电压 $v_c(t)$。

**已知公式**（后续章节会详细推导）：

- 电容阻抗：$Z_C = \frac{1}{j\omega C}$
- 串联分压：$\dot{V}_C = \dot{V}_s \cdot \frac{Z_C}{R + Z_C}$

### 分步求解

**第 1 步**：写出源电压的相量。

$$
\dot{V}_s = 10 \angle 0^\circ = 10 + j0
$$

**第 2 步**：计算角频率和电容阻抗。

$$
\omega = 1000 \text{ rad/s}
$$

$$
Z_C = \frac{1}{j \cdot 1000 \cdot 10 \times 10^{-6}} = \frac{1}{j \cdot 0.01} = \frac{1}{j 0.01}
$$

先算 $1/j = -j$，所以：

$$
Z_C = -j \cdot \frac{1}{0.01} = -j 100\ \Omega
$$

**第 3 步**：计算总阻抗。

$$
Z_{\text{总}} = R + Z_C = 100 - j100\ \Omega
$$

**第 4 步**：用分压公式计算 $\dot{V}_C$。

$$
\dot{V}_C = \dot{V}_s \cdot \frac{Z_C}{Z_{\text{总}}}
= 10 \angle 0^\circ \cdot \frac{-j100}{100 - j100}
$$

先处理分压比：

$$
\frac{-j100}{100 - j100}
= \frac{-j}{1 - j}
= \frac{-j(1 + j)}{(1 - j)(1 + j)}
= \frac{-j - j^2}{1 + 1}
= \frac{1 - j}{2}
= 0.5 - j0.5
$$

**第 5 步**：完整的 $\dot{V}_C$。

$$
\dot{V}_C = 10 \cdot (0.5 - j0.5) = 5 - j5
$$

**第 6 步**：转为极坐标。

$$
\begin{aligned}
|\dot{V}_C| &= \sqrt{5^2 + (-5)^2} = \sqrt{50} \approx 7.07 \text{ V} \\
\angle \dot{V}_C &= \operatorname{atan2}(-5, 5) = -45^\circ
\end{aligned}
$$

**第 7 步**：写回时域。

$$
v_c(t) = 7.07 \cos(1000 t - 45^\circ) \text{ V}
$$

**物理解释**：电容上的电压比源电压**小了约 30%**（从 10 V 降到 7.07 V），并且**相位滞后了 45°**。这个滞后是电容的典型行为——电压"跟不上"电流的变化。

### Python 验证

```python
# RC 串联电路相量分析
import numpy as np

# 参数
R = 100                # 电阻 (Ω)
C = 10e-6              # 电容 (F)
omega = 1000           # 角频率 (rad/s)
Vs_mag = 10            # 源电压振幅 (V)

# 第 2 步：电容阻抗
Zc = 1 / (1j * omega * C)
print(f"Zc = {Zc:.2f} Ω  (预期: -j100 Ω)")

# 第 3 步：总阻抗
Z_total = R + Zc
print(f"Z_total = {Z_total:.2f} Ω  (预期: 100 - j100 Ω)")

# 第 4-5 步：分压计算 Vc 相量
Vs_phasor = Vs_mag + 0j   # 10∠0°
Vc_phasor = Vs_phasor * Zc / Z_total
print(f"\nVc 相量 (直角坐标) = {Vc_phasor:.4f}  (预期: 5 - j5)")

# 第 6 步：极坐标
Vc_mag = np.abs(Vc_phasor)
Vc_angle_deg = np.degrees(np.angle(Vc_phasor))
print(f"Vc 相量 (极坐标)   = {Vc_mag:.2f}∠{Vc_angle_deg:.1f}°")
print(f"  预期: 7.07∠-45°")

# 第 7 步：时域验证 —— 在 t=0 时
t = 0
vc_t0 = Vc_mag * np.cos(omega * t + np.radians(Vc_angle_deg))
print(f"\nv_c(0) = {vc_t0:.2f} V  (直接从相量: Vc 实部 = {Vc_phasor.real:.2f} V)")

# 验证 vc(t) 在一个周期内与直接相量计算一致
print(f"\n时域验证（一个周期内抽样）：")
print(f"{'t (ms)':<10}{'v_c(t) 时域':<14}{'Re[Vc·e^(jωt)]':<16}{'一致?'}")
print("-" * 50)
for t_ms in [0, 0.5, 1.0, 1.5, 2.0]:
    t = t_ms / 1000
    vc_time = Vc_mag * np.cos(omega * t + np.radians(Vc_angle_deg))
    vc_phasor_method = np.real(Vc_phasor * np.exp(1j * omega * t))
    match = "✓" if abs(vc_time - vc_phasor_method) < 1e-10 else "✗"
    print(f"{t_ms:<10}{vc_time:<14.6f}{vc_phasor_method:<16.6f}{match}")

print(f"\n✓ RC 串联电路相量分析验证通过！")
```

**预期输出**：

```
Zc = 0.00-100.00j Ω  (预期: -j100 Ω)
Z_total = 100.00-100.00j Ω  (预期: 100 - j100 Ω)

Vc 相量 (直角坐标) = 5.0000-5.0000j  (预期: 5 - j5)
Vc 相量 (极坐标)   = 7.07∠-45.0°
  预期: 7.07∠-45°

v_c(0) = 5.00 V  (直接从相量: Vc 实部 = 5.00 V)

时域验证（一个周期内抽样）：
t (ms)    v_c(t) 时域    Re[Vc·e^(jωt)] 一致?
--------------------------------------------------
0         5.000000      5.000000        ✓
0.5       6.304086      6.304086        ✓
1.0       7.063170      7.063170        ✓
1.5       7.063170      7.063170        ✓
2.0       6.304086      6.304086        ✓

✓ RC 串联电路相量分析验证通过！
```

---

## 8. 思考题

### 基础题 (1-4)

**题 1**：计算 $(2 + j3) + (4 - j5) - (-1 + j2)$。

**题 2**：计算 $\frac{2 + j}{3 - j4}$，结果用 $a + jb$ 形式表示。

**题 3**：将 $z = -4 + j4\sqrt{3}$ 转为极坐标。

**题 4**：将 $z = 8 \angle 210^\circ$ 转为直角坐标。

### 进阶题 (5-6)

**题 5**：用欧拉公式化简 $(1 + j\sqrt{3})^6$，分别用直角坐标展开和极坐标两种方法，比较哪种更快。

**题 6**：已知 $z_1 = 3 + j4$，$z_2 = 5 \angle 30^\circ$。计算 $z_1 \cdot z_2$ 和 $z_1 / z_2$，结果用极坐标表示。

### 综合题 (7-8)

**题 7**：一个正弦电流 $i(t) = 5 \cos(100\pi t + 60^\circ) \text{ A}$ 流过一个 $R = 10\ \Omega$ 的电阻和一个 $L = 31.8 \text{ mH}$ 的电感串联的电路。电感的阻抗为 $Z_L = j\omega L$。求：

(a) 写出电流的相量 $\dot{I}$。  
(b) 计算电感阻抗 $Z_L$（$\omega = 100\pi$）。  
(c) 计算电感两端电压的相量 $\dot{V}_L = \dot{I} \cdot Z_L$。  
(d) 写出电感电压的时域表达式 $v_L(t)$。

**题 8**：验证 $e^{j\theta_1} \cdot e^{j\theta_2} = e^{j(\theta_1 + \theta_2)}$ 在 $\theta_1 = 40^\circ$、$\theta_2 = 50^\circ$ 时成立。由此说明：复数乘法中"幅角相加"这个规律，本质上是**指数的加法法则**在复数域的自然推广。用这个思路，不查表，只用 $e^{j\theta}$ 的形式计算 $\cos 75^\circ$ 和 $\sin 75^\circ$（提示：$75^\circ = 45^\circ + 30^\circ$）。

---

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

#### 题 1

$$
\begin{aligned}
(2 + j3) + (4 - j5) - (-1 + j2)
&= 2 + j3 + 4 - j5 + 1 - j2 \\
&= (2 + 4 + 1) + j(3 - 5 - 2) \\
&= 7 - j4
\end{aligned}
$$

#### 题 2

$$
\begin{aligned}
\frac{2 + j}{3 - j4}
&= \frac{(2 + j)(3 + j4)}{(3 - j4)(3 + j4)} \\
&= \frac{6 + j8 + j3 + j^2 4}{9 + 16} \\
&= \frac{6 + j11 - 4}{25} \\
&= \frac{2 + j11}{25} \\
&= 0.08 + j0.44
\end{aligned}
$$

#### 题 3

$z = -4 + j4\sqrt{3}$，其中 $a = -4$，$b = 4\sqrt{3} \approx 6.928$。

$$
\begin{aligned}
r &= \sqrt{(-4)^2 + (4\sqrt{3})^2} = \sqrt{16 + 48} = \sqrt{64} = 8 \\[4pt]
\theta &= \operatorname{atan2}(4\sqrt{3}, -4)
\end{aligned}
$$

在第二象限（$a < 0, b > 0$）。参考角 $\arctan(\sqrt{3}) = 60^\circ$。第二象限角 $= 180^\circ - 60^\circ = 120^\circ$。

$$
z = 8 \angle 120^\circ
$$

#### 题 4

$z = 8 \angle 210^\circ$。

注意 $210^\circ$ 在第三象限。$\cos 210^\circ = -\cos 30^\circ = -\sqrt{3}/2$，$\sin 210^\circ = -\sin 30^\circ = -1/2$。

$$
\begin{aligned}
a &= 8 \cos 210^\circ = 8 \times (-\sqrt{3}/2) = -4\sqrt{3} \approx -6.928 \\
b &= 8 \sin 210^\circ = 8 \times (-1/2) = -4
\end{aligned}
$$

$$
z = -4\sqrt{3} - j4 \approx -6.928 - j4
$$

#### 题 5

**方法一（直角坐标，硬展开）**：

$$
\begin{aligned}
(1 + j\sqrt{3})^2 &= 1 + j2\sqrt{3} + j^2 \cdot 3 = 1 + j2\sqrt{3} - 3 = -2 + j2\sqrt{3} \\
(1 + j\sqrt{3})^3 &= (1 + j\sqrt{3})(-2 + j2\sqrt{3}) \\
&= -2 + j2\sqrt{3} - j2\sqrt{3} + j^2 6 \\
&= -2 - 6 = -8
\end{aligned}
$$

$$
(1 + j\sqrt{3})^6 = (-8)^2 = 64
$$

**方法二（极坐标）**：

$$
\begin{aligned}
r &= |1 + j\sqrt{3}| = \sqrt{1 + 3} = 2 \\
\theta &= \arctan(\sqrt{3}) = 60^\circ
\end{aligned}
$$

所以 $1 + j\sqrt{3} = 2 e^{j60^\circ}$。

$$
(1 + j\sqrt{3})^6 = (2 e^{j60^\circ})^6 = 2^6 \cdot e^{j 360^\circ} = 64 \cdot 1 = 64
$$

极坐标快得多！6 次方在直角坐标下要展开三次，极坐标下只要 $2^6 = 64$，角度 $6 \times 60^\circ = 360^\circ$ 恰好转一圈回到实轴。

#### 题 6

先将 $z_1$ 转为极坐标：

$$
|z_1| = \sqrt{3^2 + 4^2} = 5,\quad \angle z_1 = \arctan(4/3) \approx 53.13^\circ
$$

所以 $z_1 = 5 \angle 53.13^\circ$。

$z_2$ 已知为 $5 \angle 30^\circ$。

**乘法**：

$$
z_1 \cdot z_2 = (5 \cdot 5) \angle (53.13^\circ + 30^\circ) = 25 \angle 83.13^\circ
$$

**除法**：

$$
\frac{z_1}{z_2} = \frac{5}{5} \angle (53.13^\circ - 30^\circ) = 1 \angle 23.13^\circ
$$

#### 题 7

**(a)** 电流相量 $\dot{I} = 5 \angle 60^\circ$。

**(b)** 电感阻抗。

$$
\omega = 100\pi \approx 314.16 \text{ rad/s}
$$

$$
Z_L = j\omega L = j \cdot 100\pi \cdot 31.8 \times 10^{-3} = j \cdot 100\pi \cdot 0.0318
$$

先算 $100\pi \cdot 0.0318 \approx 314.16 \cdot 0.0318 \approx 10$（恰好约等于 10，这是个设计好的数值）。

$$
Z_L = j10\ \Omega
$$

**(c)** 电感电压相量。

$$
\dot{V}_L = \dot{I} \cdot Z_L = 5 \angle 60^\circ \cdot j10
$$

$j10$ 就是 $10 \angle 90^\circ$（因为 $j$ 的模是 1，角度是 90°）。

$$
\dot{V}_L = 5 \angle 60^\circ \cdot 10 \angle 90^\circ = 50 \angle 150^\circ \text{ V}
$$

**(d)** 时域表达式。

$$
v_L(t) = 50 \cos(100\pi t + 150^\circ) \text{ V}
$$

物理意义：电感两端的电压比电流**超前了 90°**（电流相位 60°，电压相位 150°，差了 90°），而且电压振幅（50 V）远大于源电压——电感在交流下可以产生比电源更高的电压（类似于水锤效应）。

#### 题 8

验证 $e^{j40^\circ} \cdot e^{j50^\circ} = e^{j90^\circ}$：

$$
e^{j90^\circ} = \cos 90^\circ + j\sin 90^\circ = 0 + j = j
$$

同时 $e^{j40^\circ} \cdot e^{j50^\circ} = e^{j(40^\circ + 50^\circ)} = e^{j90^\circ}$（指数法则直接给出）。数值验证一致。

这说明复数乘法的"幅角相加"不是一条需要额外记忆的规则——它是**指数法则 $e^a \cdot e^b = e^{a+b}$** 在复数域的自然推广。这也是为什么欧拉形式比 $r\angle\theta$ 更优美：它把复数乘法还原成了你早就知道的指数运算。

计算 $\cos 75^\circ$ 和 $\sin 75^\circ$：

$$
e^{j75^\circ} = e^{j(45^\circ + 30^\circ)} = e^{j45^\circ} \cdot e^{j30^\circ}
$$

$$
\begin{aligned}
e^{j45^\circ} \cdot e^{j30^\circ}
&= (\cos 45^\circ + j\sin 45^\circ)(\cos 30^\circ + j\sin 30^\circ) \\
&= \left(\frac{\sqrt{2}}{2} + j\frac{\sqrt{2}}{2}\right)\left(\frac{\sqrt{3}}{2} + j\frac{1}{2}\right) \\
&= \frac{\sqrt{6}}{4} + j\frac{\sqrt{2}}{4} + j\frac{\sqrt{6}}{4} + j^2\frac{\sqrt{2}}{4} \\
&= \frac{\sqrt{6}}{4} - \frac{\sqrt{2}}{4} + j\left(\frac{\sqrt{2}}{4} + \frac{\sqrt{6}}{4}\right) \\
&= \frac{\sqrt{6} - \sqrt{2}}{4} + j\frac{\sqrt{6} + \sqrt{2}}{4}
\end{aligned}
$$

所以：

$$
\cos 75^\circ = \frac{\sqrt{6} - \sqrt{2}}{4} \approx 0.2588
$$

$$
\sin 75^\circ = \frac{\sqrt{6} + \sqrt{2}}{4} \approx 0.9659
$$

不用查表，不用计算器——就靠复数的指数形式和乘法展开，推导出了 $\cos 75^\circ$ 的精确值。这就是欧拉公式的威力。

</details>

---

## 本章小结

你已经从"复数是个奇怪的东西"走到了"相量分析 RC 电路"。回顾一下你本章掌握的核心工具：

1. **复数** $a + jb$ 就是二维平面上的点/箭头
2. **四则运算**：加法和向量一样，乘法多一个 $j^2 = -1$，除法用共轭
3. **极坐标** $r \angle \theta$ 和直角坐标可以互相转，乘除法在极坐标下极其简洁
4. **欧拉公式** $e^{j\theta} = \cos\theta + j\sin\theta$ 把旋转和指数联系起来
5. **相量**就是正弦信号的"身份证"：存着振幅和相位，不管旋转

这些工具在下一章会直接派上用场：你将用复数来分析电阻、电容、电感的阻抗，以及它们在串并联中的行为。别再怕 $j$ 了——它只是数学里最优雅的 90° 旋转器。

> **下一章**：[交流电路基础](./ac-basics.md) — 用复数真正算交流电路
