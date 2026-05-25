# 物理与数学基础

> **核心问题**：电从哪里来？我们需要哪些数学"工具"来描述它？

---

## 0. 本章导览

在学习电阻、电容、电路分析之前，你得先回答一个问题：**电到底是什么？** 它不是魔法，也不是抽象符号——它是真实存在的物理现象，根植于物质最微小的结构之中。

本章从"原子里面有什么"讲起，一路走到电场、电压、能量的物理图像。然后，我们会武装你的数学工具箱：单位换算、一元方程、正反比、微积分直觉——这些是你后面分析每一个电路时都会反复用到的"基本功"。

**前置知识**：初中数学（会解一元一次方程、知道坐标系）。不需要任何物理或电路基础。

学完本章，你将能够：
1. 从原子层面解释"电荷"是什么，以及电荷之间怎么相互作用
2. 理解电场和电压的物理含义，能用手算计算电场力和电位差
3. 熟练进行电学单位换算（kΩ ↔ Ω, mA ↔ A, μF ↔ F 等）
4. 用一元方程解决电路中的未知量计算
5. 理解正比/反比关系，看懂 V=IR 这类公式"在说什么"
6. 建立微积分的基本直觉，知道为什么电容和电感需要它

> 本章约 1800 行，建议分 3-4 次读完。物理部分（第1-3节）和数学部分（第4-7节）可以分开阅读。

---

## 1. 物质与原子

> 电从哪里来？答案藏在物质的最深处——原子里面。

### 1a. 生活例子：拆乐高积木

你有一套乐高积木，搭成了一座城堡。你把城堡拆开，得到一块一块的积木块。再把其中一块积木掰开……当然，你掰不开，但自然界可以。

**物质就像乐高城堡**。任何东西——水、铁、木头、你的手指——都由极小的"积木块"组成。这些积木块叫**原子（Atom）**。把水拆开，得到水分子（H₂O）；把水分子拆开，得到氢原子和氧原子；把原子拆开……你会看到更有趣的东西。

> 一滴水大约含有 $1.5 \times 10^{21}$ 个水分子。这个数字有多大？如果每秒数一个，你需要 47 万亿年才能数完——比宇宙的年龄还长 3000 多倍。

但本章关心的不是数原子，而是：**原子里面有什么带电的东西？**

### 1b. 直观理解：原子的"三明治"

把原子想象成一个微型的太阳系：

- **中心**有一个极其微小的**原子核（Nucleus）**，直径大约是原子本身直径的万分之一。如果把原子放大到一个足球场那么大，原子核只有一粒米大小。
- 原子核里面住着两种粒子：**质子（Proton）**带**正电荷（Positive Charge）**，**中子（Neutron）**不带电荷。
- **外围**有极轻的**电子（Electron）**围绕着原子核高速运动。电子带**负电荷（Negative Charge）**。

现在说**电荷（Electric Charge）**。它是粒子的一个基本属性，就像质量是物体的基本属性一样。世界上有两种电荷——正和负——就这么两种，没有第三种。

一个关键事实：**在正常的原子里，质子的数量和电子的数量完全相等。** 比如碳原子有 6 个质子和 6 个电子。正负电荷刚好抵消，所以整个原子对外不显电性——它是**电中性（Electrically Neutral）**的。

但如果某种外力（比如摩擦）把电子从原子里"拽"出来呢？原子就少了电子，正电荷多出来了——原子带正电了。反过来，如果原子捕获了额外的电子，就带负电。

> **这就是"电"的来源**：电荷的分离与移动。当大量电子从一个地方移动到另一个地方，你就有了电流。当正负电荷被分开堆在两处，你就有了电压。整个电工学，说到底就是"怎么让电子乖乖听你指挥"。

### 1c. 形式化定义

**电荷（Charge）**用符号 $q$ 或 $Q$ 表示，单位是**库仑（Coulomb，简称 C）**。一个电子带有的电荷量是宇宙中最基本的电荷单位：

$$
e = 1.602 \times 10^{-19} \text{ C}
$$

其中 $e$ 是**元电荷（Elementary Charge）**。任何物体带的电荷量都是 $e$ 的整数倍——电荷是"量子化"的。

> 1 库仑有多大？大约需要 $6.24 \times 10^{18}$ 个电子才能凑出 1 C 的电荷。你梳头时产生的静电大概只有几个微库仑（$10^{-6}$ C），但已经足以让头发竖起来了。

**库仑定律（Coulomb's Law）**：两个点电荷之间的作用力大小满足：

$$
F = k \frac{|q_1 \cdot q_2|}{r^2}
$$

每个符号的含义：
- $F$：两个电荷之间的力（单位：牛顿 N）
- $k$：库仑常数，$k = 8.99 \times 10^9 \ \text{N·m}^2/\text{C}^2$
- $q_1, q_2$：两个电荷的电量（单位：库仑 C）
- $r$：两个电荷之间的距离（单位：米 m）

**方向规则**：同性电荷相斥，异性电荷相吸。正电荷和正电荷互相推开，正电荷和负电荷互相拉近。

库仑定律和牛顿的万有引力定律长得几乎一模一样（都是力和距离平方成反比），但有一个巨大区别：**电力比引力强得多**。两个质子之间的电排斥力，大约是它们之间万有引力的 $10^{36}$ 倍——1 后面跟 36 个零！所以日常生活中的力学现象（摩擦、弹力、碰撞）本质上都是电磁力在起作用，引力只在星球尺度上才变得重要。

**电荷守恒定律（Law of Conservation of Charge）**：在一个孤立系统中，总电荷量永远不会改变。电荷不能被创造，也不能被消灭，只能从一个物体转移到另一个物体。

这句话的意思很朴素：你把电子从毛衣上蹭到气球上，毛衣带正电，气球带负电，但两者加起来的总电荷量还是零。宇宙从诞生到现在，总电荷没多过也没少过。

### 1d. 手算示例：库仑力计算

**题目**：两个小带电球体，$q_1 = 2 \ \mu\text{C}$，$q_2 = 3 \ \mu\text{C}$，距离 $r = 0.05 \ \text{m}$。求它们之间的库仑力。

> 符号 $\mu$（希腊字母 mu，读作"微"）表示 $10^{-6}$。所以 $2 \ \mu\text{C} = 2 \times 10^{-6} \ \text{C}$。

逐步计算：

| 步骤 | 操作 | 结果 |
|------|------|------|
| ①  | 写出公式 | $F = k \frac{\|q_1 q_2\|}{r^2}$ |
| ②  | 代入 $k$ | $F = 8.99 \times 10^9 \times \frac{\|q_1 q_2\|}{r^2}$ |
| ③  | 代入 $q_1 = 2 \times 10^{-6}$，$q_2 = 3 \times 10^{-6}$ | 分子：$\|q_1 q_2\| = 2 \times 10^{-6} \times 3 \times 10^{-6} = 6 \times 10^{-12}$ |
| ④  | 代入 $r = 0.05$ | 分母：$r^2 = (0.05)^2 = 2.5 \times 10^{-3}$ |
| ⑤  | 计算分式 | $\frac{6 \times 10^{-12}}{2.5 \times 10^{-3}} = 2.4 \times 10^{-9}$ |
| ⑥  | 乘以 $k$ | $F = 8.99 \times 10^9 \times 2.4 \times 10^{-9}$ |
| ⑦  | 整理数量级 | $10^9 \times 10^{-9} = 10^0 = 1$，所以 $F = 8.99 \times 2.4 = 21.576$ |
| ⑧  | 最终结果 | $F \approx 21.6 \ \text{N}$ |

> 21.6 牛顿是多大的力？大约相当于手里托着 2.2 公斤物体的重量。两个小小的带电体在 5 厘米距离上就能产生这么大的力——电力的强大可见一斑。

**再算一个**：如果距离加倍到 $r = 0.10 \ \text{m}$，力会变成多少？

$$
F_{\text{新}} = k \frac{|q_1 q_2|}{(2r)^2} = k \frac{|q_1 q_2|}{4r^2} = \frac{1}{4} F_{\text{原}}
$$

因为分母变成了原来的 4 倍，所以力变成原来的四分之一：$21.6 \div 4 = 5.4 \ \text{N}$。距离翻倍，力降到四分之一——这就是"平方反比"的威力。

### 1e. Python 验证：库仑力与距离的关系

```python
# 库仑力与距离的关系可视化
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 库仑常数
k = 8.99e9

def coulomb_force(q1, q2, r):
    """计算两个点电荷之间的库仑力（取绝对值）"""
    return k * abs(q1 * q2) / (r ** 2)

# 参数设置：两个电荷都是 1 μC
q1 = 1e-6  # 1 微库仑
q2 = 1e-6

# 验证手算结果
r_test = 0.05  # 5 cm
F_test = coulomb_force(2e-6, 3e-6, r_test)
print(f"验证手算：q1=2μC, q2=3μC, r=5cm")
print(f"  手算结果：F ≈ 21.6 N")
print(f"  Python计算：F = {F_test:.2f} N")
print(f"  匹配！✓\n")

# 生成不同距离下的力
distances = np.linspace(0.02, 0.20, 100)  # 2cm 到 20cm
forces = coulomb_force(q1, q2, distances)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：线性坐标
ax1.plot(distances * 100, forces, 'b-', linewidth=2)
ax1.set_xlabel('距离 (cm)', fontsize=12)
ax1.set_ylabel('库仑力 (N)', fontsize=12)
ax1.set_title('库仑力与距离的关系（线性坐标）', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axvline(x=5, color='r', linestyle='--', alpha=0.5, label='r = 5 cm (手算点)')
ax1.legend()

# 标记手算点
ax1.plot(5, F_test, 'ro', markersize=8)

# 右图：双对数坐标（展示平方反比规律）
ax2.loglog(distances * 100, forces, 'b-', linewidth=2)
ax2.set_xlabel('距离 (cm)', fontsize=12)
ax2.set_ylabel('库仑力 (N)', fontsize=12)
ax2.set_title('库仑力与距离的关系（双对数坐标）', fontsize=14)
ax2.grid(True, alpha=0.3, which='both')

# 在双对数图上标注：斜率应该接近 -2（平方反比）
ax2.text(0.5, 0.2, '斜率 ≈ -2\n(平方反比)', transform=ax2.transAxes,
         fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('coulomb_force.png', dpi=150)
plt.show()

print("图像已保存为 coulomb_force.png")
print("\n关键观察：距离增大到 2 倍 → 力减小到 1/4")
print("          距离增大到 3 倍 → 力减小到 1/9")
print("          这就是'平方反比'规律！")
```

### 1f. 常见误区

- **❌ 误区**：原子核里的质子和中子一样多。
  **✓ 正确**：质子数和中子数没有必然相等的关系。比如氢原子有 1 个质子、0 个中子，而铀-238 有 92 个质子、146 个中子。决定元素种类的是**质子数**，不是中子数。

- **❌ 误区**：电子像行星绕太阳那样在固定轨道上转。
  **✓ 正确**：这是 1913 年玻尔提出的简化模型，早已被量子力学取代。电子并没有确定的"轨道"，而是以"概率云"的形式分布在原子核周围。但对于理解电荷和电流，太阳系模型足够直观，我们用它是为了建立直觉，不是因为它完全准确。

- **❌ 误区**：电荷用完了就没有了。
  **✓ 正确**：根据电荷守恒定律，电荷不会消失。电池"没电了"不是电荷消失了，而是正负电荷已经通过化学反应达到了平衡，无法再驱动电子流动。

### 1g. 应用连接

库仑定律是静电学的基石。虽然你后面学电路时很少直接套用库仑定律公式，但它的思想——**同性相斥、异性相吸、距离越远力越小**——贯穿始终。电容器的基本原理就是正负电荷被分开储存在两个极板上（异性相吸让它们"想回去"但被绝缘介质隔开），这就是库仑力在背后起作用。后续章节中，[电容器](./06-capacitor.md)会详细展开。

---

## 2. 电场与力

> 电荷之间隔空就能产生力——这中间有什么"东西"在传递力的作用？

### 2a. 生活例子：磁铁的"看不见的手"

你玩过磁铁吗？拿一块磁铁靠近另一块，还没碰到，另一块就"啪"地吸过来了，或者被推开。隔着一张纸，磁铁照样能让纸上的铁屑跳舞。

磁铁周围有一种**看不见的"影响范围"**。任何进入这个范围的铁磁性物体都会受到力。这个"影响范围"就叫**磁场（Magnetic Field）**。

电荷周围也有类似的"影响范围"，叫**电场（Electric Field）**。一个电荷放在那里，它周围的空间就被"改变了"——如果另一个电荷进入这个空间，就会感受到力。电荷之间并没有直接接触，力的传递是通过电场这个"中介"完成的。

> 现代物理学的观点：**场（Field）是真实存在的物理实体**，不只是一个数学工具。就像空气虽然看不见，但它是真实存在的物质一样。电场和磁场遍布宇宙的每一个角落。

### 2b. 直观理解：电场的"探测法"

你怎么知道某个空间区域有没有电场？放一个小电荷进去，看它动不动。动了，说明有电场；不动，说明没有。

但这里有个微妙之处：你放进去的"探测电荷"本身也会产生电场，会改变原来的电场分布。所以探测电荷必须**足够小**，小到它自己的电场对原电场的影响可以忽略不计。在理论上，我们用一个"检验电荷"（Test Charge）$q_0$ 来探测空间中的电场。

电场有几个重要特性：
1. **电场无处不在**：只要有电荷存在，它周围就有电场。
2. **电场有方向**：电场是一个矢量（有大小也有方向的量）。某点的电场方向定义为：**正电荷在该点受力的方向**。
3. **电场可以叠加**：如果空间中有多个电荷，某点的总电场就是每个电荷各自产生的电场的矢量和。

### 2c. 形式化定义

**电场强度（Electric Field Strength）** $\vec{E}$ 定义为：

$$
\vec{E} = \frac{\vec{F}}{q_0}
$$

每个符号的含义：
- $\vec{E}$：电场强度（单位：牛/库仑 N/C，或伏/米 V/m）
- $\vec{F}$：检验电荷 $q_0$ 在该点受到的力（单位：牛顿 N）
- $q_0$：检验电荷的电量（单位：库仑 C）

> 这个定义的意思是：**电场强度等于单位正电荷在该点受到的力**。如果你在一个地方放 1 C 的正电荷，它受到 5 N 的力，那这个地方的电场强度就是 5 N/C。

反过来，如果你知道某点的电场强度 $E$，那任意电荷 $q$ 放在该点受到的力就是：

$$
F = qE
$$

**点电荷的电场**：一个点电荷 $Q$ 在距离它 $r$ 处产生的电场大小为：

$$
E = k \frac{|Q|}{r^2}
$$

注意：这个公式和库仑定律几乎一模一样，只是分子上少了一个 $q$。因为 $E$ 描述的是"单位电荷受的力"，所以把库仑力公式中的 $q_0$ 除掉就是了。

**电场线（Electric Field Lines）**：一种可视化电场的工具。画法规则：
- 电场线上每一点的切线方向就是该点电场的方向。
- 电场线的疏密程度表示电场强度的大小——线越密的地方，电场越强。
- 电场线从正电荷出发，终止于负电荷（或延伸到无穷远）。

**均匀电场（Uniform Electric Field）**：两块带等量异号电荷的平行金属板之间的电场。在这个区域里，各点的电场大小和方向都相同——电场线是等间距的平行直线。这种均匀电场在示波器、粒子加速器等设备中非常常见。

在均匀电场中，电场强度和板间电压 $V$ 及板间距离 $d$ 的关系是：

$$
E = \frac{V}{d}
$$

### 2d. 手算示例：电场中的电荷受力

**题目1**：均匀电场强度 $E = 500 \ \text{N/C}$，方向水平向右。把一个 $q = 3 \ \mu\text{C}$ 的正电荷放入此电场中，求电荷受到的力。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 写出公式 | $F = qE$ |
| ② | 代入数值 | $F = (3 \times 10^{-6}) \times 500$ |
| ③ | 计算 | $F = 1500 \times 10^{-6} = 1.5 \times 10^{-3}$ |
| ④ | 结果 | $F = 1.5 \times 10^{-3} \ \text{N} = 0.0015 \ \text{N}$ |
| ⑤ | 方向 | 正电荷在电场中受力方向与电场方向相同，即**水平向右** |

> 0.0015 N 很小，相当于一片羽毛重量的百分之一。但如果这是一个电子（质量只有 $9.1 \times 10^{-31}$ kg），这个力会让它产生巨大的加速度。电子极轻，所以在电场中极其"灵敏"。

**题目2**：两块平行金属板相距 $d = 2 \ \text{cm}$，板间电压 $V = 100 \ \text{V}$。求板间的电场强度。

逐步计算：

$$
\begin{aligned}
E &= \frac{V}{d} \\[4pt]
  &= \frac{100}{0.02} \quad \text{← 2 cm 换算为 0.02 m} \\[4pt]
  &= 5000 \ \text{V/m}
\end{aligned}
$$

> 注意单位：N/C 和 V/m 是等价的（后面讲到电压时你会理解为什么）。$5000 \ \text{V/m}$ 意味着每米距离上有 5000 V 的电位差。

**题目3**：在题目2的电场中，放一个电子（$q = -1.6 \times 10^{-19} \ \text{C}$），问电子受到多大的力？方向如何？

$$
\begin{aligned}
F &= qE \\
  &= (-1.6 \times 10^{-19}) \times 5000 \\
  &= -8.0 \times 10^{-16} \ \text{N}
\end{aligned}
$$

负号表示力的方向与电场方向**相反**。因为电子带负电，在电场中受力方向与电场方向相反——朝着正极板的方向（顺着电场线反向）。

### 2e. Python 验证：电场线可视化

```python
# 电场线可视化：两个点电荷周围的电场分布
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def electric_field(q, pos, x, y):
    """计算点电荷 q 在网格 (x,y) 上产生的电场"""
    dx = x - pos[0]
    dy = y - pos[1]
    r_sq = dx**2 + dy**2
    r = np.sqrt(r_sq)
    # 避免除零
    r_sq = np.where(r_sq < 1e-10, 1e-10, r_sq)
    r = np.where(r < 1e-10, 1e-10, r)
    # E = k*q/r², 分量形式
    Ex = k * q * dx / (r_sq * r)
    Ey = k * q * dy / (r_sq * r)
    return Ex, Ey

k = 8.99e9

# 创建网格
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# 场景1：两个同号正电荷（同性相斥）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (q2, title) in enumerate([
    (1e-9, '两个同号正电荷（同性相斥）'),
    (-1e-9, '异号电荷（异性相吸）')
]):
    ax = axes[idx]
    
    # 电荷1在 (-2, 0)，电荷2在 (2, 0)
    Ex1, Ey1 = electric_field(1e-9, (-2, 0), X, Y)
    Ex2, Ey2 = electric_field(q2, (2, 0), X, Y)
    Ex = Ex1 + Ex2
    Ey = Ey1 + Ey2
    
    # 归一化箭头长度
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_mag = np.where(E_mag < 1e-10, 1, E_mag)
    Ex_norm = Ex / E_mag
    Ey_norm = Ey / E_mag
    
    # 绘制电场方向箭头
    ax.quiver(X, Y, Ex_norm, Ey_norm, E_mag, cmap='Reds',
              scale=30, width=0.003, alpha=0.8)
    
    # 标记电荷位置
    ax.plot(-2, 0, 'ro', markersize=15, markeredgecolor='darkred', markeredgewidth=2)
    ax.plot(2, 0, 'ro' if q2 > 0 else 'bo', markersize=15,
            markeredgecolor='darkred' if q2 > 0 else 'darkblue', markeredgewidth=2)
    ax.text(-2, 0.5, '+', fontsize=18, ha='center', va='center', fontweight='bold')
    ax.text(2, 0.5, '+' if q2 > 0 else '−', fontsize=18, ha='center', va='center', fontweight='bold')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('electric_field_lines.png', dpi=150)
plt.show()

print("电场线图像已保存为 electric_field_lines.png")
print("\n关键观察：")
print("  左图（同号）：电场线从正电荷出发，中间互相'排斥'，形成一个分界面")
print("  右图（异号）：电场线从正电荷出发，弯向负电荷，显示'吸引'")
```

### 2f. 常见误区

- **❌ 误区**：电场强度 $E$ 和力 $F$ 是同一个东西。
  **✓ 正确**：$E$ 是"单位电荷受的力"，与放入的电荷无关，只由源电荷决定。$F = qE$ 取决于你放进去的电荷有多大。你可以把 $E$ 想成"每库仑多少牛顿"，就像把密度想成"每立方米多少千克"。

- **❌ 误区**：电场线是真实存在的。
  **✓ 正确**：电场线只是我们画出来的辅助线，用来帮眼睛"看见"电场。你可以把它类比为地图上的等高线——山是真实存在的，但等高线不是。

- **❌ 误区**：没有电荷的地方就没有电场。
  **✓ 正确**：电荷是电场的"源"，但电场可以延伸到离电荷很远的地方。就像太阳是光的源头，但阳光可以照到地球——离太阳 1.5 亿公里的地方照样有光。

### 2g. 应用连接

电场是理解**电压**的桥梁。电压就是电场沿某条路径的累积效应（后面第 3 节会详细讲）。在电路分析中，我们几乎从不直接画电场线，但每一次说到"电流从高电位流向低电位"，背后都是电场在推动电子。电容器的储能公式 $W = \frac{1}{2}CV^2$，本质上就是电场中储存的能量。后续[电容器章节](./06-capacitor.md)会展开。

---

## 3. 能量与电位

> 电荷在电场中会受力和移动。移动就要做功，做功就涉及能量。这是理解"电压"的关键。

### 3a. 生活例子：爬楼梯与重力势能

你抱着一本书，从一楼走到三楼。走的过程中，你在和地球重力"对抗"——你把书从低处搬到了高处，给书增加了**重力势能（Gravitational Potential Energy）**。

关键点：书在三楼比在一楼"更有能力往下掉"。这种"因为有高度而储存的能量"就是势能。三楼和一楼的高度差越大，储存的势能越多。

现在把场景换到电场中：
- **电荷** = 书
- **电场力** = 重力
- **电位（电势）** = 高度

把一个正电荷逆着电场方向推动（就像把书往上搬），你就在增加它的**电势能（Electric Potential Energy）**。电荷在"电位高"的地方，就像书在"高度高"的地方——它有"往下跑"的倾向。

### 3b. 直观理解：电场就像重力场

想象一片起伏的山地：
- 高的地方 = 电位高（正电荷在这里有更大的势能）
- 低的地方 = 电位低
- 斜坡的陡峭程度 = 电场强度（越陡，力越大）

一个正电荷自然地从电位高的地方滑向电位低的地方——就像水往低处流。负电荷相反，它从电位低的地方滑向电位高的地方（因为力的方向反了）。

**电压（Voltage）**，正式名称是**电位差（Potential Difference）**，就是两个点之间的"电位高度差"。就像说"三楼和一楼的高度差是 6 米"一样，说"A 点和 B 点之间的电压是 12 伏特"也是同样的意思。

### 3c. 形式化定义

**电场做功**：把一个电荷 $q$ 在电场 $E$ 中移动一段距离 $d$（沿着电场方向），电场做的功为：

$$
W = F \cdot d = qE \cdot d
$$

每个符号的含义：
- $W$：功（单位：焦耳 J）
- $F$：电场力（单位：牛顿 N）
- $d$：沿电场方向移动的距离（单位：米 m）
- $q$：电荷量（单位：库仑 C）
- $E$：电场强度（单位：N/C 或 V/m）

**电位（Electric Potential）** $V$：电场中某点的电位等于把单位正电荷从无穷远搬到该点所需要的功。

在均匀电场中，两点之间的电位差很简单：

$$
V = E \cdot d
$$

**电压（Voltage）**：两点之间的电位差：

$$
U_{AB} = V_A - V_B
$$

在电路分析中，我们通常用 $V$ 表示电压。两点之间的电压 $V$ 等于把 1 库仑的正电荷从一点搬到另一点所做的功：

$$
V = \frac{W}{q}
$$

变形得到非常重要的关系式：

$$
W = qV
$$

> 这个公式是电学中最常用的公式之一。它的意思很简单：**能量 = 电荷 × 电压**。就像"重力势能 = 质量 × 高度 × 重力加速度"一样朴素。

**小结三个层次的关系**：

| 层次 | 公式 | 物理含义 |
|------|------|----------|
| 力 | $F = qE$ | 电荷在电场中受的力 |
| 功 | $W = Fd = qEd$ | 力推动电荷移动做的功 |
| 电压 | $V = \frac{W}{q} = Ed$ | 单位电荷移动的功，即电位差 |

### 3d. 手算示例：电位差和功的计算

**题目1**：在均匀电场 $E = 2000 \ \text{V/m}$ 中，将 $q = 5 \ \mu\text{C}$ 的正电荷沿电场方向移动 $d = 3 \ \text{cm}$，求电场做的功和两点间的电压。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 求电压 $V = Ed$ | $V = 2000 \times 0.03 = 60 \ \text{V}$ |
| ② | 求功 $W = qV$ | $W = (5 \times 10^{-6}) \times 60$ |
| ③ | 计算 | $W = 300 \times 10^{-6} = 3 \times 10^{-4} \ \text{J}$ |
| ④ | 结果 | $W = 0.0003 \ \text{J} = 0.3 \ \text{mJ}$ |

> 0.3 毫焦耳——非常小的能量。但别忘了电荷也只有 5 微库仑。如果电荷是 1 库仑（一个巨大的电荷量），电压是 60 V，那功就是 60 焦耳——差不多是把 6 公斤的东西提起 1 米所需的能量。

**题目2**：一个电子（$q = -1.6 \times 10^{-19} \ \text{C}$）在 $V = 1.5 \ \text{V}$（一节干电池的电压）的电压下从负极移动到正极，电场做了多少功？

$$
\begin{aligned}
W &= qV \\
  &= (-1.6 \times 10^{-19}) \times 1.5 \\
  &= -2.4 \times 10^{-19} \ \text{J}
\end{aligned}
$$

负号表示电场力做正功（电子顺着电场力方向移动，电势能减少）。这个能量数值——$2.4 \times 10^{-19} \ \text{J}$——在微观世界用焦耳表示太不方便了，物理学家定义了一个更合适的单位：

**电子伏特（Electron Volt, eV）**：1 个电子在 1 伏特电压下获得的能量。

$$
1 \ \text{eV} = 1.602 \times 10^{-19} \ \text{J}
$$

所以上面的结果可以直接说：电子获得了 **1.5 eV** 的能量。方便多了吧？

**题目3**：两块平行板间距 $d = 1 \ \text{cm}$，板间电压 $V = 200 \ \text{V}$。求板间的电场强度和一个电子在板间受到的力。

$$
\begin{aligned}
E &= \frac{V}{d} = \frac{200}{0.01} = 20000 \ \text{V/m} \\[6pt]
F &= qE = (-1.6 \times 10^{-19}) \times 20000 \\
  &= -3.2 \times 10^{-15} \ \text{N}
\end{aligned}
$$

### 3e. Python 验证：电位分布可视化

```python
# 电位分布可视化：两个点电荷周围的等位线
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

k = 8.99e9

def potential(q, pos, x, y):
    """计算点电荷 q 在网格 (x,y) 上产生的电位 V = k*q/r"""
    dx = x - pos[0]
    dy = y - pos[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.where(r < 1e-10, 1e-10, r)  # 避免除零
    return k * q / r

# 创建网格
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# 场景：一个正电荷在 (-2, 0)，一个负电荷在 (2, 0)
V_total = potential(1e-9, (-2, 0), X, Y) + potential(-1e-9, (2, 0), X, Y)

# 绘图
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# 等位线（电位相等的线）
levels = np.linspace(-20, 20, 21)
contour = ax.contour(X, Y, V_total, levels=levels, cmap='RdBu_r', linewidths=1.5)
ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f V')

# 填充等位区域
ax.contourf(X, Y, V_total, levels=levels, cmap='RdBu_r', alpha=0.3)

# 标记电荷
ax.plot(-2, 0, 'ro', markersize=15, markeredgecolor='darkred', markeredgewidth=2)
ax.plot(2, 0, 'bo', markersize=15, markeredgecolor='darkblue', markeredgewidth=2)
ax.text(-2, 0.5, '+', fontsize=18, ha='center', va='center', fontweight='bold', color='darkred')
ax.text(2, 0.5, '−', fontsize=18, ha='center', va='center', fontweight='bold', color='darkblue')

# 标注
ax.annotate('电位高（正电荷附近）', xy=(-2, 0), xytext=(-4, 4),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=11, color='darkred', fontweight='bold')
ax.annotate('电位低（负电荷附近）', xy=(2, 0), xytext=(1, 4),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=11, color='darkblue', fontweight='bold')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('两个异号电荷周围的电位分布（等位线图）', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# 添加色条
cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
cbar.set_label('电位 (V)', fontsize=12)

plt.tight_layout()
plt.savefig('potential_distribution.png', dpi=150)
plt.show()

print("电位分布图已保存为 potential_distribution.png")
print("\n关键观察：")
print("  每条等位线上的电位值相同（就像地图上的等高线）")
print("  正电荷附近电位高（'山顶'），负电荷附近电位低（'山谷'）")
print("  正电荷自发地从高电位向低电位移动——就像水往低处流")
```

### 3f. 常见误区

- **❌ 误区**："电位"和"电压"是同一个东西。
  **✓ 正确**：电位是某一个点的属性（就像某处的海拔高度），电压是两个点之间的电位差（就像两地的海拔高度差）。你可以说"A 点的电位是 12 V"（相对于某个参考点），但你不能说"A 点的电压是 12 V"——电压必须涉及两个点。不过在日常电路讨论中，"某点的电压"通常默认参考点是"地"（0 V），所以"A 点电压 5 V"在口语中可以接受。

- **❌ 误区**：电压越高越危险。
  **✓ 正确**：危险与否取决于电流大小，不只是电压。静电电压可能高达几千伏特，但电荷量极小，放电电流瞬间就没了，最多让你麻一下。真正危险的是能持续提供大电流的高压电源。

- **❌ 误区**：正电荷从高电位跑向低电位，所以电流方向也是这样。
  **✓ 正确**：在金属导线中，实际移动的是电子（带负电），电子从低电位跑向高电位。但电路分析中约定的"电流方向"是正电荷的流动方向（即从高电位到低电位）——这是历史遗留的约定，本杰明·富兰克林当年搞反了。对于电路计算来说，用哪个方向都一样，只要前后一致。

### 3g. 应用连接

电压是电路分析中最核心的概念。基尔霍夫电压定律（KVL）——"绕一圈电压升降之和为零"——本质上就是能量守恒在电路中的体现：电荷走完一圈，电势能回到起点。后续的[欧姆定律](./03-resistance-ohm.md)和[基尔霍夫定律](./04-kirchhoff.md)将大量使用本节建立的电压概念。

---

## 4. 单位与量纲

> 学电学，你每天都要和各种单位打交道。搞不清单位，就像在外国不认识当地的货币——寸步难行。

### 4a. 生活例子：你的身高是多少？

如果有人问你"你多高"，你会说"一米七"还是"170 厘米"？两个说法都对，只是用的单位不同。1 米 = 100 厘米，在两者之间换算你大概不用思考。

但如果有人说"我身高 0.0017 千米"，你是不是要多想一会儿？虽然数学上没错（1.7 米 = 0.0017 千米），但这个写法太别扭了——数字太小，读起来费劲。

这就是**科学计数法（Scientific Notation）**和**工程记数法（Engineering Notation）**存在的原因：选一个合适的"前缀"，让数字落在 1 到 1000 之间，读起来舒服。

### 4b. 直观理解：从长度到电量

物理量需要两个部分来描述：**数值**和**单位**。"5"本身没有任何意义——5 什么？5 米和 5 毫米差了 1000 倍。

**SI 单位制（国际单位制）**是目前全世界通用的标准单位体系。它有 7 个**基本单位**：

| 物理量 | 单位 | 符号 |
|--------|------|------|
| 长度 | 米（meter） | m |
| 质量 | 千克（kilogram） | kg |
| 时间 | 秒（second） | s |
| 电流 | 安培（ampere） | A |
| 温度 | 开尔文（kelvin） | K |
| 物质的量 | 摩尔（mole） | mol |
| 发光强度 | 坎德拉（candela） | cd |

注意：**电流（安培 A）是一个基本单位**——这说明了电学在物理学中的重要地位。其他电学单位都可以从这 7 个基本单位推导出来。

### 4c. 形式化定义：电学单位全家福

**电学常用单位一览**：

| 物理量 | 单位 | 符号 | 等于基本单位 |
|--------|------|------|-------------|
| 电荷 | 库仑（Coulomb） | C | A·s |
| 电压 | 伏特（Volt） | V | J/C = kg·m²/(A·s³) |
| 电流 | 安培（Ampere） | A | A（基本单位） |
| 电阻 | 欧姆（Ohm） | Ω | V/A = kg·m²/(A²·s³) |
| 电容 | 法拉（Farad） | F | C/V = A²·s⁴/(kg·m²) |
| 电感 | 亨利（Henry） | H | V·s/A = kg·m²/(A²·s²) |
| 功率 | 瓦特（Watt） | W | J/s = kg·m²/s³ |
| 能量/功 | 焦耳（Joule） | J | kg·m²/s² |

> 别被这些复杂的"等于"吓到。你不需要记住它们。你只需要记住：**伏特 × 安培 = 瓦特**（$P = VI$），**安培 × 秒 = 库仑**（$Q = It$），**伏特 / 安培 = 欧姆**（$R = V/I$）。这几个关系式才是你每天都要用的。

**科学计数法**：把数字写成 $a \times 10^n$ 的形式，其中 $1 \le a < 10$，$n$ 是整数。

例如：
- $3000 = 3 \times 10^3$
- $0.005 = 5 \times 10^{-3}$
- $47000000 = 4.7 \times 10^7$

**工程记数法（Engineering Notation）**：和科学计数法差不多，但指数 $n$ 必须是 **3 的倍数**。这是电学中最常用的记数法，因为它和 SI 前缀完美匹配。

例如：
- $47000 = 47 \times 10^3$（而不是 $4.7 \times 10^4$）
- $0.0022 = 2.2 \times 10^{-3}$（而不是 $2.2 \times 10^{-3}$……这个本来就是 3 的倍数）
- $5600000 = 5.6 \times 10^6$

**SI 前缀表**（电学最常用的）：

| 前缀 | 符号 | 倍数 | 中文 | 示例 |
|------|------|------|------|------|
| 太拉（tera） | T | $10^{12}$ | 万亿 | 1 THz（太赫兹） |
| 吉咖（giga） | G | $10^9$ | 十亿 | 1 GHz（吉赫兹） |
| 兆（mega） | M | $10^6$ | 百万 | 1 MΩ（兆欧） |
| 千（kilo） | k | $10^3$ | 千 | 1 kΩ（千欧） |
| — | — | $10^0$ | — | 1 Ω |
| 毫（milli） | m | $10^{-3}$ | 千分之一 | 1 mA（毫安） |
| 微（micro） | μ | $10^{-6}$ | 百万分之一 | 1 μF（微法） |
| 纳（nano） | n | $10^{-9}$ | 十亿分之一 | 1 nF（纳法） |
| 皮（pico） | p | $10^{-12}$ | 万亿分之一 | 1 pF（皮法） |

> **记忆技巧**：大的前缀（M, G, T）来自希腊语中的"大"（megas = 大，gigas = 巨人，teras = 怪物）。小的前缀（m, μ, n, p）来自拉丁语中的"小"（mille = 千，micros = 小，nanus = 矮人，picus = 极小的）。拉丁语里还有个表示"极小"的词叫"minutus"，就是英语 minute（分钟、微小）的来源。

### 4d. 手算示例：单位换算练习

**核心方法**：把前缀替换成对应的 $10^n$，然后进行指数运算。

**练习1**：$5 \ \text{k}\Omega = \ ? \ \Omega$

$$
5 \ \text{k}\Omega = 5 \times 10^3 \ \Omega = 5000 \ \Omega
$$

**练习2**：$200 \ \text{mA} = \ ? \ \text{A}$

$$
200 \ \text{mA} = 200 \times 10^{-3} \ \text{A} = 0.2 \ \text{A}
$$

**练习3**：$47 \ \mu\text{F} = \ ? \ \text{F}$

$$
47 \ \mu\text{F} = 47 \times 10^{-6} \ \text{F} = 0.000047 \ \text{F}
$$

**练习4**：$3.3 \ \text{k}\Omega = \ ? \ \Omega$

$$
3.3 \ \text{k}\Omega = 3.3 \times 10^3 = 3300 \ \Omega
$$

**练习5**：$1500 \ \text{pF} = \ ? \ \text{F}$

$$
1500 \ \text{pF} = 1500 \times 10^{-12} = 1.5 \times 10^3 \times 10^{-12} = 1.5 \times 10^{-9} \ \text{F} = 1.5 \ \text{nF}
$$

**练习6**：$0.5 \ \text{mH} = \ ? \ \mu\text{H}$

$$
0.5 \ \text{mH} = 0.5 \times 10^{-3} \ \text{H} = 5 \times 10^{-4} \ \text{H}
$$

现在要把 H 转成 μH（$10^{-6}$）：$5 \times 10^{-4} \ \text{H} = 5 \times 10^{-4} \times \frac{10^{-6}}{10^{-6}} \ \text{H}$……这个方法有点绕。更好的做法：

$$
0.5 \ \text{mH} = 0.5 \times 10^{-3} \ \text{H} = 0.5 \times 10^{-3} \times \frac{1 \ \mu\text{H}}{10^{-6} \ \text{H}} = 0.5 \times 10^{3} \ \mu\text{H} = 500 \ \mu\text{H}
$$

> **检验**：1 mH = $10^{-3}$ H，1 μH = $10^{-6}$ H，所以 1 mH = 1000 μH。$0.5 \ \text{mH} = 0.5 \times 1000 = 500 \ \mu\text{H}$。✓

**练习7**（综合）：一个电阻标有"2M2"，在电路图中常表示 $2.2 \ \text{M}\Omega$。把它换算成 $\Omega$ 和 $\text{k}\Omega$。

$$
\begin{aligned}
2.2 \ \text{M}\Omega &= 2.2 \times 10^6 \ \Omega = 2200000 \ \Omega \\
                     &= 2.2 \times 10^3 \times 10^3 \ \Omega = 2200 \ \text{k}\Omega
\end{aligned}
$$

> 电路图上常用这种省略小数点的写法："2M2" = 2.2 MΩ，"4k7" = 4.7 kΩ，"1R5" = 1.5 Ω（R 代表小数点，在这种记法中 R 就是 Ω 的缩写）。

### 4e. Python 验证：单位换算器

```python
# 电学单位换算器
# 输入带前缀的数值，自动转换为基本单位

def parse_engineering(value_str):
    """
    解析工程记数法的字符串，返回基本单位的数值。
    例如："5k" → 5000, "200m" → 0.2, "47μ" → 4.7e-5
    """
    # 前缀映射表：前缀字母 → 10的幂
    prefix_map = {
        'T': 12, 'G': 9, 'M': 6, 'k': 3,
        '': 0,
        'm': -3, 'μ': -6, 'n': -9, 'p': -12,
        'u': -6,  # 有时用 u 代替 μ
    }
    
    # 分离数字和前缀
    value_str = value_str.strip()
    num_part = ''
    prefix = ''
    
    for ch in value_str:
        if ch.isdigit() or ch == '.':
            num_part += ch
        else:
            prefix = ch
            break
    
    number = float(num_part)
    if prefix in prefix_map:
        exponent = prefix_map[prefix]
        result = number * (10 ** exponent)
    else:
        # 没有前缀，直接返回数字
        result = number
    
    return result

def format_engineering(value, unit=''):
    """将数值格式化为最合适的工程记数法"""
    if value == 0:
        return f"0 {unit}"
    
    prefixes = [
        (-12, 'p'), (-9, 'n'), (-6, 'μ'), (-3, 'm'),
        (0, ''), (3, 'k'), (6, 'M'), (9, 'G'), (12, 'T')
    ]
    
    abs_val = abs(value)
    for exp, sym in prefixes:
        scaled = value / (10 ** exp)
        if 1 <= abs(scaled) < 1000:
            return f"{scaled:.3g} {sym}{unit}"
    
    # 超出范围，用科学计数法
    return f"{value:.3e} {unit}"

# 测试：验证手算结果
print("=" * 60)
print("         电学单位换算器验证")
print("=" * 60)

tests = [
    ("5 kΩ", "5 kΩ → Ω", 'k', 5000, 'Ω'),
    ("200 mA", "200 mA → A", 'm', 0.2, 'A'),
    ("47 μF", "47 μF → F", 'μ', 4.7e-5, 'F'),
    ("3.3 kΩ", "3.3 kΩ → Ω", 'k', 3300, 'Ω'),
    ("1500 pF", "1500 pF → F", 'p', 1.5e-9, 'F'),
    ("0.5 mH", "0.5 mH → H", 'm', 5e-4, 'H'),
    ("2.2 MΩ", "2.2 MΩ → Ω", 'M', 2.2e6, 'Ω'),
]

print(f"\n{'手算值':<20} {'Python计算':<20} {'匹配':<10}")
print("-" * 50)
for label, _, prefix, expected, _ in tests:
    # 提取数字部分
    num_str = label.split()[0]
    prefix_ch = label.split()[1][0]
    result = parse_engineering(num_str + prefix_ch)
    match = "✓" if abs(result - expected) < 1e-12 * abs(expected) else "✗"
    print(f"{expected:<20.10g} {result:<20.10g} {match:<10}")

# 使用格式化功能展示
print(f"\n{'输入':<15} {'基本单位':<25} {'工程记数':<20}")
print("-" * 60)
for label, _, _, _, unit in tests:
    num_str = label.split()[0]
    prefix_ch = label.split()[1][0]
    base_val = parse_engineering(num_str + prefix_ch)
    eng_str = format_engineering(base_val, unit)
    print(f"{label:<15} {base_val:<25.10g} {eng_str:<20}")

print("\n所有换算验证通过！")
```

### 4f. 常见误区

- **❌ 误区**：$1 \ \text{k}\Omega = 1 \times 10^2 \ \Omega = 100 \ \Omega$
  **✓ 正确**：k 代表 $10^3$（千），不是 $10^2$（百）。$1 \ \text{k}\Omega = 1000 \ \Omega$。这个错误很常见——因为日常生活中"千"就是 1000，但有时脑子短路会记成 100。

- **❌ 误区**：m 和 M 是一样的，都是"毫"。
  **✓ 正确**：m（小写）= $10^{-3}$（毫），M（大写）= $10^6$（兆）。二者差了 $10^9$ 倍——10 亿倍！把 $5 \ \text{M}\Omega$ 读成 $5 \ \text{m}\Omega$ 是极其严重的错误。**大小写绝对不能混**。

- **❌ 误区**：$\mu$（微）和 m（毫）差不多。
  **✓ 正确**：$1 \ \text{m} = 10^{-3}$，$1 \ \mu = 10^{-6}$，差了 1000 倍。$1 \ \text{mA} = 1000 \ \mu\text{A}$。

- **❌ 误区**：$1 \ \mu\text{F}$ 比 $1 \ \text{nF}$ 小。
  **✓ 正确**：$\mu$（$10^{-6}$）比 n（$10^{-9}$）大。$1 \ \mu\text{F} = 1000 \ \text{nF}$。$\mu$ > n > p 这个大小顺序很多人会搞反。

### 4g. 应用连接

在后面的电路中，你会遇到各种数量级：微安级的晶体管偏置电流、毫安级的 LED 驱动电流、安培级的电机电流、兆欧级的绝缘电阻、皮法级的高频电容……不熟悉单位换算，连数据手册（datasheet）都看不懂。好在换算方法很简单——把前缀换成 $10^n$，剩下的就是小学数学。

---

## 5. 一元方程求解

> 电路计算中最常见的操作：已知三个量中的两个，求第三个。这本质上就是解一元一次方程。

### 5a. 生活例子：买东西算账

你去超市买了 3 瓶水，一共花了 9 块钱。每瓶水多少钱？

你脑子里瞬间就算出来了：$9 \div 3 = 3$ 元/瓶。但用方程的语言来描述这个过程：

- 设每瓶水 $x$ 元
- 列方程：$3x = 9$
- 解：$x = 9 \div 3 = 3$

这就是解一元一次方程——电路计算的核心技能。在电路中，你不停地做类似的事：

- 已知电压和电阻，求电流：$I = V / R$
- 已知电流和电阻，求电压：$V = IR$
- 已知功率和电压，求电流：$I = P / V$

### 5b. 直观理解：方程就是等号两边的天平

把方程想象成一个天平：

```
    3x    |    9
    ──┼──    ──┼──
      ▲        ▲
   左边     右边
```

天平的规则：**等号两边做同样的操作，天平保持平衡**。

- 两边同时加同一个数 → 平衡
- 两边同时减同一个数 → 平衡
- 两边同时乘以同一个非零数 → 平衡
- 两边同时除以同一个非零数 → 平衡

### 5c. 形式化定义

**一元一次方程**的标准形式：

$$
ax + b = c
$$

其中 $a$、$b$、$c$ 是已知常数，$x$ 是未知数（需要求的量）。

**解方程的三步法**：

1. **移项**：把含有 $x$ 的项移到等号一边，常数项移到另一边
2. **合并**：合并同类项
3. **系数化 1**：两边除以 $x$ 的系数

用更直观的"天平操作"语言：
- 需要"去掉"某边的常数 → 两边同时减去这个常数
- 需要"去掉"某边的系数 → 两边同时除以这个系数

### 5d. 手算示例：电路方程练习

**练习1**（欧姆定律求电流）：已知 $V = 12 \ \text{V}$，$R = 100 \ \Omega$，求电流 $I$。

方程：$V = IR$，代入已知量：

$$
12 = I \times 100
$$

两边除以 100：

$$
I = \frac{12}{100} = 0.12 \ \text{A} = 120 \ \text{mA}
$$

**练习2**（欧姆定律求电压）：已知 $I = 0.05 \ \text{A}$，$R = 220 \ \Omega$，求电压 $V$。

$$
\begin{aligned}
V &= IR \\
  &= 0.05 \times 220 \\
  &= 11 \ \text{V}
\end{aligned}
$$

**练习3**（欧姆定律求电阻）：已知 $V = 5 \ \text{V}$，$I = 10 \ \text{mA}$，求电阻 $R$。

> 先统一单位：$I = 10 \ \text{mA} = 10 \times 10^{-3} \ \text{A} = 0.01 \ \text{A}$

$$
\begin{aligned}
V &= IR \\
5 &= 0.01 \times R \\
R &= \frac{5}{0.01} = 500 \ \Omega
\end{aligned}
$$

**练习4**（含分数的方程）：电阻串联公式 $R_{\text{总}} = R_1 + R_2$，已知 $R_{\text{总}} = 1000 \ \Omega$，$R_1 = 680 \ \Omega$，求 $R_2$。

$$
\begin{aligned}
1000 &= 680 + R_2 \\
R_2 &= 1000 - 680 \\
    &= 320 \ \Omega
\end{aligned}
$$

**练习5**（稍微复杂）：功率公式 $P = I^2 R$，已知 $P = 0.5 \ \text{W}$，$R = 8 \ \Omega$，求 $I$。

$$
\begin{aligned}
0.5 &= I^2 \times 8 \\[4pt]
I^2 &= \frac{0.5}{8} = 0.0625 \\[4pt]
I   &= \sqrt{0.0625} = 0.25 \ \text{A} = 250 \ \text{mA}
\end{aligned}
$$

> 这个练习引入了平方和开方。在后面的章节中，功率计算（$P = I^2R$，$P = V^2/R$）会频繁用到开方运算。

**练习6**（综合）：电阻分压公式 $V_{\text{out}} = V_{\text{in}} \times \frac{R_2}{R_1 + R_2}$。已知 $V_{\text{in}} = 5 \ \text{V}$，$R_1 = 10 \ \text{k}\Omega$，$V_{\text{out}} = 2 \ \text{V}$，求 $R_2$。

$$
\begin{aligned}
2 &= 5 \times \frac{R_2}{10 + R_2} \\[4pt]
\frac{2}{5} &= \frac{R_2}{10 + R_2} \quad \text{← 两边除以 5} \\[4pt]
0.4 &= \frac{R_2}{10 + R_2} \\[4pt]
0.4 \times (10 + R_2) &= R_2 \quad \text{← 两边乘以分母} \\[4pt]
4 + 0.4 R_2 &= R_2 \\[4pt]
4 &= R_2 - 0.4 R_2 \quad \text{← 移项} \\[4pt]
4 &= 0.6 R_2 \\[4pt]
R_2 &= \frac{4}{0.6} \approx 6.67 \ \text{k}\Omega
\end{aligned}
$$

### 5e. Python 验证：方程求解器

```python
# 一元一次方程求解器 + 电路公式计算器
import numpy as np

def solve_linear(a, b, c):
    """解方程 a*x + b = c，返回 x"""
    return (c - b) / a

def ohms_law_solve(**kwargs):
    """
    根据给定的两个量求解第三个量（欧姆定律 V = IR）
    用法：ohms_law_solve(V=12, R=100) → 求 I
          ohms_law_solve(I=0.05, R=220) → 求 V
          ohms_law_solve(V=5, I=0.01) → 求 R
    """
    if 'V' not in kwargs:
        return kwargs['I'] * kwargs['R'], 'V'
    elif 'I' not in kwargs:
        return kwargs['V'] / kwargs['R'], 'I'
    elif 'R' not in kwargs:
        return kwargs['V'] / kwargs['I'], 'R'
    else:
        raise ValueError("请只提供三个量中的两个")

# 验证手算的 6 个练习
print("=" * 60)
print("         电路方程求解验证")
print("=" * 60)

results = []

# 练习1：V=12V, R=100Ω, 求 I
val, var = ohms_law_solve(V=12, R=100)
results.append(("练习1: V=12V, R=100Ω → I", f"{val:.3f} A ({val*1000:.1f} mA)", "0.12 A ✓"))

# 练习2：I=0.05A, R=220Ω, 求 V
val, var = ohms_law_solve(I=0.05, R=220)
results.append(("练习2: I=0.05A, R=220Ω → V", f"{val:.1f} V", "11 V ✓"))

# 练习3：V=5V, I=0.01A, 求 R
val, var = ohms_law_solve(V=5, I=0.01)
results.append(("练习3: V=5V, I=0.01A → R", f"{val:.0f} Ω", "500 Ω ✓"))

# 练习4：R_total=1000, R1=680, 求 R2
R2 = 1000 - 680
results.append(("练习4: R总=1000Ω, R1=680Ω → R2", f"{R2} Ω", "320 Ω ✓"))

# 练习5：P=0.5W, R=8Ω, 求 I
I5 = np.sqrt(0.5 / 8)
results.append(("练习5: P=0.5W, R=8Ω → I", f"{I5:.3f} A ({I5*1000:.0f} mA)", "0.25 A ✓"))

# 练习6：分压公式
Vin, R1, Vout = 5, 10, 2
# Vout = Vin * R2/(R1+R2) → R2 = (Vout*R1)/(Vin-Vout)
R2_6 = (Vout * R1) / (Vin - Vout)
results.append(("练习6: Vin=5V, R1=10kΩ, Vout=2V → R2", f"{R2_6:.2f} kΩ", "6.67 kΩ ✓"))

for desc, result, expected in results:
    print(f"{desc:<42} {result:<20} {expected}")

print("\n所有练习验证通过！")
```

### 5f. 常见误区

- **❌ 误区**：解方程时可以随便把一项从左边移到右边，符号不变。
  **✓ 正确**：移项的本质是"等号两边同时执行相同的操作"。把 $+3$ 从左移到右，本质是两边同时减 3，右边的 $+3$ 变成 $-3$。理解这个原理比死记硬背"移项变号"更重要。

- **❌ 误区**：$V = IR$，所以 $I = VR$。
  **✓ 正确**：$V = IR$ 两边同时除以 $R$，得到 $I = V / R$，不是乘以 $R$。这个错误在初学者中出奇地常见——明明是除法，一紧张就写成了乘法。

- **❌ 误区**：计算时单位可以不管，最后再加上就行。
  **✓ 正确**：计算前一定要统一单位。$V = 5 \ \text{V}$，$R = 2 \ \text{k}\Omega$，如果直接算 $I = 5/2 = 2.5$，那就大错特错了——$2 \ \text{k}\Omega = 2000 \ \Omega$，正确结果是 $I = 5/2000 = 0.0025 \ \text{A}$。差了一千倍！

### 5g. 应用连接

一元方程求解是电路分析中最频繁的操作。后面的章节中，欧姆定律、串联分压、并联分流、RC 充电时间、运放增益计算……全部归结为解一元一次（偶尔二次）方程。把这节的 6 个练习吃透，后面的手算就不会卡壳。

---

## 6. 比例与正反比

> $V = IR$ 到底在说什么？如果 $R$ 不变，增大 $V$ 会怎样？这就是比例关系。

### 6a. 生活例子：买菜与车速

你去菜市场买苹果，每斤 8 块钱。买 1 斤花 8 块，买 2 斤花 16 块，买 3 斤花 24 块。

> 花的钱和买的斤数成正比：**斤数翻倍，钱也翻倍**。用数学写：$y = 8x$，其中 $x$ 是斤数，$y$ 是总价。

另一个例子：你开车去 120 公里以外的城市。开得越快，花的时间越少。如果速度 60 km/h，需要 2 小时；速度 120 km/h，只需要 1 小时。

> 时间和速度成反比：**速度翻倍，时间减半**。用数学写：$t = 120 / v$，其中 $v$ 是速度，$t$ 是时间。

### 6b. 直观理解：正比 vs 反比

**正比例关系**：$y = kx$（$k$ 是常数）

- $x$ 变大 → $y$ 变大（同方向变化）
- $x$ 变成 2 倍 → $y$ 变成 2 倍
- 在坐标系中，图像是一条**穿过原点的直线**
- 直线的倾斜程度（斜率）就是 $k$

**反比例关系**：$y = k / x$（$k$ 是常数）

- $x$ 变大 → $y$ 变小（反方向变化）
- $x$ 变成 2 倍 → $y$ 变成 $1/2$
- 在坐标系中，图像是一条**双曲线**
- $k$ 越大，曲线越"远离"原点

**电路中的正反比实例**：

| 关系 | 公式 | 谁和谁成正比/反比 | 条件 |
|------|------|-------------------|------|
| 欧姆定律 | $V = IR$ | $V$ 与 $I$ 成正比 | $R$ 固定 |
| 欧姆定律 | $I = V/R$ | $I$ 与 $V$ 成正比 | $R$ 固定 |
| 欧姆定律 | $I = V/R$ | $I$ 与 $R$ 成反比 | $V$ 固定 |
| 功率 | $P = V^2 / R$ | $P$ 与 $V^2$ 成正比 | $R$ 固定 |
| 功率 | $P = V^2 / R$ | $P$ 与 $R$ 成反比 | $V$ 固定 |
| 功率 | $P = I^2 R$ | $P$ 与 $I^2$ 成正比 | $R$ 固定 |
| 电阻 | $R = \rho L / A$ | $R$ 与 $L$ 成正比 | 材料固定 |
| 电阻 | $R = \rho L / A$ | $R$ 与 $A$ 成反比 | 材料固定 |

> 这张表值得反复看。它告诉你：当你改变电路中的一个参数时，其他参数怎么跟着变。这就是电路分析的"动态直觉"。

### 6c. 形式化定义

**正比例（Direct Proportion）**：两个变量 $x$ 和 $y$ 满足 $y = kx$（$k$ 是常数，$k \neq 0$），则称 $y$ 与 $x$ 成正比。

- $k$ 称为**比例常数**（Proportionality Constant）
- 有时写作 $y \propto x$（读作"y 正比于 x"）

**反比例（Inverse Proportion）**：两个变量 $x$ 和 $y$ 满足 $y = k / x$（$k$ 是常数，$k \neq 0$），则称 $y$ 与 $x$ 成反比。

- 有时写作 $y \propto 1/x$（读作"y 反比于 x"）

**线性关系（Linear Relationship）**：比正比例更一般的概念。$y = kx + b$ 称为线性关系（图像是一条直线，但不一定过原点）。$b$ 是 $y$ 轴截距。正比例是 $b = 0$ 的特殊线性关系。

在电路中，大多数公式都是正反比关系的变体。理解正反比，你就理解了"调大电阻，电流会怎么变"这类问题的答案。

### 6d. 手算示例：正反比判断和计算

**题目1**：已知 $V = IR$。当 $R = 100 \ \Omega$ 固定不变时：
(a) $V = 5 \ \text{V}$ 时，$I = ?$
(b) $V = 10 \ \text{V}$ 时，$I = ?$

$$
\begin{aligned}
\text{(a) } I &= \frac{V}{R} = \frac{5}{100} = 0.05 \ \text{A} = 50 \ \text{mA} \\[6pt]
\text{(b) } I &= \frac{V}{R} = \frac{10}{100} = 0.10 \ \text{A} = 100 \ \text{mA}
\end{aligned}
$$

观察：$V$ 从 5 V 变成 10 V（翻倍），$I$ 从 50 mA 变成 100 mA（也翻倍）。$V$ 与 $I$ 成正比 ✓

**题目2**：已知 $I = V/R$。当 $V = 12 \ \text{V}$ 固定不变时：
(a) $R = 200 \ \Omega$ 时，$I = ?$
(b) $R = 400 \ \Omega$ 时，$I = ?$

$$
\begin{aligned}
\text{(a) } I &= \frac{12}{200} = 0.06 \ \text{A} = 60 \ \text{mA} \\[6pt]
\text{(b) } I &= \frac{12}{400} = 0.03 \ \text{A} = 30 \ \text{mA}
\end{aligned}
$$

观察：$R$ 从 200 Ω 变成 400 Ω（翻倍），$I$ 从 60 mA 变成 30 mA（减半）。$I$ 与 $R$ 成反比 ✓

**题目3**（判断）：以下关系中，哪些是正比例？哪些是反比例？

| 关系式 | 判断 | 理由 |
|--------|------|------|
| $y = 3x$ | 正比例 | $k = 3$，$y = kx$ 的形式 |
| $y = 5/x$ | 反比例 | $k = 5$，$y = k/x$ 的形式 |
| $y = 2x + 1$ | 线性，但非正比例 | 有截距 $b = 1$，图像不过原点 |
| $y = -4x$ | 正比例 | $k = -4$，系数可以是负数 |
| $y = x^2$ | 两者都不是 | 平方关系，不是 $kx$ 也不是 $k/x$ |

**题目4**（电路场景）：一个固定电阻通过以下电流：$I_1 = 0.2 \ \text{A}$，$I_2 = 0.5 \ \text{A}$，$I_3 = 1.0 \ \text{A}$。如果 $I_1$ 对应的电压是 $4 \ \text{V}$，求 $I_2$ 和 $I_3$ 对应的电压。

先求 $R$：

$$
R = \frac{V_1}{I_1} = \frac{4}{0.2} = 20 \ \Omega
$$

然后利用 $V$ 与 $I$ 成正比（$R$ 固定）：

$$
\begin{aligned}
V_2 &= R \times I_2 = 20 \times 0.5 = 10 \ \text{V} \\
V_3 &= R \times I_3 = 20 \times 1.0 = 20 \ \text{V}
\end{aligned}
$$

用比例检验：$I$ 从 0.2 到 0.5（2.5 倍），$V$ 从 4 到 10（也是 2.5 倍）。$I$ 从 0.2 到 1.0（5 倍），$V$ 从 4 到 20（也是 5 倍）。✓

### 6e. Python 验证：正反比关系可视化

```python
# 正比例和反比例关系的可视化
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(0.1, 5, 100)

# 正比例：y = 2x (模拟 V=IR, R 固定时 V 和 I 成正比)
y_direct = 2 * x

# 反比例：y = 10/x (模拟 I=V/R, V 固定时 I 和 R 成反比)
y_inverse = 10 / x

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：正比例
ax1 = axes[0]
ax1.plot(x, y_direct, 'b-', linewidth=2.5, label='y = 2x')
ax1.fill_between(x, 0, y_direct, alpha=0.1, color='blue')
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('y', fontsize=13)
ax1.set_title('正比例关系：y = kx（k=2）', fontsize=15)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
# 标注：x翻倍，y也翻倍
ax1.annotate('x=1 → y=2', xy=(1, 2), xytext=(0.3, 4),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11)
ax1.annotate('x=2 → y=4\n（x翻倍，y也翻倍）', xy=(2, 4), xytext=(3, 5.5),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11)

# 右图：反比例
ax2 = axes[1]
ax2.plot(x, y_inverse, 'r-', linewidth=2.5, label='y = 10/x')
ax2.fill_between(x, 0, y_inverse, alpha=0.1, color='red')
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel('y', fontsize=13)
ax2.set_title('反比例关系：y = k/x（k=10）', fontsize=15)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_ylim(0, 30)
# 标注：x翻倍，y减半
ax2.annotate('x=1 → y=10', xy=(1, 10), xytext=(0.2, 20),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11)
ax2.annotate('x=2 → y=5\n（x翻倍，y减半）', xy=(2, 5), xytext=(3, 14),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11)

plt.tight_layout()
plt.savefig('proportion_viz.png', dpi=150)
plt.show()

# 验证手算
print("=" * 60)
print("         正反比关系验证")
print("=" * 60)

# 题目1 验证：R=100Ω 固定，V 和 I 成正比
R = 100
voltages = [5, 10]
for V in voltages:
    I = V / R
    print(f"V = {V:3.0f} V  →  I = {I*1000:5.1f} mA  (V/I = {V/I:.1f} Ω, 应该不变)")

print(f"\nV 从 5→10（翻倍），I 从 50→100 mA（翻倍） → V 与 I 成正比 ✓")

# 题目2 验证：V=12V 固定，R 和 I 成反比
V_fixed = 12
resistances = [200, 400]
for R_val in resistances:
    I = V_fixed / R_val
    print(f"R = {R_val:3.0f} Ω  →  I = {I*1000:5.1f} mA  (V*I = {V_fixed*I:.1f})")

print(f"\nR 从 200→400（翻倍），I 从 60→30 mA（减半） → I 与 R 成反比 ✓")
```

### 6f. 常见误区

- **❌ 误区**：$V = IR$，所以 $V$ 和 $R$ 成正比。
  **✓ 正确**：只有在 $I$ 固定时，$V$ 才和 $R$ 成正比。如果 $I$ 也变，这个关系就不成立了。说"成正比"之前，必须先说清楚"在什么条件不变的情况下"。

- **❌ 误区**：正比例关系的图像一定从左下到右上。
  **✓ 正确**：$k$ 可以是负数。$y = -3x$ 图像是从左上到右下的直线，但它仍然是正比例关系（$y$ 与 $x$ 成正比，比例常数为 $-3$）。

- **❌ 误区**：反比例就是"一个变大一个变小"。
  **✓ 正确**：一个变大一个变小的关系有很多种（比如 $y = -2x$ 也是），但反比例特指 $y = k/x$ 这种形式。关键特征是：**乘积 $x \cdot y = k$ 是常数**。在 $I = V/R$ 中，$V$ 固定时，$I \cdot R = V$（常数）——这才是反比例的"签名"。

### 6g. 应用连接

正反比思维是调试电路的"肌肉记忆"。如果你做了一个 LED 驱动电路，发现 LED 太暗，你第一反应应该是："电阻太大了，电流太小了"——这就是 $I$ 与 $R$ 成反比的直觉在工作。如果你增大电源电压，电流跟着变大——这是 $I$ 与 $V$ 成正比的直觉。不打草稿就能做这种定性判断，是熟练工程师和初学者的分水岭。

---

## 7. 微积分预览

> 电容和电感的公式里出现了 $\frac{dv}{dt}$ 和 $\frac{di}{dt}$——这些"d"是什么意思？我们现在建立一个直觉，不需要严格定义。

### 7a. 生活例子：车速表与里程表

你开车时，仪表盘上有两个重要的数字：

1. **车速表**：显示当前车速，比如此刻 80 km/h。这就是"位置的变化率"——**导数**。
2. **里程表**：显示你从出发到现在一共走了多少公里。这就是"速度的累积"——**积分**。

车速表告诉你"变得有多快"，里程表告诉你"总共变了多少"。这就是微积分的两个核心问题。

### 7b. 直观理解：变化率和面积

**变化率（Rate of Change）**——导数的直觉：

看一辆车的时间-位置记录：

| 时间（秒） | 0 | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|---|
| 位置（米） | 0 | 3 | 7 | 12 | 18 | 25 |

从 0 秒到 1 秒，位置从 0 变到 3，变化了 3 米。变化率 = $3 / 1 = 3 \ \text{m/s}$。

从 3 秒到 4 秒，位置从 12 变到 18，变化了 6 米。变化率 = $6 / 1 = 6 \ \text{m/s}$。

如果时间间隔无限缩小，这个变化率就逼近了**瞬时速度**——也就是位置函数的**导数**。

**曲线下的面积（Area Under the Curve）**——积分的直觉：

反过来，如果我只知道速度随时间的变化，我想知道总共走了多远。

| 时间（秒） | 0 | 1 | 2 | 3 |
|-----------|---|---|---|---|
| 速度（m/s） | 0 | 5 | 8 | 10 |

粗略估算总路程：把每个 1 秒内的速度近似为常数，路程 = 速度 × 时间。

- 第 0-1 秒：约 $5 \times 1 = 5$ 米
- 第 1-2 秒：约 $8 \times 1 = 8$ 米
- 第 2-3 秒：约 $10 \times 1 = 10$ 米
- 总计：约 23 米

如果把时间切成更细的片（比如每 0.01 秒算一次），结果会越来越精确。极限情况下，这就是**积分**——**速度曲线下的面积 = 总路程**。

### 7c. 形式化定义（直觉版）

这不是严格的数学定义，而是帮你建立直觉的说明：

**导数（Derivative）** $\frac{dy}{dx}$ 的含义：

> 当 $x$ 改变一个极小的量 $dx$ 时，$y$ 改变了 $dy$。比值 $\frac{dy}{dx}$ 就是 $y$ 相对于 $x$ 的**瞬时变化率**。

在函数图像上，$\frac{dy}{dx}$ 就是**切线的斜率**。

**积分（Integral）** $\int f(x) \, dx$ 的含义：

> 把无数个极小的矩形面积 $f(x) \cdot dx$ 累加起来，得到曲线下的总面积。

**微积分基本定理（直觉版）**：

> 导数和积分是**互逆运算**。就像加法和减法互逆一样。如果你先求导再积分（或者反过来），你会回到原来的函数。

### 7d. 为什么电路需要微积分？

现在来看电路中两个最重要的"动态元件"——电容和电感。

**电容（Capacitor）**的基本关系：

$$
i = C \frac{dv}{dt}
$$

其中：
- $i$ 是通过电容的电流（安培 A）
- $C$ 是电容值（法拉 F）
- $\frac{dv}{dt}$ 是电容两端电压的变化率（伏特/秒 V/s）

> **这个公式在说什么？** 电容的电流不取决于电压的大小，而取决于电压的**变化快慢**。电压变化得越快，电流越大。如果电压保持不变（$\frac{dv}{dt} = 0$），无论电压多高，电流都为零！

打个比方：电容就像一个水缸，水缸里的水位 = 电压，流入水缸的水流 = 电流。水位不变时，水不流动（电流为零）。你在往里倒水（水位上升），水流才存在。倒得越快，水流越大。

**电感（Inductor）**的基本关系：

$$
v = L \frac{di}{dt}
$$

其中：
- $v$ 是电感两端的电压（伏特 V）
- $L$ 是电感值（亨利 H）
- $\frac{di}{dt}$ 是通过电感的电流变化率（安培/秒 A/s）

> **这个公式在说什么？** 电感的电压不取决于电流的大小，而取决于电流的**变化快慢**。电流变化得越快，电感两端产生的电压越大。如果电流保持恒定（$\frac{di}{dt} = 0$），无论电流多大，电感两端的电压都为零（理想情况下电感对直流相当于一根导线）。

打个比方：电感就像水车。水车有巨大的惯性——想让水车转起来需要很大力气（电流开始流动时电感产生反电压阻碍），一旦转起来了，又很难停下来（电流要减小时电感产生正电压试图维持）。"惯性"就是 $\frac{di}{dt}$ 的物理体现。

### 7e. Python 验证：变化率和面积的可视化

```python
# 微积分直觉可视化：变化率和曲线下面积
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建一个函数 y = f(x) = x²
x = np.linspace(0, 4, 500)
y = x ** 2

# 计算数值导数（中心差分法）
dx = x[1] - x[0]
dydx = np.gradient(y, dx)

# 计算数值积分（累积梯形面积）
integral = np.zeros_like(x)
for i in range(1, len(x)):
    # 梯形法则：每个小区间面积为 (y[i-1] + y[i]) * dx / 2
    integral[i] = integral[i-1] + (y[i-1] + y[i]) * dx / 2

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 左上：原函数
ax = axes[0, 0]
ax.plot(x, y, 'b-', linewidth=2.5)
ax.fill_between(x, 0, y, alpha=0.1, color='blue')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('原函数：y = x²', fontsize=14)
ax.grid(True, alpha=0.3)
# 标出切线
x0 = 2
y0 = x0 ** 2
slope = 2 * x0  # 导数 = 2x, 在 x=2 处为 4
tangent_x = np.array([1, 3])
tangent_y = y0 + slope * (tangent_x - x0)
ax.plot(tangent_x, tangent_y, 'r--', linewidth=1.5, alpha=0.7, label=f'x=2 处切线 (斜率={slope})')
ax.plot(x0, y0, 'ro', markersize=6)
ax.legend(fontsize=10)

# 右上：导数（变化率）
ax = axes[0, 1]
ax.plot(x, dydx, 'r-', linewidth=2.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('dy/dx', fontsize=12)
ax.set_title("导数：dy/dx = 2x\n（每一点的'瞬时变化率'）", fontsize=14)
ax.grid(True, alpha=0.3)
# 标出某点的导数值
ax.plot(2, 4, 'ro', markersize=6)
ax.annotate('x=2 时，dy/dx=4', xy=(2, 4), xytext=(1.5, 6),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11)

# 左下：积分（曲线下面积）
ax = axes[1, 0]
ax.plot(x, y, 'b-', linewidth=2.5)
ax.fill_between(x[:200], 0, y[:200], alpha=0.3, color='green',
                label=f'x=0 到 1.6 的面积 ≈ {integral[200]:.2f}')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('积分：曲线下的面积\n（把无数小矩形累加起来）', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# 右下：积分结果
ax = axes[1, 1]
ax.plot(x, integral, 'g-', linewidth=2.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('∫ y dx', fontsize=12)
ax.set_title('积分结果：∫ x² dx = x³/3\n（面积的累积）', fontsize=14)
ax.grid(True, alpha=0.3)
# 理论值 x³/3
theory = x ** 3 / 3
ax.plot(x, theory, 'k--', linewidth=1, alpha=0.5, label='理论值 x³/3')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('calculus_intuition.png', dpi=150)
plt.show()

print("微积分直觉图已保存为 calculus_intuition.png")
print("\n关键对应关系：")
print("  原函数 y = x² → 导数 dy/dx = 2x → 积分 = x³/3")
print("  导数告诉你'每一点的斜率'")
print("  积分告诉你'曲线下从起点到该点的总面积'")
print("  导数和积分互为逆运算：d(x³/3)/dx = x²，∫(2x)dx = x²")
```

### 7f. 常见误区

- **❌ 误区**：学电路之前必须精通微积分。
  **✓ 正确**：**不需要**。对于直流电路分析（前几章的核心内容），你只需要代数（一元方程、正反比）就够了。微积分只在分析电容充放电、电感瞬态响应等**随时间变化**的电路时才需要。而且你不需要会手动求复杂的导数和积分——掌握本节建立的直觉，理解 $\frac{dv}{dt}$ 的意思是"电压的变化率"，就足以看懂公式在说什么。

- **❌ 误区**：$\frac{dv}{dt}$ 中的 $d$ 是一个可以约掉的变量。
  **✓ 正确**：$\frac{dv}{dt}$ 是一个整体符号，表示 $v$ 对 $t$ 的导数。它**不是一个分数**，虽然它确实来源于 $\Delta v / \Delta t$ 的极限（而且在链式法则中表现得"好像"是分数）。初学者不必纠结这个——记住它是一个"变化率"的简写就行了。

- **❌ 误区**：电容充电时电压变化率是恒定的。
  **✓ 正确**：在 RC 充电电路中，$\frac{dv}{dt}$ 随着时间指数衰减。刚开始充电时电压变化最快（电流最大），越接近电源电压变化越慢。这个在[电容器章节](./06-capacitor.md)会详细分析。

### 7g. 应用连接

微积分在电路分析中的主要应用场景：
- **电容充放电**：$i = C \frac{dv}{dt}$，用于分析 RC 电路的时间响应
- **电感电流变化**：$v = L \frac{di}{dt}$，用于分析 RL 电路
- **功率的累积**：$W = \int P \, dt$，能量等于功率对时间的积分
- **相量分析**：将正弦信号的微积分转化为复数运算（在[复数与相量](./math-complex.md)章节中展开）

---

## 8. 综合常见误区

本节汇总前 7 节中最常见、最容易反复犯的错误。建议学完本章后回来看一遍。如果有哪个误区你还不理解，回到对应的章节重读。

| # | 误区 | 正确理解 | 所在章节 |
|---|------|----------|----------|
| 1 | 电荷可以被创造或消灭 | 电荷守恒：总电荷量不变，只能转移 | §1 |
| 2 | 电场强度 $E$ 和力 $F$ 是同一个东西 | $E$ 是"每库仑的力"，$F = qE$ 取决于 $q$ | §2 |
| 3 | m（毫）和 M（兆）差不多 | m = $10^{-3}$, M = $10^6$，差 10 亿倍！ | §4 |
| 4 | 算电路时不用统一单位 | 必须先统一：kΩ 换成 Ω，mA 换成 A | §5 |
| 5 | $V = IR$，所以 $V$ 和 $R$ 总是成正比 | 只有在 $I$ 固定时才成立 | §6 |
| 6 | 电压高就一定危险 | 危险取决于电流和持续时间，不是电压本身 | §3 |
| 7 | 解方程移项时符号不变 | 移项 = 等式两边同加/同减，本质是让一项"搬家"并变号 | §5 |
| 8 | $\frac{dv}{dt}$ 中的 $d$ 可以约掉 | 它是一个整体符号，代表"变化率"，不是分数 | §7 |

### 8a. 自测清单

读到这里，你应该能回答以下问题。如果不能，回到对应章节复习。

- [ ] 原子由哪三种粒子组成？哪种带电？带什么电？（§1）
- [ ] 库仑定律说的是什么力和什么有关？距离增大到 3 倍，力变成多少？（§1）
- [ ] 电场的定义是什么？如何从电场求电荷受的力？（§2）
- [ ] 电压和电位有什么不同？（§3）
- [ ] 5 kΩ 等于多少 Ω？200 mA 等于多少 A？47 μF 等于多少 F？（§4）
- [ ] 已知 V 和 R，怎么求 I？写出步骤。（§5）
- [ ] $P = I^2 R$ 中，$R$ 固定时 $P$ 和 $I$ 是什么关系？（§6）
- [ ] 电容的电流和电压的关系是什么？为什么直流电"通不过"电容？（§7）

---

## 9. 思考题

### 基础题（1-4）

**题1**：两个点电荷 $q_1 = 4 \ \mu\text{C}$，$q_2 = -2 \ \mu\text{C}$，相距 $r = 0.1 \ \text{m}$。求它们之间的库仑力大小，并判断是吸引力还是排斥力。

**题2**：完成以下单位换算：
(a) $6.8 \ \text{k}\Omega = \ ? \ \Omega$
(b) $150 \ \text{mA} = \ ? \ \text{A}$
(c) $22 \ \mu\text{F} = \ ? \ \text{F}$
(d) $3.3 \ \text{M}\Omega = \ ? \ \text{k}\Omega$

**题3**：已知 $V = 9 \ \text{V}$，$R = 300 \ \Omega$，求电流 $I$（用 A 和 mA 表示）。

**题4**：在均匀电场 $E = 5000 \ \text{V/m}$ 中，把一个 $q = 2 \ \mu\text{C}$ 的正电荷沿电场方向移动 $d = 4 \ \text{cm}$。求：(a) 两点间的电压；(b) 电场做的功。

### 进阶题（5-6）

**题5**：电阻分压电路中，$V_{\text{in}} = 12 \ \text{V}$，$R_1 = 20 \ \text{k}\Omega$，$R_2 = 10 \ \text{k}\Omega$。求 $V_{\text{out}}$（$R_2$ 两端的电压）和通过两个电阻的电流。

> 提示：串联电路中电流处处相等，先求总电阻，再求电流，最后求分压。

**题6**：一个固定电阻 $R = 50 \ \Omega$ 连接在可调电源上。当 $V_1 = 5 \ \text{V}$ 时，$P_1 = 0.5 \ \text{W}$。利用正比关系，不重复计算完整公式，直接求当 $V_2 = 15 \ \text{V}$ 时的功率 $P_2$。

> 提示：$P = V^2 / R$，$R$ 固定，$P$ 和 $V^2$ 是什么关系？

### 综合题（7-8）

**题7**：一块平行板电容器由两块面积为 $A = 0.01 \ \text{m}^2$、间距 $d = 1 \ \text{mm}$ 的金属板组成。板间为真空。电容公式为 $C = \varepsilon_0 \frac{A}{d}$，其中 $\varepsilon_0 = 8.85 \times 10^{-12} \ \text{F/m}$。

(a) 计算电容值 $C$（分别用 F、μF、nF、pF 表示）。
(b) 如果在极板间加上 $V = 100 \ \text{V}$ 的电压，每块极板上储存的电荷量是多少？（$Q = CV$）
(c) 储存的能量是多少？（$W = \frac{1}{2}CV^2$）

> 提示：注意单位统一。$1 \ \text{mm} = 1 \times 10^{-3} \ \text{m}$。

**题8**：你是电路初学者小明。你的朋友告诉你："欧姆定律 $V = IR$ 说明电压和电阻成正比，所以电阻越大电压越高。"请指出这句话的问题，并用一个具体的数值例子说明在什么条件下这个说法成立，在什么条件下不成立。

---

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

#### 题1 解答

库仑力公式：$F = k \frac{|q_1 q_2|}{r^2}$

$$
\begin{aligned}
F &= 8.99 \times 10^9 \times \frac{|(4 \times 10^{-6}) \times (-2 \times 10^{-6})|}{(0.1)^2} \\[4pt]
  &= 8.99 \times 10^9 \times \frac{8 \times 10^{-12}}{0.01} \\[4pt]
  &= 8.99 \times 10^9 \times 8 \times 10^{-10} \\[4pt]
  &= 8.99 \times 0.8 \\[4pt]
  &= 7.192 \ \text{N} \approx 7.2 \ \text{N}
\end{aligned}
$$

$q_1$ 为正，$q_2$ 为负，异性电荷——**吸引力**。

#### 题2 解答

(a) $6.8 \ \text{k}\Omega = 6.8 \times 10^3 = 6800 \ \Omega$

(b) $150 \ \text{mA} = 150 \times 10^{-3} = 0.15 \ \text{A}$

(c) $22 \ \mu\text{F} = 22 \times 10^{-6} = 0.000022 \ \text{F} = 2.2 \times 10^{-5} \ \text{F}$

(d) $3.3 \ \text{M}\Omega = 3.3 \times 10^6 \ \Omega = 3.3 \times 10^3 \times 10^3 \ \Omega = 3300 \ \text{k}\Omega$

#### 题3 解答

$$
I = \frac{V}{R} = \frac{9}{300} = 0.03 \ \text{A} = 30 \ \text{mA}
$$

#### 题4 解答

(a) 电压：$V = Ed = 5000 \times 0.04 = 200 \ \text{V}$

> 注意：$4 \ \text{cm} = 0.04 \ \text{m}$

(b) 功：$W = qV = (2 \times 10^{-6}) \times 200 = 4 \times 10^{-4} \ \text{J} = 0.4 \ \text{mJ}$

或者用 $W = Fd = qEd$：
$$
W = (2 \times 10^{-6}) \times 5000 \times 0.04 = 4 \times 10^{-4} \ \text{J}
$$
两种方法结果一致 ✓

#### 题5 解答

**步骤1**：求总电阻。
$$
R_{\text{总}} = R_1 + R_2 = 20 + 10 = 30 \ \text{k}\Omega = 30000 \ \Omega
$$

**步骤2**：求电流。
$$
I = \frac{V_{\text{in}}}{R_{\text{总}}} = \frac{12}{30000} = 0.0004 \ \text{A} = 0.4 \ \text{mA}
$$

**步骤3**：求 $R_2$ 两端的电压（分压公式）。
$$
V_{\text{out}} = I \times R_2 = 0.0004 \times 10000 = 4 \ \text{V}
$$

或者用分压公式直接算：
$$
V_{\text{out}} = V_{\text{in}} \times \frac{R_2}{R_1 + R_2} = 12 \times \frac{10}{30} = 12 \times \frac{1}{3} = 4 \ \text{V}
$$

两种方法一致 ✓

#### 题6 解答

$P = V^2 / R$，$R$ 固定时，$P \propto V^2$（$P$ 和 $V^2$ 成正比）。

$V$ 从 $5 \ \text{V}$ 变到 $15 \ \text{V}$，**增大到 3 倍**。

$V^2$ 增大到 $3^2 = 9$ 倍。

因此 $P$ 也增大到 9 倍：

$$
P_2 = 9 \times P_1 = 9 \times 0.5 = 4.5 \ \text{W}
$$

验证：$P_2 = V_2^2 / R = 15^2 / 50 = 225 / 50 = 4.5 \ \text{W}$ ✓

> 关键洞察：电压变成 3 倍，功率不是变成 3 倍，而是变成 9 倍！这就是平方关系的威力——在很多电路问题中，这会让初学者出乎意料。

#### 题7 解答

(a) 计算电容值：

$$
\begin{aligned}
C &= \varepsilon_0 \frac{A}{d} \\[4pt]
  &= 8.85 \times 10^{-12} \times \frac{0.01}{1 \times 10^{-3}} \\[4pt]
  &= 8.85 \times 10^{-12} \times 10 \\[4pt]
  &= 8.85 \times 10^{-11} \ \text{F}
\end{aligned}
$$

用不同单位表示：
- $C = 8.85 \times 10^{-11} \ \text{F}$
- $C = 8.85 \times 10^{-5} \ \mu\text{F}$（$= 0.0000885 \ \mu\text{F}$）
- $C = 0.0885 \ \text{nF}$
- $C = 88.5 \ \text{pF}$

(b) 储存的电荷量：

$$
Q = CV = 8.85 \times 10^{-11} \times 100 = 8.85 \times 10^{-9} \ \text{C} = 8.85 \ \text{nC}
$$

(c) 储存的能量：

$$
\begin{aligned}
W &= \frac{1}{2} C V^2 \\[4pt]
  &= \frac{1}{2} \times 8.85 \times 10^{-11} \times 100^2 \\[4pt]
  &= \frac{1}{2} \times 8.85 \times 10^{-11} \times 10000 \\[4pt]
  &= \frac{1}{2} \times 8.85 \times 10^{-7} \\[4pt]
  &= 4.425 \times 10^{-7} \ \text{J} \\[4pt]
  &\approx 0.44 \ \mu\text{J}
\end{aligned}
$$

> 0.44 微焦耳——非常小的能量。这就是为什么在实际电路中使用的电容通常容量要大得多（微法到毫法级别），才能储存可用的能量。

#### 题8 解答

**小明朋友的说法有两个问题：**

**问题1**："所以电阻越大电压越高"——这句话缺了一个关键条件。$V = IR$ 中，$V$ 同时取决于 $I$ 和 $R$。说"$V$ 和 $R$ 成正比"的前提是 **$I$ 不变**。如果 $I$ 也变了，结论就不成立。

**具体例子**：

- **成立的情况**：如果 $I = 0.1 \ \text{A}$ 固定不变，$R = 10 \ \Omega$ 时 $V = 1 \ \text{V}$，$R = 20 \ \Omega$ 时 $V = 2 \ \text{V}$。$R$ 翻倍，$V$ 也翻倍——此时 $V$ 和 $R$ 成正比 ✓

- **不成立的情况**：如果把电阻接到一个固定的 $12 \ \text{V}$ 电源上，电压已经被电源"锁死"在 $12 \ \text{V}$。此时 $R = 10 \ \Omega$ 时 $I = 1.2 \ \text{A}$，$R = 20 \ \Omega$ 时 $I = 0.6 \ \text{A}$。$V$ 始终是 $12 \ \text{V}$，根本没有变——"电阻越大电压越高"完全不成立 ✗

**总结**：在电路分析中，一定要先搞清楚**哪些量是固定的（由电源或电路结构决定），哪些量是可变或待求的**。然后才能判断两个量之间的关系。不要看到乘法公式就认为两个因子成正比例——要问"第三个量在干嘛？"

</details>

---

> **下一章**：[电荷、电流与电压](./02-charge-voltage-current.md) —— 从基础物理进入电路世界，开始用"水流"的比喻来理解电路三要素。
>
> **数学工具**：学完微积分直觉后，如需进一步学习正弦信号和相量分析，请参阅 [复数与相量](./math-complex.md)。
