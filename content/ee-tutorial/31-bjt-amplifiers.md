# 第31章：BJT放大电路

> **核心问题**：微弱信号怎么变成"大嗓门"？

---

## 0. 本章导览

上一章你学会了 BJT 的基本原理和直流偏置——知道了怎么给晶体管"装上发动机"让它工作在合适的区间。但 BJT 的真正威力不在直流，而在**交流放大**。你的手机麦克风输出的是几毫伏的微弱信号，扬声器需要几伏的强信号才能发出声音。中间发生了什么？答案就是：BJT 放大电路。

这一章的核心任务：把第 30 章的直流 BJT 知识"升级"到交流世界。你要学会把小信号抽象成一个模型（小信号模型），然后用三种基本接法（共射、共集、共基）把信号放大、缓冲、或做高频处理。最后把多级放大器串起来、分析频率响应，画出漂亮的增益曲线。

前置章节：[BJT基础](./30-bjt-fundamentals.md)（工作区、直流偏置、Q 点、β 的含义）
下一章：[场效应管 FET](./32-fet.md)（另一种放大元件，电压控制型）

学完本章，你将能够：

1. 画出 BJT 的 π 型小信号等效电路，算出 $r_\pi$ 和 $g_m$
2. 推导共射（CE）放大器的电压增益 $A_v = -g_m R_C$，并用示波器概念解释"反相"
3. 理解共集（CC）射极跟随器的"电压不放大、电流放大"特性
4. 掌握共基（CB）的同相放大和高频优势
5. 列出三种组态的输入输出电阻、增益、相位的完整对比表
6. 计算多级放大器的总增益（级联总增益 = 各级增益之积）
7. 分析放大器的低频截止频率 $f_L$、高频截止频率 $f_H$ 和带宽 $BW$
8. 手算 6 道以上完整放大电路设计题，用 Python 画增益曲线验证
9. 避开初学者在交流分析和增益计算中最常踩的坑

> 本章约 1500 行，建议分 4-5 次读完。第 1 节（小信号模型）和第 2 节（共射 CE）是最核心的基础，必须吃透才能往后走。Python 代码在第 8 节，跑一遍曲线图会加深你对频率响应的理解。

---

## 1. 小信号模型——给交流信号"画地图"

### 1a. 生活例子：在平坦公路上骑车 vs 在起伏山路上骑车

想象你在一条笔直平坦的公路上骑车。路面的"海拔"大概是 100 米（这就像 BJT 的直流工作点，也叫 Q 点）。你骑过去，海拔不变。

现在想象这条公路其实建在起伏的丘陵上——路面有小坡、小坑，上下波动可能只有几十厘米。这几十厘米的波动就是"小信号"——它叠在 100 米的"大背景"上。

你骑车时，车把的上下晃动只取决于路面的起伏（小信号），而不是 100 米的绝对海拔（直流偏置）。**小信号分析的核心思想就是：把直流大背景先"归零"，只看交流小波动**，因为只有波动里才藏着你要放大的信息（声音、图像、数据）。

BJT 放大电路做的是同一件事：先用直流偏置把晶体管"架"到合适的工作点（第 30 章的内容），然后在这个工作点附近分析交流信号怎么被放大。直流是"舞台"，交流才是"演员"。

### 1b. 直观理解：斜坡上推箱子

另一个类比：你把一个箱子放在一个斜坡上。如果不推它，它可能往下滑（这就是直流趋势）。现在你在箱子旁边轻轻推一下、拉一下——箱子会以它当前的位置为中心来回晃动。

BJT 的交流分析就像分析这个"来回晃动"的规律：晃动幅度跟你的推力成正比，晃动方向跟你的推力方向一致（或者相反）。至于箱子的绝对位置在斜坡的哪里——这不重要，因为"晃动"才是信号。

在电路里，我们用**小信号等效电路**来把 BJT 这个非线性元件，"简化"成在 Q 点附近近似线性的元件。就像把一段弯弯曲曲的山路，在某个点附近用一段直线来近似。这段直线的斜率，就是放大的关键参数。

这里有一个非常重要的前提："小信号"必须足够小。什么叫"足够小"？一般来说，输入交流信号的幅度要远小于 $V_T$（26 mV）。如果信号太大（比如超过 10 mV），BJT 的非线性开始显现，输出的波形就不再是完美的正弦波——你会看到波峰被"压扁"或"拉长"，这就是**失真**。因此，小信号模型的适用范围通常是 $v_{be} \ll V_T$，工程上一般控制在 5 mV 以内比较安全。

### 1c. 形式化定义：π 型小信号模型

把小信号分析想象成"给 BJT 拍了三张 X 光片"，从三个角度看清它的交流行为：

**第一步：直流工作点（Q 点）**

在做任何交流分析之前，必须先算好直流 Q 点。Q 点告诉我们：
- 集电极直流电流 $I_{CQ}$（直流偏置下的集电极电流）
- 基极直流电流 $I_{BQ} = I_{CQ} / \beta$
- 集电极-发射极直流电压 $V_{CEQ}$

Q 点是"舞台"，没有它，交流"演员"站都站不稳。

**第二步：BJT 的 π 型小信号等效电路**

在 Q 点附近，BJT 可以用以下三个元件组成的电路来替代：

```
        B ──┬───── r_π ─────┬── E
            │                │
            │                │
            +                │
            │  v_be          │  g_m × v_be (受控电流源,方向从C到E)
            │  -             │
            │                │
            └────────────────┘

            C ─────────────────── E
                 ↑ g_m × v_be
                (电流源方向: C → E)
```

三个关键参数：

**（1）跨导 $g_m$（transconductance）**

$g_m$ 是小信号模型里最重要的参数。它的物理含义是：**B-E 之间的小信号电压变化 $\Delta v_{BE}$，能在 C 极产生多大的小信号电流变化 $\Delta i_C$**。

$$
g_m = \frac{\Delta i_C}{\Delta v_{BE}} \quad \text{（单位：西门子 S，或 A/V）}
$$

在室温下（$T = 300\text{ K}$，约 27°C），$g_m$ 可以通过 Q 点的集电极电流直接算出来：

$$
g_m = \frac{I_{CQ}}{V_T}
$$

其中 $V_T$ 是**热电压**（thermal voltage），室温下约 $26\text{ mV}$（即 $0.026\text{ V}$）。

看这个公式：$I_{CQ}$ 越大，$g_m$ 越大。也就是说，直流工作电流越大，BJT 对交流信号的"放大灵敏度"越高。这就像水龙头的水压越大，你拧同样的角度，出水量变化越大。

**（2）基极-发射极小信号电阻 $r_\pi$**

$r_\pi$ 表示 B-E 结在 Q 点附近的交流电阻。这个电阻不是实际存在的物理电阻，而是 PN 结 I-V 曲线在 Q 点处的斜率倒数：

$$
r_\pi = \frac{\beta}{g_m} = \frac{\beta \cdot V_T}{I_{CQ}}
$$

或者等价地：

$$
r_\pi = \frac{V_T}{I_{BQ}}
$$

注意：$r_\pi$ 通常不大。如果 $\beta = 100$、$I_{CQ} = 1\text{ mA}$，则 $r_\pi = 100 \times 0.026 / 0.001 = 2600\text{ }\Omega$（2.6 kΩ）。

**为什么是 $V_T / I_{BQ}$？因为 $r_\pi$ 本质上就是 B-E 二极管在 Q 点处的"微分电阻"。**回顾二极管的 I-V 方程 $I = I_S (e^{V/V_T} - 1)$，对其求导可得动态电阻 $r_d = V_T / I$。B-E 结就是一个二极管，所以基极电流的微分电阻就是 $V_T / I_{BQ}$。而 $r_\pi$ 把基极侧的电阻"映射"到整个 π 模型里，这就是它名字（π 型）的由来——模型里还有一个 π 形的电阻网络。

**（3）输出电阻 $r_o$（Early 效应）**

实际 BJT 并不是一个完美的电流源。当 $V_{CE}$ 增加时，$I_C$ 也会微微增加（这种现象叫 Early 效应）。这个效应可以用一个并联在 C-E 之间的电阻 $r_o$ 来建模：

$$
r_o = \frac{V_A}{I_{CQ}}
$$

其中 $V_A$ 是**Early 电压**，典型值在 50V 到 200V 之间。在初步分析中，如果 $r_o$ 远大于电路中的其他电阻，可以暂时忽略它。本章大部分手算都会忽略 $r_o$，只在需要精确分析时才引入。

**第三步：三步法——把直流和交流分开**

任何 BJT 放大电路的分析都遵循"三步法"：

1. **直流分析**：把所有交流信号源短路（电压源→短路、电流源→开路）、电容开路，算出 Q 点（$I_{CQ}$、$V_{CEQ}$ 等）
2. **交流分析**：用 $\pi$ 模型替换 BJT，所有直流电压源短路（因为直流电源的交流内阻为 0）、耦合电容和旁路电容短路（因为它们在交流频率下阻抗很低），画出小信号等效电路
3. **综合**：把小信号的计算结果（交流量）叠在 Q 点上（直流量），得到完整的电压/电流波形

这三步是本章所有内容的基础。每一节的手算都会严格执行这三步。

### 1d. 手算示例：求小信号参数

**题目**：一个 NPN 型 BJT，$\beta = 120$，Early 电压 $V_A = 100\text{ V}$。测得 Q 点 $I_{CQ} = 2\text{ mA}$。设 $V_T = 26\text{ mV}$。求 $g_m$、$r_\pi$、$r_o$，以及 $I_{BQ}$。

**解**：

| 步骤 | 公式 | 代入 | 结果 |
|------|------|------|------|
| ① 求 $I_{BQ}$ | $I_{BQ} = I_{CQ} / \beta$ | $2\text{ mA} / 120$ | $16.67\text{ }\mu\text{A}$ |
| ② 求 $g_m$ | $g_m = I_{CQ} / V_T$ | $0.002 / 0.026$ | $0.07692\text{ S}$（约 76.9 mS） |
| ③ 求 $r_\pi$ | $r_\pi = \beta / g_m$ | $120 / 0.07692$ | $1560\text{ }\Omega$（1.56 kΩ） |
| ④ 求 $r_o$ | $r_o = V_A / I_{CQ}$ | $100 / 0.002$ | $50{,}000\text{ }\Omega$（50 kΩ） |

**验证**：$r_\pi$ 也可以用 $V_T / I_{BQ} = 0.026 / (16.67 \times 10^{-6}) = 1560\text{ }\Omega$，一致。

> 观察：$I_{CQ}$ 越大 → $g_m$ 越大 → $r_\pi$ 越小。工作电流翻倍，$g_m$ 翻倍，$r_\pi$ 减半。这个关系在设计中非常重要。

### 1e. Python 验证：画出 $g_m$ 和 $r_\pi$ 随 $I_{CQ}$ 变化的曲线

```python
# -*- coding: utf-8 -*-
# 31-01: g_m 和 r_pi 随集电极电流 I_CQ 的变化曲线
# 验证公式 g_m = I_CQ / V_T  和  r_pi = β / g_m

import numpy as np
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 参数
VT = 0.026       # 热电压 26 mV
BETA = 120        # 电流增益
ICQ = np.linspace(0.1e-3, 5e-3, 100)  # I_CQ 从 0.1 mA 到 5 mA

# 计算
gm = ICQ / VT                # 跨导 (S)
r_pi = BETA / gm             # 小信号电阻 (Ω)

# 画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 左图：g_m 随 I_CQ 变化
ax1.plot(ICQ*1000, gm*1000, 'b-', linewidth=2)
ax1.set_xlabel(r'集电极直流电流 $I_{CQ}$ (mA)', fontsize=12)
ax1.set_ylabel(r'跨导 $g_m$ (mS)', fontsize=12)
ax1.set_title(r'$g_m$ 随 $I_{CQ}$ 线性增长', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=76.9, color='r', linestyle='--', alpha=0.5, label=r'手算例: $I_{CQ}$=2mA → $g_m$=76.9mS')
ax1.legend(fontsize=10)

# 右图：r_pi 随 I_CQ 变化
ax2.plot(ICQ*1000, r_pi/1000, 'b-', linewidth=2)
ax2.set_xlabel(r'集电极直流电流 $I_{CQ}$ (mA)', fontsize=12)
ax2.set_ylabel(r'基极电阻 $r_\pi$ (kΩ)', fontsize=12)
ax2.set_title(r'$r_\pi$ 随 $I_{CQ}$ 反比例减小', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1.56, color='r', linestyle='--', alpha=0.5, label=r'手算例: $I_{CQ}$=2mA → $r_\pi$=1.56kΩ')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('31-01-gm-rpi-vs-icq.png', dpi=150, bbox_inches='tight')
plt.show()
print("关键结论：I_CQ 越大 → g_m 越大（线性） → r_pi 越小（反比例）")
print("手算验证：I_CQ=2mA, β=120, V_T=26mV")
print(f"  g_m  = {2e-3/VT*1000:.1f} mS")
print(f"  r_pi = {BETA*VT/2e-3:.0f} Ω = {BETA*VT/2e-3/1000:.2f} kΩ")
```

预期输出：
```
关键结论：I_CQ 越大 → g_m 越大（线性） → r_pi 越小（反比例）
手算验证：I_CQ=2mA, β=120, V_T=26mV
  g_m  = 76.9 mS
  r_pi = 1560 Ω = 1.56 kΩ
```

### 1f. 常见误区

❌ **误区1：把 $r_\pi$ 当成实际存在的物理电阻。**
$r_\pi$ 是小信号等效参数，不代表 BJT 里有一个真实的 1.56 kΩ 电阻。它描述的是 B-E PN 结的 I-V 曲线在 Q 点处的"局部斜率"。你用万用表去量 B-E 之间是量不到这个阻值的。

✓ **正确理解**：$r_\pi$ 是"微分电阻"——$v_{BE}$ 的变化量除以 $i_B$ 的变化量。只有当交流小信号在 Q 点附近波动时，这个"等效电阻"才成立。

❌ **误区2：$g_m$ 越大越好，所以 $I_{CQ}$ 设得越大越好。**
$I_{CQ}$ 越大确实 $g_m$ 越大，但电路功耗也更大（$P = V_{CC} \times I_{CQ}$），而且 Q 点可能偏出放大区进入饱和区。

✓ **正确做法**：$I_{CQ}$ 的选择是在增益、功耗、线性范围三者之间取平衡。一般小信号放大器取 0.5 mA 到 5 mA。

### 1g. 应用连接

小信号模型是所有模拟放大器分析的"通用语言"。无论是本章的 CE/CC/CB 基本组态、第 32 章的 FET 共源放大器、还是第 33 章的运放内部电路，用的都是同一套"直流偏置→交流替代→参数提取"的流程。熟练掌握小信号模型，你就拿到了分析任何模拟电路的"万能钥匙"。

---

## 2. 共射放大电路（CE）——最常见的"反相放大器"

### 2a. 生活例子：跷跷板

公园里的跷跷板你玩过吧？你坐在一端往下压（输入信号），另一端的人就会往上翘（输出信号）。你往下、他往上——方向相反。而且如果你用力猛一点（输入大），他翘得就高（输出大）。

CE 放大器干的就是跷跷板的事：输入电压变化往上，输出电压变化往下——**反相**。而且输出的"翘起幅度"比你输入的"下压力度"大得多——**电压放大**。

还有一个更精确的类比：**杠杆**。阿基米德说"给我一个支点，我能撬动地球"。杠杆的一端你施加一个小力（输入），另一端输出一个大力（输出），但方向相反。CE 放大器中，$g_m$ 就是杠杆的"力臂比"——它决定了用多小的输入电压变化能产生多大的输出电流变化。$R_C$ 则是把这个电流变化转成电压变化的"转换器"。

### 2b. 直观理解：CE 放大器的"信号流"

看一个典型的 CE 放大电路：

```
          +Vcc
           │
           │
          ┌┴┐
          │ │ R_C   ← 集电极电阻（把电流变化转成电压变化）
          │ │
          └┬┘
           ├───── V_out（输出端,在 C 极取）
           │
         C │
    ───┤   BJT (NPN)
         E │
           │
          ┌┴┐
          │ │ R_E   ← 发射极电阻（稳定 Q 点,常并旁路电容）
          │ │
          └┬┘
           │
          ═══  GND
```

交流信号从基极（B）输入，从集电极（C）输出，发射极（E）是输入和输出的公共端——所以叫**共射**（Common Emitter）。

信号的"旅程"是这样的：

1. **输入电压 $v_{in}$ 加在 B-E 之间** → 产生 $v_{be}$（小信号分量）
2. **$v_{be}$ 通过 $g_m$ 转成集电极电流变化** → $\Delta i_c = g_m \cdot v_{be}$
3. **$\Delta i_c$ 流过集电极电阻 $R_C$** → 产生电压变化 $\Delta v_{R_C} = \Delta i_c \cdot R_C = g_m R_C \cdot v_{be}$
4. **输出电压** $v_{out} = V_{CC} - i_C R_C$，所以 $\Delta v_{out} = -\Delta i_C \cdot R_C = -g_m R_C \cdot v_{be}$

注意第 4 步的负号！$i_C$ 增加 → $R_C$ 上的压降增加 → $V_{out}$（C 极对地电压）**减小**。所以输入增大，输出减小——反相。

### 2c. 形式化定义：CE 放大器的小信号分析

**步骤一：直流分析（找 Q 点）**

设 $V_{CC} = 12\text{ V}$，$R_C = 4\text{ kΩ}$，$R_E = 1\text{ kΩ}$，$R_1 = 80\text{ kΩ}$，$R_2 = 20\text{ kΩ}$（分压偏置），$\beta = 100$。

先算 Thevenin 等效：$V_{BB} = V_{CC} \cdot R_2/(R_1+R_2) = 12 \times 20/100 = 2.4\text{ V}$，$R_{BB} = R_1 \parallel R_2 = 80 \parallel 20 = 16\text{ kΩ}$。

KVL 回路：$V_{BB} = I_{BQ} R_{BB} + V_{BE} + I_{EQ} R_E$。设 $V_{BE} = 0.7\text{ V}$，$I_{EQ} \approx I_{CQ} = \beta I_{BQ}$：

$$
\begin{aligned}
V_{BB} - V_{BE} &= I_{BQ} R_{BB} + \beta I_{BQ} R_E \\
2.4 - 0.7 &= I_{BQ} (16\text{k} + 100 \times 1\text{k}) \\
1.7 &= I_{BQ} \times 116\text{k} \\
I_{BQ} &= 14.66\text{ }\mu\text{A}
\end{aligned}
$$

则 $I_{CQ} = \beta I_{BQ} = 1.466\text{ mA}$，$V_{CEQ} = V_{CC} - I_{CQ}(R_C + R_E) = 12 - 1.466 \times 5 = 12 - 7.33 = 4.67\text{ V}$。

**步骤二：交流分析（求 $A_v$、$R_{in}$、$R_{out}$）**

交流等效电路中，$R_E$ 被旁路电容 $C_E$ 短路（$C_E$ 在交流频率下阻抗极低），所以 E 极直接接地。

小信号参数：$g_m = I_{CQ}/V_T = 0.001466 / 0.026 = 0.0564\text{ S}$（56.4 mS），$r_\pi = \beta/g_m = 100/0.0564 = 1773\text{ }\Omega$（1.77 kΩ）。

从基极看进去的输入电阻：
$$
R_{in} = R_1 \parallel R_2 \parallel r_\pi = 80\text{k} \parallel 20\text{k} \parallel 1.77\text{k} \approx 1.59\text{ kΩ}
$$

（$r_\pi$ 只有 1.77 kΩ，远小于 $R_1 \parallel R_2 = 16 \text{ kΩ}$，所以输入电阻主要由 $r_\pi$ 决定）

电压增益：
$$
A_v = \frac{v_{out}}{v_{in}} = -g_m (R_C \parallel r_o \parallel R_L)
$$

忽略 $r_o$ 且空载（$R_L = \infty$）时：
$$
A_v = -g_m R_C = -0.0564 \times 4000 = -225.6
$$

带 10 kΩ 负载时：
$$
A_v = -g_m (R_C \parallel R_L) = -0.0564 \times (4\text{k} \parallel 10\text{k}) = -0.0564 \times 2857 = -161.1
$$

输出电阻：$R_{out} = R_C \parallel r_o \approx R_C = 4\text{ kΩ}$（$r_o$ 通常远大于 $R_C$）

**步骤三：意义解释**

- 空载增益 -225.6 意味着：输入 10 mV 的正弦波 → 输出约 2.26V 的正弦波，但**相位相反**（输入正半周时输出是负半周）
- 带负载后增益降到 -161.1，因为 $R_L$ 和 $R_C$ 并联，等效交流负载变小
- 输入电阻约 1.6 kΩ，对前级信号源来说是一个"中等负载"，需要前级输出阻抗足够低才能不衰减信号

> **检查 Q 点是否合理**：$V_{CEQ}=4.67\text{ V}$，对 12V 电源来说 Q 点大约在中间偏下。$V_{CEQ}$ 大于 $V_{CE(sat)}$（约 0.2V），且在放大区。输出信号最大摆幅约 $2 \times \min(V_{CEQ} - V_{CE(sat)}, I_{CQ}R_C) \approx 2 \times \min(4.47, 5.86) \approx 8.94\text{ V}$（峰峰值），足以覆盖大部分小信号放大的需求。

### 2d. 手算示例（共 3 题）

**例题1（基础）：求 CE 放大器的 Q 点和 $A_v$**

电路：$V_{CC}=15\text{V}$，$R_C=5\text{kΩ}$，$R_E=2\text{kΩ}$，$R_1=150\text{kΩ}$，$R_2=30\text{kΩ}$，$\beta=150$，$C_E$ 将 $R_E$ 完全旁路。

| 步骤 | 计算 | 结果 |
|------|------|------|
| ① $V_{BB}$ | $15 \times 30/(150+30)$ | $2.5\text{ V}$ |
| ② $R_{BB}$ | $150 \parallel 30$ | $25\text{ kΩ}$ |
| ③ $I_{BQ}$ | $(2.5-0.7)/[25\text{k} + 150\times2\text{k}] = 1.8/325\text{k}$ | $5.54\text{ μA}$ |
| ④ $I_{CQ}$ | $150 \times 5.54\text{μ}$ | $0.831\text{ mA}$ |
| ⑤ $V_{CEQ}$ | $15 - 0.831\text{m}\times(5\text{k}+2\text{k})$ | $9.18\text{ V}$ |
| ⑥ $g_m$ | $0.831\text{m}/0.026$ | $31.96\text{ mS}$ |
| ⑦ $r_\pi$ | $150/0.03196$ | $4.69\text{ kΩ}$ |
| ⑧ $A_v$（空载） | $-g_m R_C$ | **-159.8** |
| ⑨ $R_{in}$ | $25\text{k} \parallel 4.69\text{k}$ | **3.95 kΩ** |
| ⑩ $R_{out}$ | $R_C$ | **5 kΩ** |

**例题2（进阶）：带 $R_E$ 部分旁路（发射极退化）**

有时为了降低失真或提高输入电阻，只旁路 $R_E$ 的一部分。设 $R_{E1}=200\text{ }\Omega$（未被旁路），$R_{E2}=1.8\text{ kΩ}$（被旁路），总 $R_E = 2\text{ kΩ}$。其他参数同例题1。

直流分析完全不变（$R_E$ 的旁路对直流无影响，Q 点不变）。交流分析中，$R_{E1}$ 保留在 E 极和地之间，反馈使得输入电阻增大、增益减小。

$$
A_v = -\frac{g_m R_C}{1 + g_m R_{E1}} = -\frac{0.03196 \times 5000}{1 + 0.03196 \times 200} = -\frac{159.8}{1 + 6.392} = -\frac{159.8}{7.392} = -21.6
$$

$$
R_{in} = R_1 \parallel R_2 \parallel [r_\pi + (1+\beta)R_{E1}] = 25\text{k} \parallel [4.69\text{k} + 151 \times 0.2\text{k}]
$$

先算：$r_\pi + (1+\beta)R_{E1} = 4.69\text{k} + 151\times200 = 4.69\text{k} + 30.2\text{k} = 34.89\text{ kΩ}$

$R_{in} = 25\text{k} \parallel 34.89\text{k} \approx 14.6\text{ kΩ}$（输入电阻大幅提升！）

| 参数 | 完全旁路 $R_E$ | 部分旁路 $R_{E1}$=200Ω |
|------|:------:|:------:|
| $A_v$（空载） | -159.8 | -21.6 |
| $R_{in}$ | 3.95 kΩ | 14.6 kΩ |
| 适用场景 | 追求高增益 | 需要高输入阻抗 / 低失真 |

**例题3（综合）：考虑 $r_o$ 的精算**

当 $R_C$ 较大（接近 $r_o$）时，不能忽略输出电阻 $r_o$。精确增益公式：

$$
A_v = -\frac{g_m (R_C \parallel r_o)}{1 + g_m R_{E1}} \cdot \frac{r_o}{r_o + R_C + (1+g_m r_o)R_{E1}}
$$

（当 $R_{E1}=0$ 时简化为 $A_v = -g_m(R_C \parallel r_o)$）

设 $V_A = 100\text{ V}$，$I_{CQ} = 0.831\text{ mA}$ → $r_o = 100/0.831\text{m} \approx 120.3\text{ kΩ}$。$R_C=5\text{kΩ}$，$r_o \gg R_C$，所以 $r_o$ 的影响可以忽略（$(R_C \parallel r_o) \approx R_C$）。对于一般的小信号放大器，忽略 $r_o$ 通常是安全的。

### 2e. Python 验证：CE 放大器增益 vs 负载电阻

```python
# -*- coding: utf-8 -*-
# 31-02: CE放大器电压增益随负载电阻 R_L 的变化
# 验证公式 A_v = -g_m (R_C || R_L)

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 电路参数
ICQ = 0.831e-3    # Q点集电极电流 0.831 mA
VT = 0.026         # 热电压
gm = ICQ / VT      # 跨导 31.96 mS
RC = 5000          # 集电极电阻 5 kΩ

# 负载电阻范围（1 kΩ 到 200 kΩ，200 kΩ代表接近空载）
RL = np.logspace(np.log10(500), np.log10(200e3), 100)

# 计算并联等效电阻和增益
R_equiv = (RC * RL) / (RC + RL)
Av = -gm * R_equiv

# 画图
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(RL/1000, np.abs(Av), 'b-', linewidth=2)
ax.axhline(y=159.8, color='r', linestyle='--', alpha=0.5, label='空载增益 159.8')
ax.set_xlabel(r'负载电阻 $R_L$ (kΩ)', fontsize=12)
ax.set_ylabel(r'|$A_v$| (电压增益绝对值)', fontsize=12)
ax.set_title(r'CE放大器增益随 $R_L$ 增大趋于空载值', fontsize=13)
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('31-02-ce-gain-vs-load.png', dpi=150, bbox_inches='tight')
plt.show()

# 验证几个关键负载值
RL_test = [2000, 5000, 10000, 50000, 200000]
print("=== 增益 vs 负载验证 ===")
for rl in RL_test:
    av_val = gm * (RC * rl) / (RC + rl)
    print(f"R_L = {rl/1000:5.1f} kΩ → |A_v| = {av_val:7.1f}")
print(f"\n空载 A_v = -gm×R_C = -{gm*RC:.1f}")
print(f"R_L=10kΩ → R_C||R_L = {5000*10000/(5000+10000):.0f} Ω → A_v = -{gm*3333:.1f}")
```

预期输出：
```
R_L =  2.0 kΩ → |A_v| =  45.7
R_L =  5.0 kΩ → |A_v| =  79.9
R_L = 10.0 kΩ → |A_v| = 106.6
R_L = 50.0 kΩ → |A_v| = 145.3
R_L = 200.0 kΩ → |A_v| = 156.0
```

### 2f. 常见误区

❌ **误区3：忘记负号！把 CE 当成同相放大器。**
$A_v = -g_m R_C$ 的负号不是装饰。输入正半周 → 输出负半周。如果你后面的电路期望同相（比如反馈回路），相位搞反会导致灾难。

✓ **记住口诀**："C 就是反"——从 Collector 输出就是反相（Inverting）。画波形图时一定要把输入和输出画在时间轴的上下两侧。

❌ **误区4：旁路电容 $C_E$ 对交流是"短路"，但不等于 $R_E$ 不存在。**
直流分析时 $R_E$ 必须计入（它决定 Q 点），交流分析时 $R_E$ 被旁路电容短路。初学者常犯：交流分析时也把 $R_E$ 算进去，或者反过来直流分析时漏掉 $R_E$。

✓ **三步法的价值**：第一步直流分析时 $R_E$ 在回路中；第二步交流分析时 $R_E$ 被 $C_E$ 短路移除。两步分开做，就不会混。

### 2g. 应用连接

CE 放大器是模拟电路里最常用的增益级。几乎所有音频放大器、射频前置放大器、传感器信号调理电路的第一级都是 CE 放大器。它的高增益特性使它成为"信号放大链"中的主力。缺点是高频时增益会衰减（见第 7 节频率响应）。

---

## 3. 共集放大电路（CC）——射极跟随器，电压不涨但"力气"大了

### 3a. 生活例子：经理和员工

公司里，经理（输入）下一个命令，他自己并不能一个人干完所有活。他把命令传达给一个团队，团队里有 50 个人（电流放大），大家一起干。但命令的内容（电压）没变——"把箱子搬到三楼"还是"把箱子搬到三楼"。只是执行力强了 50 倍。

CC 放大器（也叫**射极跟随器**，Emitter Follower）就是这个经理：输入电压 ≈ 输出电压（$A_v \approx 1$），但输出电流可以比输入电流大 $\beta$ 倍。它不是"扩音器"，而是"扩力器"。

再想一个更具体的场景：你用一根细吸管往一个大气球里吹气。你嘴里的气压（输入电压）和气球里的气压（输出电压）几乎一样——因为你一吹，气球就鼓起来，气压跟着你的吹力变化。但气球里存的气量（电流/功率）比你吹进去的大得多——因为你的每一口小气流，都被气球的弹性"拉伸"成了持续的大体积。射极跟随器做的就是这个"吸管吹气球"的事：电压跟随，电流倍增。

### 3b. 直观理解：从发射极"偷看"基极

CC 放大器的接法：输入在 B 极，输出在 E 极，C 极直接接 $V_{CC}$（交流接地——因为电源对交流相当于短路，所以 C 极是"公共端"）。

```
          +Vcc
           │
           │   C 极直接接电源（交流接地）
         C │
    ───┤   BJT (NPN)
         E │
           ├───── V_out（输出端,在 E 极取）
           │
          ┌┴┐
          │ │ R_E   ← 发射极电阻
          │ │
          └┬┘
           │
          ═══  GND
```

信号流：
1. $v_{in}$ 加在 B 极
2. B-E 之间只有一个 PN 结压降（约 0.7V 直流 + 小信号）
3. 所以 E 极电压总是"紧跟着"B 极——B 极涨 10 mV，E 极也涨约 10 mV
4. 输出电压 $v_{out} \approx v_{in}$（增益约等于 1）

关键洞察：BJT 的 B-E 电压基本恒定（~0.7V），所以 $v_E = v_B - 0.7\text{V}$。当 $v_B$ 变化 $\Delta v$ 时，$v_E$ 也变化 $\Delta v$。输出**跟随**输入，所以叫"跟随器"。

### 3c. 形式化定义：射极跟随器的小信号分析

**直流分析（找 Q 点）：**

$I_{BQ} = \dfrac{V_{BB} - V_{BE}}{R_{BB} + (1+\beta)R_E}$（注意分母中是 $(1+\beta)R_E$ 而不是 $\beta R_E$，因为从基极"看进去"，发射极电阻会被放大 $(1+\beta)$ 倍——这是 BJT 的阻抗反射特性）。

**交流分析：**

小信号等效电路中，C 极接地（交流地）。从 B 极到 E 极的电压增益：

$$
A_v = \frac{v_{out}}{v_{in}} = \frac{(1+\beta)(R_E \parallel R_L)}{r_\pi + (1+\beta)(R_E \parallel R_L)}
$$

因为 $(1+\beta)(R_E \parallel R_L)$ 通常远大于 $r_\pi$，所以 $A_v \approx 1$（但略小于 1，通常是 0.95 ~ 0.99）。

**输入电阻（CC 的招牌优势）：**
$$
R_{in} = R_1 \parallel R_2 \parallel [r_\pi + (1+\beta)(R_E \parallel R_L)]
$$

$(1+\beta)(R_E \parallel R_L)$ 非常大地提升了输入电阻。典型值可达 50 kΩ 到 500 kΩ，远高于 CE 的几 kΩ。

**输出电阻（CC 的另一招牌优势）：**
$$
R_{out} = R_E \parallel \frac{r_\pi + R_{sig}'}{1+\beta}
$$

其中 $R_{sig}' = R_1 \parallel R_2 \parallel R_{sig}$（信号源内阻与偏置电阻的并联）。典型 $R_{out}$ 只有几十到几百 Ω——输出阻抗极低，驱动能力强。

**为什么 $R_{out}$ 这么低？** 从输出端（E 极）往回看，基极侧的等效电阻被"反射"过来时除了 $\beta$。BJT 的电流放大作用使得发射极可以"吸取"很大的电流变化而电压变化很小——这正是低输出阻抗的定义（$\Delta V / \Delta I$ 小）。

### 3d. 手算示例

**例题4：射极跟随器设计**

$V_{CC}=12\text{V}$，$\beta=120$，$R_E=2\text{kΩ}$，$R_1=100\text{kΩ}$，$R_2=50\text{kΩ}$，$R_L=1\text{kΩ}$，信号源内阻 $R_{sig}=600\text{ }\Omega$。

| 步骤 | 计算 | 结果 |
|------|------|------|
| ① $V_{BB}$ | $12 \times 50/150$ | $4\text{ V}$ |
| ② $R_{BB}$ | $100 \parallel 50$ | $33.33\text{ kΩ}$ |
| ③ $I_{BQ}$ | $(4-0.7)/[33.33\text{k} + 121\times2\text{k}] = 3.3/275.3\text{k}$ | $11.98\text{ μA}$ |
| ④ $I_{CQ}$ | $120 \times 11.98\text{μ}$ | $1.438\text{ mA}$ |
| ⑤ $g_m$ | $1.438\text{m}/0.026$ | $55.3\text{ mS}$ |
| ⑥ $r_\pi$ | $120/0.0553$ | $2.17\text{ kΩ}$ |
| ⑦ $R_E \parallel R_L$ | $2\text{k} \parallel 1\text{k}$ | $0.667\text{ kΩ}$ |
| ⑧ $(1+\beta)(R_E \parallel R_L)$ | $121 \times 667$ | $80.7\text{ kΩ}$ |
| ⑨ $A_v$ | $80.7\text{k} / (2.17\text{k} + 80.7\text{k})$ | **0.974** |
| ⑩ $R_{in}$ | $33.3\text{k} \parallel (2.17\text{k}+80.7\text{k})$ | **23.8 kΩ** |
| ⑪ $R_{out}$ | $R_E \parallel \frac{r_\pi + R_{sig}'}{1+\beta}$ | 算 $R_{sig}' = 33.3\text{k}\parallel0.6\text{k} \approx 590\text{ }\Omega$，$\frac{2.17\text{k}+0.59\text{k}}{121} \approx 22.8\text{ }\Omega$，$2\text{k}\parallel 22.8 \approx$ **22.5 Ω** |

**关键结论**：增益 0.974（≈1），输入电阻 23.8 kΩ（较高），输出电阻仅 22.5 Ω（极低）。

### 3e. Python 验证：射极跟随器增益 vs β

```python
# -*- coding: utf-8 -*-
# 31-03: 射极跟随器电压增益 vs 晶体管 β 值
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 电路参数
ICQ = 1.438e-3    # Q点集电极电流
VT = 0.026
RE = 2000
RL = 1000
RE_equiv = RE * RL / (RE + RL)  # R_E || R_L

beta_range = np.linspace(30, 400, 100)
Av = np.zeros_like(beta_range)

for i, beta in enumerate(beta_range):
    gm = ICQ / VT
    r_pi = beta / gm
    Av[i] = (1+beta) * RE_equiv / (r_pi + (1+beta) * RE_equiv)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(beta_range, Av, 'b-', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=120, color='r', linestyle='--', alpha=0.5, label=r'例题 $\beta$=120, $A_v$=0.974')
ax.set_xlabel(r'$\beta$', fontsize=13)
ax.set_ylabel(r'电压增益 $A_v$', fontsize=13)
ax.set_title(r'射极跟随器：$\beta > 100$ 后增益趋近于 1', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('31-03-cc-gain-vs-beta.png', dpi=150, bbox_inches='tight')
plt.show()

for beta_test in [50, 100, 120, 200, 400]:
    gm = ICQ / VT
    r_pi = beta_test / gm
    Av_test = (1+beta_test) * RE_equiv / (r_pi + (1+beta_test) * RE_equiv)
    print(f"β = {beta_test:3d} → A_v = {Av_test:.4f}")
```

预期输出：
```
β =  50 → A_v = 0.9389
β = 100 → A_v = 0.9680
β = 120 → A_v = 0.9737
β = 200 → A_v = 0.9841
β = 400 → A_v = 0.9920
```

### 3f. 常见误区

❌ **误区5：CC 放大器"没有增益"所以没用。**
虽然电压增益 ≈ 1，但**电流增益 ≈ β**（约 100 倍），**功率增益 ≈ β**。射极跟随器的核心价值是**阻抗变换**——高输入阻抗（不拖累前级）+ 低输出阻抗（能驱动重负载）。

✓ **正确使用场景**：放在高输出阻抗的传感器和低输入阻抗的负载之间（比如压电传感器 → 跟随器 → 长电缆 → 下一级），或者放在 CE 放大器后面做"功率输出级"。

❌ **误区6：CC 和 CE 的 Q 点公式一样。**
不一样！CC 的输入回路方程中，$R_E$ 乘以 $(1+\beta)$ 而不是 $\beta$：$V_{BB} - V_{BE} = I_{BQ}[R_{BB} + (1+\beta)R_E]$。如果误用 $\beta$，算出的 $I_{BQ}$ 会偏小约 $1/\beta$ 的相对误差（约 1%），对 $\beta$ 很小的情况误差大。

### 3g. 应用连接

射极跟随器在模拟电路中无处不在：电压源的输出缓冲、音频功放的输出级（推挽输出）、长线驱动、ADC 输入缓冲。第 33 章运放中的电压跟随器（$A_v=1$）其实就是把射极跟随器的思路用运放实现的"升级版"。

---

## 4. 共基放大电路（CB）——同相、高频的"快枪手"

### 4a. 生活例子：传送带上的包裹分拣

想象一条传送带，包裹从一端塞进去（输入，在发射极 E），从另一端出来（输出，在集电极 C）。B 极接地，像传送带的地面——不动、不参与信号的传递，只是一个参考点。

包裹从进到出，方向一致（你从左边塞进去，右边出来的包裹标签还是正的）——这就是**同相**。而且传送带转得很快（高频特性好），适合处理高频信号。

还有一个更贴近电子的类比：**望远镜**。你把望远镜倒过来看——从目镜（E 极）往里看，物镜（C 极）那边出来的像还是正的（同相），而且视场很宽（带宽大）。而 CE 放大器则像正常用望远镜——倒像（反相），视场窄。

### 4b. 直观理解：输入在 E，输出在 C，B 极"镇守中央"

CB 放大器的接法：
- 输入信号加在 **E 极**
- 输出信号从 **C 极**取出
- **B 极**接一个固定直流电压（交流接地），是输入和输出的公共端

```
          +Vcc
           │
          ┌┴┐
          │ │ R_C
          │ │
          └┬┘
           ├───── V_out（从 C 极输出）
         C │
    ───┤   BJT (NPN)
         E │───── V_in（输入加在 E 极）
           │
          ┌┴┐
          │ │ R_E（可能）
          │ │
          └┬┘
           │
          ═══  GND
```

信号流：
1. $v_{in}$ 加在 E 极 → E 极电压变化
2. B 极电压固定 → $v_{BE}$ 变化（$v_{BE} = v_B - v_E = -v_E$，注意负号！）
3. $v_{BE}$ 的变化通过 $g_m$ 产生 $\Delta i_C = g_m \cdot \Delta v_{BE}$
4. 而 $\Delta v_{BE} = -\Delta v_{in}$，所以 $\Delta i_C = -g_m \cdot \Delta v_{in}$，输出 $\Delta v_{out} = -\Delta i_C \cdot R_C = g_m R_C \cdot \Delta v_{in}$

**增益 $A_v = +g_m R_C$（正号！同相）**。这是因为 E 极输入增加 → $v_{BE}$ 减小 → $i_C$ 减小 → $v_{out}$（C 极电压 = $V_{CC} - i_C R_C$）增加。两次反号，最终同相。

### 4c. 形式化定义：CB 放大器的小信号分析

**关键参数：**

$$
A_v = +g_m (R_C \parallel R_L) \quad \text{（同相！没有负号）}
$$

$$
R_{in} = R_E \parallel \frac{r_\pi}{1+\beta} = R_E \parallel r_e
$$

其中 $r_e = \dfrac{r_\pi}{1+\beta} \approx \dfrac{1}{g_m}$ 是从发射极"看进去"的等效小信号电阻。在室温下，$I_{CQ}=1\text{mA}$ 时 $r_e \approx 26\text{ }\Omega$——极小！

$$
R_{out} \approx R_C \quad \text{（与 CE 相同）}
$$

**CB 的三个显著特点：**

1. **输入电阻极低**（几十 Ω 量级）——因为信号从发射极注入，而发射极的等效交流电阻 $r_e$ 很小
2. **电压增益与 CE 相同**（$g_m R_C$），但同相——因为信号"绕了一圈"（E→B→C），两次反相抵消
3. **高频特性优异**——因为基极接地，切断了 C-B 之间的 Miller 电容反馈路径（详见第 7 节）

### 4d. 手算示例

**例题5：CB 放大器**

$V_{CC}=15\text{V}$，$R_C=5\text{kΩ}$，$R_E=2\text{kΩ}$（提供直流偏置），$R_1=80\text{kΩ}$，$R_2=20\text{kΩ}$（为 B 极提供固定偏压），$\beta=150$，$V_T=26\text{mV}$。

**直流分析**（跟 CE 的 Q 点计算一致）：

$V_{BB} = 15 \times 20/100 = 3\text{V}$，$R_{BB}=80\parallel20=16\text{kΩ}$

$I_{BQ} = (3-0.7)/[16\text{k} + 151 \times 2\text{k}] = 2.3/318\text{k} = 7.23\text{ }\mu\text{A}$

$I_{CQ} = 150 \times 7.23\mu = 1.085\text{ mA}$

**交流分析：**

| 参数 | 公式 | 计算 | 结果 |
|------|------|------|------|
| $g_m$ | $I_{CQ}/V_T$ | $1.085\text{m}/0.026$ | $41.7\text{ mS}$ |
| $r_\pi$ | $\beta/g_m$ | $150/0.0417$ | $3.60\text{ kΩ}$ |
| $r_e$ | $r_\pi/(1+\beta)$ | $3.60\text{k}/151$ | $23.8\text{ }\Omega$ |
| $R_{in}$ | $R_E \parallel r_e$ | $2\text{k} \parallel 23.8$ | **23.5 Ω** |
| $A_v$（空载） | $+g_m R_C$ | $+0.0417 \times 5000$ | **+208.5** |
| $R_{out}$ | $R_C$ | — | **5 kΩ** |

**对比 CE**（同样 $I_{CQ}$、$R_C$）：CE 增益是 -208.5（反相），CB 是 +208.5（同相），数值完全一样！区别只在符号和输入阻抗。

### 4e. Python 验证：三种组态增益对比

```python
# -*- coding: utf-8 -*-
# 31-04: CE/CC/CB 三种组态增益对比（相同 I_CQ 和 R_C）
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 统一参数
BETA = 150
VT = 0.026
RC = 5000
RE = 2000
ICQ_range = np.linspace(0.2e-3, 3e-3, 100)

gm_range = ICQ_range / VT
r_pi_range = BETA / gm_range

# CE 增益 (空载, 绝对值)
Av_CE = gm_range * RC

# CB 增益 (空载)
Av_CB = gm_range * RC  # 与CE绝对值相同

# CC 增益 (R_L很大)
Av_CC = (1+BETA) * RE / (r_pi_range + (1+BETA) * RE)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ICQ_range*1000, Av_CE, 'b-', linewidth=2, label='|CE| = |CB|（高增益组态）')
ax.plot(ICQ_range*1000, Av_CC*10, 'g-', linewidth=2, label='CC×10（跟随器,增益≈1）')
ax.set_xlabel(r'集电极直流电流 $I_{CQ}$ (mA)', fontsize=12)
ax.set_ylabel(r'电压增益 $|A_v|$', fontsize=12)
ax.set_title(r'三种组态增益对比（CE反相、CB同相、CC≈1）', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.annotate('CE/CB 增益随 I_CQ 线性增长', xy=(1.5, 280), fontsize=10, color='blue')
ax.annotate('CC 增益始终≈1（此处×10便于观察）', xy=(1.5, 9.6), fontsize=10, color='green')
plt.tight_layout()
plt.savefig('31-04-three-config-gain.png', dpi=150, bbox_inches='tight')
plt.show()

gm_test = 1e-3 / 0.026
print(f"=== 三种组态在 I_CQ=1mA, R_C=5kΩ, β=150 时的关键参数 ===")
print(f"CE: A_v = -{gm_test*RC:.0f} (反相), R_in≈r_pi={BETA/gm_test/1000:.1f}kΩ")
print(f"CB: A_v = +{gm_test*RC:.0f} (同相), R_in≈1/gm={1/gm_test:.0f}Ω (极低)")
print(f"CC: A_v ≈ 0.97 (同相), R_out≈1/gm={1/gm_test:.0f}Ω (极低)")
```

### 4f. 常见误区

❌ **误区7：把 CB 的增益写成 $-g_m R_C$（带负号）。**
CB 的增益公式里没有负号！E 极输入增加 → $v_{BE}$ 减小 → $i_C$ 减小 → $v_{out}$ 增加，两次反转，最终同相。$A_v = +g_m R_C$。

✓ **检验方法**：随手画一个箭头——输入 ↑ → $v_E$ ↑ → $v_{BE}$ ↓ → $i_C$ ↓ → $v_C$ ↑ → 输出 ↑。输入↑输出↑ → 同相。

### 4g. 应用连接

CB 放大器因为输入阻抗极低（~25 Ω），不常用作电压放大器（前级很难驱动这么低的阻抗）。但它在高频放大、射频电路、电流缓冲器中频繁出现。CB 的输出端（C 极）和输入端（E 极）之间几乎没有 Miller 电容，所以高频特性远优于 CE。在射频前端（比如 Wi-Fi 的 2.4 GHz 低噪声放大器）中，CB 和 cascode（CE+CB 级联）是主力拓扑。

---

## 5. 三种组态对比——一张表看清所有

### 5a. 直观理解：三种组态的"性格"

把三种组态想象成三个不同性格的工人：

- **CE（共射）**：力气大（高电压增益），但脾气倔（反相），而且不太合群（中等输入阻抗）
- **CC（共集/跟随器）**：脾气好（电压不放大但同相）、善于社交（高输入+低输出阻抗），力气用来"搬东西"（电流放大）
- **CB（共基）**：反应快（高频好），但"门槛低"（输入阻抗极低），适合接在需要速度的场合

### 5b. 形式化对比表

以 $I_{CQ}=1\text{mA}$、$\beta=150$、$R_C=5\text{kΩ}$、$R_E=2\text{kΩ}$（旁路）为基准：

| 特性 | CE（共射） | CC（共集/射极跟随器） | CB（共基） |
|------|:---:|:---:|:---:|
| **输入引脚** | B（基极） | B（基极） | E（发射极） |
| **输出引脚** | C（集电极） | E（发射极） | C（集电极） |
| **公共端** | E（发射极） | C（集电极） | B（基极） |
| **电压增益 $A_v$** | 高（$-g_m R_C$，约 -192） | ≈1（0.95~0.99） | 高（$+g_m R_C$，约 +192） |
| **相位** | **反相（180°）** | 同相（0°） | **同相（0°）** |
| **输入电阻 $R_{in}$** | 中等（~$r_\pi$，几 kΩ） | 高（几十~几百 kΩ） | **极低**（~$1/g_m$，几十 Ω） |
| **输出电阻 $R_{out}$** | 较高（~$R_C$，几 kΩ） | **极低**（几十 Ω） | 较高（~$R_C$，几 kΩ） |
| **电流增益** | 高（≈$\beta$） | 高（≈$1+\beta$） | ≈1（$i_{out} \approx i_{in}$） |
| **典型应用** | 电压放大主力 | 阻抗变换、缓冲 | 高频放大、电流缓冲 |
| **Miller 效应** | 强（限制高频） | 极弱 | **无**（基极接地隔离） |

**口诀记忆**：
- "CE 高反中" —— CE：高增益、反相、中等输入阻抗
- "CC 低同变" —— CC：低增益（≈1）、同相、阻抗变换
- "CB 高同快" —— CB：高增益、同相、速度快（高频好）

---

## 6. 多级放大——把多个放大器"串"起来

### 6a. 生活例子：接力赛跑

4×100 米接力赛，每一棒跑 100 米，总距离 = 100 + 100 + 100 + 100 = 400 米。但如果把"距离"换成"放大倍数"：

- 第一棒把信号放大 10 倍
- 第二棒把第一棒的结果再放大 10 倍
- 总放大 = 10 × 10 = 100 倍

多级放大就是"接力放大"：每一级把前一级的输出作为自己的输入，最终总增益 = 各级增益之积。

### 6b. 直观理解：为什么需要多级？

单个 CE 放大器的增益上限受限于：
- $g_m R_C$ 不能无限大（$R_C$ 太大会让 Q 点进入饱和区）
- 单级增益超过 500 容易不稳定、失真大
- 单级不能满足"高增益 + 高输入阻抗 + 低输出阻抗"的综合要求

多级放大把不同组态"串"起来：
- 第一级用 CC（高输入阻抗，不拖累信号源）
- 中间级用 CE（提供主要电压增益）
- 最后一级用 CC（低输出阻抗，驱动负载）

这就是经典的 **CC → CE → CC** 三级结构，在很多音频前放芯片内部都能见到。

### 6c. 形式化定义：级联分析

**总增益（电压）：**
$$
A_{v,\text{total}} = A_{v1} \times A_{v2} \times A_{v3} \times \cdots
$$

如果各级之间直接耦合（无隔直电容），每级的负载效应必须考虑。第 $n$ 级的**有效负载** = 本级集电极电阻 $R_{C,n}$ 并联下一级的输入电阻 $R_{in,n+1}$。

**多级增益的分贝表示：**
$$
A_{v,\text{total}}(\text{dB}) = A_{v1}(\text{dB}) + A_{v2}(\text{dB}) + A_{v3}(\text{dB}) + \cdots
$$

分贝相加比倍数相乘在数值上更方便——这就是分贝的"魔力"。

**级间耦合方式：**

| 耦合方式 | 特点 | 适用场景 |
|----------|------|----------|
| 电容耦合（阻容耦合）| 各级 Q 点独立，隔直流 | 分立元件放大器 |
| 直接耦合 | 无电容，低频可到 DC | 运放内部、IC 放大器 |
| 变压器耦合 | 阻抗匹配好，但体积大 | 射频/音频功放（老式） |

### 6d. 手算示例

**例题6：两级 CE+CC 放大器**

第一级：CE 放大器，$R_{C1}=5\text{kΩ}$，$I_{CQ1}=1\text{mA}$，$\beta=150$（空载 $A_{v1} = -g_{m1}R_{C1} = -192.3$）

第二级：CC 射极跟随器，$I_{CQ2}=2\text{mA}$，$\beta=150$，$R_{E2}=2\text{kΩ}$，$A_{v2} \approx 0.98$

**级间加载效应**：第二级 CC 的输入电阻是 CE 的负载。

$g_{m2} = 2\text{m}/0.026 = 76.9\text{ mS}$，$r_{\pi 2} = 150/0.0769 = 1.95\text{ kΩ}$

$R_{in2} = r_{\pi2} + (1+\beta)R_{E2} = 1.95\text{k} + 151\times2\text{k} = 303.95\text{ kΩ}$

第一级的有效负载：$R_{C1} \parallel R_{in2} = 5\text{k} \parallel 303.95\text{k} \approx 4.92\text{ kΩ}$

第一级的实际增益（有负载）：$A_{v1} = -g_{m1} (R_{C1} \parallel R_{in2}) = -0.03846 \times 4919 = -189.2$

| 级 | $A_v$ | 累计 $A_v$ | 累计 $A_v$ (dB) |
|----|:-----:|:----------:|:---------------:|
| CE | -189.2 | -189.2 | 45.5 dB |
| CC | 0.98 | **-185.4** | **45.4 dB** |

因为 CC 的输入阻抗极高（~304 kΩ >> $R_{C1}=5\text{kΩ}$），对前级的负载效应几乎可以忽略。CE 的实际增益（-189.2）和空载增益（-192.3）只差不到 2%。这就是为什么 CC 作为缓冲器如此好用。

### 6e. Python 验证：多级总增益

```python
# -*- coding: utf-8 -*-
# 31-05: 多级放大器级联总增益验证
import numpy as np

# 总增益 = 各级增益之积
Av_stage = np.array([-189.2, 0.98, 15.0, 0.95])  # CE + CC + CE + CC

Av_total_mult = np.prod(np.abs(Av_stage))  # 绝对值乘积
Av_total_mult_signed = np.prod(Av_stage)    # 带符号

Av_dB_per_stage = 20 * np.log10(np.abs(Av_stage))
Av_dB_total = np.sum(Av_dB_per_stage)

print("=== 四级放大器：CE → CC → CE → CC ===")
print(f"各级增益: {Av_stage}")
print(f"各级增益绝对值: {np.abs(Av_stage)}")
print(f"总增益（绝对值乘积）: {Av_total_mult:.0f}")
print(f"总增益（带符号）: {Av_total_mult_signed:.0f}")
# 相位：CE 反相(-)、CC 同相(+)、CE 反相(-)、CC 同相(+)
# 两个 CE → (-1)^2 = +1 → 最终同相
print("最终相位：同相（两个 CE 各反相 180°，互相抵消）")

print(f"\n分贝计算：")
for i, (Av_signed, Av_db) in enumerate(zip(Av_stage, Av_dB_per_stage)):
    print(f"  第{i+1}级: {Av_signed:7.1f} → {Av_db:6.1f} dB")
print(f"  总增益(dB) = {' + '.join([f'{x:.1f}' for x in Av_dB_per_stage])} = {Av_dB_total:.1f} dB")
print(f"\n验证：{Av_dB_total:.1f} dB = 10^({Av_dB_total:.1f}/20) = {10**(Av_dB_total/20):.0f} ≈ {Av_total_mult:.0f} ✓")
```

预期输出：
```
=== 四级放大器：CE → CC → CE → CC ===
总增益（绝对值乘积）: 2639
总增益（带符号）: 2639
最终相位：同相（两个 CE 各反相 180°，互相抵消）
总增益(dB) = 45.5 + -0.2 + 23.5 + -0.4 = 68.4 dB
验证：68.4 dB = 10^(68.4/20) = 2639 ≈ 2639 ✓
```

### 6e.1 补充示例：两级 CE 直耦的问题

上面例题用 CC 做第二级，级间匹配很好。但如果第二级也是 CE：

设 $R_{C1}=5\text{kΩ}$，$r_{\pi2}=2.5\text{kΩ}$（第二级 CE 的输入电阻）

有效负载 $R_{L1} = 5\text{k} \parallel 2.5\text{k} = 1.67\text{kΩ}$，只有 $R_{C1}$ 的 33%！

这说明 CE 级联 CE 时，后级的低输入阻抗会"吃掉"前级的大部分增益。这就是为什么实际设计中 CE-CE 中间通常要插一级 CC 缓冲。CC 缓冲像"阻抗变压站"——从前级看进去是高阻抗（不拖累），对后级输出是低阻抗（驱动自如）。

### 6f. 常见误区

❌ **误区8：直接把各级空载增益相乘，忽略级间负载效应。**
每级的实际增益必须用"本级的 $R_C$ 并联下一级输入电阻"作为有效负载。如果下一级输入阻抗不高（比如 CE 的 $r_\pi$ 只有几 kΩ），前级增益会明显衰减。

✓ **正确做法**：从最后一级往前算——先算最后一级的输入电阻，把它当成前一级的负载，再算前一级的增益。逐级往前推。

### 6g. 应用连接

多级放大在运放内部（第 33 章）体现得最充分：运放的内部结构就是差分级（CE 变形）→ 中间增益级（CE）→ 输出缓冲级（CC 推挽）。学会多级分析，你就拿到了理解运放内部电路的钥匙。

---

## 7. 频率响应——放大器为什么对高频"力不从心"？

### 7a. 生活例子：你在嘈杂的酒吧里和朋友聊天

在安静的咖啡馆里，你能清楚地听到朋友说的每个字（低频、中频都能"放大"）。但在嘈杂的酒吧里，低频的"嗡嗡"背景噪声让你听不清（低频被衰减），朋友尖着嗓子提高音调你也只能听个大概（高频也被衰减）——只有中间一段频率范围你能听得最清楚。

放大器也是这样：它不是说对任何频率的信号都能同样放大。在某个频率范围内增益最大且平坦（这叫**中频区**或**带宽内**），低于某个频率（$f_L$）、高于某个频率（$f_H$）增益都会下降。这种增益随频率变化的特性就叫**频率响应**。

更具体地说，一个音频放大器如果 $f_L = 100\text{ Hz}$、$f_H = 5\text{ kHz}$，那么低音（比如贝斯的 40 Hz 低频）和高音（比如镲片的 8 kHz 高频）都会被严重衰减。听感上就是"闷"——低音薄、高音暗。这就是为什么 Hi-Fi 音响的放大器必须把 $f_L$ 做到 20 Hz 以下、$f_H$ 做到 20 kHz 以上的原因。

### 7b. 直观理解：电容在"捣鬼"

放大器里有很多电容。有些是你故意加的（耦合电容、旁路电容），有些是"天生"的（BJT 内部的寄生电容，如 $C_\pi$、$C_\mu$）。

这些电容对不同频率的信号表现不同：

- **频率很低时**：电容阻抗 $X_C = 1/(2\pi fC)$ 很大，耦合电容开始阻碍信号通过 → 低频增益下降
- **频率很高时**：寄生电容的阻抗变得很小，信号被"旁路"到地 → 高频增益下降

只有在一个中间频率区间——耦合电容已经"消失"（阻抗极小），寄生电容还"没醒"（阻抗极大）——增益才是最大值，这就是**中频增益** $A_{v,\text{mid}}$。

### 7c. 形式化定义：$f_L$、$f_H$ 和带宽

**低频截止频率 $f_L$（lower cutoff frequency）：**

$f_L$ 是增益下降到中频增益的 $1/\sqrt{2}$（约 70.7%，即 -3 dB）时的低端频率。$f_L$ 主要由耦合电容和旁路电容决定：

$$
f_L \approx \sqrt{f_{L1}^2 + f_{L2}^2 + \cdots}
$$

**高频截止频率 $f_H$（upper cutoff frequency）：**

$f_H$ 是增益下降到中频增益的 $1/\sqrt{2}$（约 70.7%，即 -3 dB）时的高端频率。$f_H$ 主要由 BJT 的内部寄生电容和 Miller 效应决定：

$$
f_H \approx \frac{1}{2\pi \cdot R_{eq} \cdot C_{eq}}
$$

其中 $C_{eq}$ 包含了 Miller 放大后的 $C_\mu$。

**带宽 $BW$：**
$$
BW = f_H - f_L
$$

如果 $f_H \gg f_L$（大多数情况），近似 $BW \approx f_H$。

**增益-带宽积（Gain-Bandwidth Product, GBW）：**

对于给定的 BJT 和偏置，中频增益和带宽大致成反比：
$$
|A_{v,\text{mid}}| \times BW \approx \text{常数}
$$

这就是设计中的经典权衡：要提高增益就得牺牲带宽，要扩张带宽就得接受较低增益。

### 7d. 手算示例

**例题7：CE 放大器的低频截止频率**

CE 放大器有三个耦合/旁路电容，每个产生一个低频极点：
- 输入耦合电容 $C_{C1}$ 与 $R_{in}$：$f_{L1} = 1/(2\pi R_{in} C_{C1})$
- 输出耦合电容 $C_{C2}$ 与 $(R_C + R_L)$：$f_{L2} = 1/(2\pi (R_C+R_L) C_{C2})$
- 旁路电容 $C_E$ 与发射极等效电阻：$f_{L3} = 1/(2\pi R_{eq,E} C_E)$

设 $R_{in}=4\text{kΩ}$，$R_C=5\text{kΩ}$，$R_L=10\text{kΩ}$，$R_{eq,E}=50\text{ }\Omega$，$\beta=150$

| 电容 | 容值 | 相关电阻 | $f_{L,i}$ |
|------|:----:|------|:---:|
| $C_{C1}$ | $1\text{ }\mu\text{F}$ | $4\text{ kΩ}$ | $1/(2\pi\cdot4000\cdot10^{-6}) = 39.8\text{ Hz}$ |
| $C_{C2}$ | $1\text{ }\mu\text{F}$ | $15\text{ kΩ}$ | $1/(2\pi\cdot15000\cdot10^{-6}) = 10.6\text{ Hz}$ |
| $C_E$ | $100\text{ }\mu\text{F}$ | $50\text{ }\Omega$ | $1/(2\pi\cdot50\cdot100\times10^{-6}) = 31.8\text{ Hz}$ |

总 $f_L = \sqrt{39.8^2 + 10.6^2 + 31.8^2} \approx 52.0\text{ Hz}$

主导极点（最大的 $f_{L,i}$）是 $C_{C1}$ 产生的 39.8 Hz。

**例题8：高频截止 $f_H$ 与 Miller 效应**

CE 放大器中，C-B 之间的寄生电容 $C_\mu$（约 2 pF）因为 Miller 效应被"放大"了。

**Miller 定理**：跨接在输入和输出之间（且输出是反相放大）的电容，等效到输入端时电容值变为 $(1+|A_v|)C_\mu$。

设 CE 中频增益 $|A_{v,\text{mid}}| = 192$，$C_\mu = 2\text{ pF}$，$C_\pi = 10\text{ pF}$。

Miller 等效输入电容：
$$
C_{in} = C_\pi + (1 + |A_v|) C_\mu = 10\text{ pF} + 193 \times 2\text{ pF} = 10 + 386 = 396\text{ pF}
$$

等效输入端总电阻：$R_{eq} = R_{sig} \parallel R_1 \parallel R_2 \parallel r_\pi$

设 $R_{sig}=600\text{ }\Omega$，$R_1\parallel R_2=16\text{kΩ}$，$r_\pi=3.9\text{kΩ}$

$R_{eq} = 600 \parallel 16000 \parallel 3900 \approx 482\text{ }\Omega$

$$
f_H = \frac{1}{2\pi \cdot R_{eq} \cdot C_{in}} = \frac{1}{2\pi \times 482 \times 396 \times 10^{-12}} \approx 834\text{ kHz}
$$

**对比 CB**：CB 的基极接地，C-B 之间没有 Miller 效应。等效输入电容 ≈ $C_\pi = 10\text{ pF}$，$R_{eq}$ 是发射极等效电阻 $r_e \approx 26\text{ }\Omega$。

$f_H \approx 1/(2\pi \times 26 \times 10 \times 10^{-12}) \approx 612\text{ MHz}$——比 CE 高了近 1000 倍！

这就是 CB 高频性能优异的核心原因。

### 7e. Python 代码：画出完整的频率响应曲线（Bode 图）

```python
# -*- coding: utf-8 -*-
# 31-06: CE放大器频率响应 Bode 图
# 展示低频截止 f_L、中频平坦区、高频截止 f_H

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 参数
Av_mid = 192.3
Av_mid_dB = 20 * np.log10(Av_mid)

fL = 52.0          # 低频 -3dB 点 (Hz)
fH = 834e3         # 高频 -3dB 点 (Hz)
BW = fH - fL

# 频率范围：1 Hz 到 100 MHz
f = np.logspace(0, 8, 1000)

# 简化的传递函数模型：两个极点
# |H(f)| = Av_mid / sqrt(1+(fL/f)^2) / sqrt(1+(f/fH)^2)
mag = Av_mid / np.sqrt(1 + (fL/f)**2) / np.sqrt(1 + (f/fH)**2)
mag_dB = 20 * np.log10(mag)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.semilogx(f, mag_dB, 'b-', linewidth=2)
ax.axhline(y=Av_mid_dB, color='green', linestyle='--', alpha=0.4,
           label=f'中频增益 {Av_mid_dB:.1f} dB')
ax.axhline(y=Av_mid_dB - 3, color='red', linestyle='--', alpha=0.4, label='-3 dB')

# 标注 f_L 和 f_H
ax.axvline(x=fL, color='orange', linestyle=':', alpha=0.6)
ax.axvline(x=fH, color='orange', linestyle=':', alpha=0.6)
ax.annotate(f'$f_L$={fL:.0f} Hz', xy=(fL, Av_mid_dB-3),
            xytext=(fL*5, Av_mid_dB-8),
            arrowprops=dict(arrowstyle='->', color='orange'), fontsize=11, color='orange')
ax.annotate(f'$f_H$={fH/1e3:.0f} kHz', xy=(fH, Av_mid_dB-3),
            xytext=(fH*3, Av_mid_dB-8),
            arrowprops=dict(arrowstyle='->', color='orange'), fontsize=11, color='orange')

# 带宽标注
ax.annotate(f'带宽 = {BW/1e3:.1f} kHz', xy=(fL*100, Av_mid_dB-1.5),
            fontsize=11, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.4))

ax.set_xlabel('频率 (Hz)', fontsize=13)
ax.set_ylabel('增益 (dB)', fontsize=13)
ax.set_title(f'CE放大器频率响应 (Bode图)\n$f_L$={fL:.0f}Hz, $f_H$={fH/1e3:.0f}kHz, BW≈{BW/1e3:.1f}kHz',
             fontsize=14)
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11)
ax.set_ylim(Av_mid_dB-35, Av_mid_dB+3)
plt.tight_layout()
plt.savefig('31-06-freq-response-bode.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"中频增益: {Av_mid_dB:.1f} dB")
print(f"低频 -3dB 点: f_L = {fL:.0f} Hz")
print(f"高频 -3dB 点: f_H = {fH/1e3:.1f} kHz")
print(f"带宽 BW = f_H - f_L ≈ {BW/1e3:.1f} kHz")
print(f"增益-带宽积 GBW = {Av_mid:.0f} × {BW/1e3:.1f} kHz = {Av_mid*BW/1e6:.1f} MHz")
```

### 7f. 常见误区

❌ **误区9：以为"带宽内增益完全不变"。**
带宽内的增益确实是最大值，但"平坦"是相对的。在 $f_L$ 和 $f_H$ 的 0.1 倍和 10 倍处，增益已经开始有轻微下降（约 -0.04 dB）。真正的完全平坦只在 $f \gg f_L$ 且 $f \ll f_H$ 时。

✓ **设计经验**：如果要 20 Hz ~ 20 kHz（音频范围）内增益波动不超过 0.5 dB，设计 $f_L < 2\text{ Hz}$，$f_H > 200\text{ kHz}$。把截止频率放在目标频率范围的 10 倍之外。

❌ **误区10：把所有电容的截止频率标量相加来算 $f_L$。**
$f_L$ 是 RSS（Root-Sum-Square）合成，不是直接相加。每个电容产生的极点频率 $f_{Li}$ 先平方、再求和、最后开方：$f_L = \sqrt{\sum f_{Li}^2}$。

### 7g. 应用连接

频率响应是模拟电路设计的"必修课"。第 33 章运放的频率补偿就是利用 Miller 电容（有意加的）来降低高频增益、防止振荡。第 34 章的滤波器则用频率选择性来"只放大你想要的频率"。

---

## 8. 完整实战——从偏置到频率响应的全流程设计

### 8a. 设计目标

设计一个 CE-CC 两级音频前放，规格如下：

| 指标 | 要求 |
|------|------|
| 电源电压 | $V_{CC} = 15\text{ V}$ |
| 中频电压增益 | $\vert A_v \vert \geq 180$ |
| 输入电阻 | $\geq 10\text{ kΩ}$ |
| 输出电阻 | $\leq 100\text{ }\Omega$ |
| 低频 -3dB | $\leq 20\text{ Hz}$ |
| 高频 -3dB | $\geq 50\text{ kHz}$ |
| 负载电阻 | $R_L = 1\text{ kΩ}$ |
| 晶体管 | $\beta=150$, $V_A=100\text{V}$ |

### 8b. 拓扑选择

- **输入级用 CC**（高输入阻抗 → 满足 $R_{in} \geq 10\text{kΩ}$）
- **中间级用 CE**（提供主要电压增益 → 满足 $|A_v| \geq 180$）
- **输出级用 CC**（低输出阻抗 → 满足 $R_{out} \leq 100\text{ }\Omega$，驱动 $R_L=1\text{kΩ}$）

### 8c. 参数设计

**第一级 CC（输入缓冲）：**

选 $I_{CQ1}=1\text{mA}$，$R_{E1}=5.6\text{kΩ}$。

$V_{E1} = I_{CQ1} \cdot R_{E1} = 1\text{mA} \times 5.6\text{kΩ} = 5.6\text{V}$，$V_{B1} \approx 5.6 + 0.7 = 6.3\text{V}$

选 $R_{B1}=200\text{kΩ}$，$R_{B2}=150\text{kΩ}$ → $V_{BB1} = 15 \times 150/350 \approx 6.43\text{V}$ ✓

$r_{\pi1} = \beta V_T / I_{CQ1} = 150 \times 0.026 / 0.001 = 3.9\text{kΩ}$

$R_{in1} = R_{B1} \parallel R_{B2} \parallel [r_{\pi1} + (1+\beta)R_{E1}]$

$= 200\text{k}\parallel150\text{k}\parallel[3.9\text{k}+151\times5.6\text{k}] = 85.7\text{k} \parallel 849.5\text{k} \approx 77.8\text{ kΩ} \gg 10\text{ kΩ}$ ✓

**第二级 CE（增益级）：**

选 $I_{CQ2}=1.5\text{mA}$，$R_{C2}=6.8\text{kΩ}$，$R_E$ 被 $C_E$ 旁路。

$g_{m2} = 1.5\text{mA} / 0.026\text{V} = 57.7\text{ mS}$

空载增益 $= -g_{m2} R_{C2} = -57.7\text{m} \times 6.8\text{k} = -392$

但需要先算第三级输入电阻作为第二级的负载。

**第三级 CC（输出缓冲）：**

选 $I_{CQ3}=3\text{mA}$，$R_{E3}=1.5\text{kΩ}$。

$r_{\pi3} = 150 \times 0.026 / 0.003 = 1.3\text{kΩ}$

$R_{in3} = r_{\pi3} + (1+\beta)(R_{E3}\parallel R_L) = 1.3\text{k} + 151\times(1.5\text{k}\parallel1\text{k}) = 1.3\text{k} + 151\times600 = 91.9\text{ kΩ}$

**回算第二级 CE 的实际增益：**

有效负载 $= R_{C2} \parallel R_{in3} = 6.8\text{k} \parallel 91.9\text{k} = 6.33\text{ kΩ}$

$A_{v2} = -g_{m2} \times 6.33\text{k} = -57.7\text{m} \times 6.33\text{k} = -365$

**总增益：** $|A_{v,\text{total}}| = 0.99 \times 365 \times 0.97 \approx 350 \gg 180$ ✓

**输出电阻：** $R_{out} \approx (r_{\pi3} + R_{C2})/(1+\beta) \parallel R_{E3} = 53.6 \parallel 1500 \approx 51.7\text{ }\Omega \leq 100\text{ }\Omega$ ✓

### 8d. 频率响应验算

耦合电容选 $C_{C1}=C_{C2}=C_{C3}=1\text{ }\mu\text{F}$，旁路电容 $C_E=100\text{ }\mu\text{F}$。

最坏情况 $f_L$ ≈ $1/(2\pi \times 77.8\text{k} \times 1\mu) \approx 2\text{ Hz}$，远低于 20 Hz ✓

详细：第一级 CC 的输入耦合电容 $C_{C1}$ 与 $R_{in1}=77.8\text{kΩ}$ 形成的高通极点 $f_{L1} = 1/(2\pi \times 77.8\text{k} \times 1\mu) \approx 2.05\text{ Hz}$。这个极点主导了总 $f_L$，因为它是最大的一个。

CC-CE-CC 拓扑中每级的 Miller 效应都较小（输入级和输出级都是 CC，增益 ≈1）。第二级 CE 的增益约 -365，Miller 电容被放大到 $C_{in} \approx C_\pi + 366 \times C_\mu \approx 10\text{p} + 366 \times 2\text{p} \approx 742\text{ pF}$。输入端等效电阻约几 kΩ，$f_H \approx 1/(2\pi \times R_{eq} \times 742\text{p}) \approx 500\text{ kHz}$ ✓

所以最终 $f_L \approx 2\text{ Hz} \ll 20\text{ Hz}$ ✓，$f_H \approx 500\text{ kHz} \gg 50\text{ kHz}$ ✓。带宽远远超出音频范围，设计裕量充足。

### 8e. Python 验证完整设计

```python
# -*- coding: utf-8 -*-
# 31-07: CC-CE-CC 三级音频前放完整验证

VT = 0.026
BETA = 150

# ========= 第一级：CC 输入缓冲 =========
ICQ1 = 1e-3
gm1 = ICQ1 / VT
r_pi1 = BETA / gm1
RE1 = 5600
RB1, RB2 = 200e3, 150e3
RBB1 = (RB1 * RB2) / (RB1 + RB2)
Av1 = (1+BETA)*RE1 / (r_pi1 + (1+BETA)*RE1)
Rin1 = 1 / (1/RBB1 + 1/(r_pi1 + (1+BETA)*RE1))

# ========= 第二级：CE 增益级 =========
ICQ2 = 1.5e-3
gm2 = ICQ2 / VT
RC2 = 6800

# ========= 第三级：CC 输出缓冲 =========
ICQ3 = 3e-3
gm3 = ICQ3 / VT
r_pi3 = BETA / gm3
RE3 = 1500
RL = 1000
Req3 = (RE3 * RL) / (RE3 + RL)
Av3 = (1+BETA)*Req3 / (r_pi3 + (1+BETA)*Req3)
Rin3 = r_pi3 + (1+BETA)*Req3

# 第二级实际负载
R_load2 = (RC2 * Rin3) / (RC2 + Rin3)
Av2 = -gm2 * R_load2

# 第三级输出电阻
Rout3 = ((r_pi3 + RC2) / (1+BETA) * RE3) / ((r_pi3 + RC2) / (1+BETA) + RE3)

Av_total = Av1 * Av2 * Av3

print("="*55)
print("  CC-CE-CC 三级音频前放 — 完整设计验证")
print("="*55)
print(f"\n第一级 CC: I_CQ1={ICQ1*1000:.1f}mA, A_v1={Av1:.4f}, R_in1={Rin1/1000:.1f}kΩ ✓")
print(f"第二级 CE: I_CQ2={ICQ2*1000:.1f}mA, g_m2={gm2*1000:.1f}mS, A_v2={Av2:.0f}")
print(f"  有效负载 R_C2||R_in3 = {R_load2:.0f}Ω")
print(f"第三级 CC: I_CQ3={ICQ3*1000:.1f}mA, A_v3={Av3:.4f}, R_out3={Rout3:.1f}Ω ✓")
print(f"\n总增益 |A_v| = {abs(Av_total):.0f} (要求≥180) ✓")
print(f"总增益 (dB) = {20*np.log10(abs(Av_total)):.1f} dB")
print(f"输入电阻 = {Rin1/1000:.1f} kΩ (要求≥10kΩ) ✓")
print(f"输出电阻 = {Rout3:.1f} Ω (要求≤100Ω) ✓")
print("="*55)
```

---

## 9. 思考题

### 基础题（1-3）

**题1：** 一个 CE 放大器，$I_{CQ}=2\text{mA}$，$R_C=3.3\text{kΩ}$，$\beta=200$，$V_T=26\text{mV}$。求空载中频增益 $A_v$。

**题2：** 为什么射极跟随器（CC）的电压增益总是 < 1？从电路原理上解释（不用公式）。

**题3：** CB 放大器的输入电阻通常只有几十欧姆，这在实际应用中是优势还是劣势？在什么场景下这个特性是优势？

**题3b（补充）：** 你手头有一个信号源（内阻 10 kΩ）、一个 CB 放大器（$R_{in}=25\text{ }\Omega$、$A_v=+200$）、一个 CC 放大器（$R_{in}=100\text{ kΩ}$、$A_v\approx 1$）。如果你想用 CB 的 +200 倍同相增益来放大信号源，直接连可以吗？不行的话，应该怎么连？

### 进阶题（4-6）

**题4：** 一个 CE 放大器使用 $R_E=1\text{kΩ}$ 且完全旁路。如果移除旁路电容 $C_E$（即 $R_E$ 参与交流反馈），增益从 -200 下降到多少？设 $g_m=40\text{mS}$。

**题5：** 两级放大器：第一级 CE（空载增益 -150），第二级 CE（空载增益 -80）。第一级 $R_{C1}=4.7\text{kΩ}$，$r_{\pi1}=2.5\text{kΩ}$；第二级 $r_{\pi2}=1.8\text{kΩ}$，$R_{C2}=3.3\text{kΩ}$。考虑级间负载效应，求实际总增益和最终输出相位。

**题6：** CE 放大器中 $C_\mu=3\text{pF}$，$C_\pi=8\text{pF}$，中频增益 $|A_v|=250$，等效输入电阻 $R_{eq}=1\text{kΩ}$。估算高频 -3dB 频率 $f_H$。

### 综合题（7-8）

**题7：** 设计一个 $f_L \leq 15\text{Hz}$ 的 CE 放大器音频前级。已知 $R_{in}=5\text{kΩ}$，$R_C+R_L=12\text{kΩ}$，$R_{eq,E}=40\text{ }\Omega$。确定 $C_{C1}$、$C_{C2}$、$C_E$ 的最小取值。

**题8（开放讨论）：** 你在做音频功放，需要 1000 倍（60 dB）的总增益。两种方案：方案 A 用一级 CE（需 $A_v=-1000$），方案 B 用 CE-CE-CC 三级。从频率响应、失真、输入/输出阻抗、设计复杂度四个维度对比两种方案。你会选哪个？为什么？

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

**题1解答：**

$g_m = I_{CQ}/V_T = 0.002 / 0.026 = 76.92\text{ mS}$

$A_v = -g_m R_C = -0.07692 \times 3300 = -253.8$

答：空载中频增益约 -254（反相）。

---

**题2解答：**

射极跟随器的输出从发射极取出，而发射极电压 = 基极电压 - 0.7V（B-E 结压降）。因为 B-E 结压降几乎恒定，所以 $v_E$ 的变化量 ≈ $v_B$ 的变化量。如果增益 = 1，输出电压完美跟随输入。但实际 BJT 内部 $r_\pi$ 上会损耗一点小信号电压（$v_{be} = i_b \cdot r_\pi$），所以 $v_E$ 的变化量略小于 $v_B$ 的变化量，增益略小于 1。这是 BJT 物理特性决定的，不是设计缺陷。

---

**题3解答：**

极低的输入电阻在电压放大场景下是劣势（信号源难驱动）。但在以下场景是优势：

1. **高频放大**：低输入阻抗 + 无 Miller 效应 → 极宽带宽（CB 的 $f_H$ 可达 600 MHz 以上）
2. **电流缓冲**：CB 的电流增益 ≈ 1，输入输出电流几乎相等，适合做电流镜的一部分
3. **cascode 结构**：CE+CB 级联中，CB 的低输入阻抗为 CE 级提供了一个"虚拟地"，消除 Miller 效应

**题3b解答：**

直接连不行。信号源内阻 10 kΩ >> CB 输入电阻 25 Ω，电压几乎全部降在信号源内阻上（分压比 $25/(10000+25) \approx 0.25\%$），到达 CB 输入端的信号只有原来的 0.25%。

正确接法：在信号源和 CB 之间插入一级 CC 射极跟随器。CC 的 $R_{in}=100\text{ kΩ}$ >> 信号源内阻 10 kΩ（分压比 $100\text{k}/(100\text{k}+10\text{k}) \approx 91\%$），信号损失小；CC 的输出电阻极低（几十 Ω），正好匹配 CB 的低输入阻抗。这就是经典的 **CC → CB** 宽带缓冲-放大链。

---

**题4解答：**

增益从 $A_v = -g_m R_C$ 变为 $A_v = -\dfrac{g_m R_C}{1 + g_m R_E}$。

代入：$-200 = -g_m R_C$ → $g_m R_C = 200$。$R_C = 200/0.04 = 5\text{kΩ}$

新增益 $= -\dfrac{0.04 \times 5000}{1 + 0.04 \times 1000} = -\dfrac{200}{1 + 40} = -\dfrac{200}{41} \approx -4.88$

增益从 200 暴跌到约 4.9，下降了约 97.5%。$R_E$ 的负反馈严重压制了增益，但同时大幅提升了线性度和输入阻抗。

---

**题5解答：**

第二级的输入电阻 $R_{in2} = r_{\pi2} = 1.8\text{kΩ}$

第一级的有效负载 $R_{L1} = R_{C1} \parallel R_{in2} = 4.7\text{k} \parallel 1.8\text{k} = 1.30\text{ kΩ}$

$g_{m1} = 150/4700 = 0.0319\text{ S}$

第一级的实际增益 $A_{v1} = -0.0319 \times 1300 = -41.5$

第二级空载增益 $A_{v2} = -80$

总增益 $A_{v,\text{total}} = (-41.5) \times (-80) = +3320$

相位：两级 CE，每级反相 180°，两级 360° → **最终同相**。

关键教训：两级 CE 直接级联时，第二级的低输入电阻（1.8 kΩ）严重拖累了第一级的增益（从 -150 跌到 -41.5）。在两级之间插入一个 CC 缓冲器会好得多。

---

**题6解答：**

Miller 等效输入电容：
$C_{in} = C_\pi + (1 + |A_v|) C_\mu = 8\text{ pF} + 251 \times 3\text{ pF} = 8 + 753 = 761\text{ pF}$

高频截止频率：
$f_H = \dfrac{1}{2\pi R_{eq} C_{in}} = \dfrac{1}{2\pi \times 1000 \times 761 \times 10^{-12}}$

$= \dfrac{1}{4.78 \times 10^{-6}} \approx 209\text{ kHz}$

答：高频 -3dB 频率约 209 kHz。

---

**题7解答：**

保守设计：每个电容的极点频率 ≤ $f_L/3 = 5\text{ Hz}$

$C_{C1} \geq 1/(2\pi \times 5000 \times 5) = 6.37\text{ }\mu\text{F}$

$C_{C2} \geq 1/(2\pi \times 12000 \times 5) = 2.65\text{ }\mu\text{F}$

$C_E \geq 1/(2\pi \times 40 \times 5) = 795.8\text{ }\mu\text{F}$

工程选值（取标准值）：$C_{C1}=10\text{ }\mu\text{F}$，$C_{C2}=4.7\text{ }\mu\text{F}$，$C_E=1000\text{ }\mu\text{F}$。

验算实际 $f_L$：
$f_{L1}' = 1/(2\pi\times5000\times10\mu) = 3.18\text{ Hz}$
$f_{L2}' = 1/(2\pi\times12000\times4.7\mu) = 2.82\text{ Hz}$
$f_{L3}' = 1/(2\pi\times40\times1000\mu) = 3.98\text{ Hz}$

$f_L = \sqrt{3.18^2 + 2.82^2 + 3.98^2} \approx 5.82\text{ Hz} \ll 15\text{ Hz}$ ✓

---

**题8解答（开放讨论）：**

**方案A：单级 CE，$A_v=-1000$**

优点：电路极简，元件数最少，PCB 面积小，设计调试简单。

缺点：
- 增益 1000 → $g_m R_C = 1000$，在合理 $I_{CQ}$ 下 $R_C$ 很大（如 $I_{CQ}=2$mA → $R_C=13$kΩ），Q 点设计困难
- Miller 效应极强：$C_{in} \approx C_\pi + 1001 C_\mu$，$f_H$ 可能低到几十 kHz
- 输入阻抗只有 $r_\pi$（几 kΩ），对信号源负载重
- 输出阻抗 = $R_C$（约 13 kΩ），无法直接驱动 8Ω 扬声器
- 高增益下失真大，单级线性范围窄
- **结论：不现实**

**方案B：CE-CE-CC 三级**

优点：
- 每级增益可分配：CE1 × CE2 × CC ≈ -150 × -7 × 0.95 ≈ 997 ✓
- 输入阻抗可灵活调整（第一级加 CC）
- 输出级 CC 提供极低输出阻抗（可驱动扬声器）
- Miller 效应拆分到两级，每级增益小，$f_H$ 更高
- 失真分散到各级，整体线性度更好
- 方案成熟，业界标准做法

缺点：元件数多，功耗大，设计和调试更复杂，级间耦合电容多

**推荐选择：方案 B**。60 dB 的总增益用单级 CE 实现几乎不可行（带宽、失真、阻抗匹配全面崩溃）。CE-CE-CC 三级结构是经典方案，平衡了增益、带宽和驱动能力。

</details>

---

## 本章小结

你已经学完了 BJT 放大电路的完整知识体系。回看核心问题——"微弱信号怎么变成大嗓门？"——答案就是三个关键词：**偏置（Q 点）→ 模型（小信号 π 模型）→ 组态（CE/CC/CB 三种接法）**。

让我们用一个完整的"信号旅程"来回顾本章的核心内容。假设你对着麦克风唱了一个音：

1. 麦克风输出 5 mV 的微弱交流信号
2. 这个信号首先进入 CC 射极跟随器（输入缓冲）：电压不变（≈5 mV），但输入阻抗很高（100 kΩ 级），不拖累麦克风
3. 信号进入 CE 共射放大器：5 mV × (-200) = -1V，电压被放大 200 倍，但相位反转
4. 信号进入第二级 CE：-1V × (-30) = +30V——等等，电源只有 15V，输出会削顶！所以实际增益不能设这么高，或者第二级用 CC 缓冲
5. 最终 CC 输出级：电压几乎不变，但输出阻抗降到几十 Ω，能驱动 8Ω 的扬声器
6. 中间的耦合电容保证了各级 Q 点独立，但也导致了低频衰减（$f_L$）
7. BJT 内部的寄生电容和 Miller 效应导致了高频衰减（$f_H$）
8. 最终带宽内的信号被放大、驱动到扬声器，你听到了自己的歌声

这就是本章的全部故事。具体来说：

1. **小信号模型**：把 BJT 在 Q 点附近"线性化"，用 $g_m$、$r_\pi$、$r_o$ 三个参数描述它的交流行为。这是分析一切放大器的通用工具。

2. **CE 共射放大器**：电压增益的主力军，$A_v = -g_m R_C$（反相）。增益高，输入/输出阻抗中等，是最常用的增益级。

3. **CC 共集放大器（射极跟随器）**：电压增益 ≈ 1，但电流增益 ≈ β。高输入阻抗 + 低输出阻抗，是"阻抗变换器"。放在信号链的两端做缓冲。

4. **CB 共基放大器**：电压增益 = $+g_m R_C$（同相），输入阻抗极低，没有 Miller 效应 → 高频特性极佳。射频和高速电路的主力。

5. **多级放大**：总增益 = 各级增益之积（dB 则相加）。级联时必须考虑后级输入电阻对前级的负载效应。CC 缓冲器是解决级间匹配的利器。

6. **频率响应**：耦合电容决定 $f_L$（低频截止），寄生电容 + Miller 效应决定 $f_H$（高频截止）。$BW = f_H - f_L$。设计中频增益和带宽此消彼长。

下一章，你将遇到 BJT 的"表兄弟"——**场效应管 FET**（第 32 章）。FET 和 BJT 一样能放大信号，但它用**电压**控制而不是电流控制，输入阻抗可达 $10^{12}\text{ }\Omega$ 以上。这种"电压驱动"的特性让 FET 在数字电路和低功耗模拟电路中大放异彩。你还会发现，FET 的共源（CS）、共漏（CD）、共栅（CG）三种组态和 CE/CC/CB 几乎一一对应。

> **好了，带着这章的知识去"听懂"你手机和音响里的放大器吧！那个把微弱电信号变成震撼音乐的过程，你已经知道它内部的一砖一瓦了。



