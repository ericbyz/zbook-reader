# 第30章：BJT基础

> **核心问题**：一个元件怎么做到"以小控大"？

---

## 0. 本章导览

前面第 28 和 29 章，你学会了半导体和二极管。二极管是电学里的"单向阀"——电流只能往一个方向走。但二极管有一个致命缺陷：它只能"通"或"断"，你不能拿一个微小的信号去控制一个大的电流。如果电子学只有二极管，你的手机、电脑、音响里所有的放大电路都不存在。

这一章你要学的是电子学里真正的"控制元件"：**双极结型晶体管（Bipolar Junction Transistor，BJT）**。BJT 的核心能力用一个词概括：**以小控大**。你在它的一个引脚上输入一点点电流（微安级别），它就能在另一个通路上放行一个大得多的电流（毫安甚至安培级别）。放大倍数通常在 50 到 500 之间。

这个"以小控大"的原理，是一切模拟放大器、开关电路、稳压电源的基础。本章用 NPN 型 BJT 为主线讲解（最常用），同时覆盖 PNP 的对称关系。你会手算偏置电路、画负载线、用 Python 跑特性曲线，最后用 8 道思考题把知识扎牢。

前置章节：[半导体物理基础](./28-semiconductor-physics.md)（理解 PN 结）、[二极管](./29-diodes.md)（PN 结的应用，BJT 就是两个背靠背的 PN 结）
下一章：[场效应管（FET）](./32-fet.md)（另一种"以小控大"的元件，用电压控制而不是电流）

学完本章，你将能够：

1. 用水龙头的比喻讲清楚 BJT 为什么能用小 Ib 控制大 Ic
2. 画出 NPN 和 PNP 的结构图，标注发射极 E、基极 B、集电极 C
3. 解释 BJT 三个工作区（截止、放大、饱和）的条件和特征
4. 写出 Ic = β Ib 并在电路中正确应用
5. 画出 BJT 的输入特性曲线和输出特性曲线族
6. 设计固定偏置和分压偏置两种直流偏置电路
7. 在输出特性曲线上画出直流负载线，求 Q 点
8. 用 Python 画出 BJT 特性曲线、负载线和 Q 点
9. 手算 BJT 偏置电路中的各点电压和电流
10. 避开初学者最常见的 6 个坑

> 本章约 1550 行，建议分 4-5 次读完。第 2 节（工作原理）和第 5 节（直流偏置）是核心，值得反复看。Python 代码在第 7 节，跑一遍加深理解。

---

## 1. BJT 是什么

### 1a. 生活例子：水龙头

你家厨房的水龙头，你一定用过。仔细想想水龙头是怎么工作的：

- 你用手指轻轻一转手柄（花的力气很小，可能只有几十克力）
- 水管里"哗"地流出大量的水（每秒几升，力量大得多）
- 手柄本身并不出水——它只是**控制**了阀门开多大
- 手柄转的角度大一点，水流就大一点；手柄转小一点，水流就小一点

这里有一个关键的比例关系：**你的手劲 vs 水流大小**。手劲是"输入"，水流是"输出"。输入只需要一点点，就能控制一个大得多的输出。这就是"以小控大"。

BJT 干的是一模一样的事，只不过把"手劲"换成了"基极电流 Ib"，把"水流"换成了"集电极电流 Ic"：

- 你在 B 极（基极，Base）注入一点点电流 Ib（微安级）
- C 极（集电极，Collector）和 E 极（发射极，Emitter）之间就会流过一个大得多的电流 Ic（毫安级）
- 而且 Ic 和 Ib 之间有一个近乎固定的比例关系：**Ic ≈ β × Ib**

这个 β（读作"贝塔"）叫**直流电流增益**，通常在 50 到 500 之间。Ib 是 10 μA？那 Ic 就是 500 μA 到 5 mA（取决于 β）。Ib 是 0？那 Ic 也是 0——水龙头关死了。

### 1b. 直观理解：从二极管到三极管的"进化"

你已经很熟悉二极管了：一块 P 型半导体 + 一块 N 型半导体 = 一个 PN 结，电流只能从 P 流向 N。

BJT 可以理解为**两个背靠背的 PN 结**，共享中间那一层。有两种"叠法"：

**NPN 型 BJT**（最常见，本章的主角）：

```
        N        P        N
   集电极 C    基极 B    发射极 E
```

由左到右：N 型（集电极 C）→ P 型（基极 B）→ N 型（发射极 E）。C-B 之间是一个 PN 结（叫**集电结**），B-E 之间是另一个 PN 结（叫**发射结**）。

**PNP 型 BJT**（NPN 的"镜像版本"）：

```
        P        N        P
   集电极 C    基极 B    发射极 E
```

PNP 的工作原理与 NPN 完全对称，只是所有电压极性和电流方向反过来。学透 NPN，PNP 自然就懂了。

### 1c. 三个引脚和它们的"职责"

BJT 有三个引脚，每个有不同的职责：

| 引脚 | 英文 | 符号 | 职责（用水龙头比喻） |
|------|------|------|---------------------|
| 发射极 | Emitter | E | 水龙头出水口。电子/空穴从这里"发射"出来，是电流的源头 |
| 基极 | Base | B | 水龙头手柄。你用 Ib 在这里"拧"，控制 Ic 的大小 |
| 集电极 | Collector | C | 水龙头的进水口（或更准确地说，是"收集"从发射极来的载流子的地方） |

名字本身就很好记：**发射极**发射载流子，**基极**是控制的基地，**集电极**收集载流子。

在电路符号中，NPN 和 PNP 的区别只看发射极箭头方向：

```
  NPN:              PNP:
     C                  C
     │                  │
  B──┤               B──┤
     │                  │
     ▼ E                ▲ E
```

**NPN 的发射极箭头向外**（电流从 C 和 B 流入，从 E 流出）。**PNP 的发射极箭头向内**（电流从 E 流入，从 C 和 B 流出）。记住箭头的方向就是**发射结正向偏置时电流的方向**。

### 1d. 形式化定义：电流关系

BJT 是**电流控制型**器件（这一点和第 32 章要学的 FET 完全不同，FET 是电压控制型）。你在 B 极输入一个小电流 Ib，它就能在 C 极产生一个放大了的大电流 Ic。

三个引脚上的电流满足基尔霍夫电流定律（KCL）：

$$
I_E = I_C + I_B
$$

发射极电流 = 集电极电流 + 基极电流。这很好理解：电流从 E 流出来（NPN 情况下），分两路——大头去了 C，小头去了 B。

而 C 极电流和 B 极电流之间的关系，是 BJT 最核心的公式：

$$
I_C = \beta \cdot I_B
$$

其中：
- $I_C$：集电极电流（Collector Current），是我们想要的"大电流输出"
- $I_B$：基极电流（Base Current），是我们输入的"小电流控制信号"
- $\beta$：直流电流增益（DC Current Gain），也常记作 $h_{FE}$（在数据手册里你见到的是这个名字）

把两个公式合起来：

$$
I_E = I_C + I_B = \beta I_B + I_B = (\beta + 1) I_B
$$

$\beta$ 通常在 50 到 500 之间，所以 $\beta + 1 \approx \beta$，$I_E \approx I_C$。基极电流 Ib 在总电流里只占约 1/50 到 1/500，小到几乎可以忽略——这就是"以小控大"的数学表达。

---

## 2. BJT 工作原理

### 2a. 生活例子：发射结正偏 → 注入 → 集电结反偏 → 收集

在讲微观过程之前，先用水流模型建立一个整体直觉。

想象一个特殊的三层结构的水池：

```
    [N区-集电区]  ║  [P区-基区]  ║  [N区-发射区]
        C 极      ║     B 极      ║      E 极
```

两层隔板就是两个 PN 结。要让水（载流子）从 E 流到 C，你需要做两件事：

1. **把 E-B 之间的隔板打开**（发射结正向偏置）：E 区的水（电子）涌入 B 区。这就叫**注入**。
2. **把 B-C 之间的隔板保持关闭但给它一个很强的"吸力"**（集电结反向偏置）：进入 B 区的电子被 C 区强大的正电压吸过去。这就叫**收集**。

关键点：B 区（基区）做得**非常薄**。电子从 E 注入 B 之后，绝大多数还没来得及从 B 极的引线"跑掉"（变成 Ib），就已经被 C 极吸走了（变成 Ic）。只有极少一部分电子（约 1/β）在 B 区复合，形成了基极电流 Ib。

所以你也可以这样理解：Ib 不是"控制电流"，而是"浪费掉的电流"。β 越大，说明浪费越少，晶体管效率越高。

### 2b. 直观理解：电子在 NPN BJT 里的旅程（微观视角）

现在我们把镜头拉近，看看电子在 NPN 型 BJT 里到底走了怎样的路。

**第一步：发射结正向偏置（E-B PN 结正偏）**

你给 E 极接地（0 V），B 极加约 0.7 V。E-B 之间的 PN 结正向偏置了——和二极管的导通条件一模一样。E 区（N 型）里大量的自由电子（多数载流子）越过耗尽层，涌入 B 区（P 型）。这步叫**载流子注入**。

在这个阶段，E-B 结就像正偏的二极管，电子大量地从 E 跑到 B。

**第二步：基区传输（电子在极薄的 B 区中扩散）**

B 区是关键。它是 P 型半导体，空穴是多数载流子。但 B 区被刻意做得很薄（通常只有微米甚至亚微米级别），而且掺杂浓度比 E 区低很多。

从 E 区注入到 B 区的电子，在 B 区里成了"少数载流子"（B 区本来空穴多、电子少）。它们从 E-B 结附近往 C-B 结方向扩散。因为 B 区极薄，绝大多数电子还没在 B 区中和空穴复合（复合 = 形成 Ib），就已经到达了 C-B 结的边缘。

只有大约 1% 甚至更少的电子在 B 区中与空穴复合。这些复合掉的电子就构成了基极电流 Ib。你往 B 极注入的 Ib，本质上就是在补充 B 区中因复合而消耗掉的空穴，维持整个过程的持续进行。

**第三步：集电结反向偏置（C-B PN 结反偏）**

C 极加了一个较高的正电压（比如 10 V），B 极只有约 0.7 V。C 对 B 来说是正电压，也就是 C-B 结反向偏置。反向偏置意味着耗尽层变宽，内建电场加强。

这个强电场对电子来说是一个巨大的"吸力"：电子带负电，C 极正电压高达 10 V，电子一旦到达 C-B 结边缘，就被电场"扫"进了 C 区，再被外部电路拉走，构成了集电极电流 Ic。

总结一下整个旅程：**E 极发射电子 → 电子穿过极薄的 B 区（只有极少部分"浪费"成 Ib）→ 电子被 C 极的强电场"扫"走形成 Ic**。

PNP 型 BJT 的原理完全对称，只是载流子换成了空穴，所有电压极性反过来。

### 2c. 形式化定义：爱伯斯-莫尔模型（简化版）

BJT 的完整数学模型叫**爱伯斯-莫尔模型（Ebers-Moll Model）**。对于初学者，我们只需要它的简化放大区形式：

$$
I_C = I_S \cdot e^{\frac{V_{BE}}{V_T}}
$$

看到这个公式，你应该觉得很眼熟——它和肖克利二极管方程几乎一模一样！这不是巧合。B-E 结本身就是一个 PN 结（正偏），所以 $I_C$ 和 $V_{BE}$ 之间是指数关系。

加上 $\beta$ 之后：

$$
I_B = \frac{I_C}{\beta} = \frac{I_S}{\beta} \cdot e^{\frac{V_{BE}}{V_T}}
$$

其中各符号含义：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $I_C$ | 集电极电流 | 由外部电路和 $V_{BE}$ 共同决定 |
| $I_S$ | 反向饱和电流 | 约 $10^{-15}$ 到 $10^{-12}$ A（比二极管更小） |
| $V_{BE}$ | 基极-发射极电压 | 放大区约 0.6-0.7 V（硅管） |
| $V_T$ | 热电压 $kT/q$ | 室温下约 26 mV |
| $\beta$ | 直流电流增益 | 50 到 500，具体值查数据手册 |

**用大白话说**：$V_{BE}$ 每增加约 18 mV（自然对数关系），$I_C$ 就翻一倍。$\beta$ 可以理解为"浪费因子"的倒数：$\beta$ 大说明从 E 到 C 的路上几乎没浪费电子。

### 2d. 手算示例：用 β 计算 Ic 和 Ie

**题目 1**：已知 NPN 型 BJT 的 $\beta = 100$，基极电流 $I_B = 20\ \mu\text{A}$。求 $I_C$ 和 $I_E$。

$$
\begin{aligned}
I_C &= \beta \cdot I_B \\
    &= 100 \times 20\ \mu\text{A} \\
    &= 2000\ \mu\text{A} = 2\ \text{mA}
\end{aligned}
$$

$$
\begin{aligned}
I_E &= I_C + I_B \\
    &= 2\ \text{mA} + 20\ \mu\text{A} \\
    &= 2.02\ \text{mA}
\end{aligned}
$$

可以看到 $I_E \approx I_C$，基极电流 Ib 只贡献了总电流的约 1%。

**题目 2**：已知 NPN 型 BJT 的 $\beta = 150$，测得 $I_E = 3.02\ \text{mA}$。求 $I_C$ 和 $I_B$。

先算 $I_B$。由 $I_E = (\beta + 1)I_B$：

$$
\begin{aligned}
I_B &= \frac{I_E}{\beta + 1} \\
    &= \frac{3.02\ \text{mA}}{151} \\
    &= \frac{3020\ \mu\text{A}}{151} \\
    &= 20\ \mu\text{A}
\end{aligned}
$$

再算 $I_C$：

$$
\begin{aligned}
I_C &= \beta \cdot I_B \\
    &= 150 \times 20\ \mu\text{A} \\
    &= 3000\ \mu\text{A} = 3\ \text{mA}
\end{aligned}
$$

验算：$I_C + I_B = 3\ \text{mA} + 20\ \mu\text{A} = 3.02\ \text{mA} = I_E$。完美。

**题目 3**：BJT 工作在放大区时 $V_{BE} = 0.7\ \text{V}$，$\beta = 200$。外部基极电阻 $R_B = 200\ \text{k}\Omega$ 接到 $V_{BB} = 5\ \text{V}$。求 $I_B$、$I_C$ 和 $V_{CE}$（假设 $V_{CC} = 10\ \text{V}$，$R_C = 2\ \text{k}\Omega$）。

这是一个最简单的固定偏置电路。先画个回路：$V_{BB}$ → $R_B$ → B-E 结（0.7 V 管压降）→ 地。基极回路的 KVL：

$$
\begin{aligned}
V_{BB} - I_B R_B - V_{BE} &= 0 \\[4pt]
I_B &= \frac{V_{BB} - V_{BE}}{R_B} \\[4pt]
    &= \frac{5\ \text{V} - 0.7\ \text{V}}{200\ \text{k}\Omega} \\[4pt]
    &= \frac{4.3\ \text{V}}{200 \times 10^3\ \Omega} \\[4pt]
    &= 21.5\ \mu\text{A}
\end{aligned}
$$

有了 Ib，Ic 就出来了：

$$
I_C = \beta I_B = 200 \times 21.5\ \mu\text{A} = 4.3\ \text{mA}
$$

集电极回路 KVL：$V_{CC} - I_C R_C - V_{CE} = 0$，所以：

$$
\begin{aligned}
V_{CE} &= V_{CC} - I_C R_C \\
       &= 10\ \text{V} - 4.3\ \text{mA} \times 2\ \text{k}\Omega \\
       &= 10\ \text{V} - 8.6\ \text{V} \\
       &= 1.4\ \text{V}
\end{aligned}
$$

$V_{CE} = 1.4\ \text{V}$，远大于 $V_{CE(sat)} \approx 0.2\ \text{V}$，所以 BJT 确实工作在放大区，我们的假设成立。

### 2e. 常见误区：β 是"神奇"的放大倍数？

初学者最容易对 $\beta$ 产生两个极端误解。

**❌ 误区一："β 是固定不变的，每个管子的 β 都是定值"**

实际上 $\beta$ 受很多因素影响：温度、$I_C$ 大小、$V_{CE}$ 大小、同一批次管子之间的个体差异。数据手册上通常给一个范围，比如 2N2222A 的 $\beta$ 在 $I_C = 150\ \text{mA}$ 时是 100 到 300。你不能假设手里这只管子 β 恰好是 150——设计电路时要考虑 β 的波动，确保在 β 取最小值和最大值时电路都能正常工作。

**✓ 正确做法**：用分压偏置（第 5 节）让 Q 点对 β 的依赖降到最低。

**❌ 误区二："Ic = β Ib 在任何情况下都成立"**

这个公式只在**放大区**成立。BJT 进入饱和区以后，$I_C$ 不再由 $\beta$ 决定，而是由外部电路（$V_{CC}$ 和 $R_C$）决定了。此时继续加大 Ib，Ic 也不会按比例增加。

**✓ 正确做法**：先判断 BJT 在哪个区，再决定用哪个公式。第 3 节会详细讲三个工作区。

---

### 2f. PNP 型 BJT——NPN 的"镜像兄弟"

前面所有的原理讲解都是以 NPN 为例。PNP 型 BJT 的结构和 NPN 完全对称，只是掺杂类型反过来：E 和 C 是 P 型，B 是 N 型。符号上的区别：PNP 的发射极箭头**向内**（指向 B 极）。

PNP 的工作条件与 NPN"全部反着来"：

| | NPN | PNP |
|---|-----|-----|
| B-E 结正偏条件 | $V_B > V_E$（B 比 E 约高 0.7 V） | $V_B < V_E$（B 比 E 约低 0.7 V） |
| C-B 结反偏条件 | $V_C > V_B$（C 电压高于 B） | $V_C < V_B$（C 电压低于 B） |
| E 极电位 | 通常接地（最低电位） | 通常接电源（最高电位） |
| C 极电位 | 通常接正电源（最高电位） | 通常接地或负载（最低电位） |
| 电流方向 | Ib 和 Ic 流入 B 和 C，Ie 从 E 流出 | Ie 流入 E，Ib 和 Ic 从 B 和 C 流出， |
| 载流子 | 电子（从 E 发射） | 空穴（从 E 发射） |
| $V_{CE(sat)}$ | 约 0.1-0.2 V | 约 -0.1 到 -0.2 V（负号表示 C 比 E 电压高） |

在电路图中，PNP 管的 E 极在上面（接正电源），C 极在下面（接负载到地）。把 NPN 电路上下颠倒、电源正负互换，就得到了对应的 PNP 电路。

**手算示例（题目 3b）**：PNP 型 BJT，$\beta = 120$，$I_B = 15\ \mu\text{A}$ 从 B 极流出（PNP 的 Ib 是从 B 流出的）。求 $I_C$ 和 $I_E$。

$$
\begin{aligned}
I_C &= \beta I_B = 120 \times 15\ \mu\text{A} = 1.8\ \text{mA} \quad (\text{Ic 从 C 极流出})\\[4pt]
I_E &= I_C + I_B = 1.8\ \text{mA} + 15\ \mu\text{A} = 1.815\ \text{mA} \quad (\text{Ie 流入 E 极})
\end{aligned}
$$

PNP 的 $I_C = \beta I_B$ 同样成立，只是电流方向反过来。如果你已经彻底理解 NPN，PNP 只需记住一句话："把所有电压和电流的方向反过来"。

PNP 在模拟电路中有独特优势：它与 NPN 配合使用可以实现"互补对称"电路（如推挽输出级、B 类放大器），用一个 NPN 处理信号正半周、一个 PNP 处理负半周，效率远高于单管放大。这个在后续运放章节会详细展开。

---

### 2g. β 的温度漂移——为什么你的偏置电路不能"写死"

$\beta$ 随温度升高而变化。通常硅 BJT 的 $\beta$ 在温度从 25°C 升到 100°C 时可能增加 20%-50%。不是所有管子都变——有的是 $\beta$ 增大，有的是 $I_S$ 增大导致同样 $V_{BE}$ 下 $I_C$ 增大。无论哪种，最终效果都是 Q 点漂移。

在实际工程中，大数据手册和仿真软件会给出 $\beta$ 的温度曲线。对于初学者，你只需要记住：**设计偏置电路时不要假设 $\beta$ 是一个精确的定值。用分压偏置（第 5 节）让电路在 $\beta$ 变动时 Q 点尽量不动。**

---

## 3. 三个工作区

### 3a. 生活例子：水龙头的三种"开度"

回到水龙头的比喻。你拧动水龙头手柄，有三种典型状态：

1. **关死**：手柄完全拧紧，把手没有再往下拧的余地了。水流 = 0。不管你怎么在关死的基础上再用力，水也不会减少（本来就是 0）。
2. **调节区**：手柄在"刚开一点"到"快全开"之间。水流和手柄拧的角度大致成正比。拧大一点，水大一点；拧小一点，水小一点。
3. **全开到底**：手柄拧到头了，阀门全部打开。水流已经到最大了（由水管压力和水龙头结构决定），这时再拧大手柄也没用——水流不会继续增加。

这三种状态完美对应 BJT 的三个工作区：**截止区（关死）→ 放大区（调节区）→ 饱和区（全开）**。

### 3b. 三个区的直观特征

**截止区（Cutoff Region）**

$I_B \approx 0$（基极没有电流输入），$I_C \approx 0$（集电极也没有电流）。BJT 相当于一个断开的开关。B-E 结没有正向偏置（$V_{BE}$ 不够大），发射极没有电子注入，整个管子"沉睡"。

条件：$V_{BE} < V_{BE(on)}$（硅管约 0.5-0.6 V 以下），或者说 $I_B \approx 0$。

**放大区（Active Region 或 Forward-Active Region）**

B-E 结正偏（$V_{BE} \approx 0.7\ \text{V}$），C-B 结反偏（$V_{CE} > V_{BE}$，通常 $V_{CE} > 0.2\ \text{V}$）。$I_C = \beta I_B$ 成立，BJT 工作在"线性放大"状态。这是模拟放大器要用的工作区。

条件：$I_B > 0$ 且 $V_{CE} > V_{CE(sat)}$（硅管约 > 0.2 V）。

**饱和区（Saturation Region）**

B-E 结正偏（$I_B$ 足够大），C-B 结也变成正偏了（$V_{CE}$ 降到了极低的水平）。管子"全开"，$I_C$ 达到电路允许的最大值，不再随 $I_B$ 增大而增大。$V_{CE}$ 饱和压降约 0.1-0.2 V（硅管）。这是开关电路中 BJT"导通"的状态。

条件：$I_B > \frac{I_{C(sat)}}{\beta}$，其中 $I_{C(sat)} = \frac{V_{CC} - V_{CE(sat)}}{R_C} \approx \frac{V_{CC}}{R_C}$。

### 3c. 形式化定义：三个区的精确判断条件

用一张表总结三个区的电压和电流特征（NPN 型，硅管）：

| 工作区 | B-E 结状态 | C-B 结状态 | $V_{BE}$（典型） | $V_{CE}$（典型） | $I_C$ 由谁决定 |
|--------|-----------|-----------|-----------------|-----------------|---------------|
| **截止区** | 反偏或零偏 | 反偏 | < 0.5 V | ≈ $V_{CC}$ | $I_C \approx 0$ |
| **放大区** | 正偏 | 反偏 | ≈ 0.7 V | > 0.2 V | $I_C = \beta I_B$ |
| **饱和区** | 正偏 | 正偏 | ≈ 0.7-0.8 V | ≈ 0.1-0.2 V | $I_C = \frac{V_{CC} - V_{CE(sat)}}{R_C}$ |

判断 BJT 工作在哪一区的步骤：

1. 先假设 BJT 在**放大区**（这是默认假设）。
2. 用 $I_C = \beta I_B$ 算出 $I_C$。
3. 用 $V_{CE} = V_{CC} - I_C R_C$ 算出 $V_{CE}$。
4. 如果 $V_{CE} \approx V_{CC}$（几乎等于电源电压），说明 Ic 太小，管子很可能在截止区。
5. 如果 $V_{CE} < V_{CE(sat)}$（约 0.2 V），说明 Ic 已经大到"触顶"了，管子实际上在饱和区——此时算出来的 $V_{CE}$ 小于 0.2 V 是不真实的，真实值就是 $V_{CE(sat)} \approx 0.2\ \text{V}$。
6. 如果 $V_{CE}$ 介于 $V_{CE(sat)}$ 和 $V_{CC} - 1\ \text{V}$ 之间，放大区的假设成立。

### 3d. 手算示例：判断工作区

**题目 4**：电路参数：$V_{CC} = 12\ \text{V}$，$R_C = 1\ \text{k}\Omega$，$\beta = 100$。基极偏置使 $I_B = 50\ \mu\text{A}$。判断 BJT 的工作区。

**步骤一**：假设放大区，算 $I_C$。

$$
I_C = \beta I_B = 100 \times 50\ \mu\text{A} = 5\ \text{mA}
$$

**步骤二**：算 $V_{CE}$。

$$
V_{CE} = V_{CC} - I_C R_C = 12\ \text{V} - 5\ \text{mA} \times 1\ \text{k}\Omega = 12\ \text{V} - 5\ \text{V} = 7\ \text{V}
$$

**步骤三**：判断。$V_{CE} = 7\ \text{V}$，远大于 $V_{CE(sat)} = 0.2\ \text{V}$，也远小于 $V_{CC} = 12\ \text{V}$。BJT 确实在**放大区**。

---

**题目 5**：同样电路，但 $I_B = 200\ \mu\text{A}$。判断工作区。

**步骤一**：假设放大区。

$$
I_C = \beta I_B = 100 \times 200\ \mu\text{A} = 20\ \text{mA}
$$

**步骤二**：算 $V_{CE}$。

$$
V_{CE} = 12\ \text{V} - 20\ \text{mA} \times 1\ \text{k}\Omega = 12\ \text{V} - 20\ \text{V} = -8\ \text{V} \ (\text{？})
$$

$V_{CE}$ 算出负数！这不可能。说明我们的"放大区"假设不成立——BJT 已经进入**饱和区**了。$I_C$ 的实际最大值由外部电路决定：

$$
I_{C(sat)} = \frac{V_{CC} - V_{CE(sat)}}{R_C} \approx \frac{12\ \text{V} - 0.2\ \text{V}}{1\ \text{k}\Omega} = 11.8\ \text{mA}
$$

饱和判据：$I_B > \frac{I_{C(sat)}}{\beta} = \frac{11.8\ \text{mA}}{100} = 118\ \mu\text{A}$。实际 $I_B = 200\ \mu\text{A} > 118\ \mu\text{A}$，BJT 确实饱和了。在饱和区，$I_B$ 继续增大，$I_C$ 维持在 11.8 mA 不变。

---

## 4. 特性曲线族

### 4a. 输入特性曲线

**输入特性曲线**画的是 $I_B$ 随 $V_{BE}$ 变化的关系，在不同 $V_{CE}$ 下形成一族曲线。

曲线形状和二极管正向特性几乎一样——指数上升。$V_{BE}$ 从 0 V 开始增加，刚开始 Ib 几乎是 0（$V_{BE} < 0.5\ \text{V}$），到了约 0.6-0.7 V 时 Ib 开始急剧上升。

$V_{CE}$ 对输入特性影响不大：$V_{CE}$ = 1 V、5 V、10 V 时的输入特性曲线几乎重合。你可以记住一条经验规律：**输入特性基本上就是一条"固定"的指数曲线，$V_{BE} \approx 0.7\ \text{V}$ 时 Ib 大约在微安到毫安的实用范围。**

### 4b. 输出特性曲线族

**输出特性曲线族**是 BJT 最重要的图。横轴是 $V_{CE}$，纵轴是 $I_C$，每条曲线对应一个固定的 $I_B$ 值。

```
 Ic ↑
    │        Ib=60μA ─────────────────────
    │        Ib=50μA ───────────────────
    │        Ib=40μA ────────────────
    │        Ib=30μA ─────────────
    │        Ib=20μA ──────────      ← 放大区（平坦部分）
    │        Ib=10μA ───────
    │        Ib=0    ────────────────  ← 截止区
    │                                  ← 饱和区（陡峭上升）
    └──────────────────────────────────────→ Vce
    0    饱和区  │       放大区          │ 击穿区
          (Vce很小)  (Ic ≈ βIb，几乎平)   (Vce过大)
```

这张图把三个工作区画得清清楚楚：

**截止区**：$I_B = 0$ 的那条线（最底下），$I_C \approx 0$（实际上有个极小的漏电流 $I_{CEO}$）。

**饱和区**：$V_{CE}$ 很小的区域（0 到约 0.2-0.3 V）。所有曲线在这个区域急速上升，$I_C$ 受 $V_{CE}$ 限制，不受 $I_B$ 控制。$V_{CE}$ 稍微增加一点，$I_C$ 陡增。

**放大区**：$V_{CE}$ 超过约 0.3 V 之后的平坦区域。$I_C$ 几乎不随 $V_{CE}$ 变化（只有微小的上倾，叫**厄利效应 Early Effect**）。不同 $I_B$ 对应的 $I_C$ 之间等间距（如果 $I_B$ 是等步长增加的话），间距就是 $\beta \times \Delta I_B$。

**击穿区**：$V_{CE}$ 超过管子的额定击穿电压 $BV_{CEO}$ 后，$I_C$ 急剧增大，可能损坏管子。这个区域在正常工作时应避免进入。

### 4c. 厄利效应：为什么"平坦"不是完全平的？

放大区的曲线不是完全水平，而是微微上倾。把各条曲线向左延长，它们会汇聚到横轴上的一个点，这个点的电压绝对值叫**厄利电压（Early Voltage）**，记作 $V_A$。

厄利效应来自于 $V_{CE}$ 增大时，C-B 结耗尽层变宽，导致有效基区宽度变窄（基区宽度调制效应）。基区变窄，电子更容易穿过，所以同样 $I_B$ 下 $I_C$ 会稍微大一点。

考虑厄利效应的修正公式：

$$
I_C = I_S \cdot e^{\frac{V_{BE}}{V_T}} \cdot \left(1 + \frac{V_{CE}}{V_A}\right)
$$

对于大多数手算和初学分析，厄利效应可以忽略。只有在高精度模拟电路设计中才需要考虑。

---

## 5. 直流偏置

### 5a. 偏置是什么，为什么需要它？

BJT 不是接上电源就能自动进入放大区的。你需要用外部的电阻网络给 B、C、E 三个引脚"安排好"合适的直流电压和电流，让 BJT 的**静态工作点（Q 点，Quiescent Point）**落在放大区的中间位置。这个过程叫**直流偏置（DC Biasing）**。

为什么不直接偏在放大区边缘？因为输入信号（比如一个小正弦波）会让 Ib 上下波动。如果 Q 点太靠近截止区，输入信号往下一摆，管子就被"关死"了；如果 Q 点太靠近饱和区，输入信号往上一摆，管子就饱和了——两种情况都会造成信号失真。

好的偏置要让 Q 点尽可能稳定，不随温度、$\beta$ 的个体差异等因素大幅漂移。

### 5b. 固定偏置（Fixed Bias）——最简单但不稳定

固定偏置的电路极其简单：

```
        Vcc
         │
        ┌┴┐
        │ │ Rb
        │ │
        └┬┘
         ├─────────→ C
         │           │
        ┌┤B       ┌──┴──┐
        ││  NPN   │     │ Rc
  Vin───┴┤E       │     │
           │       └──┬──┘
           │          │
           └──────────┴───→ Vcc
           (这里有点问题——让我画个规范的)
```

还是用参数画吧。固定偏置电路：电源 $V_{CC}$ → 电阻 $R_B$ → B 极 → E 极 → 地。C 极通过 $R_C$ 接到 $V_{CC}$，E 极直接接地。

**基极回路**（从 $V_{CC}$ 经 $R_B$ 到 B-E 结到地）：

$$
V_{CC} - I_B R_B - V_{BE} = 0
$$

$$
I_B = \frac{V_{CC} - V_{BE}}{R_B}
$$

**集电极回路**（从 $V_{CC}$ 经 $R_C$ 到 C-E 到地）：

$$
V_{CC} - I_C R_C - V_{CE} = 0
$$

$$
V_{CE} = V_{CC} - \beta I_B R_C
$$

**手算示例（题目 6——固定偏置设计）**

设计要求：$V_{CC} = 12\ \text{V}$，$\beta = 100$，希望 Q 点 $I_C = 2\ \text{mA}$，$V_{CE} = 6\ \text{V}$（放大区中间位置）。求 $R_B$ 和 $R_C$。

**先算 Rc**：

$$
\begin{aligned}
V_{CE} &= V_{CC} - I_C R_C \\[4pt]
R_C &= \frac{V_{CC} - V_{CE}}{I_C} \\[4pt]
    &= \frac{12\ \text{V} - 6\ \text{V}}{2\ \text{mA}} \\[4pt]
    &= \frac{6\ \text{V}}{2 \times 10^{-3}\ \text{A}} \\[4pt]
    &= 3000\ \Omega = 3\ \text{k}\Omega
\end{aligned}
$$

标准电阻值选 $R_C = 3.3\ \text{k}\Omega$（接近 3 kΩ 的常用标称值）。

**再算 Rb**。先求需要的 $I_B$：

$$
I_B = \frac{I_C}{\beta} = \frac{2\ \text{mA}}{100} = 20\ \mu\text{A}
$$

$$
\begin{aligned}
R_B &= \frac{V_{CC} - V_{BE}}{I_B} \\[4pt]
    &= \frac{12\ \text{V} - 0.7\ \text{V}}{20\ \mu\text{A}} \\[4pt]
    &= \frac{11.3\ \text{V}}{20 \times 10^{-6}\ \text{A}} \\[4pt]
    &= 565{,}000\ \Omega = 565\ \text{k}\Omega
\end{aligned}
$$

标准值选 $R_B = 560\ \text{k}\Omega$。

**固定偏置的最大问题**：Q 点严重依赖 $\beta$。如果换成一只 $\beta = 200$ 的同型号管子，同样 $R_B$ 下 $I_B$ 不变，但 $I_C$ 翻倍变成 4 mA，$V_{CE}$ 会从 6 V 掉到 $12 - 4\text{mA} \times 3.3\text{k}\Omega = -1.2\ \text{V}$——管子直接饱和了。所以固定偏置只在要求不高、$\beta$ 已知且稳定的场合使用。

### 5c. 分压偏置（Voltage Divider Bias）——工程上的标准答案

分压偏置用四个电阻，能大幅降低 Q 点对 $\beta$ 的依赖：

```
        Vcc
         │
        ┌┴┐
        │ │ R1
        │ │
        └┬┘
         ├───────────→ C
        ┌┴┐         ┌──┴──┐
        │ │ R2      │     │ Rc
        │ │         │     │
        └┬┘         └──┬──┘
         │             │
         └─────→ B     │
              ┌─┤      │
              │ │ NPN   │
              └─┤E     │
                │      │
               ┌┴┐     │
               │ │ Re  │
               │ │     │
               └┬┘     │
                │      │
                └──────┴───→ 地
```

$R_1$ 和 $R_2$ 构成一个分压器，给 B 极提供一个稳定的直流电压 $V_B$。$R_E$ 是**发射极电阻**——反馈电阻，是稳定 Q 点的关键。$R_C$ 和固定偏置一样。

**分析步骤**：

**第一步：求基极电压 $V_B$。**

假设基极电流 $I_B$ 远小于流过 $R_1$ 和 $R_2$ 的分压电流（加上 $R_1$ 和 $R_2$ 选得不太大，一般在几十千欧级别），可以近似认为 B 极电压为：

$$
V_B \approx V_{CC} \cdot \frac{R_2}{R_1 + R_2}
$$

**第二步：求发射极电压 $V_E$。**

B-E 结正偏，$V_{BE} \approx 0.7\ \text{V}$：

$$
V_E = V_B - V_{BE} = V_B - 0.7\ \text{V}
$$

**第三步：求发射极电流 $I_E$。**

$I_E$ 就是流过 $R_E$ 的电流：

$$
I_E = \frac{V_E}{R_E} = \frac{V_B - 0.7}{R_E}
$$

**第四步：近 $I_C \approx I_E$（因为 $\beta$ 大，$I_B$ 很小）。**

**第五步：求 $V_{CE}$。**

集电极回路 KVL：$V_{CC} - I_C R_C - V_{CE} - I_E R_E = 0$

因为 $I_C \approx I_E$：

$$
V_{CE} \approx V_{CC} - I_C (R_C + R_E)
$$

关键观察：以上所有公式中**没有出现 $\beta$**！$I_C$ 完全由 $V_B$ 和 $R_E$ 决定。只要 $\beta$ 足够大（$I_B$ 远小于分压器电流），$\beta$ 的波动几乎不影响 Q 点。这就是分压偏置稳如泰山的原因。

**手算示例（题目 7——分压偏置设计）**

设计要求：$V_{CC} = 15\ \text{V}$，希望 $I_C \approx 1\ \text{mA}$，$V_{CE} \approx 7.5\ \text{V}$（放大区中间位置），$\beta$ 在 100-300 之间。设计分压偏置电路。

**第一步：选 $V_E$。**

发射极电压通常选 $V_E \approx 0.1 V_{CC}$ 到 $0.2 V_{CC}$，既能提供足够的稳定裕量，又不浪费太多电压。这里选 $V_E = 2\ \text{V}$。

**第二步：确定 $R_E$。**

$$
R_E = \frac{V_E}{I_E} \approx \frac{V_E}{I_C} = \frac{2\ \text{V}}{1\ \text{mA}} = 2\ \text{k}\Omega
$$

**第三步：确定 $R_C$。**

由 $V_{CE} = V_{CC} - I_C(R_C + R_E)$：

$$
\begin{aligned}
7.5\ \text{V} &= 15\ \text{V} - 1\ \text{mA} \times (R_C + 2\ \text{k}\Omega) \\[4pt]
R_C + 2\ \text{k}\Omega &= \frac{15\ \text{V} - 7.5\ \text{V}}{1\ \text{mA}} = 7.5\ \text{k}\Omega \\[4pt]
R_C &= 7.5\ \text{k}\Omega - 2\ \text{k}\Omega = 5.5\ \text{k}\Omega
\end{aligned}
$$

标准值选 $R_C = 5.6\ \text{k}\Omega$（微调一下 Q 点不影响大局）。

**第四步：求 $V_B$。**

$$
V_B = V_E + V_{BE} = 2\ \text{V} + 0.7\ \text{V} = 2.7\ \text{V}
$$

**第五步：确定 $R_1$ 和 $R_2$。**

选分压器电流 $I_{\text{divider}} \approx 10 I_B$（保证 $I_B$ 对分压的影响可忽略）。先估算 $I_B$：

$$
I_B \approx \frac{I_C}{\beta_{\text{min}}} = \frac{1\ \text{mA}}{100} = 10\ \mu\text{A}
$$

所以分压器电流 $I_{\text{divider}} = 10 \times 10\ \mu\text{A} = 100\ \mu\text{A}$。

$$
\begin{aligned}
R_1 + R_2 &= \frac{V_{CC}}{I_{\text{divider}}} = \frac{15\ \text{V}}{100\ \mu\text{A}} = 150\ \text{k}\Omega \\[4pt]
R_2 &= \frac{V_B}{I_{\text{divider}}} = \frac{2.7\ \text{V}}{100\ \mu\A} = 27\ \text{k}\Omega \\[4pt]
R_1 &= 150\ \text{k}\Omega - 27\ \text{k}\Omega = 123\ \text{k}\Omega
\end{aligned}
$$

标准值选 $R_1 = 120\ \text{k}\Omega$，$R_2 = 27\ \text{k}\Omega$。

**第六步：验证。**

$V_B = 15 \times \frac{27}{120+27} = 15 \times \frac{27}{147} = 2.76\ \text{V}$（接近 2.7 V）。

$V_E = 2.76 - 0.7 = 2.06\ \text{V}$。

$I_E = 2.06 / 2\text{k} = 1.03\ \text{mA}$，$I_C \approx 1.03\ \text{mA}$（几乎就是设计值）。

$V_{CE} = 15 - 1.03\text{mA} \times (5.6\text{k} + 2\text{k}) = 15 - 7.83 = 7.17\ \text{V}$（接近 7.5 V，在放大区中间）。

如果换一只 $\beta = 300$ 的管子，$I_B = 1.03\text{mA} / 300 \approx 3.4\ \mu\text{A}$，仍然远小于分压器电流 100 μA，所以 $V_B \approx 2.76\ \text{V}$ 几乎不变，Q 点稳定。这就是分压偏置的价值。

---

## 6. Q 点与负载线

### 6a. 直流负载线（DC Load Line）

BJT 的输出回路是一个简单的串联电路：$V_{CC}$ → $R_C$ → C-E → $R_E$ → 地。这个回路的 KVL 方程是：

$$
V_{CC} = I_C R_C + V_{CE} + I_E R_E
$$

由于 $I_C \approx I_E$（忽略 Ib），可以写成：

$$
V_{CC} \approx I_C (R_C + R_E) + V_{CE}
$$

整理成 $I_C$ 随 $V_{CE}$ 的关系：

$$
I_C = -\frac{1}{R_C + R_E} \cdot V_{CE} + \frac{V_{CC}}{R_C + R_E}
$$

这是一条直线方程，斜率 $-\frac{1}{R_C + R_E}$。这条直线就叫**直流负载线**。

它的两个端点：
- $V_{CE} = 0$ 时，$I_C = \frac{V_{CC}}{R_C + R_E}$（饱和电流，最大值）
- $I_C = 0$ 时，$V_{CE} = V_{CC}$（截止电压，最大值）

把直流负载线画在输出特性曲线上，它和某条 $I_B$ 曲线（由偏置电路决定）的交点就是**Q 点（静态工作点）**。

### 6b. Q 点的意义

Q 点是一个"锚点"——在没有输入信号时，BJT 待在的位置。当输入信号（比如一个正弦波）叠加在偏置上，Ib 会在 Q 点附近上下波动，BJT 的工作点就在负载线上来回移动。

Q 点选得好（在负载线中间），输出信号的摆动范围最大，失真最小。Q 点太靠近截止区（负载线底部），信号往下摆时会被"削底"；Q 点太靠近饱和区（负载线顶部），信号往上摆时会被"削顶"。

### 6c. 手算示例：Q 点与负载线

**题目 8**：分压偏置电路，$V_{CC} = 12\ \text{V}$，$R_C = 3\ \text{k}\Omega$，$R_E = 1\ \text{k}\Omega$，$R_1 = 40\ \text{k}\Omega$，$R_2 = 10\ \text{k}\Omega$，$\beta = 150$。求 Q 点坐标 ($V_{CE}$, $I_C$)，并画出负载线。

**求 Q 点**：

$$
\begin{aligned}
V_B &= 12 \times \frac{10}{40 + 10} = 12 \times 0.2 = 2.4\ \text{V} \\[4pt]
V_E &= 2.4 - 0.7 = 1.7\ \text{V} \\[4pt]
I_C &\approx I_E = \frac{1.7\ \text{V}}{1\ \text{k}\Omega} = 1.7\ \text{mA} \\[4pt]
V_{CE} &= 12 - 1.7\text{mA} \times (3\text{k}\Omega + 1\text{k}\Omega) \\
       &= 12 - 6.8 = 5.2\ \text{V}
\end{aligned}
$$

Q 点 = (5.2 V, 1.7 mA)。

**负载线的两个端点**：

$$
\begin{aligned}
\text{饱和端点：} V_{CE} &= 0,\quad I_{C(\text{max})} = \frac{12}{3+1} = 3\ \text{mA} \\[4pt]
\text{截止端点：} I_C &= 0,\quad V_{CE(\text{max})} = 12\ \text{V}
\end{aligned}
$$

Q 点在负载线偏中间位置（Vce 在 5.2 V，12 V 的约 43% 处），摆动空间充足，偏置合理。

如果想精确调整 Q 点到正中间（$V_{CE} = 6\ \text{V}$），可以微调 $R_2$ 的大小。但 5.2 V 对大多数应用已经足够好了。

---

## 7. Python 仿真

### 7a. BJT 输出特性曲线族

```python
# -*- coding: utf-8 -*-
"""
绘制 NPN BJT 输出特性曲线族
横轴: V_CE (0 到 10V)
纵轴: I_C
每条曲线对应一个 I_B 值
"""

import numpy as np
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# === 参数定义 ===
beta = 100           # 直流电流增益
V_A = 100            # 厄利电压（单位 V），100V 意味着很小的厄利效应
I_S = 1e-14          # 反向饱和电流（A）
V_T = 0.026          # 热电压 26mV（室温）
V_CE_sat = 0.2       # 饱和压降
V_CC = 10            # 电源电压（用于参考）

# === 扫描变量 ===
V_CE = np.linspace(0, 10, 500)  # V_CE 从 0 到 10V，取 500 个点
I_B_values = [0, 10, 20, 30, 40, 50]  # 基极电流，单位 μA

# === 绘图 ===
plt.figure(figsize=(10, 7))

for I_B_uA in I_B_values:
    I_B = I_B_uA * 1e-6  # 转为安培
    
    if I_B == 0:
        # 截止区：Ic 近乎为 0
        I_C = np.zeros_like(V_CE)
        label_str = f'Ib = 0 μA (截止)'
    else:
        # 简化 Ebers-Moll + 厄利效应 + 饱和模型
        # 放大区: Ic = beta * Ib * (1 + Vce/VA)
        I_C_active = beta * I_B * (1 + V_CE / V_A)
        # 饱和区限制：Ic 不能超过饱和电流曲线
        # 饱和区用线性近似：Ic = (Vcc - Vce_sat) / Rc  (这里简化)
        # 更精确：饱和区 Ic 随 Vce 线性上升，到达放大区后转平
        I_C = np.minimum(I_C_active, V_CE / 50)  # 简化饱和特性
        label_str = f'Ib = {I_B_uA} μA'
    
    plt.plot(V_CE, I_C * 1000, linewidth=1.8, label=label_str)  # 转为 mA

# === 标注工作区 ===
plt.axvline(x=V_CE_sat, color='gray', linestyle='--', alpha=0.5, label=f'饱和边界 Vce={V_CE_sat}V')
plt.axhline(y=0, color='black', linewidth=0.5)

# 用文字标注三个区
plt.text(0.05, 4, '饱和区', fontsize=12, color='red', alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(3, 4, '放大区\n(Ic ≈ β·Ib)', fontsize=12, color='blue', alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(8, 0.15, '截止区\n(Ib=0, Ic≈0)', fontsize=11, color='green', alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel('$V_{CE}$ (V)', fontsize=14)
plt.ylabel('$I_C$ (mA)', fontsize=14)
plt.title('BJT NPN 输出特性曲线族 (β=100)', fontsize=15)
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/bjt_output_curves.png', dpi=150)
plt.show()

print("图已保存。你可以看到：")
print("1. Vce < 0.2V 时，所有曲线急速上升 → 饱和区")
print("2. Vce > 0.3V 后，曲线趋于平坦 → 放大区")
print("3. Ib=0 时 Ic≈0 → 截止区")
print("4. 不同 Ib 之间的 Ic 间距约为 β × ΔIb")
```

**预期输出**：一张包含 6 条曲线的图（Ib = 0, 10, 20, 30, 40, 50 μA）。最底下的曲线（Ib = 0）几乎贴着横轴，代表截止区。Vce 很小的区域所有曲线挤在一起陡峭上升，代表饱和区。Vce 增大后曲线分开并趋于平坦，代表放大区。

### 7b. 直流负载线与 Q 点

```python
# -*- coding: utf-8 -*-
"""
在输出特性曲线上画出直流负载线，并标出 Q 点
电路：分压偏置，Vcc=12V, Rc=3kΩ, Re=1kΩ, R1=40kΩ, R2=10kΩ
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# === 电路参数 ===
V_CC = 12.0
R_C = 3000.0   # 3 kΩ
R_E = 1000.0   # 1 kΩ
R_1 = 40000.0  # 40 kΩ
R_2 = 10000.0  # 10 kΩ
beta = 150
V_BE = 0.7
V_CE_sat = 0.2

# === 计算 Q 点 ===
V_B = V_CC * R_2 / (R_1 + R_2)
V_E = V_B - V_BE
I_C_Q = V_E / R_E  # Ic ≈ Ie
V_CE_Q = V_CC - I_C_Q * (R_C + R_E)

print(f"=== Q 点计算结果 ===")
print(f"V_B  = {V_B:.2f} V")
print(f"V_E  = {V_E:.2f} V")
print(f"I_C  = {I_C_Q * 1000:.2f} mA")
print(f"V_CE = {V_CE_Q:.2f} V")
print(f"Q 点坐标: ({V_CE_Q:.2f} V, {I_C_Q * 1000:.2f} mA)")

# === 负载线 ===
V_CE_line = np.linspace(0, V_CC, 200)
R_total = R_C + R_E  # 总负载电阻
I_C_max = V_CC / R_total  # 饱和电流（Vce=0）
I_C_line = (V_CC - V_CE_line) / R_total  # 负载线方程

# === 输出特性曲线（简化绘制几条） ===
plt.figure(figsize=(10, 7))

# 绘制几条 Ib 曲线（简化）
I_B_lines = [0, 5e-6, 10e-6, 15e-6, 20e-6, 25e-6, 30e-6, 35e-6]  # A
V_A = 100  # 厄利电压
for ib in I_B_lines:
    if ib == 0:
        ic_curve = np.zeros_like(V_CE_line)
        label = 'Ib=0 μA'
    else:
        ic_curve = beta * ib * (1 + V_CE_line / V_A)
        # 饱和区限制
        ic_curve = np.minimum(ic_curve, V_CE_line / 50)
        label = f'Ib={ib*1e6:.0f} μA'
    
    plt.plot(V_CE_line, ic_curve * 1000, 'b-', alpha=0.25, linewidth=1)

# === 画负载线 ===
plt.plot(V_CE_line, I_C_line * 1000, 'r-', linewidth=2.5, label='直流负载线')

# === 标出 Q 点 ===
plt.plot(V_CE_Q, I_C_Q * 1000, 'ro', markersize=12, markerfacecolor='red',
         markeredgecolor='darkred', markeredgewidth=2, zorder=5, label=f'Q点({V_CE_Q:.1f}V, {I_C_Q*1000:.1f}mA)')

# === 标出负载线两端点 ===
plt.plot(0, I_C_max * 1000, 'ko', markersize=8, markerfacecolor='none')
plt.text(0.1, I_C_max * 1000 + 0.15, f'饱和端点\n(0V, {I_C_max*1000:.1f}mA)',
         fontsize=10, va='bottom')

plt.plot(V_CC, 0, 'ko', markersize=8, markerfacecolor='none')
plt.text(V_CC - 1.5, 0.15, f'截止端点\n({V_CC:.0f}V, 0mA)',
         fontsize=10, va='bottom')

# === 标注 ===
plt.xlabel('$V_{CE}$ (V)', fontsize=14)
plt.ylabel('$I_C$ (mA)', fontsize=14)
plt.title('BJT 直流负载线与 Q 点 (分压偏置)', fontsize=15)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, V_CC + 0.5)
plt.ylim(0, I_C_max * 1000 + 0.5)
plt.tight_layout()
plt.savefig('/tmp/bjt_loadline_qpoint.png', dpi=150)
plt.show()

print(f"\n负载线方程: Ic = ({V_CC:.0f} - Vce) / {R_total/1000:.0f}kΩ")
print(f"Q 点约在负载线 {V_CE_Q/V_CC*100:.0f}% 处，偏置合理")
```

**预期输出**：一张图，浅蓝色细线是不同 Ib 对应的输出特性曲线，红色粗线是负载线，大红点标出了 Q 点位置。两端的小圆圈标出了饱和端点和截止端点。

### 7c. 固定偏置 vs 分压偏置：β 变动时的 Q 点稳定性对比

```python
# -*- coding: utf-8 -*-
"""
对比固定偏置和分压偏置在 β 变动时的 Q 点稳定性
固定偏置：Rb=565kΩ, Rc=3kΩ, Vcc=12V
分压偏置：R1=120kΩ, R2=27kΩ, Rc=5.6kΩ, Re=2kΩ, Vcc=15V
模拟 β 从 50 到 500 变化时，两者的 Ic 和 Vce 如何漂移
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# === 固定偏置参数 ===
V_CC_fixed = 12.0
R_B_fixed = 565e3   # 565 kΩ
R_C_fixed = 3e3     # 3 kΩ
V_BE = 0.7

# === 分压偏置参数 ===
V_CC_div = 15.0
R1 = 120e3   # 120 kΩ
R2 = 27e3    # 27 kΩ
RC_div = 5.6e3   # 5.6 kΩ
RE_div = 2e3     # 2 kΩ

# === β 扫描范围 ===
beta_values = np.linspace(50, 500, 200)

# === 计算固定偏置在不同 β 下的 Q 点 ===
# 固定偏置: Ib = (Vcc - Vbe) / Rb, Ic = β * Ib
I_B_fixed = (V_CC_fixed - V_BE) / R_B_fixed  # Ib 不变（固定偏置的特点）
I_C_fixed = beta_values * I_B_fixed
V_CE_fixed = V_CC_fixed - I_C_fixed * R_C_fixed

# 饱和检测（Vce < 0.2V 时钳位）
saturated_fixed = V_CE_fixed < 0.2
I_C_fixed_clamped = np.where(saturated_fixed,
                              (V_CC_fixed - 0.2) / R_C_fixed,
                              I_C_fixed)
V_CE_fixed_clamped = np.where(saturated_fixed, 0.2, V_CE_fixed)

# === 计算分压偏置在不同 β 下的 Q 点 ===
# 分压偏置: Vb ≈ Vcc * R2/(R1+R2), Ve = Vb-0.7, Ie = Ve/Re
V_B_div = V_CC_div * R2 / (R1 + R2)
V_E_div = V_B_div - V_BE
I_E_div = V_E_div / RE_div  # 几乎不变（分压偏置的特点）

# 考虑 Ib 的修正（β 较小时）
I_B_div = I_E_div / (beta_values + 1)
# Vb 受 Ib 影响的修正（流过 R1 的电流 = 流过 R2 + Ib）
# 简化近似（精确计算需要迭代）
V_B_actual = (V_CC_div / R1 - I_B_div) * (R1 * R2) / (R1 + R2)
# 实际上分压偏置 Ib 影响不大，这里用近似
I_E_actual = (V_B_actual - V_BE) / RE_div
# 为避免过度复杂，直接用简单近似（Ib 不够大时略有偏差）
I_C_div = I_E_div * beta_values / (beta_values + 1)  # Ic = Ie - Ib = β/(β+1) * Ie
V_CE_div = V_CC_div - I_C_div * RC_div - I_E_div * RE_div

# === 绘图：Ic 对比 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：Ic 随 β 变化
ax1.plot(beta_values, I_C_fixed_clamped * 1000, 'r-', linewidth=2, label='固定偏置 Ic')
ax1.plot(beta_values, I_C_div * 1000, 'b-', linewidth=2, label='分压偏置 Ic')
ax1.axhline(y=I_E_div * 1000, color='blue', linestyle='--', alpha=0.4,
            label=f'分压偏置设计值 ≈ {I_E_div*1000:.1f} mA')
ax1.set_xlabel('β (直流电流增益)', fontsize=12)
ax1.set_ylabel('$I_C$ (mA)', fontsize=12)
ax1.set_title('Ic 随 β 变化的稳定性对比', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 标记饱和区
ax1.fill_between(beta_values, 0, (V_CC_fixed - 0.2) / R_C_fixed * 1000,
                 where=saturated_fixed, color='red', alpha=0.1)
ax1.text(200, 2, '固定偏置\n饱和了！', fontsize=10, color='red',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 右图：Vce 随 β 变化
ax2.plot(beta_values, V_CE_fixed_clamped, 'r-', linewidth=2, label='固定偏置 Vce')
ax2.plot(beta_values, V_CE_div, 'b-', linewidth=2, label='分压偏置 Vce')
ax2.axhline(y=V_CC_div / 2, color='gray', linestyle=':', alpha=0.5, label='Vcc/2 (理想 Q 点)')
ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.4, label='饱和边界 Vce=0.2V')
ax2.set_xlabel('β (直流电流增益)', fontsize=12)
ax2.set_ylabel('$V_{CE}$ (V)', fontsize=12)
ax2.set_title('Vce 随 β 变化的稳定性对比', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('固定偏置 vs 分压偏置：Q 点稳定性对比', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/bjt_bias_stability.png', dpi=150)
plt.show()

print("=== Q 点稳定性对比 ===")
print(f"固定偏置：β 从 50→500，Ic 从 {I_C_fixed_clamped[0]*1000:.1f} mA → {I_C_fixed_clamped[-1]*1000:.1f} mA")
print(f"          变化幅度：{(I_C_fixed_clamped[-1]-I_C_fixed_clamped[0])/I_C_fixed_clamped[0]*100:.0f}%")
print(f"分压偏置：β 从 50→500，Ic 从 {I_C_div[0]*1000:.1f} mA → {I_C_div[-1]*1000:.1f} mA")
print(f"          变化幅度：{(I_C_div[-1]-I_C_div[0])/I_C_div[0]*100:.0f}%")
print(f"\n结论：分压偏置的 Q 点稳定性远超固定偏置。")
print(f"这也是为什么工程上几乎 100% 使用分压偏置。")
```

**预期输出**：左右两图对比。左图 Ic-β：红线（固定偏置）随 β 直线上升，很快饱和。蓝线（分压偏置）几乎水平不变。右图 Vce-β：红线急速下跌到饱和，蓝线保持平稳。这个对比一目了然地展示了为什么分压偏置是标准答案。

---

## 8. 常见误区

### ❌ 误区一："BJT 是电压控制元件"

**错。** BJT 是**电流控制**元件。决定 Ic 大小的是 Ib，不是 Vbe。虽然 $I_C$ 和 $V_{BE}$ 之间确实有指数关系（Ebers-Moll 方程），但在电路分析和设计层面，我们用的是 $I_C = \beta I_B$，控制量是电流。

**✓** 如果你需要电压控制，请用 FET（第 32 章）。

### ❌ 误区二："BJT 在任何情况下都放大电流"

**错。** BJT 只在**放大区**才起放大作用（$I_C = \beta I_B$）。在截止区，$I_C \approx 0$——没放大。在饱和区，$I_C$ 由外部电路决定，增大 $I_B$ 也不增加 $I_C$——也没放大。

**✓** 分析任何 BJT 电路，第一步永远是判断工作区。

### ❌ 误区三："Vbe 永远是 0.7 V"

**错。** 0.7 V 是硅管在典型工作电流（毫安级）下的近似值。Ib 很小（纳安级）时，$V_{BE}$ 可能只有 0.5 V。Ib 很大（几十毫安）时，$V_{BE}$ 可能超过 0.8 V。而且 $V_{BE}$ 随温度升高而降低（约 -2 mV/°C）。

**✓** 对于手算和初学分析，0.7 V 是足够好的近似。但在精密设计中需要查数据手册或实际测量。

### ❌ 误区四："Ic = β Ib，所以 β 越大越好"

**不对。** β 太大带来的问题是：第一，β 大的管子往往热稳定性更差（温度变化时 Q 点漂移更大）。第二，高 β 管更容易自激振荡。第三，高 β 管通常耐压较低。实际选管时，β 在 100-300 之间就够用了，不需要追求极端的 β 值（比如达林顿管的 β 可上千，但那是特殊用途）。

**✓** 选 β 适中的管子 + 用分压偏置来稳定 Q 点，比追求高 β 管更靠谱。

### ❌ 误区五："饱和时 Vce = 0"

**错。** 饱和时 $V_{CE}$ 只是**很小**，不是零。硅 BJT 的饱和压降 $V_{CE(sat)}$ 约 0.1-0.2 V（小功率管），大功率管可能达到 0.5 V 甚至更高。在开关电源设计中，$V_{CE(sat)}$ 直接决定了开关损耗（$P = V_{CE(sat)} \times I_C$），必须精确考虑。

**✓** 在开关电路中，选 $V_{CE(sat)}$ 低的管子可以降低导通损耗。

### ❌ 误区六："PNP 管就是 NPN 管反过来接就行"

**错。** PNP 管的电压极性和电流方向与 NPN 完全相反，但工作原理是对称的。重点在于偏置：NPN 管的 E 极通常接低电位（地），C 极接高电位（电源）；PNP 管的 E 极接高电位（电源），C 极接低电位（地或负载）。把 PNP 当 NPN 偏置会直接烧管。

**✓** 学透 NPN 后，PNP 只需把"所有电压反过来、所有电流反过来"即可。电路图也一样，只是电源正负极互换。

---

## 9. 思考题

### 基础题

**第 1 题**：NPN 型 BJT，$\beta = 80$，测得 $I_B = 25\ \mu\text{A}$。求 $I_C$ 和 $I_E$。BJT 工作在哪个区？（假设 $V_{CC}$ 足够大，$R_C$ 不大）

**第 2 题**：固定偏置电路，$V_{CC} = 10\ \text{V}$，$R_B = 470\ \text{k}\Omega$，$R_C = 2\ \text{k}\Omega$，$\beta = 120$。求 $I_B$、$I_C$、$V_{CE}$。BJT 工作在哪个区？

**第 3 题**：上题中，如果换用一只 $\beta = 300$ 的同型号管子，$I_C$ 和 $V_{CE}$ 会变成多少？这说明固定偏置有什么问题？

### 进阶题

**第 4 题**：分压偏置电路，$V_{CC} = 15\ \text{V}$，$R_1 = 100\ \text{k}\Omega$，$R_2 = 20\ \text{k}\Omega$，$R_C = 4.7\ \text{k}\Omega$，$R_E = 1.2\ \text{k}\Omega$。求 Q 点（$I_C$、$V_{CE}$）。假设 $\beta$ 足够大。

**第 5 题**：在上题中，如果 $\beta = 80$（不太大），重新计算 Q 点（需要考虑 $I_B$ 对 $V_B$ 的影响）。与第 4 题结果对比。

### 综合题

**第 6 题**：设计一个分压偏置电路，要求：$V_{CC} = 12\ \text{V}$，Q 点 $I_C = 2\ \text{mA}$，$V_{CE} = 6\ \text{V}$，$V_E = 1.2\ \text{V}$（约 0.1 Vcc）。$\beta$ 范围 80-200。选定所有电阻值（用常用标称值）。

**第 7 题**：画出第 6 题电路的直流负载线，标出 Q 点位置。如果输入信号使 $I_B$ 在 $10\ \mu\text{A}$ 到 $30\ \mu\text{A}$ 之间摆动（$\beta = 100$），$V_{CE}$ 的摆动范围是多少？输出会不会被削波？

**第 8 题**：设计一个用 NPN BJT 驱动 LED 的开关电路。LED 正向压降 2 V，工作电流 20 mA。BJT 用 3.3 V 单片机的 GPIO 引脚控制（输出高电平时电压 3.3 V，能提供的最大电流 10 mA）。电源 $V_{CC} = 5\ \text{V}$，$\beta_{\text{min}} = 100$。求基极限流电阻 $R_B$ 和集电极限流电阻 $R_C$。BJT 应该工作在哪个区？为什么？

---

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

---

#### 第 1 题解答

$$
\begin{aligned}
I_C &= \beta I_B = 80 \times 25\ \mu\text{A} = 2000\ \mu\text{A} = 2\ \text{mA} \\[4pt]
I_E &= I_C + I_B = 2\ \text{mA} + 25\ \mu\text{A} = 2.025\ \text{mA} \\[4pt]
    &\approx 2\ \text{mA}
\end{aligned}
$$

因为 $I_B > 0$ 且题目说 $V_{CC}$ 足够大、$R_C$ 不大，意味着 $V_{CE}$ 不会太低，BJT 工作在**放大区**。

---

#### 第 2 题解答

基极回路：

$$
I_B = \frac{V_{CC} - V_{BE}}{R_B} = \frac{10\ \text{V} - 0.7\ \text{V}}{470\ \text{k}\Omega} = \frac{9.3\ \text{V}}{470 \times 10^3\ \Omega} \approx 19.8\ \mu\text{A}
$$

$$
I_C = \beta I_B = 120 \times 19.8\ \mu\text{A} \approx 2.38\ \text{mA}
$$

$$
V_{CE} = V_{CC} - I_C R_C = 10\ \text{V} - 2.38\ \text{mA} \times 2\ \text{k}\Omega = 10 - 4.76 = 5.24\ \text{V}
$$

$V_{CE} = 5.24\ \text{V} > 0.2\ \text{V}$，BJT 工作在**放大区**。

---

#### 第 3 题解答

$I_B$ 不变（B 极回路只取决于 $V_{CC}$、$R_B$、$V_{BE}$），还是 19.8 μA。

但 $\beta = 300$：

$$
I_C = \beta I_B = 300 \times 19.8\ \mu\text{A} \approx 5.94\ \text{mA}
$$

$$
V_{CE} = 10\ \text{V} - 5.94\ \text{mA} \times 2\ \text{k}\Omega = 10 - 11.88 = -1.88\ \text{V} \ (\text{不可能！})
$$

$V_{CE}$ 算出来是负数，说明放大区假设不成立。实际 BJT 已经进入**饱和区**。

饱和电流：$I_{C(sat)} = \frac{10\ \text{V} - 0.2\ \text{V}}{2\ \text{k}\Omega} = 4.9\ \text{mA}$，$V_{CE} \approx 0.2\ \text{V}$。

**结论**：β 从 120 换到 300，Q 点从放大区正中间直接掉进了饱和区。固定偏置电路对 β 极度敏感，这就是为什么工程上几乎不用它。

---

#### 第 4 题解答

$$
\begin{aligned}
V_B &= V_{CC} \cdot \frac{R_2}{R_1 + R_2} = 15 \times \frac{20}{100 + 20} \\[4pt]
    &= 15 \times \frac{20}{120} = 15 \times \frac{1}{6} = 2.5\ \text{V}
\end{aligned}
$$

$$
V_E = V_B - V_{BE} = 2.5 - 0.7 = 1.8\ \text{V}
$$

$$
I_C \approx I_E = \frac{V_E}{R_E} = \frac{1.8\ \text{V}}{1.2\ \text{k}\Omega} = 1.5\ \text{mA}
$$

$$
V_{CE} = V_{CC} - I_C (R_C + R_E) = 15 - 1.5\text{mA} \times (4.7\text{k}\Omega + 1.2\text{k}\Omega) = 15 - 8.85 = 6.15\ \text{V}
$$

Q 点：(6.15 V, 1.5 mA)，在放大区中间偏上。

---

#### 第 5 题解答

当 $\beta = 80$ 不够大时，$I_B$ 不能完全忽略。需要联立方程组。

设 $I_B$ 未知，则 $I_C = 80 I_B$，$I_E = 81 I_B$。

B 极节点电流（流入为正）：流过 $R_1$ 的电流 $I_{R1} = \frac{V_{CC} - V_B}{R_1}$，流过 $R_2$ 的电流 $I_{R2} = \frac{V_B}{R_2}$，流入 B 极的电流为 $I_B$。

KCL 在 B 点：
$$
\frac{V_{CC} - V_B}{R_1} = \frac{V_B}{R_2} + I_B
$$

另有：
$$
V_E = V_B - 0.7,\quad I_E = 81 I_B = \frac{V_E}{R_E} = \frac{V_B - 0.7}{1200}
$$

从发射极方程：
$$
I_B = \frac{V_B - 0.7}{81 \times 1200} = \frac{V_B - 0.7}{97200}
$$

代入 B 点 KCL：
$$
\frac{15 - V_B}{100000} = \frac{V_B}{20000} + \frac{V_B - 0.7}{97200}
$$

乘以公分母求解（略去中间代数），得 $V_B \approx 2.41\ \text{V}$。

$$
\begin{aligned}
V_E &= 2.41 - 0.7 = 1.71\ \text{V} \\[4pt]
I_E &= \frac{1.71}{1200} = 1.425\ \text{mA} \\[4pt]
I_C &= I_E - I_B \approx I_E - \frac{I_E}{81} = 1.425 \times \frac{80}{81} \approx 1.407\ \text{mA} \\[4pt]
V_{CE} &= 15 - 1.407\text{mA} \times 5.9\text{k}\Omega - 1.425\text{mA} \times 1.2\text{k}\Omega \\
       &\approx 15 - 8.30 - 1.71 = 4.99\ \text{V}
\end{aligned}
$$

**对比第 4 题**（假设 β 无穷大，Ic = 1.5 mA, Vce = 6.15 V）：
- β = 80 时：Ic = 1.41 mA, Vce = 4.99 V
- 偏差：Ic 偏小了约 6%，Vce 偏小了约 19%

虽然有了偏差，但 Q 点仍在放大区。相比固定偏置直接饱和，分压偏置稳得多。

---

#### 第 6 题解答

**设计要求**：
- $V_{CC} = 12\ \text{V}$
- $I_C = 2\ \text{mA}$
- $V_{CE} = 6\ \text{V}$
- $V_E = 1.2\ \text{V}$
- $\beta$ 范围 80-200

**步骤一：确定 $R_E$**

$$
R_E = \frac{V_E}{I_E} \approx \frac{V_E}{I_C} = \frac{1.2\ \text{V}}{2\ \text{mA}} = 600\ \Omega
$$

常用标称值：620 Ω 或 560 Ω。选 $R_E = 620\ \Omega$。

**步骤二：确定 $R_C$**

$$
V_{CE} = V_{CC} - I_C(R_C + R_E) \quad\Rightarrow\quad R_C = \frac{V_{CC} - V_{CE}}{I_C} - R_E
$$

$$
R_C = \frac{12 - 6}{2\ \text{mA}} - 620 = \frac{6}{0.002} - 620 = 3000 - 620 = 2380\ \Omega
$$

常用标称值：$R_C = 2.4\ \text{k}\Omega$（或 2.2 kΩ + 调整 $V_E$ 稍做补偿）。

**步骤三：确定 $V_B$**

$$
V_B = V_E + V_{BE} = 1.2 + 0.7 = 1.9\ \text{V}
$$

**步骤四：确定 $R_1$ 和 $R_2$**

取分压器电流 $I_{\text{divider}} = 10 \times I_{B(\text{max})}$（用 $\beta_{\text{min}}$ 估算最坏情况 $I_B$ 最大）：

$$
I_{B(\text{max})} = \frac{I_C}{\beta_{\text{min}}} = \frac{2\ \text{mA}}{80} = 25\ \mu\text{A}
$$

$$
I_{\text{divider}} = 10 \times 25\ \mu\text{A} = 250\ \mu\text{A}
$$

$$
R_2 = \frac{V_B}{I_{\text{divider}}} = \frac{1.9\ \text{V}}{250\ \mu\text{A}} = 7600\ \Omega
$$

常用标称值：$R_2 = 7.5\ \text{k}\Omega$。

$$
R_1 = \frac{V_{CC} - V_B}{I_{\text{divider}}} = \frac{12 - 1.9}{250\ \mu\text{A}} = \frac{10.1\ \text{V}}{0.25\ \text{mA}} = 40{,}400\ \Omega
$$

常用标称值：$R_1 = 39\ \text{k}\Omega$。

**最终参数**：$R_E = 620\ \Omega$，$R_C = 2.4\ \text{k}\Omega$，$R_1 = 39\ \text{k}\Omega$，$R_2 = 7.5\ \text{k}\Omega$。

---

#### 第 7 题解答

**第一问：直流负载线**

总负载电阻 $R_{\text{total}} = R_C + R_E = 2.4\text{k} + 0.62\text{k} = 3.02\ \text{k}\Omega$。

负载线两端点：
- 饱和端：$V_{CE} = 0$，$I_C = \frac{12}{3.02\text{k}} \approx 3.97\ \text{mA}$
- 截止端：$I_C = 0$，$V_{CE} = 12\ \text{V}$

Q 点（从第 6 题设计值）：$V_{CE} = 6\ \text{V}$，$I_C = 2\ \text{mA}$。正好在负载线正中间，非常理想。

**第二问：输出摆动范围**

$\beta = 100$。$I_B$ 在 10 μA 到 30 μA 之间摆动。

$$
\begin{aligned}
I_{C(\text{min})} &= 100 \times 10\ \mu\text{A} = 1\ \text{mA} \\
I_{C(\text{max})} &= 100 \times 30\ \mu\text{A} = 3\ \text{mA}
\end{aligned}
$$

对应的 $V_{CE}$：

$$
\begin{aligned}
V_{CE(\text{min})} &= 12 - 3\ \text{mA} \times 3.02\ \text{k}\Omega = 12 - 9.06 = 2.94\ \text{V} \\[4pt]
V_{CE(\text{max})} &= 12 - 1\ \text{mA} \times 3.02\ \text{k}\Omega = 12 - 3.02 = 8.98\ \text{V}
\end{aligned}
$$

$V_{CE}$ 摆动范围：2.94 V ~ 8.98 V，峰峰值约 6 V。

$I_C$ 最小 1 mA，离截止（0 mA）还有距离。$V_{CE}$ 最小 2.94 V，离饱和（0.2 V）还有距离。**输出不会被削波**。

---

#### 第 8 题解答

**问题分析**：单片机的 GPIO 输出 3.3 V，最大 10 mA。我们要用这个信号控制 BJT 去开关一个需要 20 mA 的 LED。显然单片机直接驱动不了 LED（电流不够），需要 BJT 来"以小控大"。

**BJT 工作区选择**：开关电路需要 BJT 要么完全**截止**（LED 灭），要么完全**饱和**（LED 亮）。所以 BJT 工作在**截止区 ↔ 饱和区**之间切换，不经过放大区。

**选择 $R_C$（LED 限流电阻）**：

LED 回路：$V_{CC}$ → $R_C$ → LED → C → E → 地。BJT 饱和时 $V_{CE} \approx 0.2\ \text{V}$。

$$
\begin{aligned}
V_{CC} &= I_{LED} R_C + V_{LED} + V_{CE(sat)} \\[4pt]
5\ \text{V} &= 20\ \text{mA} \times R_C + 2\ \text{V} + 0.2\ \text{V} \\[4pt]
R_C &= \frac{5 - 2 - 0.2}{20\ \text{mA}} = \frac{2.8\ \text{V}}{0.02\ \text{A}} = 140\ \Omega
\end{aligned}
$$

常用标称值：$R_C = 150\ \Omega$。

**选择 $R_B$（基极限流电阻）**：

基极回路：GPIO → $R_B$ → B → E → 地。BJT 饱和需要 $I_B > \frac{I_C}{\beta_{\text{min}}}$。

$$
I_{B(\text{min})} = \frac{I_{LED}}{\beta_{\text{min}}} = \frac{20\ \text{mA}}{100} = 0.2\ \text{mA} = 200\ \mu\text{A}
$$

为了确保可靠饱和，通常取 **过驱动因子** 为 2-5 倍。取 3 倍：

$$
I_B = 3 \times 200\ \mu\text{A} = 600\ \mu\text{A}
$$

基极回路 KVL（GPIO 高电平 3.3 V，B-E 结 0.7 V）：

$$
\begin{aligned}
3.3\ \text{V} &= I_B R_B + 0.7\ \text{V} \\[4pt]
R_B &= \frac{3.3 - 0.7}{600\ \mu\text{A}} = \frac{2.6\ \text{V}}{0.0006\ \text{A}} \approx 4333\ \Omega
\end{aligned}
$$

常用标称值：$R_B = 4.3\ \text{k}\Omega$。

**验证 GPIO 驱动能力**：$I_B = \frac{3.3 - 0.7}{4.3\text{k}} \approx 0.605\ \text{mA} = 605\ \mu\text{A}$，远小于 GPIO 最大输出 10 mA，安全。

**最终参数**：$R_C = 150\ \Omega$，$R_B = 4.3\ \text{k}\Omega$。

**工作状态**：
- GPIO 高电平（3.3 V）→ BJT 饱和 → LED 亮（约 20 mA）
- GPIO 低电平（0 V）→ BJT 截止 → LED 灭

</details>

---

## 9a. 应用连接：BJT 在真实世界中的位置

学完 BJT 的基础，你可能想知道这个东西到底用在哪。答案是：**无处不在**。

**模拟放大器**：任何需要把小信号放大的地方都有 BJT。你的吉他效果器、麦克风前置放大器、音响功放的第一级，几乎都是 BJT 放大电路。后续的共射极、共集电极（射极跟随器）、共基极三种基本放大组态（将在下一阶段章节中讲解）各有用武之地。

**开关电源（Switching Power Supply）**：你手机充电器里、电脑主板上的 DC-DC 变换器，里面的功率开关管很多就是 BJT（或 MOSFET，第 32 章）。BJT 在截止和饱和之间高速切换（每秒几十万次），配合电感和电容实现高效率的电压变换。

**数字逻辑电路**：虽然现代 CPU 主要由 MOSFET 构成，但最早的计算机用的是 BJT。TTL（Transistor-Transistor Logic）逻辑门——你在第 21 章学过的 7400 系列芯片——内部就是一堆 BJT 组成的与非门、或非门。直到今天，TTL 芯片在某些工业和航天领域仍在服役，因为它们比 CMOS 更耐辐射。

**温度传感器**：B-E 结的 $V_{BE}$ 随温度线性变化（约 -2 mV/°C）。利用这个特性可以造出极便宜的温度传感器——一个 BJT 加几个电阻就是一个测温电路。很多芯片内部的过温保护电路就是这个原理。

**恒流源**：用一个 BJT、一个电阻、一个二极管（或稳压管），就能做出一个输出电流非常稳定的恒流源。LED 驱动、电池充电电路中都大量使用 BJT 恒流源。

**电机和继电器驱动**：单片机的 GPIO 引脚输出电流通常不超过 10-20 mA，远远不够直接驱动一个电机或继电器。这时候用一个小小的 BJT 做开关（正是思考题第 8 题的 LED 驱动方案的放大版），单片机用小电流控制 BJT 的基极，BJT 再用大电流去驱动负载。

你可能会问：那 MOSFET 和 BJT 有什么区别？什么时候该用 BJT，什么时候该用 FET？简单说：BJT 是电流控制（输入是电流信号），FET 是电压控制（输入是电压信号，几乎不消耗电流）。BJT 在低噪声放大、高频模拟电路中仍有巨大优势。FET 在数字电路、大功率开关中占主导。两种都要学，两种都要会。

等你学完 FET（第 32 章）和运放（第 33 章），你就能根据实际需求，在 BJT、FET、运放三者之间自由选择。而这一切的起点，就是本章你能手算的 $I_C = \beta I_B$。

---

## 10. 本章总结

这一章你学会了 BJT 的核心秘密：一个小电流（Ib）控制一个大电流（Ic）。NPN 和 PNP 两种结构、三个工作区（截止、放大、饱和）的条件和判断方法、两种直流偏置电路（固定 vs 分压）、Q 点和负载线的概念——这些都是模拟电子学的"基本功"。

如果只记住三件事，就记这三件：

1. **放大区：$I_C = \beta I_B$**，B-E 正偏，C-B 反偏。这是 BJT 作为放大器的核心工作区。
2. **分压偏置比固定偏置稳得多**。$I_C \approx V_E / R_E$，几乎不依赖 $\beta$。工程上默认用分压偏置。
3. **会用负载线找 Q 点**。Q 点在负载线中间 → 输出摆幅最大、失真最小。Q 点偏到两边 → 削波。

下一章是[场效应管（FET）](./32-fet.md)。FET 也是一种"以小控大"的元件，但它用**电压**控制而不是电流。学完 FET，你就会拥有两种放大器的设计工具，可以根据需求自由选择。

---

> **致读者**：如果你跟着本章一步步手算了 8 道题、跑了 Python 代码，恭喜你——你已经能独立分析和设计 BJT 偏置电路了。这是模拟电路设计入门的第一个里程碑。后面的 FET、运放、放大器章节会在本章基础上层层搭建，你会发现 BJT 的知识无处不在。
