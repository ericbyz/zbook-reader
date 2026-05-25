# 第13章：RL电路的动态响应

> **核心问题**：电感也像电容一样"慢慢来"吗？

---

## 0. 本章导览

上一章你学会了 RC 电路的充放电：电容的电压不能突变，充放电按指数曲线走，速度由 $\tau = RC$ 决定。这一章我们把电容换成电感（Inductor），看看会发生什么。你会发现，电感的"性格"和电容刚好"镜像对称"：电容是电压不能突变，电感是**电流不能突变**；电容的稳态是断路，电感的稳态是**短路**。

如果你已经掌握了第 12 章的三要素法和指数响应的思路，这一章你学起来会快很多。RL 电路的分析方法和 RC 一模一样——列微分方程、解指数函数、三要素法，只是 $\tau$ 变成了 $L/R$。

学完本章，你将能够：

1. 写出 RL 电路在开关动作后的电流响应（零输入、零状态、全响应）
2. 理解时间常数 $\tau = L/R$ 的物理意义
3. 用三要素法直接写出 RL 电路的响应表达式
4. 掌握 RC 与 RL 的对偶关系（C $\leftrightarrow$ L，R $\leftrightarrow$ 1/G，v $\leftrightarrow$ i）
5. 理解阶跃响应和单位阶跃函数 $u(t)$
6. 用 Python 对比 RC 和 RL 的动态行为

> 本章约 1200 行，建议分 2-3 次读完。如果你已经熟悉了 RC 电路，RL 电路的核心就是"镜像类比"。

前置章节：[RC 电路的充放电](./12-first-order-rc.md)
下一章：[二阶 RLC 电路与振荡](./14-second-order-rlc.md)

---

## 1. 电感：和电容"对偶"的储能元件

### 1a. 生活例子：旋转的水车

在第 12 章里，电容的比喻是水桶（存水量 ↔ 存电荷）。那电感呢？最形象的比喻是**旋转的水车**。

想象一条水渠里有一个大水车。水车很重，有很大的惯性。刚开始水不流的时候，水车静止不动。你把水闸打开，水开始流（相当于有了电流）。但水车太重了，它不会立刻飞速旋转。水流推着它，慢慢加速，越来越快。如果水流突然停止，水车也不会马上停下来——它靠惯性继续转，反过来推动水流。

电感就是电路里的"水车"：**电流开始流动时，电感阻挡电流变化；电流稳定之后，电感就"不碍事"了（相当于导线）；电流要停下来时，电感又阻挡它停下来，试图维持电流继续流。**

在电路比喻里：
- **水车的转动速度** ↔ 电路里的**电流 i_L**
- **水车的惯性（质量）** ↔ 电感的**感值 L**（感值越大，惯性越大）
- **推水车的水流压力** ↔ 电感两端的**电压 v_L**（电压是"驱动力"）

### 1b. 直观理解：电感凭什么"慢慢来"？

电感的核心物理原理是**法拉第电磁感应定律**：变化的电流产生变化的磁场，变化的磁场又感应出电压来反对电流的变化。这个"反对"的电压叫**反电动势（Back EMF）**。

用电感的 V-I 关系表达：

$$
v_L(t) = L \frac{di_L(t)}{dt}
$$

翻译成人话：**电感两端的电压，正比于电流的变化率**。电流变得越快，电感两端产生的电压越大。如果电流稳定不变（$di_L/dt = 0$），电感两端电压就是零——电感在直流稳态下相当于**短路（导线）**。

反过来，电感电流是电压的积分：

$$
i_L(t) = i_L(0) + \frac{1}{L} \int_0^t v_L(\tau) \, d\tau
$$

这个式子说：**电感当前的电流 = 初始电流 + 从 0 到当前时刻电感两端承受的"电压冲量"（伏秒积）除以电感值**。

### 1c. RC 和 RL 的对偶关系

把 RC 和 RL 并排放在一起比较，你会发现一种美妙的对称：

| 特性 | RC 电路 | RL 电路 |
|------|---------|---------|
| 储能元件 | 电容 C | 电感 L |
| 不能突变的量 | 电压 $v_C$ | 电流 $i_L$ |
| V-I 关系 | $i_C = C \frac{dv_C}{dt}$ | $v_L = L \frac{di_L}{dt}$ |
| 直流稳态行为 | 断路（开路） | 短路（导线） |
| 时间常数 | $\tau = RC$ | $\tau = L/R$ |
| 单位验算 | $\Omega \times \text{F} = \text{s}$ | $\text{H} / \Omega = \text{s}$ |
| 水比喻 | 水桶（存水量） | 水车（存惯性） |

这种对称叫**对偶性（Duality）**。如果把 RC 电路中的所有物理量做以下替换：
- 电容 C $\leftrightarrow$ 电感 L
- 电阻 R $\leftrightarrow$ 电导 $G = 1/R$
- 电压 v $\leftrightarrow$ 电流 i
- 串联 $\leftrightarrow$ 并联

那么 RC 的公式就变成了 RL 的公式。掌握了一个，就掌握了两个。

### 1d. 对偶性的深入理解

只看表格可能觉得"哦，换一下字母就行"。但对偶性比你想象的更深。实际上整个电路理论都能在对偶变换下保持成立。

举个例子：戴维南定理说的是"任意线性一端口网络可以等效为一个电压源串电阻"。如果你做对偶变换（电压 ↔ 电流，串联 ↔ 并联），就得到了**诺顿定理**："任意线性一端口网络可以等效为一个电流源并电阻"。这不是巧合，是结构性的对称。

再举个例子：基尔霍夫电压定律（KVL）说"闭合回路电压和为 0"。对偶变换后就是基尔霍夫电流定律（KCL）："闭合曲面上电流和为 0"。

所以当你学会 RC 电路后，RL 电路不是"另一套东西"，它是同一套数学结构的对偶体现。你在 RC 里建立的直觉（指数曲线、时间常数、三要素法）在 RL 里完全复用，只是"谁是主角"从电压换成了电流。

### 1e. 单位验证小技巧

对于 RL 电路，有一种简单的方法记住 $\tau = L/R$ 而不是 $L \times R$：

- 直觉法：$R$ 大意味着能量消耗快（$P = I^2 R$），所以时间应该**短**。$L/R$ 符合这个直觉（$R \uparrow \Rightarrow \tau \downarrow$）。如果是 $L \times R$，那 $R$ 大时间反而长，这违背物理直觉。
- 单位法：$L$ 的单位是 $\text{V} \cdot \text{s} / \text{A}$，$R$ 的单位是 $\text{V} / \text{A}$。要得到秒，只能是 $L/R$。

这个方法在面对不熟悉的公式时非常好用——先检查单位，再检查直觉。

---

## 2. RL 电路的零输入响应（电流衰减）

### 2a. 场景：电感里有初始电流，断开电源让它"自生自灭"

电路：一个电感 $L$ 和一个电阻 $R$ 串联（或并联——但在串联 RL 里更自然）。$t<0$ 时电感里已经有稳定的电流 $I_0$ 在流动（比如之前接在电源上跑了一段时间）。$t=0$ 时把电源断开，让电感和电阻自己构成一个闭合回路。

这个过程中电感里储存的磁能通过电阻消耗掉——这就是 RL 的**零输入响应**（Zero-Input Response）。

```
         t=0 断开电源，R和L自成回路
              ┌───────┐
              │       │
              │  t=0  │
         R    │  断开 │
    ───[~~~]──┴───o o─┴───
    │                     │
    │                     │
   (+)                  (~~~) L
    │  Vs                  │
   (-)                     │
    │                     │
    └─────────────────────┘

    t<0: 开关闭合，Vs 供电，电感电流稳定在 I0 = Vs/R
    t=0: 开关断开，电感通过R放电（电流衰减）
```

> 注意：电感和电阻的"放电回路"有点像 RC 放电——RC 是电容通过 R 放掉电荷，RL 是电感通过 R 放掉磁能。但关键区别：RC 放电关注的是**电压**衰减，RL 放电关注的是**电流**衰减。

### 2b. 列微分方程

$t>0$ 时电源断开，R 和 L 串联成一个回路。根据 KVL：

$$
v_L(t) + v_R(t) = 0
$$

代入 $v_L = L \cdot di_L/dt$ 和 $v_R = i_L \cdot R$（串联回路中电流相同）：

$$
L \frac{di_L}{dt} + R \, i_L(t) = 0
$$

除以 $L$ 整理成标准形式：

$$
\frac{di_L}{dt} + \frac{R}{L} i_L(t) = 0
$$

这和 RC 放电方程 $\frac{dv_C}{dt} + \frac{1}{RC} v_C = 0$ 结构完全相同，只是把 $v_C$ 换成 $i_L$，把 $1/(RC)$ 换成了 $R/L$。

### 2c. 解微分方程

分离变量：

$$
\frac{di_L}{i_L} = -\frac{R}{L} dt
$$

两边积分：

$$
\int_{I_0}^{i_L(t)} \frac{di_L}{i_L} = \int_0^t -\frac{R}{L} dt
$$

得到：

$$
\ln\!\left(\frac{i_L(t)}{I_0}\right) = -\frac{R}{L} t
$$

两边取指数：

$$
i_L(t) = I_0 \cdot e^{-(R/L)t}
$$

### 2d. 引入 RL 时间常数

定义 RL 电路的**时间常数**：

$$
\tau = \frac{L}{R}
$$

单位验证：$L$ 的单位是亨利（H），$R$ 的单位是欧姆（$\Omega$）。亨利的定义：$\text{H} = \text{V} \cdot \text{s} / \text{A}$。所以：

$$
\frac{\text{H}}{\Omega} = \frac{\text{V} \cdot \text{s} / \text{A}}{\text{V} / \text{A}} = \text{s}
$$

单位也是秒！所以 RL 的零输入响应是：

$$
\boxed{i_L(t) = I_0 \cdot e^{-t/\tau}, \quad \tau = \frac{L}{R}}
$$

### 2e. 对比 RC 和 RL 放电

| 项目 | RC 放电 | RL 放电 |
|------|---------|---------|
| 衰减的量 | 电容电压 $v_C$ | 电感电流 $i_L$ |
| 公式 | $v_C(t) = V_0 e^{-t/RC}$ | $i_L(t) = I_0 e^{-(R/L)t}$ |
| 时间常数 | $\tau = RC$ | $\tau = L/R$ |
| 每 $\tau$ 剩多少 | 36.8% | 36.8%（一样！） |
| 5$\tau$ 法则 | 5$\tau$ 基本完成 | 5$\tau$ 基本完成（一样！） |

**关键直觉**：

- RC 中，$R$ 越大→ $\tau$ 越大→放电越**慢**（因为电阻大限制了电流，电荷泄放得慢）
- RL 中，$R$ 越大→ $\tau$ 越小→放电越**快**（因为电阻大消耗能量快，磁能散得快）

RL 的 $\tau$ 表达式里 $R$ 在分母上，所以 $R$ 的作用和 RC 中相反！这是初学者的头号陷阱。

---

## 3. RL 电路的零状态响应（电流建立）

### 3a. 场景：电感初始没有电流，接上直流电源

电路：$V_s$、$R$、$L$ 串联。$t<0$ 时开关断开，电感没有电流。$t=0$ 时闭合开关，直流电源开始"推"电流通过电感。电感因为有惯性，电流不能立刻跳到最终值 $V_s/R$，而是慢慢爬上去。

```
         R
    ───[~~~]────┬────
    │           │
   (+)        (~~~) L
    │  Vs        │
   (-)          │
    │           │
    └───────────┘
```

### 3b. 列微分方程

根据 KVL：

$$
V_s = v_R(t) + v_L(t) = R \cdot i_L(t) + L \frac{di_L}{dt}
$$

整理成标准形式：

$$
\frac{di_L}{dt} + \frac{R}{L} i_L(t) = \frac{V_s}{L}
$$

这个方程和 RC 充电方程 $\frac{dv_C}{dt} + \frac{1}{RC} v_C = \frac{V_s}{RC}$ 结构完全对应。

### 3c. 解微分方程

非齐次方程的全解 = 齐次通解 + 特解。

齐次通解：$i_{L,h}(t) = A \cdot e^{-(R/L)t}$。

特解：$t \to \infty$ 时 $di_L/dt = 0$（电流稳定），代入方程得 $0 + (R/L) i_L = V_s/L$，所以 $i_L = V_s/R$。这就是最终稳态电流（电感变成导线后的电流）。

全解：

$$
i_L(t) = A \cdot e^{-(R/L)t} + \frac{V_s}{R}
$$

初始条件：$i_L(0^+) = 0$（电感电流不能突变，初始为 0）。

$$
0 = A \cdot 1 + \frac{V_s}{R} \quad \Rightarrow \quad A = -\frac{V_s}{R}
$$

代回：

$$
\boxed{i_L(t) = \frac{V_s}{R} \left(1 - e^{-(R/L)t}\right) = I_{\text{final}} \left(1 - e^{-t/\tau}\right)}
$$

其中 $I_{\text{final}} = V_s/R$ 是最终稳态电流，$\tau = L/R$。

### 3d. 电感电压

电感两端的电压也可以求出来。用 $v_L = L \cdot di_L/dt$ 对上面的电流公式求导：

$$
v_L(t) = L \cdot \frac{d}{dt}\left[\frac{V_s}{R}\left(1 - e^{-(R/L)t}\right)\right] = L \cdot \frac{V_s}{R} \cdot \frac{R}{L} e^{-(R/L)t} = V_s \, e^{-t/\tau}
$$

初始时刻 $t=0$，$v_L(0) = V_s$。全部电源电压都降在电感上（因为初始电流为 0，电阻上压降为 0）。随着电流增加，电阻上的压降增大，电感上的压降按指数衰减，最终趋近于 0（稳态时电感相当于导线）。

### 3e. 对比 RC 充电和 RL 充电

| 项目 | RC 充电 | RL 充电 |
|------|---------|---------|
| 上升的量 | $v_C(t)$ | $i_L(t)$ |
| 终值 | $V_s$ | $V_s/R$ |
| 公式 | $v_C(t) = V_s(1 - e^{-t/RC})$ | $i_L(t) = (V_s/R)(1 - e^{-t/(L/R)})$ |
| 63.2% 点 | $t = RC$ 时 $v_C = 0.632V_s$ | $t = L/R$ 时 $i_L = 0.632 \cdot V_s/R$ |
| 另一元件的电压 | $v_R = V_s e^{-t/RC}$ | $v_L = V_s e^{-t/(L/R)}$ |

---

## 4. RL 全响应与三要素法

### 4a. RL 的全响应公式

和 RC 一样，RL 电路如果有初始电流 $I_0 \neq 0$ 又接了电源，就是全响应。利用叠加原理：

$$
i_L(t) = \underbrace{I_0 \cdot e^{-t/\tau}}_{\text{零输入}} + \underbrace{\frac{V_s}{R} \left(1 - e^{-t/\tau}\right)}_{\text{零状态}}
$$

合并：

$$
\boxed{i_L(t) = \frac{V_s}{R} + \left(I_0 - \frac{V_s}{R}\right) e^{-t/\tau}, \quad \tau = \frac{L}{R}}
$$

这和 RC 的全响应公式 $v_C(t) = V_s + (V_0 - V_s) e^{-t/\tau}$ 是一一对应的：

- RC 的 $V_s$（终值电压）↔ RL 的 $V_s/R$（终值电流）
- RC 的 $V_0$（初值电压）↔ RL 的 $I_0$（初值电流）

### 4b. RL 电路的三要素法

和 RC 的三要素法一模一样，公式完全相同：

$$
f(t) = f(\infty) + \left[f(0^+) - f(\infty)\right] e^{-t/\tau}
$$

唯一不同的是 $\tau$ 的求法：

- **RC**：$\tau = R_{th} \cdot C$（从电容看进去的等效电阻 × 电容值）
- **RL**：$\tau = L / R_{th}$（电感值 ÷ 从电感看进去的等效电阻）

求 $R_{th}$ 的方法不变：独立源置零（电压源短路、电流源开路），从储能元件（这里是电感）两端看进去的等效电阻。

求初值 $f(0^+)$ 时的关键原则：**电感电流不能突变**——$i_L(0^+) = i_L(0^-)$。在 $t=0^+$ 的等效电路中，把电感用一个电流源替代（电流值等于 $i_L(0^+)$）。

求终值 $f(\infty)$ 时的关键原则：稳态下电感相当于**短路（导线）**。

### 4c. 三要素法对比总结

| 步骤 | RC 电路 | RL 电路 |
|------|---------|---------|
| 不能突变的量 | $v_C$ | $i_L$ |
| $t=0^+$ 等效 | 电容 → 电压源 | 电感 → 电流源 |
| $t=\infty$ 等效 | 电容 → 断路 | 电感 → 短路 |
| 时间常数 | $\tau = R_{th} \cdot C$ | $\tau = L / R_{th}$ |
| 三要素公式 | 完全一样：$f(\infty) + [f(0^+) - f(\infty)]e^{-t/\tau}$ |

---

## 5. 手算实战

### 5a. 例题 1：RL 零状态响应（电流建立）

**电路**：$V_s = 12\text{ V}$，$R = 4\text{ }\Omega$，$L = 2\text{ H}$（亨利）。R 和 L 串联接在 Vs 两端。$t=0$ 时闭合开关。求 $i_L(t)$ 和 $v_L(t)$，并计算 $t = 0.5\text{ s}$ 时的电流。

**解：**

**第 1 步**：求 $\tau$。

$$
\tau = \frac{L}{R} = \frac{2}{4} = 0.5\text{ s}
$$

**第 2 步**：求初值和终值。

$i_L(0^+) = 0$（初始无电流）。

$i_L(\infty) = V_s / R = 12 / 4 = 3\text{ A}$（稳态，电感为导线，电流完全由 R 决定）。

**第 3 步**：三要素法。

$$
i_L(t) = 3 + (0 - 3) e^{-t/0.5} = 3 \left(1 - e^{-2t}\right)
$$

注意：$1/\tau = 1/0.5 = 2$。

**第 4 步**：求 $t = 0.5\text{ s}$ 时的电流。

$$
i_L(0.5) = 3 \left(1 - e^{-2 \times 0.5}\right) = 3 (1 - e^{-1}) \approx 3 \times 0.632 = 1.896\text{ A}
$$

巧合：$t = 0.5\text{ s} = \tau$，正好是终值的 63.2%。

**第 5 步**：电感电压。

用 KVL：$v_L(t) = V_s - v_R(t) = V_s - R \cdot i_L(t) = 12 - 4 \times 3(1 - e^{-2t}) = 12 - 12(1 - e^{-2t}) = 12 e^{-2t}$。

或者直接用 $v_L = L \cdot di_L/dt$：结果一样。

验证：$t=0$ 时 $v_L(0) = 12\text{ V}$（全部电压在电感上）；$t \to \infty$ 时 $v_L \to 0$。

---

### 5b. 例题 2：RL 零输入响应（电流衰减）

**电路**：一个 $L = 0.1\text{ H}$ 的电感里已经有 $I_0 = 5\text{ A}$ 的稳定电流在流动。$t=0$ 时把电源断开，同时把 $R = 10\text{ }\Omega$ 的电阻并在电感两端形成放电回路。求 $i_L(t)$ 和电阻上的初始电压。

**解：**

**第 1 步**：求 $\tau$。

$$
\tau = \frac{L}{R} = \frac{0.1}{10} = 0.01\text{ s} = 10\text{ ms}
$$

**第 2 步**：初值 $i_L(0^+) = I_0 = 5\text{ A}$（电流不能突变）。

终值 $i_L(\infty) = 0$（磁能耗尽）。

**第 3 步**：三要素法。

$$
i_L(t) = 0 + (5 - 0) e^{-t/0.01} = 5 e^{-100t}
$$

注意指数上 $1/0.01 = 100$。电流衰减得非常快——$5\tau = 50\text{ ms}$ 后仅剩约 0.034 A。

**第 4 步**：电阻初始电压。

$t=0^+$ 时 $i_L = 5\text{ A}$，$v_R(0^+) = i_L \cdot R = 5 \times 10 = 50\text{ V}$！注意这个电压远大于原电源电压（如果有的话）。这也是为什么继电器线圈两端要并联一个"续流二极管"——线圈是电感，断开瞬间会产生很高的感应电压，可能击穿驱动晶体管。二极管给电感电流提供了一个安全的续流回路。

---

### 5c. 例题 3：含分流的 RL 电路（含两个电阻）

**电路**：$V_s = 10\text{ V}$，$R_1 = 2\text{ }\Omega$，$R_2 = 8\text{ }\Omega$，$L = 0.4\text{ H}$。R1 串 Vs 后接电感 L，R2 并在 L 两端。$t=0$ 时闭合开关。求 $i_L(t)$。

```
         R1(2Ω)
    ────[~~~]────┬────
    │            │
   (+)        (~~~) L=0.4H
    │  Vs=10V    │
   (-)        R2(8Ω)
    │            │
    └────────────┴────
```

**解：**

**第 1 步**：求终值 $i_L(\infty)$。稳态下电感短路（导线）。R1 和 R2 并联？不对。电感短路后，电流从 Vs 经 R1 直接到地（电感短接了 R2）。所以 $i_L(\infty) = V_s / R_1 = 10 / 2 = 5\text{ A}$。

**第 2 步**：求 $\tau$。关掉 Vs（短路），从电感两端看进去。Vs 短路后，R1 的上端接地。电感两端"看到"的是 R1 和 R2 并联：

$$
R_{th} = R_1 \parallel R_2 = \frac{2 \times 8}{2+8} = \frac{16}{10} = 1.6\text{ }\Omega
$$

$$
\tau = \frac{L}{R_{th}} = \frac{0.4}{1.6} = 0.25\text{ s}
$$

**第 3 步**：初值 $i_L(0^+) = 0$。

**第 4 步**：三要素法。

$$
i_L(t) = 5 + (0 - 5) e^{-t/0.25} = 5 \left(1 - e^{-4t}\right)
$$

$5\tau = 1.25\text{ s}$ 后基本达到稳态 5 A。

---

### 5d. 例题 4：RL 先建立后衰减（开关切换）

**电路**：$V_s = 20\text{ V}$，$R_1 = 5\text{ }\Omega$，$R_2 = 15\text{ }\Omega$，$L = 0.2\text{ H}$。$t<0$ 时开关在位置 1（Vs 通过 R1 给 L 供电，R2 断开），电感电流已稳定。$t=0$ 时拨到位置 2（Vs 断开，L 通过 R1+R2 串联放电）。求 $i_L(t), t \ge 0$。

**解：**

**第 1 步**：求 $i_L(0^-)$。$t<0$ 时 L 的稳态电流（L 短路）：$i_L(0^-) = V_s / R_1 = 20 / 5 = 4\text{ A}$。

$i_L(0^+) = 4\text{ A}$（电流不能突变）。

**第 2 步**：$t>0$ 时 L 通过 $R_1+R_2 = 20\text{ }\Omega$ 放电。终值 $i_L(\infty) = 0$。

**第 3 步**：求 $\tau$。

$$
\tau = \frac{L}{R_1 + R_2} = \frac{0.2}{20} = 0.01\text{ s} = 10\text{ ms}
$$

**第 4 步**：三要素法。

$$
i_L(t) = 0 + (4 - 0) e^{-t/0.01} = 4 e^{-100t}
$$

$5\tau = 50\text{ ms}$ 后基本放光。

**注意**：$t=0^+$ 时电阻 $R_1 + R_2$ 上的电压为 $4 \times 20 = 80\text{ V}$！这远大于原电源 20 V。这就是电感"反击"的威力——断开电感回路时要小心高压。

---

## 6. 单位阶跃函数 $u(t)$

### 6a. 什么是阶跃函数？

到目前为止，我们描述开关动作都用文字："$t=0$ 闭合开关，$t<0$ 时开关断开"。这在简单电路里没问题，但在复杂电路和各种教科书里，这种文字描述太啰嗦了。数学家和工程师们发明了一个简洁的函数——**单位阶跃函数（Unit Step Function）**，记作 $u(t)$：

$$
u(t) = \begin{cases}
0, & t < 0 \\
1, & t > 0
\end{cases}
$$

在 $t=0$ 这一点，$u(0)$ 通常定义为 $1/2$，但在电路分析中这并不重要（我们只关心 $0^+$ 之后的行为）。

$u(t)$ 就像一扇在 $t=0$ 时刻打开的门。$t<0$ 时值为 0（"没发生"），$t>0$ 时值为 1（"发生了"）。

### 6b. 用 $u(t)$ 表示开关动作

一个在 $t=0$ 时接入的 10 V 直流电源，可以写成 $10\,u(t)\text{ V}$。意思是：$t < 0$ 时电压为 0，$t > 0$ 时电压为 10 V。

同理，一个在 $t=0$ 时断开的电源可以写成 $10[1 - u(t)] = 10\,u(-t)\text{ V}$。

RC 充电的电压响应可以写成：

$$
v_C(t) = V_s \left(1 - e^{-t/RC}\right) u(t)
$$

RL 充电的电流响应可以写成：

$$
i_L(t) = \frac{V_s}{R} \left(1 - e^{-(R/L)t}\right) u(t)
$$

乘以 $u(t)$ 确保 $t<0$ 时响应为零（开关还没合上）。

### 6c. 阶跃响应的概念

一个电路在单位阶跃输入下的输出称为**阶跃响应（Step Response）**。RC 和 RL 电路的充放电过程本质上就是阶跃响应。阶跃响应是测试电路动态特性的标准方法之一——你给电路一个"突然的变化"，看它怎么反应。这就像医生用橡皮锤敲一下你的膝盖，观察膝跳反射。

在控制工程里，阶跃响应是衡量一个系统"快不快""振荡不振荡""有稳态误差吗"的标准手段。你在这一章学的 $1 - e^{-t/\tau}$ 形状就是**一阶系统的阶跃响应**，它是所有动态系统中最基本、最简单的一种。后面学二阶 RLC 振荡时你会看到更丰富的阶跃响应形态：可能有过冲、可能有震荡。

---

### 6d. 延时阶跃和脉冲响应

如果阶跃不是发生在 $t=0$ 而是发生在 $t = t_0$，写成 $u(t - t_0)$：

$$
u(t - t_0) = \begin{cases}
0, & t < t_0 \\
1, & t > t_0
\end{cases}
$$

一个在 $t=2$ 秒时接入的 5 V 电源可以写成 $5\,u(t-2)\text{ V}$。

两个阶跃函数相减可以得到一个**矩形脉冲**：

$$
\text{rect}(t) = u(t) - u(t - T)
$$

这个脉冲在 $0 < t < T$ 之间为 1，其余地方为 0。这在数字电路和信号处理中非常常用。

再延伸一步：**冲激函数** $\delta(t)$ 是阶跃函数的导数：$\delta(t) = du/dt$。冲激函数在 $t=0$ 处面积为 1、宽度为 0、高度为无穷大。虽然听起来很数学化，但在实际中非常有用——你可以把它理解为"极短时间内的巨大冲击"，比如静电放电（ESD）或者一道闪电。

---

## 7. Python 仿真

### 7a. RL 电流建立与衰减

```python
# ============================================================
# RL 电路仿真：电流建立（充电）和衰减（放电）
# 对比不同 L/R 时间常数
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 电路参数
Vs = 10.0        # 电源电压 (V)
R = 10.0          # 电阻 (Ohm)
I_final = Vs / R  # 最终稳态电流 (A)

# 三种不同电感值 (H)，对应不同的时间常数
L_values = [0.5, 2.0, 5.0]
tau_values = [L/R for L in L_values]
colors = ['#2196F3', '#FF9800', '#4CAF50']

t_max = 6 * max(tau_values)
t = np.linspace(0, t_max, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- 左图：电流建立（RL充电） ----
ax1 = axes[0]
for L, tau, color in zip(L_values, tau_values, colors):
    iL_rise = I_final * (1 - np.exp(-t / tau))
    ax1.plot(t, iL_rise, color=color, linewidth=2.5,
             label=f'L = {L} H, tau = {tau:.1f} s')

# 63.2% 标记
v_632 = I_final * (1 - np.exp(-1))
for tau, color in zip(tau_values, colors):
    ax1.axvline(x=tau, color=color, linestyle=':', alpha=0.5)
    ax1.scatter([tau], [v_632], s=40, color=color, zorder=5)

ax1.axhline(y=I_final, color='red', linestyle='--', alpha=0.6, label=f'终值 I_final = {I_final} A')
ax1.set_xlabel('时间 t (s)', fontsize=12)
ax1.set_ylabel('电感电流 i_L (A)', fontsize=12)
ax1.set_title('RL 电流建立：i_L(t) = I_final*(1 - e^{-t/tau})', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ---- 右图：电流衰减（RL放电） ----
ax2 = axes[1]
I0 = 5.0  # 初始电流 (A)
for L, tau, color in zip(L_values, tau_values, colors):
    iL_decay = I0 * np.exp(-t / tau)
    ax2.plot(t, iL_decay, color=color, linewidth=2.5,
             label=f'L = {L} H, tau = {tau:.1f} s')

v_368 = I0 * np.exp(-1)
for tau, color in zip(tau_values, colors):
    ax2.axvline(x=tau, color=color, linestyle=':', alpha=0.5)
    ax2.scatter([tau], [v_368], s=40, color=color, zorder=5)

ax2.set_xlabel('时间 t (s)', fontsize=12)
ax2.set_ylabel('电感电流 i_L (A)', fontsize=12)
ax2.set_title('RL 电流衰减：i_L(t) = I0 * e^{-t/tau}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout(pad=2)
plt.savefig('/tmp/rl_current_response.png', dpi=150, bbox_inches='tight')
plt.show()

print("RL 电流建立与衰减曲线已绘制。")
print("注意：L 越大 -> tau 越大 -> 电流变化越慢（惯性越大）。")
```

预期输出：左图三条上升曲线，右图三条衰减曲线。$L$ 越大（$\tau$ 越大），电流变化越慢。

### 7b. RC vs RL 对比

```python
# ============================================================
# RC 与 RL 的充电/建立过程对比
# 相同的时间常数 tau = 1s
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

tau = 1.0         # 统一时间常数 (s)
Vs = 10.0          # 电源电压 (V)

t = np.linspace(0, 5*tau, 300)

# RC 充电：v_C(t) 上升
vc = Vs * (1 - np.exp(-t/tau))

# RL 充电：i_L(t) 上升（标幺化到 0-1 范围，便于对比）
iL_normalized = 1 - np.exp(-t/tau)

# RC 放电：v_C(t) 衰减（归一化）
vc_decay = np.exp(-t/tau)

# RL 放电：i_L(t) 衰减（归一化）
iL_decay = np.exp(-t/tau)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图：充电/建立（归一化）
ax1 = axes[0]
ax1.plot(t, vc, 'b-', linewidth=2.5, label='RC: v_C(t) / Vs')
# iL 归一化：乘以 Vs 是为了在同图上对比形状
ax1.plot(t, iL_normalized * Vs, 'r--', linewidth=2.5, label='RL: i_L(t)*R / Vs (归一化)')
ax1.axhline(y=Vs, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(x=tau, color='green', linestyle=':', alpha=0.5, label=f'tau = {tau} s')
ax1.set_xlabel('时间 t (s)', fontsize=12)
ax1.set_ylabel('归一化响应', fontsize=12)
ax1.set_title('RC 充电 vs RL 电流建立（tau 相同时完全一致）', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 右图：放电/衰减（归一化）
ax2 = axes[1]
ax2.plot(t, vc_decay * Vs, 'b-', linewidth=2.5, label='RC: v_C(t) / V0 (归一化)')
ax2.plot(t, iL_decay * Vs, 'r--', linewidth=2.5, label='RL: i_L(t) / I0 (归一化)')
ax2.axvline(x=tau, color='green', linestyle=':', alpha=0.5, label=f'tau = {tau} s')
ax2.set_xlabel('时间 t (s)', fontsize=12)
ax2.set_ylabel('归一化响应', fontsize=12)
ax2.set_title('RC 放电 vs RL 电流衰减（tau 相同时完全一致）', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/rc_vs_rl_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("RC vs RL 对比图已绘制。")
print("关键结论：只要 tau 相同，RC 和 RL 的归一化响应曲线完全重合。")
print("区别仅在于：RC 关心电压，RL 关心电流。这是对偶性的直观体现。")
```

预期输出：左右两图都显示 RC 和 RL（归一化后）的曲线完全重合。tau 相同时，指数函数的形状完全一样。

### 7c. R 对 RL 和 RC 的相反影响

```python
# ============================================================
# 关键对比：R 增大对 RC 和 RL 的影响方向相反！
# RC: R 越大 -> tau = RC 越大 -> 充放电越慢
# RL: R 越大 -> tau = L/R 越小 -> 充放电越快
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 固定参数
Vs = 10.0          # 电源电压 (V)
C = 100e-6         # 100 uF
L = 100e-3         # 100 mH

# 三种不同的 R 值
R_values = [10, 100, 1000]  # Ohm

t = np.linspace(0, 0.5, 500)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ---- 左上：RC 充电，不同 R ----
ax = axes[0, 0]
for R in R_values:
    tau = R * C
    vc = Vs * (1 - np.exp(-t / tau))
    ax.plot(t, vc, linewidth=2, label=f'R={R} Ohm, tau={tau*1000:.1f} ms')
ax.axhline(y=Vs, color='gray', linestyle=':', alpha=0.5)
ax.set_title('RC充电：R越大 -> tau越大 -> 越慢', fontsize=13, fontweight='bold')
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('v_C (V)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ---- 右上：RL 充电，不同 R ----
ax = axes[0, 1]
for R in R_values:
    tau = L / R
    i_final = Vs / R
    iL = i_final * (1 - np.exp(-t / tau))
    ax.plot(t, iL, linewidth=2, label=f'R={R} Ohm, tau={tau*1000:.1f} ms')
ax.set_title('RL电流建立：R越大 -> tau越小 -> 越快', fontsize=13, fontweight='bold')
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('i_L (A)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ---- 左下：RC 放电，不同 R ----
ax = axes[1, 0]
V0 = 10.0
for R in R_values:
    tau = R * C
    vc = V0 * np.exp(-t / tau)
    ax.plot(t, vc, linewidth=2, label=f'R={R} Ohm, tau={tau*1000:.1f} ms')
ax.set_title('RC放电：R越大 -> tau越大 -> 越慢', fontsize=13, fontweight='bold')
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('v_C (V)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ---- 右下：RL 放电，不同 R ----
ax = axes[1, 1]
I0 = 1.0
for R in R_values:
    tau = L / R
    iL = I0 * np.exp(-t / tau)
    ax.plot(t, iL, linewidth=2, label=f'R={R} Ohm, tau={tau*1000:.1f} ms')
ax.set_title('RL放电：R越大 -> tau越小 -> 越快', fontsize=13, fontweight='bold')
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('i_L (A)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(pad=2)
plt.savefig('/tmp/rc_vs_rl_r_effect.png', dpi=150, bbox_inches='tight')
plt.show()

print("四张图对比了 R 增大对 RC 和 RL 的影响方向。")
print("核心规律：")
print("  RC: R 在 tau=RC 的分子 -> R大则tau大 -> 慢")
print("  RL: R 在 tau=L/R 的分母 -> R大则tau小 -> 快")
print("这是因为在RC中R限制了电荷搬运，在RL中R加速了磁能耗散。")
```

预期输出：四张子图。上面两张（RC）中 R 越大越慢；下面两张（RL）中 R 越大越快。这种"相反"是初学者的重要区分点。

---

## 8. 应用连接

### 8a. 继电器线圈的续流二极管

继电器内部有一个线圈（电感）。当你给线圈通电时，它产生磁场吸合触点；当你断电时，磁场迅速消失，线圈两端会产生很高的感应电压（几十甚至几百伏），可能击穿驱动它的晶体管。

解决办法：在线圈两端反向并联一个二极管。正常通电时二极管反向截止（不影响电路）；断电时电感产生的反电动势使二极管正向导通，给电流提供一个"续流"回路，把磁能安全消耗掉。这就是你在很多电路板上看到继电器旁边都有一个二极管的原因。

### 8b. DC-DC 开关电源中的电感

几乎所有开关电源（Buck、Boost 转换器）的核心元件就是电感。电源芯片以高频（几十 kHz 到几 MHz）不断通断，电感在每个周期里"充电"（储存磁能）和"放电"（释放磁能），配合电容滤波，实现高效率的电压转换。你在这一章学到的 $di_L/dt = V/L$ 就是开关电源设计的基础公式。

举个例子：在 Buck 转换器中，开关导通时电感两端电压为 $V_{in} - V_{out}$，电流以斜率 $(V_{in} - V_{out})/L$ 线性上升；开关断开时电感两端电压为 $-V_{out}$（通过续流二极管），电流以斜率 $-V_{out}/L$ 线性下降。电感值 $L$ 的选择直接决定了电流纹波的大小。这些都会在后面的电源章节详细展开。

### 8c. 汽车点火线圈

最早的"升压转换器"其实就在你的汽车里。汽车点火线圈本质上就是一个 RL 电路：12 V 电池给一个高感值线圈（初级绕组）通电建立电流。当点火信号触发时，一个机械触点（白金）突然断开电路，线圈里的电流瞬间被切断。

还记得吗？$v_L = L \cdot di_L/dt$。初级绕组里的电流从几安培在微秒级别内降到零，$di_L/dt$ 极其巨大，所以 $v_L$ 可以高达几千伏。这个高压通过匝数比更大的次级绕组进一步升压到 2-3 万伏，在火花塞间隙打出火花，点燃汽油-空气混合物。

这就是电感"反击"的经典应用：用一个低压直流源，产生一个短暂的高压脉冲。

### 8d. RL 电路的非理想因素

实际电感不只是一个纯 L，它还有：

1. **绕线电阻（DCR，DC Resistance）**：电感是用铜线绕的，铜线本身有电阻（从几十 m$\Omega$ 到几 $\Omega$ 不等）。在分析 RL 电路时，这个 DCR 串联在 L 上，会改变 $\tau$。精密计算时 $R_{th}$ 应包含 DCR。

2. **磁芯饱和（Core Saturation）**：铁氧体或铁粉芯电感在大电流下会磁饱和，此时感值 L 急剧下降。这意味着你的 $\tau = L/R$ 在大电流时会变小——电路突然变快。开关电源设计中必须确保电感不会在工作电流范围内饱和。

3. **寄生电容**：电感线圈的匝间存在微小的寄生电容，在高频下会和 L 形成并联谐振。这就是为什么"电感"在某个高频以上反而表现得像个电容。在讨论 kHz 以上的 RL 动态时需要考虑这个效应。

这些细节不会影响你对 RL 基本原理的理解，但当你设计实际电路遇到"为什么和算的不一样"时，知道往哪个方向排查。

---

## 9. 常见误区

### ❌ 误区 1："RL 的 $\tau = L \times R$"

这只是 RC 的公式！RL 的 $\tau = L/R$，$R$ 在**分母**上。$R$ 越大，$\tau$ 越小，放电越快。和 RC 刚好反过来。

✓ 正确：$\tau = L / R_{th}$。单位验证：$\text{H}/\Omega = (\text{V} \cdot \text{s} / \text{A})/(\text{V/A}) = \text{s}$。

### ❌ 误区 2："电感稳态相当于断路"

电容的稳态是断路，电感的稳态是**短路（导线）**。因为稳态电流不变，$di_L/dt = 0$，所以 $v_L = L \cdot 0 = 0$。两端电压为零，相当于导线。千万别搞反。

✓ 电容：$t \to \infty$ 时断路（开路）。电感：$t \to \infty$ 时短路（导线）。

### ❌ 误区 3："电感电流不能突变和电容电压不能突变是两回事"

本质上是同一回事——都是储能元件的"惯性"体现。电容的电场能不能突变→电压不能突变。电感的磁能不能突变→电流不能突变。两者是对偶的，原理完全相同。

### ❌ 误区 4："求 RL 的 $\tau$ 时，$R_{th}$ 的求法和 RC 不一样"

完全一样。都是独立源置零后从储能元件两端看进去的等效电阻。只是 RC 中 $\tau = R_{th} \cdot C$，RL 中 $\tau = L / R_{th}$。

---

## 10. 本章小结：一阶电路统一框架

恭喜！你已经掌握了 RC 和 RL 两种一阶电路。现在把它们放在一个统一框架下回顾，你会发现它们其实是一回事。

**统一的三要素公式**：

$$
f(t) = f(\infty) + \left[f(0^+) - f(\infty)\right] e^{-t/\tau}
$$

这个公式对 RC 和 RL 完全通用。唯一的区别是 $\tau$ 的表达式：

| 电路类型 | 储能元件 | 不能突变量 | $t \to \infty$ 稳态等效 | $\tau$ |
|----------|---------|-----------|------------------------|--------|
| RC | 电容 C | 电压 $v_C$ | 断路 | $R_{th} \cdot C$ |
| RL | 电感 L | 电流 $i_L$ | 短路 | $L / R_{th}$ |

**核心直觉清单**：

1. 一阶电路的响应永远是**指数曲线**，不是直线、不是台阶、不是正弦。
2. 每过一个 $\tau$，当前值与终值的差距缩小到原来的 37%。
3. $5\tau$ 后基本完成（99.3% 的变化）。
4. 对偶性让你只用学一套方法，就能处理两种电路。
5. 求 $\tau$ 的关键永远是"从储能元件看进去的等效电阻 $R_{th}$"。

这些规律不仅适用于 RC 和 RL，也适用于任何一阶系统——无论是电路、热力学、化学反应动力学还是经济模型。指数衰减和指数上升是自然界最普遍的动态行为之一。你现在已经掌握了描述它们的数学工具。

---

## 11. 思考题

### 基础题

**题 1**：$L = 0.2\text{ H}$，$R = 40\text{ }\Omega$。求 RL 电路的时间常数 $\tau$。

**题 2**：一个 RL 电路中 $I_0 = 3\text{ A}$，$\tau = 0.05\text{ s}$。求 $t = 0.1\text{ s}$ 时电感电流（零输入响应）。

**题 3**：$V_s = 24\text{ V}$，$R = 6\text{ }\Omega$，$L = 0.3\text{ H}$ 串联。$t=0$ 闭合开关。写出 $i_L(t)$ 表达式。

### 进阶题

**题 4**：RL 电路中，$R$ 从 $10\text{ }\Omega$ 变成 $100\text{ }\Omega$。$\tau$ 是变大了还是变小了？放电是变快了还是变慢了？解释为什么。

**题 5**：$V_s = 30\text{ V}$，$R_1 = 10\text{ }\Omega$，$R_2 = 15\text{ }\Omega$，$L = 0.5\text{ H}$。$R_1$ 串联 $V_s$ 和 $L$，$R_2$ 并联在 $L$ 两端。$t=0$ 闭合开关。求 $i_L(t)$。

**题 6**：一个 RL 电路中 $L = 0.1\text{ H}$，初始电流 $I_0 = 10\text{ A}$。如果要让电流在 $0.02\text{ s}$ 内衰减到初始值的 5% 以下，R 至少需要多大？

### 综合题

**题 7**：对比 RC 充电和 RL 电流建立过程。如果 RC 的 $\tau = 1\text{ s}$，RL 的 $\tau = 1\text{ s}$，它们充到各自终值的 95% 分别需要多久？答案有区别吗？为什么？

**题 8**：一个 RL 电路中 $L = 0.05\text{ H}$，$R = 2\text{ }\Omega$，初始电流 $I_0 = 6\text{ A}$。求：(a) 电流衰减到 1 A 所需的时间；(b) 在这段时间内电阻上消耗的总能量；(c) 电感中最初存储的磁能（$\frac{1}{2} L I_0^2$）。比较 (b) 和 (c) 的数值并解释。

---

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

#### 题 1

$$
\tau = \frac{L}{R} = \frac{0.2}{40} = 0.005\text{ s} = 5\text{ ms}
$$

**答**：$\tau = 5\text{ ms}$。

---

#### 题 2

放电公式：$i_L(t) = I_0 \cdot e^{-t/\tau}$。

$$
i_L(0.1) = 3 \times e^{-0.1/0.05} = 3 \times e^{-2} \approx 3 \times 0.1353 = 0.406\text{ A}
$$

**答**：约 0.406 A。

---

#### 题 3

$$
\tau = \frac{L}{R} = \frac{0.3}{6} = 0.05\text{ s}
$$

终值：$I_{\text{final}} = V_s / R = 24 / 6 = 4\text{ A}$。

初值：$i_L(0^+) = 0$。

三要素法：

$$
i_L(t) = 4 + (0 - 4) e^{-t/0.05} = 4 \left(1 - e^{-20t}\right)
$$

**答**：$i_L(t) = 4 \left(1 - e^{-20t}\right)\text{ A}$。

---

#### 题 4

原来：$\tau_1 = L/10$。

现在：$\tau_2 = L/100 = \tau_1/10$。

$\tau$ 变小了！放电**变快了**。

解释：在 RL 中 $R$ 在分母上。$R$ 越大意味着耗能越快（$P = I^2 R$ 更大），磁能消耗得更快，所以电流衰减更快。这和 RC 放电相反——RC 中 $R$ 越大，放电越慢（因为 $R$ 限制了电荷泄放的速度）。

**答**：$\tau$ 变小为原来的 $1/10$，放电变快。因为 $R$ 越大 → 磁能耗散越快 → $\tau = L/R$ 越小。

---

#### 题 5

稳态下电感短路。L 短路后 R2 被短路，电流全部经 R1 到地：$i_L(\infty) = V_s / R_1 = 30 / 10 = 3\text{ A}$。

求 $\tau$：关掉 Vs，从 L 看进去 $R_{th} = R_1 \parallel R_2$。

$$
R_{th} = \frac{10 \times 15}{10+15} = \frac{150}{25} = 6\text{ }\Omega
$$

$$
\tau = \frac{L}{R_{th}} = \frac{0.5}{6} \approx 0.0833\text{ s}
$$

$i_L(0^+) = 0$。

三要素法：

$$
i_L(t) = 3 + (0 - 3) e^{-t/0.0833} = 3 \left(1 - e^{-12t}\right)
$$

**答**：$i_L(t) = 3 \left(1 - e^{-12t}\right)\text{ A}$。

---

#### 题 6

要求 $t = 0.02\text{ s}$ 时 $i_L(0.02) \le 0.05 I_0$。

$$
I_0 \cdot e^{-0.02 \cdot R / 0.1} \le 0.05 I_0
$$

$$
e^{-0.2R} \le 0.05
$$

$$
-0.2R \le \ln(0.05) \approx -2.996
$$

$$
R \ge \frac{2.996}{0.2} \approx 14.98\text{ }\Omega
$$

**答**：$R$ 至少需要约 $15\text{ }\Omega$。

验证：$R = 15\text{ }\Omega$ → $\tau = 0.1/15 \approx 6.67\text{ ms}$ → $t=20\text{ ms} \approx 3\tau$ → 剩约 5%，符合要求。

---

#### 题 7

两者都是指数响应。充到终值的 95% 意味着 $1 - e^{-t/\tau} = 0.95$，即 $e^{-t/\tau} = 0.05$。

$$
-\frac{t}{\tau} = \ln(0.05) \approx -2.996
$$

$$
t \approx 3\tau
$$

因为两者的 $\tau = 1\text{ s}$ 相同，所以都需要约 **3 秒**才能充到 95%。RC 和 RL 的响应形状完全一样（都是 $1 - e^{-t/\tau}$ 或 $e^{-t/\tau}$），只是变量不同（RC 是电压，RL 是电流）。这就是对偶性的核心含义。

**答**：都需要约 3 秒（$3\tau$）。没有区别——指数函数的形式由时间常数完全决定，和储能元件的类型（C 还是 L）无关。

---

#### 题 8

**(a) 求衰减到 1 A 的时间**

$$
\tau = \frac{L}{R} = \frac{0.05}{2} = 0.025\text{ s} = 25\text{ ms}
$$

$$
i_L(t) = 6 e^{-t/0.025} = 6 e^{-40t}
$$

求 $i_L = 1$：

$$
1 = 6 e^{-40t} \quad \Rightarrow \quad e^{-40t} = \frac{1}{6} \quad \Rightarrow \quad -40t = \ln\!\left(\frac{1}{6}\right) = -\ln 6
$$

$$
t = \frac{\ln 6}{40} \approx \frac{1.7918}{40} \approx 0.0448\text{ s} = 44.8\text{ ms}
$$

**(b) 电阻耗能**

$$
W_R = \int_0^t i_L^2(\tau) \cdot R \, d\tau = R \int_0^t (6 e^{-40\tau})^2 \, d\tau = 2 \times 36 \int_0^t e^{-80\tau} \, d\tau
$$

$$
= 72 \times \left[-\frac{1}{80} e^{-80\tau}\right]_0^t = 72 \times \frac{1}{80} (1 - e^{-80t}) = 0.9 (1 - e^{-80t})
$$

代入 $t = 0.0448\text{ s}$：$e^{-80 \times 0.0448} = e^{-3.584} \approx 0.0278$。

$$
W_R = 0.9 \times (1 - 0.0278) = 0.9 \times 0.9722 \approx 0.875\text{ J}
$$

**(c) 电感初始储能**

$$
W_L = \frac{1}{2} L I_0^2 = \frac{1}{2} \times 0.05 \times 6^2 = 0.025 \times 36 = 0.9\text{ J}
$$

比较：$W_R \approx 0.875\text{ J}$，$W_L = 0.9\text{ J}$。两者差 0.025 J，恰好等于 $t=0.0448\text{ s}$ 时电感中剩余的磁能：$\frac{1}{2} L \cdot 1^2 = 0.025\text{ J}$。

符合能量守恒：初始磁能 = 电阻消耗 + 电感残留。√

</details>
