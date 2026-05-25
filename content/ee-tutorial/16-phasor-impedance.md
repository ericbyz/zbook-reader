# 第16章：相量法与阻抗 ★本章是交流电路的核心★

> **核心问题**：交流电路有正弦波、有微积分、有相位差——能不能把这些都"打包"，让交流电路变得像直流电路一样简单？

---

## 0. 本章导览

上一章你学会了描述正弦交流电：$v(t) = V_m \sin(\omega t + \varphi)$，三要素（振幅、频率、相位），RMS 有效值。你已经能用这些来分析单个正弦信号了。

但真正的电路不是只有一个信号。一个电路里有电阻、电容、电感，欧姆定律变成微分方程，KVL 变成一堆三角函数的加减……难道每次都要解微分方程、做三角恒等变换？上一章思考题第 7 题你应该尝到了手算两个正弦波相加有多痛苦。

本章介绍的**相量法（Phasor Method）**，就是工程上用来"驯服"交流电路的核武器。它的思路出奇地简单：把旋转的正弦波"冻结"成静止的箭头（相量），把电阻/电容/电感统一成"复数的电阻"（阻抗），然后整个交流电路就退化成了"复数版的直流电路"——欧姆定律、KCL、KVL、节点法、网孔法、戴维南定理，全部原样适用！

这一章是交流电路分析的**绝对核心**。学好这一章，后面功率分析、滤波器、谐振、三相电……全部建立在相量和阻抗之上。

**前置知识**：你已掌握正弦交流电的三要素和相位差（[第 15 章](./15-ac-sinusoidal.md)）、复数运算和极坐标（[复数与相量](./math-complex.md)）、以及直流电路分析的全部方法（第 2-9 章）。如果你对复数的加减乘除、模和幅角、直角坐标与极坐标的转换还不太熟，建议先快速回顾复数那一章。

学完本章，你将能够：

1. 把任意正弦电压/电流写成相量形式
2. 手算 R、L、C 在任意频率下的阻抗
3. 用阻抗 + 相量将交流电路当成"复数直流电路"来分析
4. 画相量图，直观判断电压电流的相位关系
5. 分析 RLC 串联和并联电路的等效阻抗
6. 用 Python 验证相量计算并绘制阻抗-频率曲线
7. 理解为什么相量 + 阻抗让 KCL/KVL/节点法/网孔法/戴维南全部适用

> 本章约 1500 行，建议分 3-4 次读完。第 1-2 节（相量概念和三大元件阻抗）可以一次读完；第 3-4 节（广义阻抗和串联/并联计算）是计算核心，建议慢慢看，每一步都跟着手算；第 5-6 节是实战和练习。

---

## 1. 相量：把"旋转"冻成"箭头"

### 1a. 生活例子：旋转木马的影子

你小时候玩过旋转木马。木马绕中心柱子匀速转圈。太阳从正上方照下来，木马在地上的影子有什么特点？

影子在地上来回移动：木马转到最东边时，影子在最东边；木马转到最西边时，影子在最西边；木马转回中间时，影子在中间。如果你记录影子位置随时间的变化曲线，它恰好是一条**正弦波**。

把这个发现反过来：你看到的任何一个正弦波，都可以想象成"一个匀速旋转的箭头在地面上的投影"。

这个旋转箭头，就是**相量（Phasor）**。

### 1b. 直观理解：冻结旋转

观察一个旋转箭头：它的**长度**是固定的（就是正弦波的振幅 $V_m$），它绕原点**匀速旋转**（角速度 $\omega$），它在 $t = 0$ 时的**起始角度**就是初相位 $\varphi$。

这个旋转箭头包含了一个正弦波的全部信息：振幅、频率、初相位。在任意时刻 $t$，箭头转过的角度是 $\omega t + \varphi$，箭头在垂直轴上的投影就是 $V_m \sin(\omega t + \varphi)$——恰好等于正弦波的瞬时值。

现在关键一步来了：**因为电路中所有正弦波频率都相同（$\omega$ 一样），所有箭头都以相同的角速度同步旋转。它们之间的相对位置——谁超前谁、谁滞后谁——是永恒不变的。**

既然大家都以同样的速度旋转，旋转本身可以"暂不考虑"，只看它们**在某一瞬间的相对位置**。就像拍一张照片：所有箭头都在转，但在快门按下的那一刻，每个箭头停在一个特定的角度——这张"照片"里的箭头集合，就是相量。

> **相量 = 旋转箭头在 $t=0$ 时刻的"快照"。**

频率 $\omega$ 没有消失——它被"记录在案"，但在做加减运算时不需要显式参与。当你需要回到时域波形时，把相量重新"启动旋转"就行了。

### 1c. 形式化定义

一个正弦电压 $v(t) = V_m \sin(\omega t + \varphi)$ 对应的**相量**（记作 $\dot{V}$ 或 $\vec{V}$，本书用 $\dot{V}$）定义为：

$$
\dot{V} = V_m \angle \varphi
$$

其中：
- $V_m$ 是相量的**模（Magnitude）**，等于正弦波的振幅
- $\varphi$ 是相量的**幅角（Angle）**，等于正弦波的初相位

你也可以用复数形式写：

$$
\dot{V} = V_m e^{j\varphi} = V_m (\cos\varphi + j \sin\varphi)
$$

从相量回到时域波形很简单——把相量"重新旋转"并取虚部（因为我们统一用正弦）：

$$
v(t) = \operatorname{Im}\left( \dot{V} e^{j\omega t} \right) = \operatorname{Im}\left( V_m e^{j(\omega t + \varphi)} \right) = V_m \sin(\omega t + \varphi)
$$

如果你习惯用余弦，就取实部（$\operatorname{Re}$）。两种映射只要统一就行。

**重要约定**：在交流电路分析中，相量的模一般用**RMS 值**而不是峰值。也就是说：

$$
\dot{V} = V_{\text{rms}} \angle \varphi
$$

比如 220 V 市电的相量是 $\dot{V} = 220 \angle 0°$ V（注意这是 RMS，峰值是 $220\sqrt{2} \approx 311$ V）。

为什么用 RMS 而不用峰值？因为功率计算。$P = V_{\text{rms}} I_{\text{rms}} \cos\theta$ 直接就是 RMS 相乘，不需要额外的 $\sqrt{2}$ 因子。如果相量的模是峰值，算功率时就要处处除以 2。工程上当然选方便的。

> **本书约定**：从现在起，相量的模统一用 **RMS 值**，除非特别注明。

### 1d. 手算：正弦 → 相量转换

**题 1**：将以下正弦信号转换为相量（用极坐标 $\text{RMS} \angle \varphi$）：

| 时域表达式 | 转换步骤 | 相量 |
|-----------|---------|------|
| $v(t) = 311 \sin(100\pi t)$ V | $V_m = 311$，$V_{\text{rms}} = 220$，$\varphi = 0$ | $\dot{V} = 220 \angle 0°$ V |
| $i(t) = 5 \sin(100\pi t + 30°)$ A | $I_m = 5$，$I_{\text{rms}} = 5/\sqrt{2} \approx 3.54$，$\varphi = 30°$ | $\dot{I} = 3.54 \angle 30°$ A |
| $v(t) = 10 \sin(200\pi t - 45°)$ V | $V_m = 10$，$V_{\text{rms}} \approx 7.07$，$\varphi = -45°$ | $\dot{V} = 7.07 \angle -45°$ V |
| $v(t) = 14.14 \sin(100\pi t + 90°)$ V | $V_m = 14.14$，$V_{\text{rms}} = 10$，$\varphi = 90°$ | $\dot{V} = 10 \angle 90°$ V |

**题 2**：将以下相量还原为时域正弦表达式（设 $\omega = 100\pi$ rad/s）：

| 相量（RMS 模） | 还原步骤 | 时域表达式 |
|---------------|---------|-----------|
| $\dot{V} = 220 \angle 0°$ V | $V_m = 220\sqrt{2}$，$\varphi = 0$ | $v(t) = 311 \sin(100\pi t)$ V |
| $\dot{I} = 2 \angle -60°$ A | $I_m = 2\sqrt{2} \approx 2.83$，$\varphi = -60°$ | $i(t) = 2.83 \sin(100\pi t - 60°)$ A |
| $\dot{V} = 5 \angle 120°$ V | $V_m = 5\sqrt{2} \approx 7.07$，$\varphi = 120°$ | $v(t) = 7.07 \sin(100\pi t + 120°)$ V |

---

## 2. R、L、C 三大元件的阻抗

有了相量把正弦波"冻结"成箭头，接下来要回答的问题是：**在交流电路中，电阻、电容、电感的表现各是什么？** 结果会惊人地统一：每种元件都遵循"电压相量 = 阻抗 × 电流相量"。

### 2a. 生活例子：三种"路障"

这条路有三种"路障"，你对它们都很熟悉了。

- **阻力（电阻 R）**：像沙地。不管你跑多快，阻碍力都一样。你推过去花了力，力没有"存"下来，全变成了热。
- **惯性（电感 L）**：像很重的推车。你想让它加速时它抗拒（电流增大时电感产生反电动势抵抗），你想让它减速时它也抗拒（电流减小时电感释放能量维持电流）。它不消耗能量，而是"不愿意改变状态"。
- **弹性（电容 C）**：像弹簧。你推它的时候它储存能量（充电），你松手的时候它释放能量（放电）。它也不消耗能量，而是"暂存"能量。

用车的类比：
- 电阻：刹车片（消耗能量为热）
- 电感：飞轮（阻碍转速变化，储存动能为惯性）
- 电容：弹簧（储存势能，释放时反弹）

这也就是为什么电阻是"有损耗"的（发热），而电感和电容是"无损耗"的（理想情况下）：电感和电容只是暂时借走能量又还回来，不把能量变成热。

### 2b. 电阻的阻抗：$Z_R = R$

电阻的 V-I 关系是欧姆定律（$v = iR$），与时间和频率都无关。在交流中，电阻上电压和电流的**瞬时关系**依然是：

$$
v(t) = R \cdot i(t)
$$

如果 $i(t) = I_m \sin(\omega t)$，那么：

$$
v(t) = R \cdot I_m \sin(\omega t) = (R I_m) \sin(\omega t)
$$

电压和电流是**同相**的——它们同时过零点、同时到峰值。用相量表示：

$$
\dot{V} = R \cdot \dot{I}
$$

定义电阻的**阻抗（Impedance）** $Z_R$：

$$
Z_R = R
$$

阻抗是一个复数，对电阻来说它是一个**纯实数**（虚部为零）：

$$
Z_R = R + j0
$$

它的模 $|Z_R| = R$，幅角 $\angle Z_R = 0°$（电压和电流同相）。

> 在相量形式下，电阻的欧姆定律精确保留：$\dot{V}_R = Z_R \dot{I}$，且 $V_{\text{rms}} = R \cdot I_{\text{rms}}$。

### 2c. 电感的阻抗：$Z_L = j\omega L$

电感的 V-I 关系是：

$$
v_L(t) = L \frac{di_L}{dt}
$$

电压不是与电流成正比，而是与电流的**变化率**成正比。假设电流是正弦波 $i_L(t) = I_m \sin(\omega t)$：

$$
\begin{aligned}
v_L(t) &= L \frac{d}{dt} \left[ I_m \sin(\omega t) \right] \\[4pt]
       &= L \cdot I_m \omega \cos(\omega t) \\[4pt]
       &= \omega L I_m \sin(\omega t + 90°)
\end{aligned}
$$

发现了什么？电压也是一个同频正弦波，但：
- 振幅是 $\omega L I_m$（不是 $R I_m$，与频率有关！）
- 相位**超前**电流 90°（$\sin(\omega t + 90°)$ 比 $\sin(\omega t)$ 早到峰值）

换句话说：**在电感中，电压超前电流 90°。** 电感抗拒电流的变化，所以当电流变化率最大（过零点时），电压达到峰值；当电流在峰值（变化率为零）时，电压为零。

用相量表示：
- $\dot{I}_L = I_{\text{rms}} \angle 0°$（以电流相位为参考）
- $\dot{V}_L = \omega L \cdot I_{\text{rms}} \angle 90°$

相量除法：

$$
\frac{\dot{V}_L}{\dot{I}_L} = \frac{\omega L \cdot I_{\text{rms}} \angle 90°}{I_{\text{rms}} \angle 0°} = \omega L \angle 90°
$$

而 $\omega L \angle 90°$ 在复数形式下就是 $j\omega L$（因为 $j$ 就是 $1 \angle 90°$）。

因此，电感的**阻抗（Impedance）** $Z_L$ 为：

$$
Z_L = j\omega L = j X_L
$$

其中 $X_L = \omega L = 2\pi f L$ 叫做**感抗（Inductive Reactance）**，单位也是欧姆（Ω）。

| 属性 | 值 |
|------|-----|
| 阻抗 $Z_L$ | $j\omega L$ |
| 模 $|Z_L|$ | $\omega L$ |
| 幅角 $\angle Z_L$ | $+90°$ |
| V-I 相位关系 | V 超前 I 90° |

关键结论：**感抗 $X_L$ 与频率成正比。频率越高，电感对交流的"阻碍"越大。**

- 直流（$f = 0$，$\omega = 0$）：$X_L = 0$，电感对直流相当于短路（理想情况下）
- 高频（$f \to \infty$）：$X_L \to \infty$，电感对高频相当于开路

这就是电感在电路中的核心作用：**通低频、阻高频。**

### 2d. 电容的阻抗：$Z_C = \frac{1}{j\omega C}$

电容的 V-I 关系是：

$$
i_C(t) = C \frac{dv_C}{dt}
$$

假设电压是正弦波 $v_C(t) = V_m \sin(\omega t)$：

$$
\begin{aligned}
i_C(t) &= C \frac{d}{dt} \left[ V_m \sin(\omega t) \right] \\[4pt]
       &= C \cdot V_m \omega \cos(\omega t) \\[4pt]
       &= \omega C V_m \sin(\omega t + 90°)
\end{aligned}
$$

电流也超前电压 90°！或者说，**在电容中，电流超前电压 90°**（电压滞后电流 90°）。

理解这个关系：电容两端的电压不能突变（因为 $v = q/C$，电荷不能瞬间改变），所以电压变化"慢一拍"。当电压过零点（变化率最大）时，充电电流达到峰值；当电压在峰值（变化率为零）时，充电电流为零。

用相量：
- $\dot{V}_C = V_{\text{rms}} \angle 0°$（以电压相位为参考）
- $\dot{I}_C = \omega C V_{\text{rms}} \angle 90°$

阻抗：

$$
\frac{\dot{V}_C}{\dot{I}_C} = \frac{V_{\text{rms}} \angle 0°}{\omega C V_{\text{rms}} \angle 90°} = \frac{1}{\omega C} \angle -90°
$$

$\frac{1}{\omega C} \angle -90°$ 写成复数形式：

$$
\frac{1}{\omega C} \angle -90° = \frac{1}{j\omega C}
$$

（因为 $\frac{1}{j} = -j = 1 \angle -90°$）

因此，电容的**阻抗（Impedance）** $Z_C$ 为：

$$
Z_C = \frac{1}{j\omega C} = -j \frac{1}{\omega C} = -j X_C
$$

其中 $X_C = \frac{1}{\omega C}$ 叫做**容抗（Capacitive Reactance）**，单位也是欧姆（Ω）。

| 属性 | 值 |
|------|-----|
| 阻抗 $Z_C$ | $\frac{1}{j\omega C} = -j X_C$ |
| 模 $|Z_C|$ | $\frac{1}{\omega C}$ |
| 幅角 $\angle Z_C$ | $-90°$ |
| V-I 相位关系 | I 超前 V 90°（V 滞后 I 90°） |

关键结论：**容抗 $X_C$ 与频率成反比。频率越高，电容对交流的"阻碍"越小。**

- 直流（$f = 0$，$\omega = 0$）：$X_C \to \infty$，电容对直流相当于开路（隔直）
- 高频（$f \to \infty$）：$X_C \to 0$，电容对高频相当于短路

这就是电容在电路中的核心作用：**通高频、阻低频、隔直流。**

### 2e. 三大元件阻抗速查表

| 元件 | 时域 V-I 关系 | 阻抗 $Z$ | 模 $|Z|$ | 幅角 | V-I 相位 |
|------|-------------|----------|-----------|------|----------|
| 电阻 R | $v = iR$ | $Z_R = R$ | $R$ | 0° | 同相 |
| 电感 L | $v = L \frac{di}{dt}$ | $Z_L = j\omega L$ | $\omega L$ | +90° | V 超前 I 90° |
| 电容 C | $i = C \frac{dv}{dt}$ | $Z_C = \frac{1}{j\omega C} = -j\frac{1}{\omega C}$ | $\frac{1}{\omega C}$ | -90° | I 超前 V 90°（V 滞后 I 90°） |

> **核心观察**：三种元件在相量域中全部遵循 $\dot{V} = Z \dot{I}$。电阻的阻抗是实数，电感和电容的阻抗是纯虚数。这个统一形式是你用"直流方法"分析交流电路的基础。

### 2f. 手算：阻抗计算

**题 1**：计算以下元件在 $f = 50$ Hz 和 $f = 10$ kHz 下的阻抗（给出复数形式和极坐标形式）：

**(a) 电感 $L = 10$ mH**

$f = 50$ Hz：$\omega = 2\pi \times 50 = 100\pi \approx 314.16$ rad/s

$$
Z_L = j\omega L = j \times 314.16 \times 0.01 = j 3.1416 \ \Omega
$$
$$
Z_L = 3.14 \angle 90° \ \Omega
$$

$f = 10$ kHz：$\omega = 2\pi \times 10000 = 20000\pi \approx 62832$ rad/s

$$
Z_L = j \times 62832 \times 0.01 = j 628.32 \ \Omega
$$
$$
Z_L = 628.32 \angle 90° \ \Omega
$$

频率从 50 Hz 升到 10 kHz（200 倍），感抗也从 3.14 Ω 升到 628 Ω（200 倍）。感抗和频率是线性的。

**(b) 电容 $C = 100$ μF**

$f = 50$ Hz：

$$
\omega C = 314.16 \times 100 \times 10^{-6} = 0.031416
$$

$$
Z_C = \frac{1}{j\omega C} = -j \frac{1}{0.031416} = -j 31.83 \ \Omega
$$
$$
Z_C = 31.83 \angle -90° \ \Omega
$$

$f = 10$ kHz：

$$
\omega C = 62832 \times 100 \times 10^{-6} = 6.2832
$$

$$
Z_C = -j \frac{1}{6.2832} = -j 0.159 \ \Omega
$$
$$
Z_C = 0.159 \angle -90° \ \Omega
$$

频率从 50 Hz 升到 10 kHz（200 倍），容抗从 31.83 Ω 降到 0.159 Ω（1/200 倍）。容抗和频率成反比。

---

**题 2（综合）**：电路中有 $R = 10$ Ω 和 $L = 5$ mH 串联，电流 $i(t) = 2 \sin(1000 t)$ A。求总电压 $v(t)$。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 求角频率 | $\omega = 1000$ rad/s |
| ② | 电流相量（RMS 模） | $\dot{I} = 2/\sqrt{2} \angle 0° = 1.414 \angle 0°$ A |
| ③ | 电阻阻抗 | $Z_R = 10 \ \Omega$ |
| ④ | 电感阻抗 | $Z_L = j\omega L = j \times 1000 \times 0.005 = j5 \ \Omega$ |
| ⑤ | 总阻抗（串联相加） | $Z = Z_R + Z_L = 10 + j5 \ \Omega$ |
| ⑥ | 总阻抗的模和幅角 | $|Z| = \sqrt{10^2 + 5^2} = \sqrt{125} \approx 11.18 \ \Omega$ |
|  |  | $\angle Z = \arctan(5/10) = \arctan(0.5) \approx 26.57°$ |
| ⑦ | 电压相量 $\dot{V} = Z \dot{I}$ | $\dot{V} = (10 + j5) \times 1.414 \angle 0° = 1.414 \times 11.18 \angle 26.57°$ |
| ⑧ | 计算电压相量 | $\dot{V} = 15.81 \angle 26.57°$ V（RMS） |
| ⑨ | 还原时域（峰值 = RMS × √2） | $V_m = 15.81 \times \sqrt{2} \approx 22.36$ V |
| ⑩ | 写出时域表达式 | $v(t) = 22.36 \sin(1000 t + 26.57°)$ V |

**验证**：电压超前电流 26.57°（感性电路，电压超前电流，符合预期）。

---

## 3. 广义阻抗 $Z = R + jX$

### 3a. 形式化定义

一个元件或一段电路的**阻抗（Impedance）** 定义为电压相量与电流相量之比：

$$
Z = \frac{\dot{V}}{\dot{I}}
$$

阻抗是一个复数，一般写成直角坐标形式：

$$
Z = R + jX
$$

其中：
- **实部 $R$**：电阻（Resistance），单位 Ω。对应能量消耗（发热）
- **虚部 $X$**：电抗（Reactance），单位 Ω。对应能量存储（在电感磁场或电容电场中来回交换）
  - $X > 0$：感性电抗（电感贡献）
  - $X < 0$：容性电抗（电容贡献）
  - $X = 0$：纯电阻

也可以写成极坐标形式：

$$
Z = |Z| \angle \theta
$$

其中：
- **模 $|Z| = \sqrt{R^2 + X^2}$**：阻抗的大小，单位 Ω
- **幅角 $\theta = \arctan(X/R)$**：阻抗角（Impedance Angle），也是电压超前电流的相位角

> $\theta > 0$ → 感性（电压超前电流），$\theta < 0$ → 容性（电流超前电压），$\theta = 0$ → 纯电阻（同相）。

### 3b. 阻抗三角形

$R$、$X$、$|Z|$ 三者构成一个直角三角形：

```
        |Z|  ╱|
            ╱  |
          ╱    | X (电抗)
        ╱ θ    |
      ──────────
          R (电阻)
```

- 斜边：$|Z|$（阻抗的模）
- 水平直角边：$R$（电阻）
- 垂直直角边：$X$（电抗，向上为正 = 感性，向下为负 = 容性）
- 夹角 $\theta$：阻抗角

$$
\begin{aligned}
|Z| &= \sqrt{R^2 + X^2} \\[4pt]
R &= |Z| \cos\theta \\[4pt]
X &= |Z| \sin\theta \\[4pt]
\theta &= \arctan\left(\frac{X}{R}\right)
\end{aligned}
$$

### 3c. 导纳 $Y = G + jB$

有时候用阻抗的倒数更方便，尤其是分析并联电路时。**导纳（Admittance）** 定义为：

$$
Y = \frac{1}{Z} = \frac{\dot{I}}{\dot{V}}
$$

导纳也写成直角坐标形式：

$$
Y = G + jB
$$

其中：
- **实部 $G$**：电导（Conductance），单位 S（西门子，旧称 ℧ 姆欧）
- **虚部 $B$**：电纳（Susceptance），单位 S
  - $B > 0$：容性电纳（电容贡献，因为 $Y_C = j\omega C$）
  - $B < 0$：感性电纳（电感贡献，因为 $Y_L = 1/(j\omega L) = -j/(\omega L)$）

对于单个元件：

| 元件 | 阻抗 $Z$ | 导纳 $Y = 1/Z$ |
|------|----------|----------------|
| 电阻 R | $R$ | $G = \frac{1}{R}$ |
| 电感 L | $j\omega L$ | $Y_L = \frac{1}{j\omega L} = -j\frac{1}{\omega L}$ → $B_L = -\frac{1}{\omega L}$ |
| 电容 C | $\frac{1}{j\omega C}$ | $Y_C = j\omega C$ → $B_C = \omega C$ |

导纳在分析并联电路时非常方便，因为并联元件的导纳**直接相加**（就像串联的阻抗直接相加一样）。

### 3d. 手算：阻抗和导纳转换

**题 3**：已知阻抗 $Z = 3 + j4$ Ω。求：$|Z|$、$\theta$、$Y = G + jB$。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 模 | $|Z| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$ Ω |
| ② | 幅角 | $\theta = \arctan(4/3) \approx 53.13°$ |
| ③ | 极坐标形式 | $Z = 5 \angle 53.13°$ Ω |
| ④ | 导纳 $Y = 1/Z$ | $Y = \frac{1}{5 \angle 53.13°} = 0.2 \angle -53.13°$ S |
| ⑤ | 转直角坐标 | $G = 0.2 \cos(-53.13°) = 0.2 \times 0.6 = 0.12$ S |
|  |  | $B = 0.2 \sin(-53.13°) = 0.2 \times (-0.8) = -0.16$ S |
| ⑥ | 导纳直角坐标 | $Y = 0.12 - j 0.16$ S |

也可以用公式直接算：$Y = 1/(3 + j4) = (3 - j4)/(3^2 + 4^2) = (3 - j4)/25 = 0.12 - j0.16$ S。✓

---

## 4. 串联与并联的阻抗计算

### 4a. 核心原理：阻抗的串并联规则和电阻一模一样

这是相量法最强大的特性：

| 连接方式 | 直流电阻 | 交流阻抗 | 规则 |
|----------|---------|---------|------|
| 串联 | $R_{\text{eq}} = R_1 + R_2 + \cdots$ | $Z_{\text{eq}} = Z_1 + Z_2 + \cdots$ | **完全相同！** |
| 并联 | $\frac{1}{R_{\text{eq}}} = \frac{1}{R_1} + \frac{1}{R_2} + \cdots$ | $\frac{1}{Z_{\text{eq}}} = \frac{1}{Z_1} + \frac{1}{Z_2} + \cdots$ | **完全相同！** |

两个阻抗并联的快捷公式：

$$
Z_{\text{eq}} = \frac{Z_1 Z_2}{Z_1 + Z_2}
$$

和电阻的 $R_{\text{eq}} = \frac{R_1 R_2}{R_1 + R_2}$ 完全对应。

> 这意味着你之前学过的所有直流电路分析技巧——串联分压、并联分流、Y-Δ 变换——**全部适用于交流阻抗电路**，只是把实数 $R$ 换成复数 $Z$。

### 4b. 手算：串联 RLC 电路

**题 4（RLC 串联）**：$R = 20$ Ω，$L = 0.1$ H，$C = 50$ μF 串联。电源频率 $f = 50$ Hz，电压 $\dot{V} = 220 \angle 0°$ V（RMS）。求：总阻抗、电流、各元件上的电压。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 角频率 | $\omega = 2\pi \times 50 = 100\pi \approx 314.16$ rad/s |
| ② | 各元件阻抗 | $Z_R = 20$ Ω |
|  |  | $Z_L = j\omega L = j \times 314.16 \times 0.1 = j31.42$ Ω |
|  |  | $Z_C = \frac{1}{j\omega C} = -j\frac{1}{314.16 \times 50 \times 10^{-6}} = -j\frac{1}{0.01571} = -j63.66$ Ω |
| ③ | 总阻抗（串联相加） | $Z = 20 + j31.42 - j63.66 = 20 - j32.24$ Ω |
| ④ | 极坐标 | $|Z| = \sqrt{20^2 + 32.24^2} = \sqrt{400 + 1039} = \sqrt{1439} \approx 37.94$ Ω |
|  |  | $\theta = \arctan(-32.24/20) \approx -58.2°$ |
|  |  | $Z = 37.94 \angle -58.2°$ Ω |
| ⑤ | 电流相量 $\dot{I} = \dot{V} / Z$ | $\dot{I} = \frac{220 \angle 0°}{37.94 \angle -58.2°} = 5.80 \angle 58.2°$ A |
| ⑥ | 各元件电压（相量） | $\dot{V}_R = Z_R \dot{I} = 20 \times 5.80 \angle 58.2° = 116.0 \angle 58.2°$ V |
|  |  | $\dot{V}_L = Z_L \dot{I} = 31.42 \angle 90° \times 5.80 \angle 58.2° = 182.2 \angle 148.2°$ V |
|  |  | $\dot{V}_C = Z_C \dot{I} = 63.66 \angle -90° \times 5.80 \angle 58.2° = 369.2 \angle -31.8°$ V |

**几个重要观察**：

1. 总阻抗角是负的（$-58.2°$）——电容的容抗（63.66 Ω）大于电感的感抗（31.42 Ω），整体呈**容性**。电流超前电压 58.2°。

2. 电阻电压 $\dot{V}_R = 116$ V，电感电压 $\dot{V}_L = 182.2$ V，电容电压 $\dot{V}_C = 369.2$ V。这三个电压的 RMS 值加起来远大于电源的 220 V！这不是违反 KVL——KVL 是**相量相加**，不是 RMS 简单相加。验证：
   $$
   \dot{V}_R + \dot{V}_L + \dot{V}_C = 116 \angle 58.2° + 182.2 \angle 148.2° + 369.2 \angle -31.8°
   $$
   这三个相量的和应当等于 $220 \angle 0°$ V。相量有方向，不是简单数值相加。

3. $V_C = 369.2$ V，远大于电源电压 220 V！这在纯直流电路中不可想象，但在交流 RLC 电路中完全正常——这就是**串联谐振**附近的现象（本章不展开谐振，后面有专门章节）。

### 4c. 手算：并联 RLC 电路

**题 5（RLC 并联）**：同样的元件值（$R = 20$ Ω，$L = 0.1$ H，$C = 50$ μF），改为并联。电源电压 $\dot{V} = 220 \angle 0°$ V，$f = 50$ Hz。求：总导纳、总阻抗、总电流、各支路电流。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 各元件导纳 | $Y_R = 1/R = 1/20 = 0.05$ S |
|  |  | $Y_L = \frac{1}{j\omega L} = -j\frac{1}{31.42} = -j0.0318$ S |
|  |  | $Y_C = j\omega C = j \times 314.16 \times 50 \times 10^{-6} = j0.0157$ S |
| ② | 总导纳（并联相加） | $Y = 0.05 - j0.0318 + j0.0157 = 0.05 - j0.0161$ S |
| ③ | 总导纳极坐标 | $|Y| = \sqrt{0.05^2 + 0.0161^2} \approx 0.0525$ S |
|  |  | $\angle Y = \arctan(-0.0161/0.05) \approx -17.8°$ |
| ④ | 总阻抗 $Z = 1/Y$ | $Z = 1/0.0525 \angle 17.8° = 19.05 \angle 17.8°$ Ω |
| ⑤ | 总电流 $\dot{I} = \dot{V} \cdot Y$ | $\dot{I} = 220 \angle 0° \times 0.0525 \angle -17.8° = 11.55 \angle -17.8°$ A |
| ⑥ | 各支路电流 | $\dot{I}_R = Y_R \dot{V} = 0.05 \times 220 \angle 0° = 11.0 \angle 0°$ A |
|  |  | $\dot{I}_L = Y_L \dot{V} = 0.0318 \angle -90° \times 220 \angle 0° = 7.00 \angle -90°$ A |
|  |  | $\dot{I}_C = Y_C \dot{V} = 0.0157 \angle 90° \times 220 \angle 0° = 3.45 \angle 90°$ A |
| ⑦ | 验证 KCL | $\dot{I}_R + \dot{I}_L + \dot{I}_C = \ ?$ |

验证 KCL（转直角坐标）：
- $\dot{I}_R = 11.0 + j0$
- $\dot{I}_L = 0 - j7.00$
- $\dot{I}_C = 0 + j3.45$

和 $= 11.0 - j3.55$。转极坐标：$|I| = \sqrt{11.0^2 + 3.55^2} \approx 11.56$ A，$\angle = \arctan(-3.55/11.0) \approx -17.9°$。

与 $\dot{I} = 11.55 \angle -17.8°$ A 一致（四舍五入误差）。✓ KCL 在相量域成立！

### 4d. 相量图（Phasor Diagram）

相量图是交流电路分析的**视觉武器**。在复平面上画出各电压和电流相量，相位关系一目了然。

对于上面的 RLC 串联电路：
- 电流 $\dot{I} = 5.80 \angle 58.2°$ —— 画在 58.2° 方向
- $\dot{V}_R$ 与 $\dot{I}$ 同方向（同相）——长度 116
- $\dot{V}_L$ 超前 $\dot{I}$ 90° —— 在 $\dot{I}$ 逆时针转 90° 方向，长度 182.2
- $\dot{V}_C$ 滞后 $\dot{I}$ 90° —— 在 $\dot{I}$ 顺时针转 90° 方向，长度 369.2
- 三个电压相量首尾相接，合成为 $\dot{V} = 220 \angle 0°$

画相量图的步骤：
1. 先画参考相量（通常是电流，串联电路中电流处处相同）
2. 根据各元件 V-I 相位关系，画出各电压相量
3. 按 KVL 首尾相接，验证闭合

相量图让你**不用算**就能定性判断电路行为：在这个 RLC 串联电路中，$\dot{V}_C$ 最长（容抗最大），总电压落后于电流（总阻抗呈容性）。

---

## 5. 核心结论：相量 + 阻抗 = 交流变直流

现在我们可以回答本章的核心问题了：**能不能把交流电路变得像直流一样来算？**

答案是：**能。而且极其简单。**

在直流电阻电路中，你的分析工具箱包括：
- 欧姆定律：$V = IR$
- KCL：$\sum I = 0$
- KVL：$\sum V = 0$
- 串联：$R_{\text{eq}} = R_1 + R_2 + \cdots$
- 并联：$\frac{1}{R_{\text{eq}}} = \frac{1}{R_1} + \frac{1}{R_2} + \cdots$
- 节点电压法
- 网孔电流法
- 戴维南/诺顿等效
- 叠加定理

在交流电路中，将上述所有规则中的：
- 实数电压 $V$ → 复数电压相量 $\dot{V}$
- 实数电流 $I$ → 复数电流相量 $\dot{I}$
- 实数电阻 $R$ → 复数阻抗 $Z$

**全部定理和定律原样适用。**

这是因为：
1. KCL 和 KVL 在相量域中成立（可以由时域 KCL/KVL 的线性性推导出来）
2. 欧姆定律 $\dot{V} = Z \dot{I}$ 在相量域中取和时域 $v = iR$ 完全一样的形式
3. 阻抗的串并联规则来自 KCL/KVL + 欧姆定律，既然前两者成立，串并联也成立
4. 所有网络定理（节点法、网孔法、戴维南、叠加）的推导只依赖 KCL、KVL 和线性元件的 V-I 关系。而这三者在相量域全部保留。因此所有网络定理自动继承。

**唯一的变化**：数字从实数变成复数。你需要会复数加减乘除（参考复数数学基础那一章），以及直角坐标和极坐标之间的转换。其余的，你之前学的所有方法照用不误。

这就是为什么工程上没有人手算正弦波的微分方程——大家都用相量法，把一切都变成复数代数。

### 5b. 快速验证：分压公式

在直流电路中，两个电阻串联的分压公式是：

$$
V_{R2} = V_{\text{总}} \times \frac{R_2}{R_1 + R_2}
$$

在交流电路中，两个阻抗串联的分压公式是：

$$
\dot{V}_{Z2} = \dot{V}_{\text{总}} \times \frac{Z_2}{Z_1 + Z_2}
$$

完全一样，只是 $R$ 换成了 $Z$。我们来验证一下：

**题**：$Z_1 = 10$ Ω（纯电阻），$Z_2 = j20$ Ω（纯电感），电源 $\dot{V} = 100 \angle 0°$ V。求 $\dot{V}_{Z2}$。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 分压公式 | $\dot{V}_{Z2} = 100 \angle 0° \times \frac{j20}{10 + j20}$ |
| ② | 分母极坐标 | $10 + j20 \rightarrow |Z| = \sqrt{100+400} = \sqrt{500} \approx 22.36$，$\angle = \arctan(2) \approx 63.43°$ |
| ③ | 分压比 | $\frac{20 \angle 90°}{22.36 \angle 63.43°} = 0.894 \angle 26.57°$ |
| ④ | $\dot{V}_{Z2}$ | $100 \times 0.894 \angle 26.57° = 89.4 \angle 26.57°$ V |

**验证**：用欧姆定律。$Z = 10 + j20 = 22.36 \angle 63.43°$，$\dot{I} = 100 \angle 0° / 22.36 \angle 63.43° = 4.47 \angle -63.43°$ A。$\dot{V}_{Z2} = Z_2 \dot{I} = j20 \times 4.47 \angle -63.43° = 89.4 \angle (90° - 63.43°) = 89.4 \angle 26.57°$ V。✓

分压公式完全成立。

### 5c. 戴维南定理在相量域中的应用

如果你还记得第 8 章的戴维南定理：任何含源线性二端网络，都可以等效为一个电压源串联一个电阻。在交流电路中，同样适用——只是"电阻"变成"阻抗"，"直流电压源"变成"交流电压相量源"。

**戴维南等效步骤（交流版）**：
1. 开路电压 $\dot{V}_{\text{oc}}$ = 负载端开路时的电压相量
2. 等效阻抗 $Z_{\text{th}}$ = 将所有独立源置零（电压源短路、电流源开路）后，从负载端看进去的阻抗
3. 等效电路 = $\dot{V}_{\text{oc}}$ 串联 $Z_{\text{th}}$

**题（戴维南验证）**：电路中有 $\dot{V}_s = 50 \angle 0°$ V，$Z_1 = 20$ Ω（电阻），$Z_2 = j30$ Ω（电感）。$Z_1$ 和 $Z_2$ 串联在电源上，负载 $Z_L$ 从 $Z_2$ 两端取出。求 $Z_L = 10 - j10$ Ω 时的负载电流（用戴维南定理）。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 开路电压 $\dot{V}_{\text{oc}}$（$Z_L$ 断开） | $\dot{V}_{\text{oc}} = \dot{V}_s \times \frac{Z_2}{Z_1 + Z_2} = 50 \angle 0° \times \frac{j30}{20 + j30}$ |
|  | 分母极坐标 | $20 + j30 \rightarrow \sqrt{400+900} = \sqrt{1300} \approx 36.06$，$\angle = \arctan(30/20) \approx 56.31°$ |
|  | 分压比 | $\frac{30 \angle 90°}{36.06 \angle 56.31°} = 0.832 \angle 33.69°$ |
|  | $\dot{V}_{\text{oc}}$ | $50 \times 0.832 \angle 33.69° = 41.6 \angle 33.69°$ V |
| ② | 等效阻抗 $Z_{\text{th}}$（电压源短路 → $Z_1 \parallel Z_2$） | $Z_{\text{th}} = Z_1 \parallel Z_2 = \frac{Z_1 Z_2}{Z_1 + Z_2} = \frac{20 \times j30}{20 + j30}$ |
|  | 分子 | $j600$ |
|  | 分母 | $20 + j30$（上面算过，$36.06 \angle 56.31°$） |
|  | $Z_{\text{th}}$ | $\frac{600 \angle 90°}{36.06 \angle 56.31°} = 16.64 \angle 33.69°$ Ω |
|  | 转直角坐标 | $Z_{\text{th}} = 16.64 \cos 33.69° + j 16.64 \sin 33.69° = 13.85 + j9.23$ Ω |
| ③ | 戴维南等效电路 | $\dot{V}_{\text{oc}} = 41.6 \angle 33.69°$ V 串联 $Z_{\text{th}} = 13.85 + j9.23$ Ω |
| ④ | 接入负载 $Z_L = 10 - j10$ | 总阻抗 $Z_{\text{total}} = Z_{\text{th}} + Z_L = (13.85 + j9.23) + (10 - j10) = 23.85 - j0.77$ Ω |
| ⑤ | 负载电流 | $\dot{I}_L = \frac{\dot{V}_{\text{oc}}}{Z_{\text{total}}} = \frac{41.6 \angle 33.69°}{23.85 - j0.77}$ |
|  | $|Z_{\text{total}}|$ | $\sqrt{23.85^2 + 0.77^2} \approx 23.86$ Ω |
|  | $\angle Z_{\text{total}}$ | $\arctan(-0.77/23.85) \approx -1.85°$ |
|  | $\dot{I}_L$ | $\frac{41.6}{23.86} \angle (33.69° - (-1.85°)) = 1.743 \angle 35.54°$ A |

**验证（不用戴维南，直接算）**：
$Z_2 \parallel Z_L = \frac{j30 \times (10 - j10)}{j30 + 10 - j10} = \frac{j300 + 300}{10 + j20} = \frac{300(1 + j)}{10(1 + j2)} = \frac{30(1+j)}{1+j2}$

分子分母同乘共轭：$\frac{30(1+j)(1-j2)}{1+4} = \frac{30(1 - j2 + j + 2)}{5} = 6(3 - j) = 18 - j6$

总阻抗：$Z_1 + (Z_2 \parallel Z_L) = 20 + 18 - j6 = 38 - j6$ Ω

总电流：$\dot{I} = 50/(38 - j6) = \frac{50(38+j6)}{38^2+6^2} = \frac{1900 + j300}{1444+36} = \frac{1900 + j300}{1480} = 1.284 + j0.203$ A

分流求 $\dot{I}_L$：$\dot{I}_L = \dot{I} \times \frac{Z_2}{Z_2 + Z_L} = (1.284 + j0.203) \times \frac{j30}{10 + j20}$

$\frac{j30}{10 + j20} = \frac{30 \angle 90°}{22.36 \angle 63.43°} = 1.342 \angle 26.57° = 1.200 + j0.600$

$\dot{I}_L = (1.284 + j0.203)(1.200 + j0.600) = 1.541 + j0.770 + j0.244 - 0.122 = 1.419 + j1.014$

$|\dot{I}_L| = \sqrt{1.419^2 + 1.014^2} \approx 1.744$ A，$\angle \approx 35.5°$。与戴维南结果 $1.743 \angle 35.54°$ 一致。✓

**实战意义**：戴维南定理在交流电路中让你能把复杂网络等效化简，然后只对接入的负载做分析。这在滤波器设计、阻抗匹配、最大功率传输等问题中极其有用。步骤和直流一模一样——开路电压（相量）、等效阻抗（复数）——会一个就会全部。

---

## 6. 完整实战：用相量法做网孔分析

前面你看到了串联和并联的阻抗计算。这一节展示相量法真正的威力：**把网孔电流法照搬到交流电路上**。如果你还记得第 7 章的网孔法，这一节会让你觉得"这不就是换了数字嘛"——没错，就是换了数字。

### 6a. 题目：两网孔交流电路

电路如下（两个网孔，含 RLC 混合）：

```
     R1 = 4Ω     L1 = j3Ω
  ┌───/\/\/\──────(L)────┐
  │                       │
  │     C1 = -j2Ω         │
(+)                      (+)  V2 = 10∠30° V (RMS)
V1 = 20∠0° V            (-)
(-)   R2 = 5Ω           │
  │     /\/\/\            │
  │      │    L2 = j4Ω    │
  └──────┴──────(L)───────┘
```

- 左网孔：$V_1 = 20 \angle 0°$ V（RMS），$R_1 = 4$ Ω，$Z_L = j3$ Ω，$Z_C = -j2$ Ω，$R_2 = 5$ Ω（$R_2$ 在公共支路上）
- 右网孔：$V_2 = 10 \angle 30°$ V（RMS），$R_2 = 5$ Ω（公共），$Z_L = j4$ Ω

设网孔电流 $I_1$（左网孔，顺时针）和 $I_2$（右网孔，顺时针）。

### 6b. 逐步手算

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 左网孔 KVL（顺时针） | $V_1 - I_1 R_1 - I_1 Z_L - I_1 Z_C - (I_1 - I_2)R_2 = 0$ |
| ② | 整理左网孔方程 | $V_1 = I_1(R_1 + Z_L + Z_C + R_2) - I_2 R_2$ |
|  | 代入数值 | $20 \angle 0° = I_1(4 + j3 - j2 + 5) - I_2 \cdot 5$ |
|  |  | $20 \angle 0° = I_1(9 + j1) - 5I_2$ |
| ③ | 右网孔 KVL（顺时针） | $-V_2 - I_2 R_2 - I_2 Z_{L2} - (I_2 - I_1)R_2 = 0$ |
|  | 注意极性 | $V_2$ 在右网孔中遇到的极性与绕行方向相反，记为负 |
| ④ | 整理右网孔方程 | $-V_2 = I_2(R_2 + Z_{L2}) - I_1 R_2$ |
|  | 代入数值 | $-10 \angle 30° = I_2(5 + j4) - I_1 \cdot 5$ |
|  | 移项 | $5I_1 - (5 + j4)I_2 = 10 \angle 30°$ |
| ⑤ | 方程组 | $\begin{cases} (9 + j1) I_1 - 5 I_2 = 20 \angle 0° \\ 5 I_1 - (5 + j4) I_2 = 10 \angle 30° \end{cases}$ |

这是一个二元一次复数方程组。消元法求解：

| 步骤 | 操作 | 结果 |
|------|------|------|
| ⑥ | 由 Eq.1 解出 $I_2$ | $5 I_2 = (9 + j1) I_1 - 20 \angle 0°$ |
|  |  | $I_2 = \frac{(9 + j1) I_1 - 20}{5}$ |
| ⑦ | 代入 Eq.2 | $5 I_1 - (5 + j4) \cdot \frac{(9 + j1) I_1 - 20}{5} = 10 \angle 30°$ |
| ⑧ | 两边乘 5 | $25 I_1 - (5 + j4)[(9 + j1) I_1 - 20] = 50 \angle 30°$ |
| ⑨ | 展开乘积项 | $(5 + j4)(9 + j1) = 45 + j5 + j36 - 4 = 41 + j41$ |
|  |  | $(5 + j4) \times (-20) = -100 - j80$ |
| ⑩ | 代入 | $25 I_1 - [(41 + j41) I_1 - 100 - j80] = 50 \angle 30°$ |
| ⑪ | 合并 $I_1$ 系数 | $25 I_1 - (41 + j41) I_1 + 100 + j80 = 50 \angle 30°$ |
|  |  | $(-16 - j41) I_1 + 100 + j80 = 50 \angle 30°$ |
| ⑫ | 转直角坐标 $50 \angle 30°$ | $50 \cos 30° + j50 \sin 30° = 43.30 + j25.00$ |
| ⑬ | 移项 | $(-16 - j41) I_1 = 43.30 + j25.00 - 100 - j80$ |
|  |  | $(-16 - j41) I_1 = -56.70 - j55.00$ |
| ⑭ | 求解 $I_1$ | $I_1 = \frac{-56.70 - j55.00}{-16 - j41}$ |
|  | 分子分母同乘分母共轭 | $I_1 = \frac{(-56.70 - j55.00)(-16 + j41)}{(-16)^2 + (-41)^2}$ |
|  | 分母 | $256 + 1681 = 1937$ |
|  | 分子展开 | $(-56.70)(-16) + (-56.70)(j41) + (-j55.00)(-16) + (-j55.00)(j41)$ |
|  |  | $= 907.2 - j2324.7 + j880.0 + 2255$ |
|  |  | $= 3162.2 - j1444.7$ |
| ⑮ | 除法 | $I_1 = \frac{3162.2 - j1444.7}{1937} \approx 1.633 - j0.746$ A |
|  | 转极坐标 | $|I_1| = \sqrt{1.633^2 + 0.746^2} \approx 1.796$ A |
|  |  | $\angle I_1 = \arctan(-0.746/1.633) \approx -24.6°$ |
|  |  | $\dot{I}_1 = 1.80 \angle -24.6°$ A |
| ⑯ | 求 $I_2$（回代 Eq.1） | $I_2 = \frac{(9 + j1)(1.633 - j0.746) - 20}{5}$ |
|  | $(9 + j1)(1.633 - j0.746)$ | $= 14.70 + j1.633 - j6.714 - j^2 0.746$ |
|  |  | $= 14.70 + 0.746 - j5.081$ |
|  |  | $= 15.45 - j5.081$ |
|  | 减去 20 | $15.45 - j5.081 - 20 = -4.554 - j5.081$ |
|  | 除以 5 | $I_2 = -0.911 - j1.016$ A |
|  | 转极坐标 | $|I_2| = \sqrt{0.911^2 + 1.016^2} \approx 1.364$ A |
|  |  | $\angle I_2 = \arctan(-1.016/-0.911)$ → 第三象限，约 $-132°$ 或 $228°$ |
|  |  | $\dot{I}_2 = 1.36 \angle -132°$ A（约） |
| ⑰ | 验证（代回 Eq.1） | $(9+j1)(1.633-j0.746) - 5(-0.911-j1.016)$ |
|  |  | $= 15.45 - j5.081 + 4.555 + j5.080$ |
|  |  | $\approx 20.0 + j0 \approx 20 \angle 0°$ ✓ |

### 6c. 实战反思

这个例子展示了几个关键点：

1. **网孔法完全照搬**：写 KVL 方程、设网孔电流、消元求解——步骤和在直流电路中一模一样。唯一的区别是数字是复数而不是实数。一个 $2 \times 2$ 的方程组手工解已经很繁琐，这就是为什么 Python 的 `numpy.linalg.solve` 在交流电路分析中如此重要。

2. **负阻抗是有意义的**：上面的容抗是 $-j2$ Ω（负虚部），它和感抗 $+j3$ Ω 直接代数相加。不要害怕负号——它只是表示相位关系。

3. **角度在第三象限**：$I_2$ 的实部和虚部都是负的（$-0.911 - j1.016$），所以它的幅角在第三象限，用 $\arctan$ 需要加 $180°$。正确的角度是 $\arctan(1.016/0.911) + 180° \approx 48° + 180° = 228°$，等价于 $-132°$。

4. **相量的正负号有物理意义**：$I_2$ 的实部为负，说明 $I_2$ 的实际方向和假设的参考方向（顺时针）相反——这和直流网孔法中的"负电流"完全一样。

### 6d. Python 验证

```python
# 用矩阵法求解两网孔交流电路（复数系数矩阵）
import numpy as np

# 复数系数矩阵 A·I = B
# Eq1: (9+j1)*I1 - 5*I2 = 20
# Eq2: 5*I1 - (5+j4)*I2 = 10∠30°

# 转直角坐标：10∠30° = 10*cos30° + j*10*sin30°
b2 = 10 * (np.cos(np.deg2rad(30)) + 1j * np.sin(np.deg2rad(30)))

A = np.array([
    [9 + 1j,  -5 + 0j],        # Eq1 系数
    [5 + 0j,  -(5 + 4j)]       # Eq2 系数
])
B = np.array([20 + 0j, b2])

# 求解复数线性方程组
I = np.linalg.solve(A, B)

print("=== 两网孔交流电路 矩阵求解 ===")
print(f"I1 = {np.abs(I[0]):.4f} ∠ {np.angle(I[0], deg=True):.2f}° A")
print(f"    直角坐标: {I[0].real:.4f} + j{I[0].imag:.4f} A")
print()
print(f"I2 = {np.abs(I[1]):.4f} ∠ {np.angle(I[1], deg=True):.2f}° A")
print(f"    直角坐标: {I[1].real:.4f} + j{I[1].imag:.4f} A")
print()

# 验证残差
residual = A @ I - B
print(f"验证残差: |A·I - B| = {np.max(np.abs(residual)):.2e}")
```

**预期输出**：
```
=== 两网孔交流电路 矩阵求解 ===
I1 = 1.7958 ∠ -24.63° A
    直角坐标: 1.6326 + j-0.7484 A

I2 = 1.3681 ∠ 228.09° A
    直角坐标: -0.9129 + j-1.0190 A

验证残差: |A·I - B| = 2.22e-15
```

---

## 7. 常见误区

### 误区 1：把阻抗当成实数

- ❌ "$Z_L = \omega L$，所以对于 50 Hz，10 mH 电感的阻抗是 3.14 Ω。"
- ✓ 阻抗是**复数**。$Z_L = j\omega L = j3.14$ Ω，不是 3.14 Ω。丢掉 $j$ 就丢掉了相位信息——电感不只是"阻碍"电流，它让电压**超前**电流 90°。这两个效果同样重要。用阻抗做串并联计算时，如果忘了 $j$，结果完全是错的。

### 误区 2：串联 RLC 中各电压 RMS 直接相加

- ❌ "$V_R = 116$ V，$V_L = 182$ V，$V_C = 369$ V，总和 $116 + 182 + 369 = 667$ V > 220 V，这不可能！"
- ✓ KVL 说的是**相量和为零**，不是 RMS 值直接相加。这三个电压方向不同（$V_R$ 与 I 同相，$V_L$ 超前 I 90°，$V_C$ 滞后 I 90°），它们的相量和等于电源电压 220 V。用 RMS 值直接相加就像"往东走 3 米 + 往北走 4 米 = 走了 7 米"一样荒谬——你实际只离起点 5 米。相量有方向！

### 误区 3：$Z_L$ 和 $Z_C$ 的符号搞反

- ❌ 记成 "$Z_L = -j\omega L$，$Z_C = j/(\omega C)$"
- ✓ 一个口诀帮记忆：**ELI the ICE man**。
  - **ELI**：在电感（L）中，电动势 E（即电压）超前电流 I
  - **ICE**：在电容（C）中，电流 I 超前电动势 E（电压）
  - 超前 = 正角度 → $j$。所以电感阻抗是 $+j\omega L$（V 超前 I → Z 的幅角为正）。
  - 电容阻抗是 $-j/(\omega C)$（I 超前 V → V 滞后 I → Z 的幅角为负）。

### 误区 4：不管频率，随便用阻抗公式

- ❌ "$Z_L = j\omega L$，但这个 $\omega$ 是信号的角频率吗？还是元件自带的什么频率？"
- ✓ $\omega$ 是你正在分析的**信号的角频率**，不是元件的属性。同一个电感，在 50 Hz 时感抗是 $j3.14$ Ω，在 10 kHz 时感抗是 $j628$ Ω。阻抗是**频率的函数**，不是元件的固定属性。只有电阻的阻抗不随频率变化。

### 误区 5：相量的模用峰值和 RMS 混用

- ❌ 前面用 RMS 定义相量，算功率时又在用峰值，弄混。
- ✓ 整章统一用一个标准。本书约定相量的模用 **RMS 值**。好处是计算功率时 $P = V_{\text{rms}} I_{\text{rms}} \cos\theta$ 一步到位。唯一的代价是从相量还原时域波形时要乘 $\sqrt{2}$：$v(t) = \sqrt{2} V_{\text{rms}} \sin(\omega t + \varphi)$。

---

## 8. Python 验证

### 7a. 相量运算可视化

```python
# 相量图：RLC串联电路中各电压和电流的相位关系
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 参数（与手算题4一致）
R, L, C, f = 20, 0.1, 50e-6, 50
omega = 2 * np.pi * f
V_rms = 220

# 计算阻抗
Z_R = R
Z_L = 1j * omega * L
Z_C = 1 / (1j * omega * C)
Z_total = Z_R + Z_L + Z_C

# 计算电流和电压相量
I = V_rms / Z_total  # 复数除法
V_R = Z_R * I
V_L = Z_L * I
V_C = Z_C * I

# 绘图：相量图
fig, ax = plt.subplots(figsize=(8, 8))
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_aspect('equal')

# 绘制各电压相量（从原点出发，用于理解相位关系）
phasors = [
    (I, 'I', '#333333'),
    (V_R, 'V_R', '#e74c3c'),
    (V_L, 'V_L', '#2ecc71'),
    (V_C, 'V_C', '#3498db'),
]

for phasor, label, color in phasors:
    ax.arrow(0, 0, np.real(phasor), np.imag(phasor),
             head_width=4, head_length=6, fc=color, ec=color,
             linewidth=2, alpha=0.8)
    # 标注
    x, y = np.real(phasor), np.imag(phasor)
    if 'V' in label:
        lbl = f'{label}\n{np.abs(phasor):.1f} V'
    else:
        lbl = f'{label}\n{np.abs(phasor):.2f} A'
    ax.annotate(lbl, xy=(x, y), xytext=(x*1.15, y*1.15),
                fontsize=10, color=color, fontweight='bold')

# 用虚线画出电流参考方向
theta = np.angle(I)
ax.plot([0, 200 * np.cos(theta)], [0, 200 * np.sin(theta)],
        'k--', linewidth=0.5, alpha=0.3)

ax.set_xlabel('实部 (V/A)')
ax.set_ylabel('虚部 (V/A)')
ax.set_title(f'RLC串联电路相量图 (f={f} Hz)')
ax.grid(alpha=0.3)
limit = max(np.abs(V_R), np.abs(V_L), np.abs(V_C)) * 1.3
ax.set_xlim([-limit*0.3, limit*1.1])
ax.set_ylim([-limit*0.5, limit*0.8])
plt.tight_layout()
plt.show()

# 打印数值
print(f"总阻抗 Z = {Z_total.real:.2f} + j{Z_total.imag:.2f} = {np.abs(Z_total):.2f}∠{np.angle(Z_total, deg=True):.1f}° Ω")
print(f"电流 I   = {np.abs(I):.2f}∠{np.angle(I, deg=True):.1f}° A")
print(f"V_R       = {np.abs(V_R):.1f}∠{np.angle(V_R, deg=True):.1f}° V")
print(f"V_L       = {np.abs(V_L):.1f}∠{np.angle(V_L, deg=True):.1f}° V")
print(f"V_C       = {np.abs(V_C):.1f}∠{np.angle(V_C, deg=True):.1f}° V")
print(f"V_R + V_L + V_C = {V_R + V_L + V_C:.1f} (应等于 220∠0°)")
```

**预期输出**：
```
总阻抗 Z = 20.00 + j-32.24 = 37.94∠-58.2° Ω
电流 I   = 5.80∠58.2° A
V_R       = 116.0∠58.2° V
V_L       = 182.2∠148.2° V
V_C       = 369.2∠-31.8° V
V_R + V_L + V_C = 220.0+0.0j (应等于 220∠0°)
```

### 7b. 阻抗 vs 频率曲线

```python
# 感抗、容抗随频率变化的对比曲线
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 元件值
L = 0.1   # H
C = 50e-6 # F

# 频率范围：1 Hz 到 10 kHz，对数刻度
f = np.logspace(0, 4, 500)  # 1 Hz ~ 10 kHz
omega = 2 * np.pi * f

# 计算感抗和容抗
XL = omega * L
XC = 1 / (omega * C)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 线性刻度：看清楚低频段
ax1.plot(f, XL, 'r-', linewidth=2, label=f'感抗 X_L (L={L} H)')
ax1.plot(f, XC, 'b-', linewidth=2, label=f'容抗 X_C (C={C*1e6} muF)')
ax1.set_xlabel('频率 (Hz)')
ax1.set_ylabel('电抗 (Omega)')
ax1.set_title('感抗与容抗 vs 频率（线性坐标）')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim([0, 200])
ax1.set_ylim([0, 500])

# 对数刻度：看清宽频率范围的行为
ax2.loglog(f, XL, 'r-', linewidth=2, label=f'感抗 X_L (L={L} H)')
ax2.loglog(f, XC, 'b-', linewidth=2, label=f'容抗 X_C (C={C*1e6} muF)')
# 标注 50 Hz 点
f50 = 50
XL50 = 2 * np.pi * f50 * L
XC50 = 1 / (2 * np.pi * f50 * C)
ax2.loglog(f50, XL50, 'ro', markersize=8)
ax2.loglog(f50, XC50, 'bo', markersize=8)
ax2.annotate(f'50 Hz\nX_L={XL50:.1f} Ohm', xy=(f50, XL50),
             xytext=(80, XL50*2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red')
ax2.annotate(f'50 Hz\nX_C={XC50:.1f} Ohm', xy=(f50, XC50),
             xytext=(80, XC50*0.5),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=9, color='blue')

ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('电抗 (Omega)')
ax2.set_title('感抗与容抗 vs 频率（对数坐标）')
ax2.legend()
ax2.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()
```

**预期输出**：左图（线性坐标）显示低频段感抗和容抗的变化；右图（对数坐标）覆盖 1 Hz 到 10 kHz。感抗（红线）随频率直线上升，容抗（蓝线）随频率反比例下降。在某个频率处两条曲线会相交——这就是谐振频率，后续章节会详细讨论。

### 7c. 相量法验证 KCL

```python
# 验证并联RLC电路中的KCL（复数形式）
import numpy as np

# 参数（与手算题5一致）
R, L, C, f = 20, 0.1, 50e-6, 50
omega = 2 * np.pi * f
V_rms = 220

# 各支路导纳
Y_R = 1 / R
Y_L = 1 / (1j * omega * L)
Y_C = 1j * omega * C

# 各支路电流相量
I_R = Y_R * V_rms
I_L = Y_L * V_rms
I_C = Y_C * V_rms

# 总电流（各支路相量和）
I_KCL = I_R + I_L + I_C

# 总电流（用总导纳计算）
Y_total = Y_R + Y_L + Y_C
I_total = Y_total * V_rms

# 打印对比
print("=== 并联RLC电路 KCL验证 ===")
print(f"I_R = {np.abs(I_R):.3f} Angle {np.angle(I_R, deg=True):.2f} deg. A")
print(f"I_L = {np.abs(I_L):.3f} Angle {np.angle(I_L, deg=True):.2f} deg. A")
print(f"I_C = {np.abs(I_C):.3f} Angle {np.angle(I_C, deg=True):.2f} deg. A")
print()
print(f"I_R + I_L + I_C (KCL) = {np.abs(I_KCL):.3f} Angle {np.angle(I_KCL, deg=True):.2f} deg. A")
print(f"I_total (Y_total*V)      = {np.abs(I_total):.3f} Angle {np.angle(I_total, deg=True):.2f} deg. A")
print(f"差值 |DeltaI| = {np.abs(I_KCL - I_total):.2e} A")
print()
if np.abs(I_KCL - I_total) < 1e-10:
    print("验证结果: KCL成立")
else:
    print("验证结果: KCL不成立")
```

**预期输出**：
```
=== 并联RLC电路 KCL验证 ===
I_R = 11.000 Angle 0.00 deg. A
I_L = 7.003 Angle -90.00 deg. A
I_C = 3.454 Angle 90.00 deg. A

I_R + I_L + I_C (KCL) = 11.554 Angle -17.83 deg. A
I_total (Y_total*V)      = 11.554 Angle -17.83 deg. A
差值 |DeltaI| = 0.00e+00 A

验证结果: KCL成立
```

### 8d. 阻抗模和相位 vs 频率（Bode 图雏形）

```python
# RLC串联电路的阻抗模和阻抗角随频率变化（Bode图的雏形）
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 元件值
R, L, C = 20, 0.1, 50e-6

# 频率范围：对数坐标
f = np.logspace(0, 4, 1000)  # 1 Hz 到 10 kHz
omega = 2 * np.pi * f

# 计算阻抗
Z = R + 1j * omega * L + 1 / (1j * omega * C)
Z_mag = np.abs(Z)        # 阻抗模
Z_phase = np.angle(Z, deg=True)  # 阻抗角（度）

# 找谐振频率（感抗 = 容抗，虚部为零）
f_res = 1 / (2 * np.pi * np.sqrt(L * C))
Z_at_res = np.abs(R + 1j * 2 * np.pi * f_res * L + 1 / (1j * 2 * np.pi * f_res * C))

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 上图：阻抗模（对数刻度）
ax1.loglog(f, Z_mag, 'b-', linewidth=2)
ax1.axvline(x=f_res, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.axhline(y=R, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax1.annotate(f'谐振频率 f₀ = {f_res:.1f} Hz\n|Z| = {Z_at_res:.1f} Ω (最小)',
             xy=(f_res, Z_at_res), xytext=(f_res*2, Z_at_res*5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red')
ax1.set_ylabel('阻抗模 |Z| (Ω)')
ax1.set_title(f'RLC串联电路 频率响应（R={R}Ω, L={L}H, C={C*1e6:.0f}μF）')
ax1.legend(['|Z|'], loc='upper left')
ax1.grid(alpha=0.3, which='both')

# 下图：阻抗角
ax2.semilogx(f, Z_phase, 'r-', linewidth=2)
ax2.axvline(x=f_res, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.axhline(y=90, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
ax2.axhline(y=-90, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
ax2.fill_between(f, Z_phase, 0, where=(Z_phase > 0), color='red', alpha=0.1, label='感性区 (V超前I)')
ax2.fill_between(f, Z_phase, 0, where=(Z_phase < 0), color='blue', alpha=0.1, label='容性区 (I超前V)')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('阻抗角 θ (度)')
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3, which='both')
ax2.set_ylim([-100, 100])

plt.tight_layout()
plt.show()

print(f"谐振频率 f₀ = {f_res:.2f} Hz")
print(f"谐振时: X_L = {2*np.pi*f_res*L:.2f} Ω, X_C = {1/(2*np.pi*f_res*C):.2f} Ω, |Z| = R = {R} Ω")
```

**预期输出**：两幅图。上图：阻抗模 $|Z|$ 在谐振频率处达到最小值 $R$（容抗和感抗抵消）。下图：阻抗角在低频为负（容性），在高频为正（感性），在谐振频率处穿过 0°。这种幅频 + 相频双图就是 Bode 图的雏形，是滤波器分析的核心工具。

---

## 9. DC vs AC 分析方法对照表

为了让"交流 → 直流"的对应关系一目了然，这里给出完整的对照表：

| 分析工具 | 直流电路 | 交流电路（相量域） | 变化 |
|----------|---------|-------------------|------|
| **基本量** | $V$（实数） | $\dot{V}$（复数相量） | 实数 → 复数 |
| **元件特性** | $R$（实数） | $Z$（复数阻抗） | 实数 → 复数 |
| **欧姆定律** | $V = IR$ | $\dot{V} = Z \dot{I}$ | 同形 |
| **KCL** | $\Sigma I = 0$ | $\Sigma \dot{I} = 0$ | 同形 |
| **KVL** | $\Sigma V = 0$ | $\Sigma \dot{V} = 0$ | 同形 |
| **串联** | $R_{\text{eq}} = R_1 + R_2$ | $Z_{\text{eq}} = Z_1 + Z_2$ | 同形 |
| **并联** | $\frac{1}{R_{\text{eq}}} = \frac{1}{R_1} + \frac{1}{R_2}$ | $\frac{1}{Z_{\text{eq}}} = \frac{1}{Z_1} + \frac{1}{Z_2}$ | 同形 |
| **分压** | $V_2 = V \frac{R_2}{R_1+R_2}$ | $\dot{V}_2 = \dot{V} \frac{Z_2}{Z_1+Z_2}$ | 同形 |
| **分流** | $I_2 = I \frac{R_1}{R_1+R_2}$ | $\dot{I}_2 = \dot{I} \frac{Z_1}{Z_1+Z_2}$ | 同形 |
| **节点法** | 实数矩阵 | 复数矩阵 | 同形 |
| **网孔法** | 实数矩阵 | 复数矩阵 | 同形 |
| **戴维南** | $V_{\text{th}}$ + $R_{\text{th}}$ | $\dot{V}_{\text{th}}$ + $Z_{\text{th}}$ | 同形 |
| **叠加定理** | 对每个源单独分析 | 对每个频率分量单独分析 | 同形（但要注意不同频率不能相量叠加） |
| **功率** | $P = VI = I^2R$ | $P = V_{\text{rms}} I_{\text{rms}} \cos\theta$（第 17 章） | 不同 |

**唯一的新东西**：你需要会复数运算。但复数运算就和带小数的算术一样，多练习几次就熟了。不会复数运算不是放弃的理由——它只是一个工具，就像你不会因为不熟计算器就放弃算术一样。

> 学会了直流分析 + 复数运算 = 学会了交流分析。

---

## 10. 应用连接：后续章节的基石

相量和阻抗是你进入交流电路世界的"护照"。从这里开始，之前学的所有直流分析方法——KCL、KVL、节点法、网孔法、戴维南——全部可以用在交流电路上。

- **第 17-18 章（交流功率）**：有功功率 $P = V_{\text{rms}} I_{\text{rms}} \cos\theta$，无功功率 $Q = V_{\text{rms}} I_{\text{rms}} \sin\theta$，功率因数 $\cos\theta$。这里的 $\theta$ 就是阻抗角。
- **第 19 章（频率响应与滤波器）**：用阻抗分析 RC、RL、RLC 电路在不同频率下的行为，画出幅频特性和相频特性（Bode 图）。
- **第 20 章（谐振）**：当 $\omega L = 1/(\omega C)$ 时，感抗和容抗相互抵消，电路进入谐振状态——这只能通过阻抗分析来理解。
- **第 21 章（三相交流电）**：三根相量各差 120°，用相量法分析星形/三角形接法。

如果你对相量和阻抗还不完全熟，建议现在就回头把手算题和 Python 代码全部跑一遍。从现在开始的每一章都建立在本章的基础上。

---

## 11. 思考题

### 基础题

**题 1**：将以下正弦信号写成相量形式（RMS 模，极坐标）：
(a) $v(t) = 170 \sin(377t)$ V (美国市电)
(b) $i(t) = 2\sqrt{2} \sin(100\pi t + 45°)$ A
(c) $v(t) = 311 \sin(100\pi t - 30°)$ V

**题 2**：计算以下元件在 $f = 60$ Hz 和 $f = 1$ MHz 下的阻抗（复数形式）：
(a) $R = 100$ Ω
(b) $L = 47$ μH
(c) $C = 10$ nF

**题 3**：已知阻抗 $Z = 8 + j6$ Ω，求 $|Z|$、$\theta$、$Y = G + jB$。

### 进阶题

**题 4（RLC 串联）**：$R = 30$ Ω，$L = 0.2$ H，$C = 20$ μF 串联接在 $\dot{V} = 120 \angle 0°$ V、$f = 60$ Hz 的电源上。求：总阻抗 $Z$、电流 $\dot{I}$、各元件电压 $\dot{V}_R$、$\dot{V}_L$、$\dot{V}_C$。并判断电路呈感性还是容性。

**题 5（RLC 并联）**：同样的元件值，改为并联。求总导纳 $Y$、总阻抗 $Z$、总电流 $\dot{I}$、各支路电流。

**题 6**：针对题 4 的 RLC 串联电路，画出相量图（以电流为参考），标注各电压相量的大小和方向。

### 综合/思考题

**题 7**：一个包含三个阻抗 $Z_1 = 10 + j20$ Ω，$Z_2 = 5 - j15$ Ω，$Z_3 = 20 + j10$ Ω 的电路，$Z_1$ 和 $Z_2$ 并联后与 $Z_3$ 串联。电源 $\dot{V} = 100 \angle 0°$ V。求：总阻抗、总电流、$Z_1$ 和 $Z_2$ 各自的分流。

**题 8（思考）**：为什么用相量法分析交流电路时，不需要显式地考虑频率 $\omega$（它被"吸收"到阻抗的数值里了）？如果电源中包含两个不同频率的信号（比如 $v(t) = 100 \sin(100\pi t) + 50 \sin(300\pi t)$），相量法还能直接用吗？为什么？

---

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

#### 题 1 解答

| 时域表达式 | RMS 计算 | 相量 |
|-----------|---------|------|
| $v(t) = 170 \sin(377t)$ | $V_{\text{rms}} = 170/\sqrt{2} \approx 120.2$ | $\dot{V} = 120.2 \angle 0°$ V |
| $i(t) = 2\sqrt{2} \sin(100\pi t + 45°)$ | $I_m = 2\sqrt{2}$，$I_{\text{rms}} = 2$ | $\dot{I} = 2 \angle 45°$ A |
| $v(t) = 311 \sin(100\pi t - 30°)$ | $V_{\text{rms}} = 311/\sqrt{2} \approx 220$ | $\dot{V} = 220 \angle -30°$ V |

---

#### 题 2 解答

**(a) $R = 100$ Ω**：阻抗不随频率变化。$Z_R = 100$ Ω（任意频率）。

**(b) $L = 47$ μH**：

$f = 60$ Hz：$\omega = 2\pi \times 60 = 377$ rad/s
$$
Z_L = j \times 377 \times 47 \times 10^{-6} = j0.0177 \ \Omega
$$

$f = 1$ MHz：$\omega = 2\pi \times 10^6 \approx 6.283 \times 10^6$ rad/s
$$
Z_L = j \times 6.283 \times 10^6 \times 47 \times 10^{-6} = j295.3 \ \Omega
$$

从 60 Hz 到 1 MHz，感抗从 0.0177 Ω 涨到 295 Ω——四个数量级的跨越。

**(c) $C = 10$ nF**：

$f = 60$ Hz：$\omega = 377$ rad/s
$$
Z_C = \frac{1}{j \times 377 \times 10 \times 10^{-9}} = -j \frac{1}{3.77 \times 10^{-6}} = -j 265,252 \ \Omega \approx -j265 \ \text{kΩ}
$$

$f = 1$ MHz：$\omega = 6.283 \times 10^6$ rad/s
$$
Z_C = \frac{1}{j \times 6.283 \times 10^6 \times 10 \times 10^{-9}} = -j \frac{1}{0.06283} = -j15.92 \ \Omega
$$

从 60 Hz 到 1 MHz，容抗从 265 kΩ 降到 15.9 Ω。

---

#### 题 3 解答

$$
|Z| = \sqrt{8^2 + 6^2} = \sqrt{64 + 36} = \sqrt{100} = 10 \ \Omega
$$

$$
\theta = \arctan\left(\frac{6}{8}\right) = \arctan(0.75) = 36.87°
$$

$$
Z = 10 \angle 36.87° \ \Omega
$$

导纳 $Y = 1/Z = \frac{1}{10 \angle 36.87°} = 0.1 \angle -36.87°$ S

转直角坐标：
$$
\begin{aligned}
G &= 0.1 \cos(-36.87°) = 0.1 \times 0.8 = 0.08 \ \text{S} \\
B &= 0.1 \sin(-36.87°) = 0.1 \times (-0.6) = -0.06 \ \text{S}
\end{aligned}
$$

$$
Y = 0.08 - j0.06 \ \text{S}
$$

验证：$Y = 1/(8+j6) = (8-j6)/(8^2+6^2) = (8-j6)/100 = 0.08 - j0.06$ S。✓

---

#### 题 4 解答

$f = 60$ Hz，$\omega = 2\pi \times 60 = 377$ rad/s。

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 感抗 | $X_L = \omega L = 377 \times 0.2 = 75.40$ Ω |
| ② | 容抗 | $X_C = 1/(\omega C) = 1/(377 \times 20 \times 10^{-6}) \approx 132.63$ Ω |
| ③ | 各阻抗 | $Z_R = 30$ Ω，$Z_L = j75.40$ Ω，$Z_C = -j132.63$ Ω |
| ④ | 总阻抗 | $Z = 30 + j75.40 - j132.63 = 30 - j57.23$ Ω |
| ⑤ | 极坐标 | $|Z| = \sqrt{30^2 + 57.23^2} \approx 64.61$ Ω |
|  |  | $\theta = \arctan(-57.23/30) \approx -62.3°$ |
| ⑥ | 电流 | $\dot{I} = 120 \angle 0° / 64.61 \angle -62.3° = 1.857 \angle 62.3°$ A |
| ⑦ | $\dot{V}_R$ | $\dot{V}_R = 30 \times 1.857 \angle 62.3° = 55.72 \angle 62.3°$ V |
| ⑧ | $\dot{V}_L$ | $\dot{V}_L = 75.40 \angle 90° \times 1.857 \angle 62.3° = 140.0 \angle 152.3°$ V |
| ⑨ | $\dot{V}_C$ | $\dot{V}_C = 132.63 \angle -90° \times 1.857 \angle 62.3° = 246.3 \angle -27.7°$ V |

$Z = 30 - j57.23$，$X < 0$，电路呈**容性**（容抗大于感抗）。电流超前电压 62.3°。

---

#### 题 5 解答

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | 各导纳 | $Y_R = 1/30 = 0.03333$ S |
|  |  | $Y_L = -j/(\omega L) = -j/75.40 = -j0.01326$ S |
|  |  | $Y_C = j\omega C = j \times 377 \times 20 \times 10^{-6} = j0.00754$ S |
| ② | 总导纳 | $Y = 0.03333 - j0.01326 + j0.00754 = 0.03333 - j0.00572$ S |
| ③ | 极坐标 | $|Y| = \sqrt{0.03333^2 + 0.00572^2} \approx 0.03382$ S |
|  |  | $\angle Y = \arctan(-0.00572/0.03333) \approx -9.74°$ |
| ④ | 总阻抗 | $Z = 1/Y = 1/0.03382 \angle 9.74° = 29.57 \angle 9.74°$ Ω |
| ⑤ | 总电流 | $\dot{I} = \dot{V} \cdot Y = 120 \angle 0° \times 0.03382 \angle -9.74° = 4.058 \angle -9.74°$ A |
| ⑥ | 支路电流 | $\dot{I}_R = Y_R \dot{V} = 0.03333 \times 120 \angle 0° = 4.000 \angle 0°$ A |
|  |  | $\dot{I}_L = Y_L \dot{V} = 0.01326 \angle -90° \times 120 \angle 0° = 1.591 \angle -90°$ A |
|  |  | $\dot{I}_C = Y_C \dot{V} = 0.00754 \angle 90° \times 120 \angle 0° = 0.905 \angle 90°$ A |

---

#### 题 6 解答（相量图描述）

以电流 $\dot{I} = 1.857 \angle 62.3°$ 为参考方向（画在 62.3° 方向，长度按比例）。

各电压相量：
- $\dot{V}_R = 55.7$ V，与 $\dot{I}$ 同方向（62.3°）
- $\dot{V}_L = 140.0$ V，超前 $\dot{I}$ 90°（在 $62.3° + 90° = 152.3°$）
- $\dot{V}_C = 246.3$ V，滞后 $\dot{I}$ 90°（在 $62.3° - 90° = -27.7°$）

三个相量首尾相接：从原点出发画 $\dot{V}_R$，从 $\dot{V}_R$ 的末端接着画 $\dot{V}_L$，从 $\dot{V}_L$ 的末端接着画 $\dot{V}_C$。$\dot{V}_C$ 的末端应该落在 $120 \angle 0°$——即正实轴上 120 V 处。

因为 $X_C > X_L$，$\dot{V}_C$ 比 $\dot{V}_L$ 长，所以合成后落在实轴上偏下的位置（这和电源电压 $\angle 0°$ 一致）。

---

#### 题 7 解答

| 步骤 | 操作 | 结果 |
|------|------|------|
| ① | $Z_1 \parallel Z_2$ | $Z_{12} = \frac{Z_1 Z_2}{Z_1 + Z_2} = \frac{(10+j20)(5-j15)}{(10+j20)+(5-j15)}$ |
| ② | 分子展开 | $(10+j20)(5-j15) = 50 - j150 + j100 + 300 = 350 - j50$ |
| ③ | 分母 | $(10+j20)+(5-j15) = 15 + j5$ |
| ④ | 除法 | $Z_{12} = \frac{350 - j50}{15 + j5} = \frac{(350-j50)(15-j5)}{15^2 + 5^2}$ |
|  |  | $= \frac{5250 - j1750 - j750 - 250}{250} = \frac{5000 - j2500}{250} = 20 - j10$ Ω |
| ⑤ | $Z_{12}$ 与 $Z_3$ 串联 | $Z = (20 - j10) + (20 + j10) = 40 + j0 = 40$ Ω |
| ⑥ | 总电流 | $\dot{I} = 100 \angle 0° / 40 = 2.5 \angle 0°$ A |
| ⑦ | 分流 $I_1 = I \cdot Z_2/(Z_1+Z_2)$ | $\dot{I}_1 = 2.5 \angle 0° \times \frac{5-j15}{15+j5}$ |
|  |  | $\frac{5-j15}{15+j5} = \frac{(5-j15)(15-j5)}{250} = \frac{75 - j25 - j225 - 75}{250} = -j = 1 \angle -90°$ |
|  |  | $\dot{I}_1 = 2.5 \angle -90°$ A |
| ⑧ | $\dot{I}_2 = \dot{I} - \dot{I}_1$ | $\dot{I}_2 = 2.5 - (-j2.5) = 2.5 + j2.5$ |
|  |  | $|\dot{I}_2| = \sqrt{2.5^2 + 2.5^2} = 2.5\sqrt{2} \approx 3.54$ A，$\angle = 45°$ |

这个电路非常巧：并联部分的等效阻抗虚部（$-j10$）恰好与 $Z_3$ 的虚部（$+j10$）相互抵消，总阻抗变成了纯电阻 40 Ω。虽然单个阻抗有电抗，但组合起来"中和"了——这就是阻抗匹配的基本思想。

---

#### 题 8 解答

相量法之所以不需要显式处理频率 $\omega$，是因为它假设电路中**所有信号都是同一频率**。频率被"吸收"到了每个元件的阻抗数值里（$Z_L = j\omega L$，$Z_C = 1/(j\omega C)$）。阻抗本身已经是"在某特定频率下的复数阻值"。当我们做 $\dot{V} = Z \dot{I}$ 运算时，$\omega$ 已经隐含在 $Z$ 的数值里了，不需要再拿出来。

如果电源包含两个不同频率的信号（比如基波 50 Hz 和三次谐波 150 Hz），相量法**不能直接用**。因为：
- 50 Hz 的电压和 150 Hz 的电压不能"相量相加"——它们的旋转速度不同，相对相位不断变化。
- 同一个电感对 50 Hz 和对 150 Hz 的阻抗不同（$j\omega L$ 随频率变化），所以电路对两个频率的响应也不同。

正确的做法是**叠加定理**：分别对每个频率分量单独用相量法分析，然后把时域结果加起来。这也是为什么傅里叶分析（把任意波形拆成不同频率的正弦波）在电路分析中如此重要——对于复杂的非正弦信号，先拆成若干正弦分量，对每个分量用相量法，最后再合成。

</details>

---

> **本章小结**：相量法将正弦交流电"冻结"成复数箭头，阻抗 $Z = R + jX$ 将 R/L/C 三大元件的 V-I 关系统一成 $\dot{V} = Z\dot{I}$。电阻阻抗为实数 $R$（同相），电感阻抗为 $j\omega L$（V 超前 I 90°），电容阻抗为 $1/(j\omega C)$（I 超前 V 90°）。有了相量和阻抗，KCL、KVL、节点法、网孔法、戴维南定理——你在直流电路中学到的所有分析方法——全部原样适用，只需把实数换成复数。这就是交流电路分析的"统一场论"。
>
> **下章预告**：第 17 章将介绍**交流功率**，包括有功功率（真正做功的部分）、无功功率（在电感和电容之间来回"摆"的能量）和视在功率。你会发现，交流电路中的功率不再是简单的 $P = VI$，而需要用 $P = V_{\text{rms}} I_{\text{rms}} \cos\theta$——那个 $\cos\theta$ 就是阻抗角的余弦，叫**功率因数**。
