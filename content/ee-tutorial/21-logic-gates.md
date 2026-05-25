# 第21章：逻辑门电路

> **核心问题**：用什么"积木"来搭建数字世界？这些积木本身是怎么工作的？

---

## 0. 本章导览

上一章我们用布尔代数描述"是"和"否"的推理，这一章我们要把这些抽象的运算变成看得见、摸得着的电路。布尔运算的每一个操作（AND、OR、NOT）都对应一个实际的电路模块，叫**逻辑门**（Logic Gate）。

就像搭乐高积木一样，每个逻辑门就是一块积木。你学会每一块积木长什么样、有什么功能，然后就能用它们拼出任何数字电路——计算器、游戏机、手机芯片，归根结底都是这些积木搭出来的。

学完本章，你将能够：

1. 认出六种基本逻辑门的符号、说出它们的真值表和表达式
2. 理解为什么 NAND 和 NOR 被称为"万能门"，并用 NAND 实现所有基本逻辑
3. 画出 XOR 和 XNOR 的组合逻辑实现
4. 了解 CMOS 晶体管怎么"拼"出一个逻辑门
5. 看懂电气特性参数：传输延迟、扇出、噪声容限
6. 认识 74 系列标准逻辑芯片
7. 用 Python 写一个逻辑门模拟器

> 本章约 1400 行，建议分 3 次读完。画逻辑图是本章的核心手工活。

前置章节：[布尔代数](./20-boolean-algebra.md)
下一章：[组合逻辑电路设计](./22-combinational-design.md)

---

## 1. 六种基本逻辑门

### 1a. 生活例子：积木块和大门

想象你要搭一座城堡。你不会从挖沙子开始，你会先拿出积木块。有的积木是方的，有的是三角的，有的是窗框。每种积木有不同的功能，但拼法都是标准的。

数字电路的设计也是一样。你不用管晶体管怎么工作（那是芯片制造厂的事），你只需要知道每种"逻辑积木"的功能和用法，然后把它们接在一起就行。

逻辑门的命名很有意思。英文里 gate 既是"大门"也是"门控"。想象一道大门，有两个守卫。大门打开（输出=1）的条件可能是"两个守卫都同意"（AND），也可能是"任意一个同意就行"（OR）。这就是逻辑门的原始比喻。

### 1b. 直观理解：NOT、AND、OR 三个基础门

**NOT 门（非门/反相器）**

最简单的门。一个输入、一个输出。输出永远是输入的反面。
- 输入 0 → 输出 1
- 输入 1 → 输出 0

符号：三角形后面带一个小圆圈。小圆圈代表"取反"。
表达式：$Y = \overline{A}$

生活中的 NOT 门：自动门在"没人"的时候关着（输入=0，没检测到人），实际电机是通电锁住的（输出=1，锁住）。逻辑上就是 NOT 的关系。

**AND 门（与门）**

两个（或多个）输入、一个输出。**全 1 才 1。**

| A | B | Y = A·B |
|---|---|---------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

符号：D 形平底，输入从左入、输出从右出。

生活例子：银行保险柜需要两把钥匙同时转动才能打开。一把钥匙不行，必须两把都到位。这就是 AND。

**OR 门（或门）**

两个（或多个）输入、一个输出。**有 1 就 1。**

| A | B | Y = A+B |
|---|---|---------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

符号：弧形尖底（像一个盾牌），输入从左入、输出从右出。

生活例子：走廊灯被两个开关控制，任何一个开关打开灯都亮。

### 1c. 三个"带圈"的门：NAND、NOR、XOR/XNOR

**NAND 门（与非门）**

AND 门的输出再取反。**全 1 才 0。**（和 AND 全部反过来）

| A | B | Y = ¬(A·B) |
|---|---|------------|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

符号：AND 门符号 + 输出端的小圆圈。

NAND 门是数字电路中最重要、最常用的门。为什么？因为用晶体管搭一个 NAND 门比搭一个 AND 门更快、更省面积、更省功耗。所以几乎所有数字芯片的内部都是以 NAND 门为主的。

**NOR 门（或非门）**

OR 门的输出再取反。**有 1 就 0。**

| A | B | Y = ¬(A+B) |
|---|---|------------|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 0 |

符号：OR 门符号 + 输出端的小圆圈。

**XOR 门（异或门）**

两个输入。**相同为 0，不同为 1。**

| A | B | Y = A⊕B |
|---|---|----------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

表达式：$Y = A\overline{B} + \overline{A}B$

生活例子：走廊两端的两个开关控制同一盏灯。任何一个开关改变状态，灯就改变状态（开变关、关变开）。这就是 XOR 逻辑。

**XNOR 门（同或门）**

XOR 的输出再取反。**相同为 1，不同为 0。**

| A | B | Y = A⊙B |
|---|---|----------|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

表达式：$Y = AB + \overline{A}\overline{B}$

### 1d. 逻辑门符号与真值表汇总

完整真值表汇总（两输入情况）：

| A | B | AND | OR | NAND | NOR | XOR | XNOR |
|---|---|-----|----|------|-----|-----|------|
| 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 |
| 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |

符号示意（简笔画）：
```
NOT:  A ──▷∘── Y = ¬A
AND:  A ──┐
          ├── Y = A·B
      B ──┘
OR:   A ──┐
          ├─▷── Y = A+B
      B ──┘
NAND: A ──┐
          ├─▷∘─ Y = ¬(A·B)
      B ──┘
NOR:  A ──┐
          ├─▷∘─ Y = ¬(A+B)
      B ──┘
XOR:  A ──┐
          ├─▷── Y = A⊕B (=1)
      B ──┘
```

### 1e. Python 验证

```python
# 逻辑门模拟器

def gate_not(a): return 1 - a
def gate_and(a, b): return a & b
def gate_or(a, b): return a | b
def gate_nand(a, b): return 1 - (a & b)
def gate_nor(a, b): return 1 - (a | b)
def gate_xor(a, b): return (a & (1-b)) | ((1-a) & b)
def gate_xnor(a, b): return 1 - gate_xor(a, b)

# 生成完整真值表
print("=== 两输入逻辑门真值表 ===")
print("A B | AND  OR  NAND NOR XOR XNOR")
print("-" * 40)
for a in [0, 1]:
    for b in [0, 1]:
        r = [gate_and(a,b), gate_or(a,b), gate_nand(a,b), gate_nor(a,b), gate_xor(a,b), gate_xnor(a,b)]
        print(f"{a} {b} |  {r[0]}    {r[1]}    {r[2]}     {r[3]}    {r[4]}    {r[5]}")
```

### 1f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "NAND = 两个 NOT 加一个 AND" | 虽然功能等价，但实际芯片里 NAND 是一个独立的基本门，不是 NOT+AND 拼出来的。直接用 NAND 更快更省。 |
| "OR 门弧形那边是输出" | 弧形是输入侧，尖头/D 形是输出侧。记住：**信号从左流到右**。 |
| "XOR 门是基本门" | XOR 实际上是用 AND、OR、NOT 组合出来的。没有独立的"XOR 晶体管"，它是个复合门。 |

### 1g. 应用连接

这六个门是你整个数字电路工具箱的全部内容。后面的加法器（第 23 章）、计数器（第 25 章）、状态机（第 26 章），不管多复杂，全是用这几个门拼出来的。

---

## 2. NAND 和 NOR：万能门的秘密

### 2a. 生活例子：只用一种积木搭出所有形状

有一种特殊的乐高套装，里面的积木只有一种形状——长方形。但设计师证明了，只用长方形积木就能拼出正方形、三角形、圆形（近似）。虽然有时候拼起来麻烦一点，但确实能拼。

NAND 门就是数字电路里的"万能长方形积木"。**只用 NAND 门，可以搭出 NOT、AND、OR、NOR、XOR、XNOR 所有门。**同样，只用 NOR 门也可以。

### 2b. 直观理解：为什么 NAND 万能？

NAND 的表达式是 $Y = \overline{AB}$。注意到两点：
1. 当 $A = B$ 时，$Y = \overline{A \cdot A} = \overline{A}$。所以 NAND 接成"两个输入短接"就成了 NOT 门。
2. NOT + NAND 可以变成 AND：$\overline{\overline{AB}} = AB$。在 NAND 后面接一个 NOT（即另一个 NAND 接成的 NOT），就成了 AND 门。
3. 有了 AND 和 NOT，通过德摩根定律，就能得到 OR：$A + B = \overline{\overline{A} \cdot \overline{B}}$。对 A 和 B 各加 NOT，再 NAND，再 NOT，就是 OR。

NOR 为什么也万能？同样的道理，NOR 接成"两个输入短接"就是 NOT，NOT+NOR=OR，德摩根变出 AND。

### 2c. 用 NAND 实现所有门

| 目标门 | NAND 实现方案 | 推导 |
|--------|-------------|------|
| NOT | NAND(A, A) | $\overline{A \cdot A} = \overline{A}$ |
| AND | NOT(NAND(A, B)) | $\overline{\overline{AB}} = AB$ |
| OR | NAND(NOT A, NOT B) | $\overline{\overline{A} \cdot \overline{B}} = A + B$ |
| NOR | NOT(OR) | OR 后再 NOT |
| XOR | 4 个 NAND | 见下文 |
| XNOR | XOR + NOT | 在 XOR 后加 NAND NOT |

**用 4 个 NAND 实现 XOR**（最经典的数字电路习题）：

设 $X = \overline{AB}$（NAND1），$Y = \overline{AX}$（NAND2），$Z = \overline{BX}$（NAND3），输出 $W = \overline{YZ}$（NAND4）。

推导：$W = \overline{\overline{AX} \cdot \overline{BX}} = AX + BX = A\overline{AB} + B\overline{AB}$
$= A(\overline{A} + \overline{B}) + B(\overline{A} + \overline{B}) = A\overline{A} + A\overline{B} + B\overline{A} + B\overline{B}$
$= A\overline{B} + \overline{A}B = A \oplus B$ ✓

**用 NOR 实现所有门**：

| 目标门 | NOR 实现方案 |
|--------|-------------|
| NOT | NOR(A, A) |
| OR | NOT(NOR(A, B)) |
| AND | NOR(NOT A, NOT B) |

### 2d. 手算示例

**例1**：只用 NAND 门实现 $F = AB + CD$。

$F = AB + CD = \overline{\overline{AB} \cdot \overline{CD}}$（德摩根）。

3 个 NAND：NAND1 输出 $\overline{AB}$，NAND2 输出 $\overline{CD}$，NAND3 输入两路得 $\overline{\overline{AB} \cdot \overline{CD}} = AB + CD$ ✓

任何 AND-OR 表达式（SOP 形式）都可以直接转换成两级 NAND 电路：第一级 NAND 产生各积项（带反），第二级 NAND 把它们 AND 再取反，效果等于 OR。

**例2**：只用 NAND 门实现 $F = \overline{A} + BC$。

用德摩根：$\overline{A} + BC = \overline{A \cdot \overline{BC}}$（因为 $\overline{X \cdot \overline{Y}} = \overline{X} + Y$）

验证：令 X=A, Y=BC。$\overline{A \cdot \overline{BC}} = \overline{A} + \overline{\overline{BC}} = \overline{A} + BC$ ✓

所以只需 2 个 NAND：NAND1 输入 B、C 得 $\overline{BC}$，NAND2 输入 A 和 NAND1 输出得 $\overline{A \cdot \overline{BC}} = \overline{A} + BC$。只用 2 个 NAND！

**例3**：只用 NAND 门实现 $F = \overline{A}B + A\overline{B}$（XOR）。前面已推导，4 个 NAND。验证：输入 A=1, B=0。

NAND1: $\overline{1 \cdot 0} = 1$（X=1）
NAND2: $\overline{1 \cdot 1} = 0$（Y=0）
NAND3: $\overline{0 \cdot 1} = 1$（Z=1）
NAND4: $\overline{0 \cdot 1} = 1$（输出=1）✓（XOR(1,0)=1）

### 2e. Python 验证

```python
# 用 NAND 函数验证万能门实现

def nand(a, b): return 1 - (a & b)

def not_from_nand(a): return nand(a, a)
def and_from_nand(a, b): return not_from_nand(nand(a, b))
def or_from_nand(a, b): return nand(not_from_nand(a), not_from_nand(b))

# 4 个 NAND 实现 XOR
def xor_from_nand(a, b):
    x = nand(a, b)
    y = nand(a, x)
    z = nand(b, x)
    return nand(y, z)

print("=== NAND 万能门验证 ===")
for a in [0, 1]:
    for b in [0, 1]:
        print(f"A={a} B={b}: NOT(A)={not_from_nand(a)}(预期{1-a}), AND={and_from_nand(a,b)}(预期{a&b}), OR={or_from_nand(a,b)}(预期{a|b}), XOR={xor_from_nand(a,b)}(预期{a^b})")
```

### 2f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "NAND 门接两次输入是 AND" | NAND(A, A) = $\overline{A}$，是 NOT 不是 AND。 |
| "任何 SOP 表达式转 NAND-NAND 电路需要把每个变量取反" | 不需要。SOP 表达式直接映射到两级 NAND 电路。 |
| "NOR 门也能万能，所以 NAND 和 NOR 完全等价" | 功能上都万能，但 CMOS 实现上 NAND 比 NOR 快（电子迁移率比空穴高），所以实际芯片里 NAND 用得更多。 |

---

## 3. CMOS：逻辑门的"肉身"

### 3a. 生活例子：怎样用开关搭 AND 和 OR？

两个普通开关、一个灯泡、一节电池。怎么接线才能实现"两个开关都闭合才亮灯"？把两个开关**串联**在电路中。任何一个开关断开，电路就不通。这就是一个 AND 门。

如果要求"任意开关闭合就亮灯"呢？把两个开关**并联**。任何一个闭合都有电流通路。这就是一个 OR 门。

逻辑门的物理实现，本质上就是用晶体管做"电子开关"，并按照串联（AND）或并联（OR）的方式连接。

### 3b. CMOS 反相器（NOT 门）

**CMOS**（Complementary Metal-Oxide-Semiconductor）是现代数字芯片的标准工艺。用两种互补晶体管：NMOS（N 沟道）和 PMOS（P 沟道）。

- NMOS：栅极高电压 → 导通；低电压 → 截止
- PMOS：栅极**低**电压 → 导通；高电压 → 截止（和 NMOS 相反）

**CMOS NOT 门**只用两个晶体管：

```
        VDD（电源=1）
          │
    ┌─────┘
    │  PMOS ← 栅极接输入A
    ├───── Y（输出）
    │  NMOS ← 栅极接输入A
    ├─────┘
        GND（地=0）
```

工作分析：
- A=0 → PMOS 导通，NMOS 截止 → Y 通过 PMOS 接 VDD → Y=1 ✓
- A=1 → PMOS 截止，NMOS 导通 → Y 通过 NMOS 接 GND → Y=0 ✓

两个晶体管一个通一个断，永远不会同时通（否则 VDD 和 GND 短路烧芯片）。静态功耗几乎为零（只在翻转瞬间消耗能量）。这就是 CMOS 低功耗的秘密。

### 3c. CMOS NAND 门

用 4 个晶体管：两个 PMOS 并联在上，两个 NMOS 串联在下。

- NMOS 串联在下：A=1 AND B=1 时两个 NMOS 都导通，Y 被拉到 GND（0）。任何一个为 0，NMOS 通路就断了。
- PMOS 并联在上：A=0 OR B=0 时至少一个 PMOS 导通，Y 被拉到 VDD（1）。

两者互补：PMOS 并联（OR 行为）+ NMOS 串联（AND 行为）= NAND。

为什么 CMOS 里 NAND 比 AND 更"自然"？因为 NMOS 是"强者"（导通电阻小），AND 功能天然适合用串联 NMOS 实现（全 1 才通=AND，但输出拉低→NAND）。所以 NAND 是自然的，AND 需要再加一个反相器（多花 2 个晶体管）。

同理，NOR 门是 NMOS 并联 + PMOS 串联。这也是为什么数字芯片内部以 NAND 和 NOR 为主，而非 AND 和 OR。

### 3d. 逻辑门的电气特性

**传输延迟**（Propagation Delay, $t_{pd}$）：信号从输入变化到输出响应所需的时间。通常几个皮秒到几十纳秒（取决于工艺）。门级联越多，总延迟越大。

**扇出**（Fan-out）：一个门能驱动多少个同类门。扇出太大，驱动能力不够，输出电压下降，可能导致逻辑错误。

**扇入**（Fan-in）：一个门最多能接多少个输入。一般不超过 4-8（串联太多晶体管会变慢）。

**噪声容限**（Noise Margin）：输入信号可以偏离理想 0 和 1 多少而不会导致输出错误。

标准 CMOS 的电压阈值（VDD=5V）：
- $V_{IH}$（Input High 最低电压）：约 70% VDD = 3.5V
- $V_{IL}$（Input Low 最高电压）：约 30% VDD = 1.5V
- 1.5V~3.5V 之间 → **不确定区**，输出不可预测（禁止！）

噪声容限越大，电路越抗干扰。

### 3e. 74 系列：标准逻辑芯片

实际搭电路时（面包板实验），不需要自己焊晶体管。有现成的**标准逻辑芯片**。最经典的是 **74 系列**（TTL 工艺）和 **74HC 系列**（高速 CMOS）。

常用 74 系列芯片：

| 型号 | 功能 | 说明 |
|------|------|------|
| 7400 | 四个 2 输入 NAND 门 | 最常用的门芯片 |
| 7402 | 四个 2 输入 NOR 门 | |
| 7404 | 六个 NOT 门 | 反相器 |
| 7408 | 四个 2 输入 AND 门 | |
| 7432 | 四个 2 输入 OR 门 | |
| 7486 | 四个 2 输入 XOR 门 | |
| 74138 | 3-8 译码器 | 第 22 章用到 |
| 7474 | 双 D 触发器 | 第 24 章用到 |

一个 7400 芯片有 14 个引脚：VCC（电源）和 GND（地）各 1 个，剩下 12 个正好是 4 组 × 3 引脚（每组：A 输入、B 输入、Y 输出）。

### 3f. Python 验证

```python
# 逻辑门传输延迟模拟

class LogicGate:
    """带传输延迟的逻辑门"""
    def __init__(self, name, func, delay_ns=10):
        self.name = name
        self.func = func
        self.delay_ns = delay_ns
    def evaluate(self, *inputs):
        return self.func(*inputs)

# 模拟 4 级 NAND 级联电路的总延迟
nand_func = lambda a, b: 1 - (a & b)
g1 = LogicGate("NAND1", nand_func, 8)
g2 = LogicGate("NAND2", nand_func, 8)
g3 = LogicGate("NAND3", nand_func, 8)
g4 = LogicGate("NAND4", nand_func, 8)
total_delay = 4 * 8
print(f"4 级 NAND 级联总延迟: {total_delay} ns")
print(f"等效最大工作频率: {1 / (total_delay * 1e-9) / 1e6:.1f} MHz")

# 扇出模拟
print("\n=== 扇出与信号质量 ===")
for fo in [1, 2, 4, 8, 16]:
    load = 0.08 * fo
    quality = max(0, 1 - load)
    s = "✓ 正常" if quality > 0.5 else ("⚠ 临界" if quality > 0.2 else "✗ 不足")
    print(f"  扇出={fo:2d}: 信号质量={quality:.2f} {s}")
```

### 3g. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "逻辑门输出就是理想的 0V 和 5V" | 实际输出有容差。高电平可能是 4.5-5V，低电平可能是 0-0.5V。 |
| "传输延迟可以忽略" | 在高频电路（GHz 级）中，1 ns 的延迟在 1 GHz 时钟下就是一个完整周期。 |
| "CMOS 门输入悬空相当于 0" | 悬空输入的电平不确定，会导致不稳定行为。所有不用的输入必须接地或接电源。 |

---

## 4. 完整实战

**实战1**：只用 2 输入 NAND 门实现 $F = A\overline{B} + \overline{A}B$（XOR）。画出逻辑图。

<details>
<summary><b>点击展开解答</b></summary>

4 个 NAND：
```
A ──┬──────┐
    │      ├─ NAND1 ── X ──┬──── NAND2 ── Y ──┐
B ──┴──────┘               │                   ├─ NAND4 ── F
                           └──── NAND3 ── Z ──┘
```
NAND1 输出 X = ¬(AB)，NAND2 输出 Y = ¬(AX)，NAND3 输出 Z = ¬(BX)，NAND4 输出 F = ¬(YZ) = A⊕B。
</details>

**实战2**：用 NOR 门实现 XOR。

<details>
<summary><b>点击展开解答</b></summary>

XOR 的一种 NOR 实现（5 个 NOR）：
- NOR1: NOR(A,A) = ¬A
- NOR2: NOR(B,B) = ¬B
- NOR3: NOR(A,B) = ¬(A+B)
- NOR4: NOR(¬A, ¬B) = ¬(¬A+¬B) = AB
- NOR5: NOR(NOR3_out, NOR4_out) = ¬(¬(A+B) + AB) = A⊕B ✓
</details>

---

## 5. 思考题

### 基础题

**题1**：写出以下门电路的逻辑表达式：
(a) 两个 AND 门的输出送入一个 OR 门
(b) 一个 AND 门的两个输入分别是 ¬A 和 B⊕C

**题2**：只用 2 输入 NAND 门实现 $F = AB + CD$（两级 NAND-NAND）。

**题3**：画出 CMOS NAND 门的晶体管级电路，标注所有晶体管类型和连接。当 A=1, B=0 时分析每个晶体管的状态并确定输出。

### 进阶题

**题4**：只用 2 输入 NOR 门实现 $F = AB$（用德摩根）。

**题5**：证明 $AB + \overline{A}C + BC$ 可以用三级 NAND 实现（提示：用 SOP→NAND-NAND 方法）。

**题6**：一个 4 输入 AND 门用 2 输入 AND 门级联实现，需要几个门？如果用 NAND 实现呢？

### 综合题

**题7**：只用 2 输入 NAND 门实现一个 3 输入 XOR：$F = A \oplus B \oplus C$。验证你的设计。

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

**题1**(a)：$F = A_1 B_1 + A_2 B_2$。(b)：$F = \overline{A} \cdot (B \oplus C)$。

**题2**：3 个 NAND。NAND1: A,B → ¬(AB)。NAND2: C,D → ¬(CD)。NAND3: ¬(AB), ¬(CD) → ¬(¬(AB)·¬(CD)) = AB+CD。

**题3**：CMOS NAND：两个 PMOS 并联（栅极 A 和 B），两个 NMOS 串联（栅极 A 和 B）。A=1,B=0：A-PMOS 断，B-PMOS 通→Y 接 VDD；A-NMOS 通，B-NMOS 断→串联断→Y 不接地。结果 Y=1（NAND(1,0)=1 ✓）。

**题4**：$F = AB = \overline{\overline{AB}} = \overline{\overline{A} + \overline{B}}$。所以：NOR1(A,A)=¬A，NOR2(B,B)=¬B，NOR3(¬A,¬B)=¬(¬A+¬B)=AB。3 个 NOR。

**题5**：SOP 转 NAND-NAND：$F = \overline{\overline{AB} \cdot \overline{\overline{A}C}}$（一致律消去 BC 后）。实际上 SOP 三项可以直接用三级：一级产生 ¬(AB)、¬(¬AC)、¬(BC)；二级各做 NAND 产生各积项的反；三级做最终 NAND。

**题6**：4 输入 AND 用 2 输入 AND 级联：3 个 AND 门（树形）。用 NAND：需要前两级产生 ¬(AB) 和 ¬(CD)，第三级 NAND 再做 NAND 得 AB+CD? 不，4 输入 AND 用 NAND：第一级两个 NAND→¬(AB) 和 ¬(CD)，第二级 NAND→¬(¬(AB)·¬(CD))=AB+CD，第三级 NOT(NAND)→AB·CD? 不对。4 输入 AND = ABCD。用 NAND：NAND1(A,B)→¬(AB)，NAND2(C,D)→¬(CD)，NAND3(¬(AB),¬(CD))→AB+CD（这是 OR，不是 AND）。

正确方案：要产生 ABCD。$ABCD = \overline{\overline{AB} \cdot \overline{CD}}$ 的取反 = 4 输入 AND。所以用两个 NAND 产生 ¬(AB) 和 ¬(CD)，第三个 NAND 得 ¬(¬(AB)·¬(CD)) = AB+CD。这得到的是 OR 不是 AND。说明仅靠 2 输入 NAND 实现 4 输入 AND 需要多个门。最简单：3 个 2 输入 AND 级联（树形），或者：NOT(¬(AB)) AND NOT(¬(CD)) 需要 5 个 NAND。

**题7**：$F = A \oplus B \oplus C$。用 8 个 NAND（两级 XOR）：先用 4 个 NAND 实现 $X = A \oplus B$，再用 4 个 NAND 实现 $F = X \oplus C$。总计 8 个。没有更少的标准 2 输入 NAND 方案。
</details>

---

下一章：[组合逻辑电路设计](./22-combinational-design.md)


## 补充专题：用NAND搭建XOR的详细推导

前面提到用4个NAND可以搭出XOR。这里用完全展开的方式一步步推导，确保你每一步都看懂。

**目标**：$F = A \oplus B$，只用2输入NAND门。

**步骤1**：NAND1, 输入A和B, 输出 $X = \overline{AB}$

NAND的定义是NOT(AND)，所以 $X = \overline{A \cdot B}$。如果A=1且B=1，则X=0；否则（只要有任何一个为0），X=1。

**步骤2**：NAND2, 输入A和X, 输出 $Y = \overline{AX}$

代入X：$Y = \overline{A \cdot \overline{AB}} = \overline{A\overline{AB}}$

用德摩根展开里面的部分：$A \cdot \overline{AB} = A(\overline{A} + \overline{B}) = A\overline{A} + A\overline{B} = 0 + A\overline{B} = A\overline{B}$。

所以 $Y = \overline{A\overline{B}}$。

检查四种情况：
- A=0,B=0: $A\overline{B}=0\cdot 1=0$, Y=¬0=1
- A=0,B=1: $A\overline{B}=0\cdot 0=0$, Y=¬0=1
- A=1,B=0: $A\overline{B}=1\cdot 1=1$, Y=¬1=0
- A=1,B=1: $A\overline{B}=1\cdot 0=0$, Y=¬0=1

所以 $Y = \overline{A\overline{B}}$。也就是说，Y=0 **当且仅当** A=1且B=0。

**步骤3**：NAND3, 输入B和X, 输出 $Z = \overline{BX}$

同理：$Z = \overline{B \cdot \overline{AB}} = \overline{B(\overline{A}+\overline{B})} = \overline{B\overline{A} + B\overline{B}} = \overline{\overline{A}B}$。

验证：
- A=0,B=0: $\overline{A}B = 1\cdot 0 = 0$, Z=¬0=1
- A=0,B=1: $\overline{A}B = 1\cdot 1 = 1$, Z=¬1=0
- A=1,B=0: $\overline{A}B = 0\cdot 0 = 0$, Z=¬0=1
- A=1,B=1: $\overline{A}B = 0\cdot 1 = 0$, Z=¬0=1

所以 Z=0 **当且仅当** A=0且B=1。

**步骤4**：NAND4, 输入Y和Z, 输出 $F = \overline{YZ}$

$F = \overline{YZ} = \overline{Y} + \overline{Z}$（德摩根）。

$\overline{Y}$：Y=0 当 A=1,B=0，否则 Y=1。所以 ¬Y=1 当 A=1,B=0，否则 ¬Y=0。即 $\overline{Y} = A\overline{B}$。

$\overline{Z}$：Z=0 当 A=0,B=1，否则 Z=1。所以 ¬Z=1 当 A=0,B=1，否则 ¬Z=0。即 $\overline{Z} = \overline{A}B$。

所以 $F = A\overline{B} + \overline{A}B = A \oplus B$。✓

## 补充专题：CMOS传输门与三态输出

除了基本逻辑门，数字电路中还有一个重要概念叫**三态输出**（Tri-State Output）。普通门的输出要么是0要么是1。三态门多了一个状态：**高阻态**（High Impedance, Hi-Z），相当于"断开连接"。

生活中类比：一个水龙头，要么出水(1)，要么不出水(0)。三态水龙头还有第三种状态：把水龙头拧下来拿走——管道里没有水龙头了，既不出水也不堵水，就是"断开"。

三态输出用一个使能信号 EN 控制：
- EN=1：正常输出（0或1）
- EN=0：高阻态（输出端仿佛不存在）

三态门在总线结构（多个设备共享一根数据线）中非常关键。一条数据线上挂了很多设备，同一时刻只允许一个设备"说话"（输出），其他设备必须处于高阻态（"闭嘴"），否则会产生冲突（短路）。

在CMOS工艺中，三态输出通常用**传输门**（Transmission Gate）实现：一个PMOS和一个NMOS并联，栅极接相反的使能信号。EN=1时两个管子同时导通，信号通过；EN=0时两个管子同时截止，输出端悬空（高阻）。

## 补充实战：用不同门实现同一功能

设计一个电路实现 $F = \overline{AB + \overline{C}}$，分别用：
1. AND、OR、NOT门
2. 只用NAND门
3. 只用NOR门

<details>
<summary><b>点击展开解答</b></summary>

**方案1（AND-OR-NOT）**：
- AND门：输入A和B，输出AB
- NOT门：输入C，输出¬C
- OR门：输入AB和¬C → 输出AB+¬C
- NOT门：取反 → F = ¬(AB+¬C)

**方案2（全NAND）**：
$F = \overline{AB + \overline{C}} = \overline{AB} \cdot C$（德摩根：$\overline{X+Y} = \overline{X} \cdot \overline{Y}$，其中 $\overline{\overline{C}} = C$）

所以：NAND1(A,B) → ¬(AB)，NAND2(¬(AB), C) → ¬(¬(AB)·C) = AB + ¬C … 不对。

重新用德摩根：$\overline{AB + \overline{C}} = \overline{AB} \cdot \overline{\overline{C}} = \overline{AB} \cdot C$。

所以：NAND1 → ¬(AB)。然后我们需要把 ¬(AB) 和 C 做 AND。AND 用 NAND+NOT 实现：NAND(¬(AB), C) → ¬(¬(AB)·C) = AB + ¬C。这是原式的反！

所以要得到 F = ¬(AB+¬C)，需要在 NAND(¬(AB), C) 后再接一个 NOT（NAND 短接）。

全 NAND 方案：NAND1(A,B)→X=¬(AB)，NAND2(X,C)→Y=¬(XC)=AB+¬C，NAND3(Y,Y)→F=¬Y=¬(AB+¬C)。3个NAND。

注意：NAND1 到 NAND2 之间没有取反。X=¬(AB) 直接作为 NAND2 的输入。完美。

**方案3（全NOR）**：
用德摩根把表达式逐步转为 NOR 形式。

$F = \overline{AB + \overline{C}} = \overline{AB} \cdot C = (\overline{A} + \overline{B}) \cdot C$
$= \overline{A}C + \overline{B}C$
$= \overline{\overline{\overline{A}C} \cdot \overline{\overline{B}C}}$（德摩根变 NOR）

$\overline{\overline{A}C} = \overline{\overline{A} + \overline{C}}$ … 这就比较复杂了。全 NOR 实现较繁琐，不如 NAND 直接。
</details>

## 自测练习：逻辑门综合练习题

**练习1**：只用NAND门实现以下函数，并画出逻辑图：
(a) $F = \overline{A}$
(b) $F = AB$
(c) $F = A + B$
(d) $F = \overline{A}B + A\overline{B}$ (XOR)

<details>
<summary><b>点击展开答案</b></summary>

(a) NAND(A, A) = ¬A。1个NAND。
(b) NAND(A, B) = ¬(AB)，再接 NAND(¬(AB), ¬(AB)) = AB。2个NAND。
(c) NAND(¬A, ¬B) = A+B。需要先产生¬A和¬B（各一个NAND），然后第三个NAND做¬(¬A·¬B)=A+B。共3个NAND。
(d) 4个NAND（经典XOR实现，见正文）。
</details>

**练习2**：分析以下CMOS电路，判断输出逻辑功能。

一个CMOS门有以下结构：
- 上拉网络（PMOS）：两个PMOS并联，栅极分别接A和B
- 下拉网络（NMOS）：两个NMOS串联，栅极分别接A和B

这个门是什么功能？

<details>
<summary><b>点击展开答案</b></summary>

PMOS并联（A=0 OR B=0时至少一个通）→上拉Y到VDD。
NMOS串联（A=1 AND B=1时两个都通）→下拉Y到GND。

当A=0或B=0时Y=1；当A=1且B=1时Y=0。这正是NAND门！✓
</details>

**练习3**：分析NOR门的CMOS结构。

<details>
<summary><b>点击展开答案</b></summary>

上拉网络（PMOS）：两个PMOS串联，栅极接A和B。只有A=0 AND B=0时两个都通→上拉Y到VDD。

下拉网络（NMOS）：两个NMOS并联，栅极接A和B。A=1 OR B=1时至少一个通→下拉Y到GND。

当A=B=0时Y=1；当A=1或B=1时Y=0。这正是NOR门！✓
</details>

**练习4**：如果74系列芯片的扇出为10，你需要用1个7404（六反相器）的输出驱动12个74系列门的输入，怎么办？

<details>
<summary><b>点击展开答案</b></summary>

扇出不够（10 < 12），需要加缓冲。可以用同一个7404芯片上的两个反相器并联驱动：把12个负载分成两组，每组6个，各用一个反相器驱动。输入信号同时接两个反相器的输入。注意两个反相器的输出不能直接短接（会产生冲突），所以必须把负载分成两组，分别接到两个反相器上。

更好的方案是用一个缓冲器芯片（如7407，有更高的驱动能力），或者用两个非门级联（第一个驱动第二个，第二个的扇出足够驱动12个）。但注意这样总延迟会增加。
</details>

## 深入：逻辑门的传输延迟级联分析

在实际电路中，多个逻辑门级联后的总延迟决定了电路的最高工作频率。

**例**：一个电路路径上有3个门：NAND(tpd=8ns) → NOT(tpd=5ns) → NOR(tpd=10ns)。总延迟 = 8+5+10 = 23ns。最大工作频率 = 1/23ns ≈ 43.5 MHz。

关键路径（Critical Path）：电路中从输入到输出延迟最长的路径。关键路径决定了整个电路的速度上限。在CPU设计中，确定关键路径并优化它是时序工程师的核心工作。

**传播延迟的温度和电压依赖性**：
- 温度升高 → 晶体管载流子迁移率下降 → 门延迟增加（变慢）
- 电压降低 → 晶体管驱动电流减小 → 门延迟增加（变慢）

这就是为什么CPU在高温或低电压时会自动降频（保护机制），以及为什么超频玩家要加电压（提高驱动能力，降低延迟）。

## 深入：开路/短路故障模型

在芯片测试中，常见的物理缺陷可以用逻辑门的故障模型来描述：

- **固定0故障（Stuck-at-0）**：某个节点永远为0，不管输入如何。相当于该节点接地。
- **固定1故障（Stuck-at-1）**：某个节点永远为1。相当于该节点接电源。
- **桥接故障（Bridging Fault）**：两个不该相连的节点短路在一起。

一个NAND门的输入A如果发生固定1故障，对输出有什么影响？

正常NAND: Y=¬(A·B)。A固定为1: Y=¬(1·B)=¬B。输入A失去了控制能力，NAND退化成一个NOT门（对B取反）。

芯片出厂前，自动测试设备（ATE）会运行成千上万个测试向量（test patterns），检测这些故障。这就是可测试性设计（DFT，Design for Testability）的范畴。

## 逻辑门的符号标准

数字电路中有两套逻辑门符号标准：

**ANSI/IEEE标准（美式）**：
- AND: 平底D形
- OR: 弧形尖底
- NOT: 三角形+小圆圈

**IEC标准（欧式）**：
- 用矩形框+内部标注（&=AND, ≥1=OR, 1=NOT）

平时所见多用美式符号。但在国际标准文档和某些EDA工具中会见到IEC符号。两者表达的逻辑完全相同。

## 逻辑门芯片的实际使用注意事项

**1. 未使用的输入引脚**：CMOS门的输入不能悬空！悬空的输入端电平不确定（受附近信号耦合影响），会导致：
- 输出不稳定（可能振荡）
- 静态功耗增大（PMOS和NMOS同时部分导通）
- 芯片可能损坏（大电流）

处理：不用的输入端接地（对AND/NAND）或接VDD（对OR/NOR），或和已用的输入端并接。

**2. 去耦电容**：每个逻辑芯片的VCC和GND之间应接一个0.1μF的陶瓷电容（去耦电容），越靠近芯片越好。这可以滤除电源线上的高频噪声，防止逻辑错误。

**3. 信号完整性**：高速逻辑信号在PCB走线上会遇到反射、串扰等问题。阻抗匹配、终端端接、差分信号等技术用于保证信号完整。

## 自测练习：10道逻辑门综合题

**题1**：用逻辑门实现一个"两人意见一致检测器"。两个输入A、B，当两人意见一致（都为0或都为1）时输出1。这个电路叫什么？需要什么门？

**题2**：用CMOS晶体管实现一个3输入NAND门。需要几个PMOS？几个NMOS？画出结构。

**题3**：解释为什么CMOS门的静态功耗几乎为0。

**题4**：一个逻辑门的扇出为5，每个负载的输入电容为10fF。如果门的输出从0→1的充电电流为1mA，估算输出上升时间。

**题5**：用NOR门实现一个NOT门。画出连接方式。

**题6**：用NAND门实现一个AND门。需要几个NAND？为什么不能直接用1个？

**题7**：两个XOR门级联，输入A、B、C（第一个XOR输入A、B，输出接第二个XOR的输入，第二个XOR的另一个输入接C）。写出最终的逻辑表达式。

**题8**：解释为什么NAND和NOR被称为"通用门"而AND和OR不是。

**题9**：给定一个未知的2输入逻辑门，测得其真值表为（00→1, 01→1, 10→1, 11→0）。这是什么门？

**题10**：一个74HC00（四NAND门）芯片，VCC=5V。不用的两个门的输入应该怎么处理？为什么？

<details>
<summary><b>点击展开答案</b></summary>

**1.** XNOR门（同或门）。$F = \overline{A \oplus B} = AB + \overline{A}\overline{B}$。需要2个NOT + 2个AND + 1个OR，或直接用一个XNOR门（7486+NOT）。

**2.** 3输入NAND：3个PMOS并联（上拉），3个NMOS串联（下拉）。共6个晶体管。

**3.** CMOS门在稳态下，PMOS和NMOS网络中总有一个是断开的（VDD到GND没有直流通路）。电流只在翻转瞬间流过（对负载电容充放电），静态电流极低（仅漏电流，纳安级别）。

**4.** 总负载电容 C = 5 × 10fF = 50fF。上升时间 ≈ C × VDD / I = 50fF × 5V / 1mA = 250ps。

**5.** NOR(A, A) = ¬(A+A) = ¬A。输入A同时接NOR的两个输入端。

**6.** 需要2个NAND。NAND1输出¬(AB)，NAND2输入¬(AB)和¬(AB) → ¬(¬(AB)) = AB。

**7.** $(A \oplus B) \oplus C = A \oplus B \oplus C$。三个输入中奇数个1时输出1。

**8.** 因为NAND/NOR可以单独实现NOT，然后通过德摩根定律实现AND和OR。而AND不能单独实现NOT（因为没有取反能力），OR也不能。NAND和NOR自带"取反"功能。

**9.** 00→1, 01→1, 10→1, 11→0。这正是NAND门！

**10.** 不用的输入端接地（对NAND门，输入端接地→输出恒为1，静态功耗最小）。如果悬空，输入端电平不确定，可能导致振荡或大功耗。
</details>
