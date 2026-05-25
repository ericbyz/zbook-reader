# 第27章：存储器与数模转换

> **核心问题**：数字世界和模拟世界怎么"对话"？

---

## 0. 本章导览

到目前为止，我们学的都是"纯粹"的数字电路——里面跑的全是 0 和 1，干净利落。但真实的世界不是这样的。你说话的声音是连续变化的声波（模拟信号），你手机拍照的亮度也是连续变化的光强（模拟信号）。计算机要处理这些信号，必须先把它变成 0 和 1（ADC），算完之后再变回模拟信号送给耳机或者屏幕（DAC）。

在这个"转换"的中间，数据和程序需要有个地方**存起来**。用什么存？存储器（Memory）。从你打开一个 App 到它出现在屏幕上，数据在 RAM 和 ROM 之间来回搬运，速度之快远超你的直觉——DDR5 内存每秒钟可以读写数十亿次。但存储器不只有速度这一个维度：有些存储器掉电后数据还在（比如 U 盘里的 Flash），有些存储器一断电就清空（比如电脑内存里的 DRAM），有些甚至连断电都不怕、还能用紫外线擦除（老式 EPROM）。这一章会把这些不同类型的存储器一网打尽。

这一章分三个大块：**存储器**（数字世界的"仓库"）、**DAC**（数字→模拟）、**ADC**（模拟→数字）。你不仅能搞清 U 盘为什么能存东西不丢、电脑内存为什么一关机就清空，还能理解你的手机是怎么把你的声音变成数字信号、又是怎么把数字音乐变回声音的。

学完本章，你将能够：

1. 区分 ROM 和 RAM 的基本原理，知道各自用在什么地方
2. 理解不同 ROM 类型（PROM/EPROM/EEPROM/Flash）的擦写方式差异
3. 对比 SRAM（6T 锁存器）和 DRAM（1T1C 电容）的结构，理解为什么 DRAM 需要刷新
4. 进行存储器字扩展和位扩展的计算（几片拼成大存储器）
5. 分析 R-2R 梯形 DAC 电路并计算输出电压，理解分辨率
6. 理解 ADC 的采样定理（Nyquist）和逐次逼近（SAR）、并联比较（Flash）两种转换方式
7. 计算量化误差和信噪比
8. 用 Python 模拟 DAC 输出波形和 ADC 量化过程

> 本章约 1450 行，建议分 3-4 次读完。概念密集但手算不多。

前置章节：[计数器与寄存器](./25-counters-registers.md)、[时序逻辑设计](./26-sequential-design.md)
后续：本教程的数字电路部分到此结束。接下来进入模拟电路的世界！

---

## 1. 存储器：计算机的记忆

### 1a. 生活例子：书架和便签本

你有一个大书架，上面摆满了印好的书。你只能从书架上**读**书，不能在上面写东西，更不能把书的内容擦掉重写。这些书的内容不会因为停电而消失——过了十年拿出来，字迹依然清晰。

你还有一个便签本，放在桌面上。你随时可以在上面写东西、画草图，写满了撕掉一页再写新的。便签本用起来极其方便，但一旦一阵风吹来（断电），所有写在上面但没来得及誊抄到别处的东西就全没了。

**ROM**（Read-Only Memory，只读存储器）就是那个**书架**——数据出厂时写好（或一次性烧录），掉电不丢失（非易失性）。**RAM**（Random Access Memory，随机存取存储器）就是那个**便签本**——可随时任意读写，速度快到惊人，但一掉电内容就清空（易失性）。

注意"Random Access"这个名字可能让人误会。"随机存取"不是"随机地存取"，而是说：**访问任何一个地址的时间都差不多**，不像磁带那样要倒带到指定位置。其实 ROM 也是随机存取的，但历史上这个名字被 RAM 占了。

### 1b. 直观理解：存储矩阵

不管 ROM 还是 RAM，内部核心结构都像一个**巨大的二维表格**——存储矩阵。

```
         列选择线 (Column Select)
           ↓   ↓   ↓   ↓   ↓
         ┌───┬───┬───┬───┬───┐
 行选择  │ ● │   │   │   │   │  ← 每个交叉点存一个比特
 线 →    ├───┼───┼───┼───┼───┤
         │   │ ● │   │   │   │
         ├───┼───┼───┼───┼───┤
         │   │   │   │   │ ● │
         └───┴───┴───┴───┴───┘
```

怎么找到某一个比特？靠"地址"。比如一个 1K×8（1024 个地址，每个地址 8 位）的存储器：

- 10 根地址线（因为 $2^{10} = 1024$），其中高 5 位送给**行译码器**选择哪一行，低 5 位送给**列译码器**选择哪一列（或多位列同时选，因为一次读 8 位）。
- 行和列交叉的那一个（或那一组）存储单元被激活，数据从那里读出来、或写进去。
- 这就是存储器的核心工作流程——**给定地址，找到位置，读写数据**。

### 1c. 形式化定义：存储器的数学模型

存储器可以看成一个大的映射函数：

$$
\text{Mem}: \{0, 1, \dots, 2^n - 1\} \to \{0, 1, \dots, 2^m - 1\}
$$

- $n$ = 地址线条数，$2^n$ = 总地址数（存储空间大小）
- $m$ = 数据线条数（字宽），每个地址存 $m$ 位

读操作：$Data = \text{Mem}[Addr]$
写操作：$\text{Mem}[Addr] \leftarrow Data_{in}$（仅 RAM）

存储器容量常用单位：
- 1 Kb（Kilobit）= $2^{10} = 1024$ 比特
- 1 Mb（Megabit）= $2^{20} = 1,048,576$ 比特
- 1 Gb（Gigabit）= $2^{30}$ 比特
- 1 KB（Kilobyte）= $8 \times 1024 = 8192$ 比特

注意：存储芯片常用**bit**标称（如 1Gb 芯片 = 128MB），而操作系统用**Byte**显示。两者差了 8 倍。

### 1d. 手算示例：存储芯片参数

**例 1**：一个存储芯片标注"64K×8"。问：
- (a) 有几根地址线？
- (b) 有几根数据线？
- (c) 总容量是多少字节？

解答：
- (a) $64K = 64 \times 1024 = 65536 = 2^{16}$，需要 16 根地址线。
- (b) "×8"表示每个地址 8 位数据 → 8 根数据线。
- (c) 总容量 = $64K \times 8 \text{ bits} = 512K \text{ bits} = 64K \text{ Bytes}$。

**例 2**：一个单片机有 16 根地址线、8 根数据线。最大能接多大存储器？

地址线 16 根 → $2^{16} = 65536$ 个地址。每个地址 8 位。总容量 = $65536 \times 8 = 524288$ bits = 64 KB。

**例 3**：用 1K×4 的芯片构成一个 4K×8 的存储系统，需要多少片？

思路：
- 首先解决数据宽度：每个地址要 8 位，但每片只有 4 位 → 需要 2 片做**位扩展**（Bit Expansion），两片并排组成 1K×8。
- 然后解决地址空间：需要 4K 个地址，但每套（2 片）只提供 1K 个地址 → 需要 $4K / 1K = 4$ 套做**字扩展**（Word Expansion）。
- 总片数 = $2 \times 4 = 8$ 片。

或者换个思路：总容量目标 = $4K \times 8 \text{ bits} = 32K \text{ bits}$。每片容量 = $1K \times 4 \text{ bits} = 4K \text{ bits}$。$32K / 4K = 8$ 片。殊途同归。

### 1e. Python 验证：地址线计算

```python
def analyze_memory(org_str):
    """解析 '64Kx8' 这样的存储器组织，返回关键参数"""
    import re
    match = re.match(r'(\d+)(K|M|G)?\s*[x×]\s*(\d+)', org_str.replace(' ',''))
    if not match:
        return "格式错误，请使用如 '64Kx8' 格式"
    num = int(match.group(1))
    unit = match.group(2) or ''
    width = int(match.group(3))
    
    multiplier = {'K': 1024, 'M': 1024**2, 'G': 1024**3, '': 1}
    total_addrs = num * multiplier.get(unit, 1)
    
    # 地址线数 = ceil(log2(地址数))
    addr_lines = 0
    while (1 << addr_lines) < total_addrs:
        addr_lines += 1
    
    total_bits = total_addrs * width
    total_bytes = total_bits // 8
    
    return {
        '地址数': total_addrs,
        '地址线': addr_lines,
        '数据线': width,
        '总容量(bits)': total_bits,
        '总容量(Bytes)': total_bytes,
    }

# 测试
tests = ['64Kx8', '1Mx4', '256Kx16', '4Gx8']
print("=== 存储芯片参数分析 ===")
for t in tests:
    r = analyze_memory(t)
    print(f"\n  {t}:")
    for k, v in r.items():
        print(f"    {k}: {v:,}")
```

### 1f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "1 KB = 1000 字节" | 计算机中 1KB = 1024 字节（$2^{10}$）。硬盘厂商有时用 1000（十进制），这是 1TB 硬盘在电脑显示约 931GB 的原因。 |
| "ROM 不能写" | 部分 ROM（EEPROM, Flash）可以写，只是慢且次数有限。真正的"只读"只有掩模 ROM。 |
| "地址线数 = $\log_2(容量)$" | 只在数据宽度为 1 位时成立。正确公式：地址线数 = $\log_2(地址数)$。 |

### 1g. 应用连接

存储器是计算机体系结构中最核心的组成部分之一。我们后面会回到"存储器层次结构"（Memory Hierarchy）这个概念——从速度最快容量最小的 CPU 寄存器，到 Cache（SRAM），到主存（DRAM），到固态硬盘（Flash），到机械硬盘，每一层都在"速度"和"容量"之间做权衡。理解 ROM 和 RAM 的区别是理解这一切的起点。

在实际的嵌入式系统（比如 Arduino 或 STM32 单片机）中，你会直接接触到这两种存储器：程序代码烧录在 Flash（充当 ROM）中，掉电不丢失；运行时的变量和堆栈存在 SRAM 中，掉电即消失。这就是为什么你写完 Arduino 程序后拔掉电源再插上，程序还在（Flash），但上次运行中计的数丢了（SRAM）。

### 1h. 常见误区补充

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "存储器的速度只看读写时间" | 还有地址建立时间、数据保持时间、预充电时间等。实际系统的存储控制器在设计时序时，要满足一整套时序参数要求（类似 DRAM 的 CL/tRCD/tRP）。 |
| "Flash 就是慢速的 RAM" | Flash 的写入是以"块"为单位的（4KB-128KB），不能像 RAM 一样直接改写单个字节。改写一个字节需要：读出整个块→修改那个字节→擦除整个块→写回整个块。这称为"读-改-写"（Read-Modify-Write）。 |

---

## 2. ROM：只读存储器的家族

### 2a. 生活例子：五种"不能改"的记事方式

假设你要记下一个重要密码"8362"。有五种记法：

1. **印在出生证明上**（掩模 ROM）：从你出生就印好了，永远改不了。大批量生产时成本极低，但内容固定。
2. **刻在石板上**（PROM）：你拿凿子一次刻好，刻错就废了，没法改。只能写一次。
3. **用铅笔写在纸上**（EPROM）：可以擦掉重写，但需要用橡皮（紫外线）擦整个纸面，不能只擦一个字。擦一次要半小时。
4. **用可擦笔写**（EEPROM）：可以用"电橡皮"（电压）按字擦除重写，但写起来比较慢。可擦几万到几十万次。
5. **用白板笔写**（Flash）：擦写都快，但必须整个白板（一整块）擦，不能只擦一个角落。性价比极高。

这五种 ROM 代表了五种"只读存储器"的实际产品形态。

### 2b. 各种 ROM 详解

**掩模 ROM（Mask ROM）**

在芯片制造的最后一步——金属布线（Metalization）——用光刻模板（Mask）直接把数据"刻"进去。特定位置连上或断开，对应 1 或 0。

- 优点：大批量生产时成本极低（每片几分钱）、可靠性极高。
- 缺点：数据在制造时就固定了，后面完全不能改；设计周期长（改一行代码要重新做一套掩模，耗时数周、花费数十万美元）。
- 用途：大批量的消费电子产品（如早期游戏卡带、计算器固件）。

**PROM（Programmable ROM，可编程 ROM）**

出厂时每个存储单元都是 1（或全 0），内部每个比特位有一根细小的**熔丝**（Fuse）。用户用一个叫"编程器"的设备，在需要写 0 的位置施加高电压脉冲，烧断熔丝——烧断后不可恢复。

- 优点：用户可以自己写数据，不需要光刻厂。
- 缺点：只能写一次（OTP，One-Time Programmable），写错就报废。
- 用途：小批量产品固件、设备序列号、加密密钥。

**EPROM（Erasable PROM，可擦除 PROM）**

用**浮栅晶体管**存储数据。芯片顶部有一个石英玻璃窗口。擦除方式：把芯片放到紫外线灯下照射 20-30 分钟，紫外线光子给浮栅上的电子"充能"，使它们逃逸——整片芯片全部擦成 1。

- 优点：可重复擦写（几十到几百次），开发调试时不用报废芯片。
- 缺点：擦除慢（半小时）、需要专用紫外线擦除器、不能按字节擦除（必须整片擦）、石英窗口很贵。
- 用途：已被 EEPROM 和 Flash 取代。你只能在老式设备（90 年代电脑主板 BIOS 芯片上有个小玻璃窗的）上见到它。

**EEPROM（Electrically Erasable PROM，电可擦除 PROM）**

不需要紫外线了！用**电压**擦除（通过量子力学中的 Fowler-Nordheim 隧穿效应，给浮栅充电或放电）。可以**按字节**擦写。

- 优点：电擦除、按字节擦写、方便好用。
- 缺点：擦写速度慢（毫秒级，比 RAM 慢 1000 倍以上）、擦写次数有限（约 10 万~100 万次）、容量不大。
- 用途：存储配置信息（电脑 BIOS 设置、显示器校准参数、智能卡芯片）。

**Flash Memory（闪存）**

EEPROM 的"高速大容量进化版"。关键改进：
- 按**块**（Block，一般 4KB~128KB）擦除，而不是按字节。牺牲灵活性换速度。
- 每个存储单元可以只用一个晶体管（不像 EEPROM 需要两个），密度大幅提升。
- 分为 NOR Flash 和 NAND Flash 两种。

**NOR Flash**：每个存储单元独立接出，可以像 RAM 一样**随机读取任意地址**（速度快、可以做 XIP——片内执行代码，eXecute In Place）。写入和擦除较慢。用于存储嵌入式系统的程序代码（单片机的"硬盘"）。

**NAND Flash**：存储单元串联成串，读取以"页"为单位（顺序读取极快），随机访问慢。但密度极高、成本极低。用于 U 盘、SSD 固态硬盘、手机存储、SD 卡。

简单记忆：**NOR 读快适合做 ROM，NAND 容量大适合做硬盘。**

### 2c. 手算示例：Flash 擦写寿命

**例 4**：一个 SSD 有 256GB 的 NAND Flash，标称擦写寿命 1000 P/E Cycles（Program/Erase Cycles，擦写次数）。如果每天写入 50GB 数据，能用多少年？

- 总可写入量 = $256 \text{ GB} \times 1000 = 256,000 \text{ GB} = 250 \text{ TB}$（TBW, Total Bytes Written）。
- 每天 50GB → 每年 $50 \times 365 = 18,250 \text{ GB}$。
- 使用寿命 = $256,000 / 18,250 \approx 14.0$ 年。

实际中，SSD 内部有**磨损均衡**（Wear Leveling）算法，将写入均匀分布到所有块，避免某些块过早写坏。

### 2d. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "Flash 就是 EEPROM，只是名字不同" | Flash 按块擦除，EEPROM 按字节擦除。Flash 单元结构更简单（1T），EEPROM 更复杂（2T）。速度、容量差异巨大。 |
| "ROM 里的数据绝对不会丢" | EPROM 长期放在阳光下可能被擦除（紫外线）；Flash 在高温下浮栅电荷会缓慢泄漏（数据保持约 10 年）。ROM 也非"绝对"永久。 |
| "SSD 写不坏" | NAND Flash 的氧化层在反复擦写中会磨损。消费级 SSD 寿命约 100-1000 TBW。但现在控制器聪明，普通用户用 5-10 年没问题。 |

---

## 3. RAM：随时读写的"便签本"

### 3a. 生活例子：两种不同的便签

**SRAM（静态 RAM）**：6 个人手拉手围成一圈，记住一个秘密（0 或 1）。只要这 6 个人不松手（不断电），秘密就一直保持。人多（面积大），但反应快（谁问都能立刻回答），不累（静态功耗低？错——其实有漏电，后面会讲）。

**DRAM（动态 RAM）**：一个人端着一杯水（电容充电）站岗。水杯满 = 1，水杯空 = 0。但水杯有一个小裂缝——水慢慢渗漏。所以每隔几十毫秒，必须有人过来看一眼水位，快漏光了就重新倒满。这个人少（面积小、成本低），但需要不断"伺候"（刷新）。

### 3b. SRAM 的 6T 存储单元

SRAM 的 1 个比特存储单元由 **6 个 MOS 晶体管**（6T）组成，本质上是一个**用两个反相器交叉耦合构成的双稳态锁存器**，加上两个"访问晶体管"，类似于前面学过的 D 锁存器结构。

- **存储核心**（4 个晶体管）：两个 PMOS + 两个 NMOS 组成两个互相驱动的反相器。反相器 1 的输出接反相器 2 的输入，反相器 2 的输出接反相器 1 的输入 → 形成正反馈锁存。稳定状态只有两个：Q=1, ¬Q=0 或 Q=0, ¬Q=1。
- **访问晶体管**（2 个 NMOS）：栅极接字线（Word Line, WL），源漏分别接存储节点和位线（Bit Line, BL / ¬BL）。

**读操作过程**：
1. 两条位线（BL 和 ¬BL）预充电到一个中间电压（通常 $V_{DD}/2$）。
2. 字线 WL 激活 → 两个访问晶体管导通。
3. 存储单元中存的 0 或 1 通过访问晶体管对位线产生微小电压差（可能只有几十到几百毫伏）。
4. **灵敏放大器**（Sense Amplifier）检测这个微小电压差，把它放大到全逻辑摆幅（0 或 $V_{DD}$）。

**写操作过程**：
1. 字线 WL 激活 → 访问晶体管导通。
2. 外部写驱动器强制驱动两条位线：BL 强制为要写入的值，¬BL 强制为相反值。
3. 写驱动器的驱动能力强于锁存器中的晶体管，强制翻转存储内容。

SRAM 的关键参数：速度快（0.5~5 ns 访问时间），面积大（6 管/bit），静态功耗有漏电（工艺越先进漏电越严重）。用于 CPU 的一级、二级、三级缓存（L1/L2/L3 Cache）。

### 3c. DRAM 的 1T1C 存储单元

DRAM 只用 **1 个晶体管 + 1 个电容**（1T1C）。结构极简：

- 电容 Cₛ 存储电荷。充到一定电压 = 逻辑 1，放电到接近 0 = 逻辑 0。
- 晶体管做开关：栅极接字线 WL，源漏分别接位线 BL 和电容 Cₛ。

**写操作**：字线激活 → 晶体管导通。位线上的电压通过晶体管对电容充电（写 1）或放电（写 0）。然后字线关闭，晶体管断开，电容上的电荷被"困"住——这就是存储。

**读操作**：先给位线预充电。字线激活 → 晶体管导通。电容上的电荷和位线上的电荷**分享**（Charge Sharing）。根据电容上原来存的是 1 还是 0，位线电压会略微上升或下降（变化量大约在 50~200 mV）。灵敏放大器检测这个变化，放大到全摆幅。

**但是**，读操作是**破坏性**的！读取过程中，电容上的电荷被位线的寄生电容分走了一部分。所以读完之后必须立即把数据**重新写回**电容（这部分由 DRAM 芯片内部电路自动完成）。

**刷新（Refresh）**：DRAM 的电容会漏电。典型情况下，电容上的电荷在 **64 ms** 内就会漏到无法可靠检测的程度。所以 DRAM 控制器必须每隔 64ms 把所有行都"读一次再写回去"（刷新）。刷新操作按行进行——激活字线→灵敏放大器读出→写回→关闭字线。一次只刷新一行，逐行轮转。

**例 5**：一个 DRAM 芯片有 8192 行（13 位行地址），刷新周期 64 ms。每行刷新间隔是多少？

平均间隔 = $64 \text{ ms} / 8192 \approx 7.81 \mu\text{s}$。即大约每 7.8 微秒就要刷新一行。如果每次刷新操作耗时 50 ns，则总刷新时间 = $8192 \times 50 \text{ ns} = 409.6 \mu\text{s}$。刷新开销 = $409.6 \mu\text{s} / 64 \text{ ms} \approx 0.64\%$。非常小，几乎不影响性能。

### 3d. SRAM vs DRAM 总结对比

| 特性 | SRAM | DRAM |
|------|------|------|
| 每 bit 晶体管数 | 6 个（6T） | 1 个（1T1C） |
| 存储原理 | 双稳态锁存器（正反馈） | 电容充放电 |
| 速度 | 极快（0.5-5 ns） | 较慢（10-70 ns） |
| 密度 | 低 | 高（约为 SRAM 的 4-8 倍） |
| 刷新 | 不需要 | 需要（每 64ms） |
| 功耗 | 静态漏电大 | 刷新动态功耗 |
| 成本/bit | 高 | 低 |
| 典型用途 | CPU Cache（L1/L2/L3） | 主内存（DDR4/DDR5） |
| 典型容量 | MB 级 | GB 级 |

### 3e. 手算示例：DRAM 相关计算

**例 6**：DDR4-3200 内存（数据率 3200 MT/s，Mega-Transfers per second），数据总线 64 位。理论最大带宽是多少？

带宽 = $3200 \times 10^6 \times 64 \text{ bits/s} = 204.8 \times 10^9 \text{ bits/s} = 25.6 \text{ GB/s}$。

这就是为什么你的电脑内存速度标注为"PC4-25600"——25600 MB/s = 25.6 GB/s。

**例 7**：一个 DRAM 存储单元电容 $C_s = 30 \text{ fF}$（飞法，$10^{-15}$F），位线寄生电容 $C_{BL} = 300 \text{ fF}$。存储 1 时电容上电压 $V_{DD} = 1.2\text{ V}$，存储 0 时 0 V。读取时位线预充到 $V_{DD}/2 = 0.6\text{ V}$。读取 1 后，位线电压变化量 $\Delta V = ?$

用电荷守恒：读取前，电容上电荷 $Q_s = C_s \cdot V_s$，位线上电荷 $Q_{BL} = C_{BL} \cdot V_{pre}$。晶体管导通后，电荷重新分配，最终电压为 $V_f$。

总电荷 = $C_s V_s + C_{BL} V_{pre} = (C_s + C_{BL}) V_f$

$V_f = \frac{C_s V_s + C_{BL} V_{pre}}{C_s + C_{BL}}$

读 1 时（$V_s = 1.2\text{ V}$）：
$V_f = \frac{30 \times 1.2 + 300 \times 0.6}{30 + 300} = \frac{36 + 180}{330} = \frac{216}{330} \approx 0.655\text{ V}$

$\Delta V = 0.655 - 0.6 = 0.055\text{ V} = 55\text{ mV}$。

只产生了 55 mV 的信号！这就是为什么 DRAM 需要高灵敏度的灵敏放大器。读 0 时 $\Delta V = -55\text{ mV}$（位线电压从 0.6V 降到约 0.545V）。

### 3f. Python 验证：DRAM 位线电压模拟

```python
def dram_read_voltage(Cs_fF, Cbl_fF, Vstored, Vprecharge):
    """计算 DRAM 读取后位线电压和变化量"""
    Cs = Cs_fF * 1e-15  # 转法拉
    Cbl = Cbl_fF * 1e-15
    
    Vf = (Cs * Vstored + Cbl * Vprecharge) / (Cs + Cbl)
    delta_V = Vf - Vprecharge
    
    return Vf, delta_V

# 例7的参数
Cs, Cbl = 30, 300
Vdd, Vpre = 1.2, 0.6

print("=== DRAM 读取位线电压模拟 ===")
Vf1, dV1 = dram_read_voltage(Cs, Cbl, Vdd, Vpre)
Vf0, dV0 = dram_read_voltage(Cs, Cbl, 0, Vpre)

print(f"  存储电容 Cs = {Cs} fF, 位线电容 Cbl = {Cbl} fF")
print(f"  Vdd = {Vdd}V, 预充电压 = {Vpre}V")
print(f"\n  读 '1': 最终电压 = {Vf1*1000:.1f} mV, ΔV = {dV1*1000:.1f} mV")
print(f"  读 '0': 最终电压 = {Vf0*1000:.1f} mV, ΔV = {dV0*1000:.1f} mV")
print(f"\n  信号摆幅仅 {abs(dV1)*1000:.0f} mV，对灵敏放大器要求极高！")

# 不同 Cbl/Cs 比值的影响
print("\n=== Cbl/Cs 比值对信号的影响 ===")
for ratio in [2, 5, 10, 20, 50]:
    Cbl_new = Cs * ratio
    _, dV = dram_read_voltage(Cs, Cbl_new, Vdd, Vpre)
    print(f"  Cbl/Cs = {ratio:2d}: ΔV = {dV*1000:6.1f} mV")
```

### 3g. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "SRAM 不需要电" | SRAM 需要持续供电维持锁存器状态（静态功耗）。虽然不需要刷新时钟，但有漏电流。在先进工艺下，SRAM 的漏电功耗甚至超过 DRAM 的刷新功耗。 |
| "DRAM 比 SRAM 慢是因为需要刷新" | DRAM 慢主要是因为电容充放电 + 灵敏放大器检测需要时间。刷新只是额外开销，不是主因。 |
| "DDR 内存就是 DRAM" | DDR（Double Data Rate）是 DRAM 的一种高速接口标准。内芯还是 DRAM 的 1T1C 单元，只是接口在时钟上下沿都传数据（双倍速率）。 |

---

## 4. 存储器扩展：用小芯片拼大存储器

### 4a. 生活例子：拼积木

你有一堆积木（每块 2×4 的乐高板），想拼出一个 8×8 的大底板。

两种拼法：
- **横着拼**：两块并排，从 4 宽变成 8 宽（这就是**位扩展**——每块负责一部分数据位）。
- **竖着拼**：两块上下接，从 2 行变成 4 行（这就是**字扩展**——增加地址空间）。

实际存储器扩展就是把多片小容量芯片拼成大容量存储系统。

### 4b. 位扩展（Bit Expansion）：增加数据宽度

场景：有 1K×4 的芯片（1024 个地址，每个地址 4 位数据），需要 1K×8 的存储。

做法：两片 1K×4 芯片并排。**地址线共享**（两片的地址线全接在一起），**控制信号共享**（片选 CS、读/写 R/W 一起接）。第一片的数据线接 D₀-D₃（低 4 位），第二片的数据线接 D₄-D₇（高 4 位）。

访问地址 A 时，两片同时被选中，各自输出 4 位，拼成 8 位数据。

### 4c. 字扩展（Word Expansion）：增加地址空间

场景：有 1K×8 的芯片，需要 4K×8 的存储。

做法：4 片 1K×8 芯片。地址线的低 10 位（A₀-A₉）同时接所有芯片（芯片内部用这 10 根线选片内地址）。地址线的高 2 位（A₁₀, A₁₁）送给一个 **2-4 译码器**，译码器的 4 个输出分别接 4 片芯片的**片选**（CS，Chip Select）引脚。

- A₁₁A₁₀ = 00 → 选第一片（地址范围 0~1023）
- A₁₁A₁₀ = 01 → 选第二片（地址范围 1024~2047）
- A₁₁A₁₀ = 10 → 选第三片（地址范围 2048~3071）
- A₁₁A₁₀ = 11 → 选第四片（地址范围 3072~4095）

任何时候只有一片被选中（片选有效），其他三片的数据线呈高阻态（不影响总线）。这样就实现了 4K 的地址空间。

### 4d. 字位同时扩展

场景：用 1K×4 芯片构造 4K×8 存储器。

- 位扩展：2 片 1K×4 → 1 组 1K×8。
- 字扩展：需要 4K 地址 → 需要 4 组（每组 1K）。
- 总片数 = $2 \times 4 = 8$ 片。

地址线连接：低 10 位 A₀-A₉ 接所有芯片。A₁₀, A₁₁ 接 2-4 译码器，输出接每组的片选。数据线：D₀-D₃ 接每组的第一片，D₄-D₇ 接每组的第二片。

### 4e. 手算示例

**例 8**：有 16K×1 的 DRAM 芯片。要构成 64K×8 的存储系统，需要多少片？画出结构。

- 容量目标 = $64K \times 8 = 512K$ bits。
- 单片容量 = $16K \times 1 = 16K$ bits。
- 片数 = $512K / 16K = 32$ 片。

结构：位扩展需要 8 片（1 bit → 8 bits），组成一组 16K×8。字扩展需要 $64K / 16K = 4$ 组。总片数 $8 \times 4 = 32$ 片 ✓。

地址线：低 14 位 A₀-A₁₃ 接所有芯片（$2^{14}=16K$）。高 2 位 A₁₄, A₁₅ 接 2-4 译码器选择组。

**例 9**：一个处理器有 16 根地址线、8 根数据线。用 2K×4 的 RAM 芯片，最多能构成多大存储空间？需要多少片？

处理器地址空间 = $2^{16} = 64K$ 地址。数据线要求 8 位。
每片 2K×4 → 位扩展需 2 片（→2K×8）。字扩展需 $64K / 2K = 32$ 组。
总片数 = $2 \times 32 = 64$ 片。
最终存储 = 64K×8 = 64 KB，恰好占满处理器地址空间。

### 4f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "所有芯片共享所有地址线" | 字扩展时，高位地址线通过译码器间接选择芯片（片选），不直接连到所有芯片。只有片内地址（低位）直连。 |
| "数据线也共享就行" | 只有字扩展（数据宽度不变）时数据线才共享。位扩展时各芯片数据线独立接不同数据位，不能直接并联（会冲突）。 |
| "片选就是 CE(Chip Enable)" | 正确。不同厂商叫法不同：CS = Chip Select, CE = Chip Enable。功能一样——决定该芯片是否响应总线请求。 |

---

## 5. DAC：把数字变成模拟

### 5a. 生活例子：音量旋钮和电灯调光器

你拧音响的音量旋钮。旋钮本身是连续可调的（模拟量），但你手机的音量却是 0-100 的整数（数字量）。怎么从"第 47 格"变成"刚好那么响"？内部有一个 DAC（Digital-to-Analog Converter，数模转换器），把数字 47 翻译成一个对应的模拟电压，驱动扬声器。

同样，你家的 LED 调光灯——0-255 级亮度，每级对应不同的电流（模拟量）。DAC 把数字亮度值变成驱动 LED 的电流。

DAC 的核心任务：输入是一串二进制数字（如 8 位 10101100），输出是一个**和数字成正比的模拟电压**。输入数字越大，输出电压越高。

### 5b. 权电阻型 DAC

**原理**：每一位数字信号控制一个开关。开关连通一个电阻到运放的反相输入端。电阻的阻值按该位的"权重"（Weight）分配——高位（MSB）对应的电阻小（因为高位权重高，需要大电流），低位（LSB）对应的电阻大（权重低、电流小）。

以 4 位为例。参考电压 $V_{ref}$，4 个电阻值分别为 $R$, $2R$, $4R$, $8R$，分别对应 D₃(MSB)、D₂、D₁、D₀(LSB)。当某位数字为 1 时，开关接通 $V_{ref}$，该支路电流 $I_i = V_{ref}/R_i$ 流入运放反相输入端。反相输入端虚地，所以：

$$
V_{out} = -R_f \cdot (I_3 + I_2 + I_1 + I_0) = -R_f \cdot V_{ref} \cdot \left( \frac{D_3}{R} + \frac{D_2}{2R} + \frac{D_1}{4R} + \frac{D_0}{8R} \right)
$$

取 $R_f = R$，化简为：

$$
V_{out} = -V_{ref} \cdot \left( D_3 + \frac{D_2}{2} + \frac{D_1}{4} + \frac{D_0}{8} \right)
$$

这就把二进制数 $(D_3 D_2 D_1 D_0)_2$ 变成了对应的模拟电压！

推广到 n 位：$V_{out} = -V_{ref} \cdot \frac{D_{in}}{2^n}$，其中 $D_{in}$ 是 n 位输入数字的十进制值（0 到 $2^n-1$）。

**权电阻 DAC 的致命缺点**：高位和低位的电阻值相差 $2^{n-1}$ 倍！8 位时最大电阻是最小电阻的 128 倍。在集成电路中，精确匹配 1:128 的电阻值极其困难。而且大电阻占用面积大、温度系数难匹配。

### 5c. R-2R 梯形 DAC

为了解决权电阻 DAC 的电阻范围问题，工程师发明了 **R-2R 梯形 DAC**。整个电路只用两种阻值的电阻——**R 和 2R**，不管多少位都一样！

结构：一串 R-2R 的梯形网络，像一个无限的楼梯。每个节点的电阻往"左"看和往"右"看，等效电阻都是 $R$。这个特性保证了每一位对输出贡献的**权重恰好是前一位的一半**——2 倍的关系自动满足，不用做 128 倍的大电阻。

n 位 R-2R DAC 的输出电压公式（$R_f = R$）：

$$
V_{out} = -V_{ref} \cdot \frac{D_{in}}{2^n}
$$

和权电阻型公式完全一样！但电阻匹配要求大大降低——只需要 R 和 2R 两种电阻做好比例匹配（2:1），所有位的电阻条件一致。

**R-2R 网络为什么等效电阻总是 R？** 这是一个经典的递归证明。从最右端（LSB 端）往左看：最右边是一个 2R 电阻接地。往左一个节点，看进去是 2R ∥ 2R = R（两个 2R 并联），再串联 R → R + R = 2R。再往左一个节点，又是 2R ∥ 2R = R……以此类推。无论从哪个节点看，等效电阻永远是 R。这就是 R-2R 网络的美妙之处。

### 5d. DAC 性能指标

**分辨率（Resolution）**：DAC 的位数 n。n 越大，"台阶"越细。8 位 = 256 级，16 位 = 65536 级。1 LSB（Least Significant Bit，最低有效位对应的电压增量）：

$$
1 \text{ LSB} = \frac{V_{ref}}{2^n}
$$

**精度（Accuracy）**：实际输出与理论输出的偏差。来源：电阻匹配误差、运放失调电压、参考电压漂移等。

**微分非线性（DNL，Differential Non-Linearity）**：相邻两个输入码之间的输出电压增量与理想 1 LSB 的偏差。如果 DNL < -1 LSB，意味着输入码增加时输出电压反而减小了——**非单调**，这是严重缺陷。

**积分非线性（INL，Integral Non-Linearity）**：实际输出与理想直线之间的最大偏差。INL 是所有 DNL 的累积效果。

**建立时间（Settling Time）**：输入数字变化后，输出稳定到最终值 ±½LSB 以内所需的时间。影响 DAC 的最高工作频率。

### 5e. R-2R 网络电流分配详解（手算推导）

让我们用 3 位 R-2R DAC 为例，**一步一步**推导输出电压，让你彻底理解这种"只用两种电阻"的巧妙设计。

假设 $V_{ref} = 8\text{ V}$，所有 R=1kΩ，2R=2kΩ。运放反馈电阻 $R_f = R = 1\text{k}\Omega$。输入数字 $D_2 D_1 D_0 = 101$（二进制，即十进制的 5）。

**第 1 步：分析 R-2R 网络中的电阻关系**

R-2R 网络有三个节点：节点 A（MSB，D₂控制）、节点 B（D₁控制）、节点 C（LSB，D₀控制）。每个节点通过一个 2R 电阻连接到一个开关。开关根据数字位是 1 还是 0，连接 $V_{ref}$（8V）或地（0V）。

从节点 C（最右端）往右看，有一个 2R 电阻接地（另一个 2R 是 LSB 开关电阻，但已计入开关）。节点 C 往右的等效电阻 = 2R = 2kΩ。

从节点 B 往右看：2R 电阻并联节点 C 往右的 2R → 等效电阻 = 2R ∥ 2R = R = 1kΩ。然后串联上 R = 1kΩ → 总往右看 = 2kΩ = 2R。

从节点 A 往右看：同样，2R ∥ 2R = R = 1kΩ，串联 R = 1kΩ → 往右看也是 2kΩ = 2R。

**第 2 步：分析每一位对运放输入端的电流贡献**

利用叠加原理。运放反相输入端为虚地（0V）。

先看 D₂（MSB，数字码的 bit2=1，开关接 8V）。从 8V 通过 2R 到虚地，电流 $I_2 = 8 / 2\text{k} = 4\text{ mA}$。

再看 D₁（bit1=0，开关接地）。接地 → 没有电流流入虚地。$I_1 = 0$。

最后看 D₀（LSB，bit0=1，开关接 8V）。但 D₀ 的电流到虚地要经过两个 R 和一个 2R 的分压。从节点 B 看，D₀ 的电压经过两个 R 的衰减。让我们算清楚。

D₀ 的 8V 经过 2R 电阻到节点 C，从节点 C 到节点 B 经过 R，再从节点 B 到节点 A 经过 R，然后节点 A 通过另一个 2R 到虚地。

等效电路：D₀ 的 8V 源内阻 2R，经过 R+R=2R 串联，再经 2R 到地。

用戴维南等效：D₀→C 的等效电压源 = $8 \times (2R)/(2R+2R) = 4\text{ V}$（如果 D₀ 经过 2R 到一个由节点 C 往后看的 2R 等效电阻的话）。

实际上，R-2R 网络有个优雅的特性：**每一位对输出电流的贡献恰好是前一位的一半**。

MSB（bit2）贡献电流 = $V_{ref} / (2R) \times 1/1 = 8 / 2\text{k} = 4\text{ mA}$。
bit1 贡献电流 = $4 / 2 = 2\text{ mA}$（如果 bit1=1 的话）。
LSB（bit0）贡献电流 = $2 / 2 = 1\text{ mA}$（如果 bit0=1 的话）。

输入 101 → $I_{total} = 1 \times 4\text{mA} + 0 \times 2\text{mA} + 1 \times 1\text{mA} = 5\text{ mA}$。

**第 3 步：计算输出电压**

运放反相输入端虚地，反馈电阻 $R_f = 1\text{k}\Omega$：
$$
V_{out} = -I_{total} \cdot R_f = -5\text{ mA} \times 1\text{k}\Omega = -5\text{ V}
$$

用公式验证：
$$
V_{out} = -V_{ref} \cdot \frac{D_{in}}{2^n} = -8 \cdot \frac{5}{8} = -5\text{ V} \quad \checkmark
$$

完美吻合。你现在明白为什么公式是 $V_{ref} \cdot D_{in} / 2^n$ 了吧？总电流 = $V_{ref}/(2R) \cdot D_{in} / 2^{n-1}$（每一位比前一位电流减半，总和恰好是 $D_{in}$ 倍的 LSB 电流），输出 = $-I_{total} \cdot R$。

### 5f. 更多手算示例

**例 10**：4 位 R-2R DAC，$V_{ref} = 5\text{ V}$，$R_f = R$。输入数字 1001（二进制）= 9（十进制），求 $V_{out}$ 和 1 LSB。

$$
V_{out} = -5 \cdot \frac{9}{16} = -5 \times 0.5625 = -2.8125\text{ V}
$$

$$
1 \text{ LSB} = \frac{5}{16} = 0.3125\text{ V}
$$

负号表示运放反相输出。实际应用中如果需要正电压，可以再串一级增益为 -1 的反相器，或使用同相 R-2R 拓扑。

**例 11**：8 位 DAC，$V_{ref} = 3.3\text{ V}$。最大输出电压？1 LSB？输入 128（0x80）时输出？

最大输出（输入 255 = $2^8-1$）：$V_{max} = 3.3 \times 255/256 \approx 3.287\text{ V}$。
1 LSB = $3.3/256 \approx 0.01289\text{ V} = 12.89\text{ mV}$。
输入 128：$V_{out} = 3.3 \times 128/256 = 1.65\text{ V}$，正好是 $V_{ref}$ 的一半。

**例 12**：12 位 DAC，$V_{ref} = 2.5\text{ V}$。要输出 $1.000\text{ V}$，输入数字应该是多少？

每个 LSB = $2.5 / 4096 = 0.0006104\text{ V}$。
需要的数字码 = $1.000 / 0.0006104 \approx 1638.4$，取整为 1638（0x0666）。
实际输出 = $2.5 \times 1638 / 4096 = 0.9998\text{ V}$，误差约 0.2 mV。

### 5g. Python 验证：DAC 输出特性

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def r2r_dac(digital_code, n_bits, Vref=5.0):
    """R-2R 梯形 DAC 理想输出"""
    return Vref * digital_code / (2**n_bits)

# 不同位数的 DAC 传输特性
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate([2, 3, 4]):
    codes = np.arange(0, 2**n)
    vout = [r2r_dac(c, n, 5.0) for c in codes]
    
    ax = axes[idx]
    ax.step(codes, vout, where='mid', linewidth=2, color='#2E86AB')
    ax.plot(codes, vout, 'o', markersize=3, color='#A23B72')
    ax.set_xlabel('输入数字码', fontsize=11)
    ax.set_ylabel('输出电压 (V)', fontsize=11)
    ax.set_title(f'{n} 位 DAC\n1 LSB = {5.0/(2**n):.3f} V', fontsize=11)
    ax.set_xticks(codes)
    ax.grid(True, alpha=0.3)

plt.suptitle('不同位数 DAC 的传输特性（台阶效果）', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 打印 4 位 DAC 的完整查找表
print("=== 4位 R-2R DAC (Vref=5V) 完整查找表 ===")
print(f"{'二进制':>6}  {'十进制':>6}  {'Vout(V)':>10}")
print("-" * 30)
for i in range(16):
    v = r2r_dac(i, 4, 5.0)
    print(f"  {i:04b}     {i:3d}       {v:8.4f}")
```

### 5h. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "DAC 输出是完全连续的模拟信号" | DAC 输出本质上是阶梯状的（零阶保持），每个输入码对应一个恒定电压。平滑的模拟信号需要后续的**重建滤波器**（低通滤波）滤除高频台阶。 |
| "R-2R 比权电阻好，所以权电阻没用了" | 权电阻 DAC 在低速、低位（≤6位）应用中结构更简单。R-2R 的电阻数量是权电阻的两倍。各有取舍。 |
| "位数越多越好" | 16 位 DAC 的 1 LSB 在 $V_{ref}=5\text{V}$ 时只有 76 μV。此时电路噪声、电阻匹配精度、温度漂移等因素可能已经超过这个值。盲目堆位数没有意义——"有效位数"（ENOB）才是关键。 |

---

## 6. ADC：把模拟变成数字

### 6a. 生活例子：电子温度计

你房间里的电子温度计显示"26.3°C"。但感温元件（热敏电阻或热电偶）输出的只是一个和温度成正比的**连续变化的电压**（模拟量）。要把这个电压变成"26.3"这个数字，就需要 ADC（Analog-to-Digital Converter，模数转换器）。

所有 ADC 内部**必定包含一个 DAC**。为什么？因为 ADC 的判定过程本质上是"猜一个数字→用 DAC 转成模拟→和输入比较→调整猜测→再比较"。DAC 是 ADC 的"试金石"。

### 6b. 采样定理（Nyquist-Shannon 定理）

把连续的模拟信号变成离散的数字信号，第一步是**采样**（Sampling）：每隔 $\Delta t$ 时间记录一次信号的瞬时电压值。采样频率 $f_s = 1/\Delta t$。

**奈奎斯特采样定理**（Nyquist-Shannon Sampling Theorem）：

> 要无失真地从采样点恢复原始连续信号，采样频率必须**严格大于**信号最高频率的两倍。
>
> $$f_s > 2 \cdot f_{max}$$

$f_{max}$ 是信号中存在的最高频率分量。$f_s/2$ 被称为**奈奎斯特频率**（Nyquist Frequency）。

**为什么是 2 倍？** 直观理解：正弦波一个周期至少需要采两个点才能确定它的频率和幅度。如果只采一个点，你完全不知道这是高频信号还是低频信号——这就是**混叠**（Aliasing）。

混叠举例：一个频率 9 kHz 的正弦波，用 10 kHz 采样。采样后的点看起来像是 1 kHz 的正弦波（10-9=1 kHz）！高频"伪装"成了低频。这就是为什么在 ADC 之前一定要有**抗混叠低通滤波器**（Anti-Aliasing Filter），把高于 $f_s/2$ 的频率分量滤干净。

实际中的采样频率：
- **电话**：人声最高约 3.4 kHz，采样 8 kHz（8k > 2×3.4k=6.8k ✓）。
- **CD 音质**：人耳上限约 20 kHz，采样 44.1 kHz（44.1k > 40k ✓，留出约 4.1k 过渡带给抗混叠滤波器）。
- **高解析度音频**：采样 96 kHz 或 192 kHz（捕捉超声波，或简化滤波器设计）。

### 6c. 并联比较型 ADC（Flash ADC）

**原理**：最"暴力"也最快的方法。用 $2^n - 1$ 个比较器，每个比较器的参考电压由精密电阻分压网络提供，台阶间隔恰好为 1 LSB。输入模拟电压同时送到所有比较器的正输入端。比较器们一起工作：输入电压超过参考电压的比较器输出 1，没超过的输出 0。这串"温度计码"（Thermometer Code，如 00001111）经过一个编码器转成 n 位二进制输出。

**3 位 Flash ADC 示例**：需要 $2^3 - 1 = 7$ 个比较器。参考电压分别为 $V_{ref}/8, 2V_{ref}/8, \dots, 7V_{ref}/8$。

输入电压 $V_{in} = 0.43 V_{ref}$ 时：0.43×8=3.44，高于 1/8, 2/8, 3/8 的比较器输出 1，低于 4/8, 5/8, 6/8, 7/8 的输出 0 → 温度计码 = 0000111 → 编码为二进制 011（=3），对应 $3 \times V_{ref}/8 = 0.375 V_{ref}$。量化误差 = $0.43 - 0.375 = 0.055 V_{ref}$。

**优点**：极致速度（所有比较器并行工作 + 一个编码器延迟，总时间可小于 1 ns，采样率可达 GHz 级别）。

**缺点**：比较器数量随位数指数增长——8 位需要 255 个，10 位需要 1023 个！功耗和芯片面积爆炸式增长。一般只用于需要极高速度但位数要求不高的场景（示波器前端、雷达接收机、高速通信），通常 6-8 位。

### 6d. 逐次逼近型 ADC（SAR ADC）

**原理**：二分搜索法（Binary Search）。就像猜数字游戏（1-100）——你每次猜中间值，对方告诉你"大了"或"小了"，你据此缩小范围。n 次猜测就确定 n 位。

**SAR ADC 工作流程**（n 位）：

1. SAR（Successive Approximation Register，逐次逼近寄存器）先把**最高位 MSB** 置 1，其余位全 0。这个"试探值"送给内部 DAC。
2. DAC 把试探值转成模拟电压 $V_{DAC}$。
3. 比较器比较 $V_{in}$ 和 $V_{DAC}$：
   - 如果 $V_{in} > V_{DAC}$ → "猜小了" → MSB 保持 1。
   - 如果 $V_{in} < V_{DAC}$ → "猜大了" → MSB 清 0。
4. 把**下一位**置 1，重复步骤 2-3。
5. 经过 n 次比较后，SAR 寄存器中就是最终的 n 位结果。

**手算示例（4 位 SAR ADC）**：$V_{ref} = 8\text{ V}$，$V_{in} = 5.3\text{ V}$，1 LSB = 0.5 V。

| 步骤 | 试探值 | DAC输出(V) | 与 $V_{in}=5.3$ 比较 | 决策 |
|------|--------|-----------|---------------------|------|
| 1 | 1000(=8) | $8 \times 8/16 = 4.0$ | $4.0 < 5.3$，小了 | b₃=1 ✓ |
| 2 | 1100(=12) | $8 \times 12/16 = 6.0$ | $6.0 > 5.3$，大了 | b₂=0 ✗ |
| 3 | 1010(=10) | $8 \times 10/16 = 5.0$ | $5.0 < 5.3$，小了 | b₁=1 ✓ |
| 4 | 1011(=11) | $8 \times 11/16 = 5.5$ | $5.5 > 5.3$，大了 | b₀=0 ✗ |

最终结果：1010 = 10 → $10 \times 0.5 = 5.0\text{ V}$。量化误差 = $5.3 - 5.0 = 0.3\text{ V}$。

**SAR ADC 特点**：速度中等（一次转换需 n 个时钟周期），精度中等到高（8-18 位），功耗低，成本低。绝大多数单片机（MCU）内置的 ADC 都是 SAR 型。STM32、Arduino、ESP32 的模拟输入引脚，都靠 SAR ADC 工作。

### 6e. 量化误差与信噪比

ADC 把连续的模拟量变成离散数字量，精度有限——这个有限精度引入的误差叫**量化误差**（Quantization Error）。

理想的均匀量化器，最大量化误差为 **±½ LSB**。因为当输入电压刚好在两个量化台阶正中间时，无论往哪边舍入误差都是 ½ LSB。

$$
\text{量化误差} = V_{in} - V_{quantized},\quad |\text{误差}| \le \frac{1}{2} \text{LSB} = \frac{V_{ref}}{2^{n+1}}
$$

量化误差可以建模为一种加性噪声（量化噪声）。理想 ADC 的信噪比（SNR，Signal-to-Noise Ratio）近似为：

$$
\text{SNR(dB)} \approx 6.02n + 1.76
$$

这个公式非常有价值——它说**每增加 1 位分辨率，SNR 约提高 6 dB**。

- 8 位 ADC：SNR ≈ 49.9 dB
- 12 位 ADC：SNR ≈ 74.0 dB
- 16 位 ADC：SNR ≈ 98.1 dB
- 24 位 ADC：SNR ≈ 146.2 dB（实际受其他噪声限制，达不到这个理论值）

### 6f. 手算示例：ADC 参数

**例 13**：10 位 ADC，$V_{ref} = 5\text{ V}$，$V_{in} = 2.35\text{ V}$。量化结果和误差？

1 LSB = $5 / 1024 \approx 0.004883\text{ V} = 4.883\text{ mV}$。
数字码 = $\text{round}(2.35 / 0.004883) = \text{round}(481.3) = 481$。
量化电压 = $481 \times 0.004883 \approx 2.3486\text{ V}$。
误差 = $2.35 - 2.3486 = 0.0014\text{ V} = 1.4\text{ mV} < \frac{1}{2}\text{LSB} = 2.44\text{ mV}$ ✓

**例 14**：要分辨 1 mV 的电压变化，$V_{ref}=3.3\text{ V}$，需要多少位 ADC？

要求 1 LSB ≤ 1 mV → $3.3 / 2^n \le 0.001$ → $2^n \ge 3300$。
$2^{11}=2048 < 3300$，$2^{12}=4096 \ge 3300$。
至少需要 12 位。

**例 15**：16 位 ADC 的理论 SNR？如果实际 SNR 只有 86 dB，有效位数（ENOB）是多少？

理论 SNR = $6.02 \times 16 + 1.76 = 98.08\text{ dB}$。
实际 SNR 86 dB → ENOB = $(86 - 1.76) / 6.02 \approx 14.0$ 位。

这个 ADC 虽然标称 16 位，但由于电路噪声、非线性等因素，有效只有约 14 位。

### 6g. Python 验证：ADC 量化模拟

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def adc_quantize(vin, n_bits, Vref=5.0):
    """理想 ADC 量化"""
    lsb = Vref / (2**n_bits)
    digital = int(np.clip(round(vin / lsb), 0, 2**n_bits - 1))
    return digital, digital * lsb

# 对比不同位数 ADC 的量化效果
n_list = [2, 3, 4]
Vref = 5.0
vin = np.linspace(0, Vref, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate(n_list):
    qv = np.array([adc_quantize(v, n, Vref)[1] for v in vin])
    error = vin - qv
    lsb = Vref / (2**n)
    
    ax = axes[idx]
    ax.plot(vin, vin, 'k--', alpha=0.3, label='理想 y=x')
    ax.step(vin, qv, where='mid', linewidth=2, color='#2E86AB', label='量化输出')
    ax.set_xlabel('输入电压 (V)', fontsize=11)
    ax.set_ylabel('量化后电压 (V)', fontsize=11)
    ax.set_title(f'{n} 位 ADC\n1 LSB = {lsb:.3f}V, 最大误差=±{lsb/2:.3f}V', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('不同位数 ADC 的量化效果对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# SAR ADC 逐次逼近模拟
def sar_adc_simulate(Vin, Vref, n_bits):
    """模拟 SAR ADC 的逐次逼近过程"""
    lsb = Vref / (2**n_bits)
    sar = 0
    print(f"\n=== {n_bits}位 SAR ADC 逼近过程 ===")
    print(f"Vref={Vref}V, Vin={Vin}V, 1LSB={lsb:.4f}V")
    print(f"{'步':>3} {'试探值':>8} {'DAC(V)':>10} {'比较':>12} {'决策':>8}")
    print("-" * 50)
    
    for i in range(n_bits-1, -1, -1):
        test = sar | (1 << i)
        dac_v = test * lsb
        
        if dac_v <= Vin:
            decision = "保持 1"
            sar = test
        else:
            decision = "清 0"
        
        comp = f"{dac_v:.4f} <= {Vin}" if dac_v <= Vin else f"{dac_v:.4f} > {Vin}"
        print(f"  {n_bits-i:2d}  {test:0{n_bits}b}({test:3d})  {dac_v:10.4f}  {comp:>12}  {decision:>8}")
    
    print(f"\n结果: {sar:0{n_bits}b} = {sar} → {sar*lsb:.4f}V")
    print(f"量化误差: {abs(Vin - sar*lsb):.4f}V")
    return sar

# 测试
sar_adc_simulate(Vin=5.3, Vref=8.0, n_bits=4)
sar_adc_simulate(Vin=1.8, Vref=3.3, n_bits=8)
```

---

## 7. 完整实战：从模拟信号到数字再回到模拟

### 实战题

有一个 100 Hz 的正弦波信号，幅度 ±2V，偏置在 2.5V（即信号电压在 0.5V ~ 4.5V 之间摆动）。用一个 8 位 ADC（$V_{ref}=5\text{V}$）以 1 kHz 采样。然后在数字域不做任何处理，直接用一个 8 位 DAC（$V_{ref}=5\text{V}$）还原。

要求：
1. 计算 ADC 的 1 LSB。
2. 判断采样频率是否满足 Nyquist 定理。
3. 用 Python 生成采样数据、量化、DAC 还原，画出原始信号和还原信号的对比图。
4. 分析还原信号和原始信号的差异来自哪里。

<details>
<summary><b>点击展开解答</b></summary>

1. 1 LSB = 5/256 ≈ 0.01953 V = 19.53 mV。
2. 信号最高频率 100 Hz，奈奎斯特频率 200 Hz。采样 1 kHz > 200 Hz ✓，满足条件（5 倍过采样）。
3. Python 实现见下。
4. 差异来源：(a) 量化误差——8 位精度有限，±½LSB 的误差对 4V 峰峰值信号来说就是约 ±0.24% 的失真；(b) DAC 输出的零阶保持效应——阶梯状的输出在信号快速变化处产生高频谐波。

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 信号参数
f_sig = 100       # 信号频率 100Hz
f_sample = 1000   # 采样频率 1kHz
Vref = 5.0        # ADC/DAC 参考电压
n_bits = 8        # 分辨率
duration = 0.03   # 显示 30ms（3个周期）

t_fine = np.linspace(0, duration, 10000)
# 原始信号: 振幅2V, 偏置2.5V
v_orig = 2.5 + 2.0 * np.sin(2 * np.pi * f_sig * t_fine)

# 采样时刻
t_sample = np.arange(0, duration, 1/f_sample)
v_sample = 2.5 + 2.0 * np.sin(2 * np.pi * f_sig * t_sample)

# ADC 量化
lsb = Vref / (2**n_bits)
digital_codes = np.clip(np.round(v_sample / lsb), 0, 255).astype(int)
v_quantized = digital_codes * lsb

# DAC 还原（零阶保持——每个采样值保持到下一个采样点）
v_dac = np.zeros_like(t_fine)
for i, t in enumerate(t_fine):
    idx = int(t * f_sample)
    if idx < len(v_quantized):
        v_dac[i] = v_quantized[idx]
    else:
        v_dac[i] = v_quantized[-1]

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t_fine * 1000, v_orig, 'b-', linewidth=1.5, alpha=0.7, label='原始模拟信号')
ax1.step(t_fine * 1000, v_dac, 'r-', linewidth=1.5, where='post', label='DAC 还原信号')
ax1.plot(t_sample * 1000, v_quantized, 'ko', markersize=3, label='ADC 采样量化点')
ax1.set_xlabel('时间 (ms)')
ax1.set_ylabel('电压 (V)')
ax1.set_title(f'ADC→DAC 信号链 ({n_bits}位, fs={f_sample}Hz, 1LSB={lsb*1000:.1f}mV)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 误差曲线
error = v_orig - v_dac
ax2.plot(t_fine * 1000, error * 1000, 'm-', linewidth=1)
ax2.axhline(y=lsb/2*1000, color='gray', linestyle='--', label=f'+½LSB = {lsb/2*1000:.1f}mV')
ax2.axhline(y=-lsb/2*1000, color='gray', linestyle='--', label=f'-½LSB = {-lsb/2*1000:.1f}mV')
ax2.fill_between(t_fine * 1000, -lsb/2*1000, lsb/2*1000, alpha=0.1, color='green')
ax2.set_xlabel('时间 (ms)')
ax2.set_ylabel('误差 (mV)')
ax2.set_title('量化误差 (原始 - DAC还原)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"量化误差峰峰值: {np.max(np.abs(error))*1000:.2f} mV")
print(f"理论最大误差 ±½LSB: ±{lsb/2*1000:.2f} mV")
```
</details>

---

## 8. 思考题

### 基础题（推荐全部做完）

**题 1**：比较 ROM 和 RAM 的主要区别（至少四条）。

**题 2**：一个存储芯片标注"256K×16"。(a) 地址线数？(b) 数据线数？(c) 总容量（KB）？

**题 3**：下面哪种存储器掉电后数据不丢失？哪些可多次擦写？(A) SRAM (B) DRAM (C) EEPROM (D) Flash (E) 掩模 ROM。

**题 4**：6 位 R-2R DAC，$V_{ref} = 3.3\text{ V}$。输入 25，$V_{out}=?$，1 LSB = ?，最大输出 = ?

### 进阶题

**题 5**：用 2K×4 的 SRAM 芯片构建 8K×8 的存储系统。(a) 需要多少片？(b) 画出结构框图（地址线、数据线、片选译码器如何连接）。

**题 6**：8 位 SAR ADC，$V_{ref}=5\text{ V}$，$V_{in}=2.35\text{ V}$。写出完整的逐次逼近过程（8 步），给出最终数字码和量化误差。

**题 7**：一个模拟信号最高频率 15 kHz，要求 ADC 量化信噪比至少 60 dB。问：(a) 最小采样频率？(b) 最少需要多少位 ADC？

### 综合题

**题 8**：用 Python 模拟一个非理想 ADC，其中每一个量化台阶的实际电压有 ±0.3 LSB 的随机偏差（模拟 DNL 误差）。输入一个斜坡信号（0→Vref 线性增长），画出理想量化曲线和非理想量化曲线，并绘制 DNL 和 INL 曲线。

---

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

**题 1**：

| 特性 | ROM | RAM |
|------|-----|-----|
| 可写性 | 只读或有限擦写 | 可无限次快速读写 |
| 掉电后 | 数据保留（非易失性） | 数据丢失（易失性） |
| 写入速度 | 慢（微秒~毫秒级） | 快（纳秒级） |
| 擦写次数 | 有限（几十到百万次） | 几乎无限 |
| 成本/bit | 低（大批量掩模 ROM） | DRAM 低，SRAM 高 |
| 典型用途 | 固件、BIOS、程序存储 | 主内存、缓存 |

**题 2**：(a) $256K = 2^{18}$，18 根地址线。(b) 16 根数据线。(c) $256K \times 16 \text{ bits} = 4096K \text{ bits} = 512K \text{ Bytes}$。

**题 3**：掉电不丢（非易失）：C(EEPROM)、D(Flash)、E(掩模 ROM)。可多次擦写：C(EEPROM 约 10 万次)、D(Flash 约 1000-10 万次)。A(SRAM)和 B(DRAM)掉电即丢。

**题 4**：$V_{out} = 3.3 \times 25/64 = 1.289\text{ V}$。1 LSB = $3.3/64 = 0.0516\text{ V} = 51.6\text{ mV}$。最大输出（输入 63）= $3.3 \times 63/64 = 3.248\text{ V}$。

**题 5**：(a) 位扩展需 2 片（4bit→8bit），字扩展需 8K/2K=4 组，总 8 片。(b) A₀-A₁₀（11位片内地址）接所有芯片。A₁₁,A₁₂ 接 2-4 译码器，输出分别接 4 组的 CS。D₀-D₃ 接每组第一片，D₄-D₇ 接每组第二片。

**题 6**：1 LSB = 5/256 ≈ 0.01953 V。

| 步 | 试探 | DAC(V) | 比较 | 决策 |
|----|------|--------|------|------|
| 1 | 128(10000000) | 2.500 | >2.35,大 | b₇=0 |
| 2 | 64(01000000) | 1.250 | <2.35,小 | b₆=1 |
| 3 | 96(01100000) | 1.875 | <2.35,小 | b₅=1 |
| 4 | 112(01110000) | 2.188 | <2.35,小 | b₄=1 |
| 5 | 120(01111000) | 2.344 | <2.35,小 | b₃=1 |
| 6 | 124(01111100) | 2.422 | >2.35,大 | b₂=0 |
| 7 | 122(01111010) | 2.383 | >2.35,大 | b₁=0 |
| 8 | 121(01111001) | 2.363 | >2.35,大 | b₀=0 |

结果：01111000 = 120。量化电压 = 120×0.01953 = 2.344 V。误差 = 2.35-2.344 = 0.006 V < ½LSB(0.0098V) ✓

**题 7**：(a) Nyquist 要求 $f_s > 2 \times 15\text{k} = 30\text{ kHz}$。最小采样频率略大于 30 kHz（实际一般选 40-48 kHz 留余量）。(b) SNR ≥ 60 dB，$6.02n + 1.76 \ge 60$ → $n \ge (60-1.76)/6.02 \approx 9.67$。至少 10 位。

**题 8**：参考第 7 节的 Python 代码，在 adc_quantize 函数中加入随机 DNL 误差即可。

```python
def nonideal_adc(vin, n_bits, Vref=5.0, dnl_std=0.3):
    """带 DNL 误差的非理想 ADC"""
    lsb = Vref / (2**n_bits)
    # 每个量化台阶加随机偏差
    thresholds = np.array([(i + 0.5) * lsb + np.random.normal(0, dnl_std * lsb) 
                           for i in range(2**n_bits - 1)])
    digital = np.searchsorted(thresholds, vin)
    return digital, digital * lsb
```

完整 DNL/INL 绘图代码较长，思路：计算每个数字码对应的实际电压台阶宽度，DNL = (实际台阶 - 1LSB)/LSB，INL = DNL 的累积和。

</details>

---

## 补充专题：三种主流 ADC 架构对比

前面讲了 Flash ADC、SAR ADC 和 Σ-Δ ADC。为了帮你更好地记住它们的特点和适用场景，这里做一个综合对比。

### 工作原理对比

| ADC 类型 | Flash（并联比较） | SAR（逐次逼近） | Σ-Δ（Sigma-Delta） |
|----------|-----------------|---------------|-------------------|
| 核心思路 | 所有比较器同时比较 | 二分搜索逐一试探 | 过采样+噪声整形+数字滤波 |
| 内部 DAC 精度 | $2^n-1$ 个比较器分压 | 需要 n 位 DAC | 只需 1 位 DAC |
| 每次转换时钟数 | 1 | n（位数） | 非常高（过采样） |
| 典型位数 | 6-8 位 | 8-18 位 | 16-24 位 |
| 最高采样率 | >1 GHz | 1-10 MHz | <1 MHz |
| 功耗 | 极高（大量比较器） | 低到中等 | 中等 |
| 芯片面积 | 极大 | 小 | 中等（数字部分为主） |
| 线性度 | 受比较器匹配限制 | 受 DAC 精度限制 | 非常好（不依赖元件匹配） |
| 典型应用 | 示波器、雷达、高速通信 | MCU内置ADC、传感器 | 音频、精密测量 |

### 为什么没有一种"全能"ADC？

因为**速度、精度、功耗、成本**四者此消彼长：
- Flash 用"面积换速度"（数百个比较器→GHz 级采样）。
- SAR 用"时间换精度"（逐次比较→中高精度、低成本）。
- Σ-Δ 用"频率换分辨率"（超高采样率→24位精度，但速度慢）。

选型就一句话：**你需要多快？需要多准？给多少预算和功耗？** 三个答案决定了你选哪种 ADC。

---

## 补充专题：DRAM 时序参数入门

如果你买过内存条，可能看到过这些数字：DDR4-3200 CL16-18-18-38。这些"时序参数"（Timings）其实就是 DRAM 各个操作之间的等待时间，以时钟周期为单位。

- **CL（CAS Latency，列地址选通延迟）**：从发出"读列地址"到数据出现在数据总线上，需要等多少个时钟周期。CL16 表示等 16 个时钟周期。这是最重要的时序参数，直接影响内存延迟。
- **tRCD（RAS to CAS Delay）**：从发出行地址到可以发出列地址的等待时间。因为先要激活一整行，灵敏放大器把整行数据"锁存"后，才能选列。
- **tRP（RAS Precharge）**：关闭当前行、准备激活下一行的等待时间。需要给位线预充电。
- **tRAS（Active to Precharge）**：行激活后到可以关闭该行的最短时间。

这些参数背后是电容充放电、灵敏放大器建立等物理限制。时序数字越小，内存越快，但制造难度和成本也越高。

**例**：DDR4-3200，时钟频率 1600 MHz（因为 DDR 在上下沿都传数据，所以"3200 MT/s"对应的时钟是 1600 MHz），时钟周期 = 0.625 ns。CL16 意味着读延迟 = 16 × 0.625 = 10 ns。

---

## 补充专题：新型非易失存储技术简介

除了我们熟悉的 ROM、SRAM、DRAM、Flash，学术界和产业界正在研发多种"下一代存储器"，目标是同时做到 SRAM 的速度和 Flash 的非易失性。这里简要介绍几种：

**MRAM（磁阻 RAM，Magnetoresistive RAM）**：
利用**磁性隧道结**（MTJ）存储数据。两个铁磁层中，一层磁化方向固定，另一层可以翻转。两层磁化方向平行→低电阻（0），反平行→高电阻（1）。通过电流产生磁场或自旋转移矩（STT）来写入。速度接近 SRAM（~10 ns）、非易失、擦写寿命几乎无限（>10¹⁵次）。已在部分嵌入式 MCU 和航天级芯片中商用。

**RRAM / Memristor（忆阻器 RAM）**：
利用某些金属氧化物在电压作用下电阻可逆变化的特性。加正向电压→低电阻（SET），加反向电压→高电阻（RESET）。结构简单（金属-氧化物-金属三层）、密度极高、功耗低。被看好用于存算一体（In-Memory Computing）和神经网络加速。

**FeRAM（铁电 RAM，Ferroelectric RAM）**：
利用铁电材料（如 PZT）的极化方向存储数据。加正电场→正极化（1），加负电场→负极化（0）。读取时检测极化翻转产生的电流。写入速度极快（~10 ns）、功耗极低（不需要高电压）。但密度受限于材料工艺，容量通常较小（几 MB）。适合 IoT 设备（极低功耗是关键）。

**3D XPoint（Intel Optane / Micron QuantX）**：
一种商用化的"存储级内存"（SCM, Storage Class Memory），性能介于 DRAM 和 NAND Flash 之间。比 Flash 快约 1000 倍，比 DRAM 密度高、成本低、非易失。通过 3D 堆叠和相变或类似机制存储数据。已用于数据中心加速（Redis、数据库缓存等场景）。不过 Intel 在 2021 年已停止消费级 Optane 产品线。

这些新技术如果成熟并大规模量产，可能会模糊"内存"和"硬盘"之间的传统界限——计算机的存储层次结构可能从目前的 5-6 层简化到 2-3 层。

---

## 补充专题一：Σ-Δ (Sigma-Delta) ADC

### 为什么需要 Σ-Δ ADC？

前面讲的 SAR ADC 和 Flash ADC 各有局限：SAR 做到了中高精度但速度中等；Flash 做到了极致速度但位数上不去。如果想要 **16 位、甚至 24 位的超高精度**，怎么办？

答案是用**不同的思路**：不以"精确地量出每个采样点的电压"为目标，而是用**过采样（Oversampling）+ 噪声整形（Noise Shaping）**技术，以极高的采样率换取极高的分辨率。

### 噪声整形的直觉

想象你要用一个很粗糙的秤（只能精确到 1 kg）来称一个 0.37 kg 的苹果。单独称一次，只能得到 0 kg（误差 0.37 kg）。但如果你把苹果放上去，快速取下来、再放上去、再取下来……连续称 1000 次，苹果在秤上的时间占比是 37%。然后你对这 1000 次的 0/1 读数取平均——平均值就是 0.37！

Σ-Δ ADC 的核心思路和这个一模一样：用一个 1 位 ADC（只是一个比较器！）以极高的频率（比如信号带宽的 256 倍）采样。每一次比较判断"当前输入比参考大还是小"。大多数读数会是错的（1 位精度嘛），但经过**数字低通滤波取平均**后，平均值极其精确。

那"噪声整形"又是什么？Σ-Δ ADC 内部有一个反馈环路，专门把 1 位量化引入的量化噪声"推"到高频区域。因为采样率远超信号带宽，高频噪声可以轻松用数字滤波器去掉。去掉噪声后，低频部分的分辨率极高——这就是 Σ-Δ ADC 能达到 24 位的原因。

### Σ-Δ ADC 的特点

| 特性 | Σ-Δ ADC | SAR ADC | Flash ADC |
|------|---------|---------|-----------|
| 分辨率 | 极高（16-24 位） | 中高（8-18 位） | 低（6-8 位） |
| 速度 | 慢（kHz 级） | 中等（MHz 级） | 极快（GHz 级） |
| 内部 ADC | 1 位比较器 | n 位 DAC | $2^n-1$ 个比较器 |
| 核心技术 | 过采样+噪声整形 | 二分搜索 | 并行比较 |
| 典型应用 | 音频录音、精密测量 | MCU 内置 ADC | 示波器、雷达 |

你在录音声卡、专业音频接口、数字万用表里见到的 ADC，几乎全是 Σ-Δ 型。

---

## 补充专题二：DAC 的非线性指标详解

### 微分非线性（DNL）和积分非线性（INL）

前面提到了 DNL 和 INL，这里用图来更直观地理解。

**DNL（Differential Non-Linearity，微分非线性）**：

理想 DAC 中，输入码每增加 1，输出电压恰好增加 1 LSB。DNL 衡量的是**实际的电压增量与理想 1 LSB 的偏差**。

$$
\text{DNL}(i) = \frac{V_{out}(i) - V_{out}(i-1) - 1\text{ LSB}}{1\text{ LSB}}
$$

- DNL = 0 → 完美（每步恰好 1 LSB）
- DNL = +0.5 → 这一步比理想大了 0.5 LSB
- DNL = -0.3 → 这一步比理想小了 0.3 LSB
- DNL < -1 → **失码**（Missing Code）！这个数字码的输出电压比上一个还低——这是灾难性缺陷，意味着 DAC 非单调。

**INL（Integral Non-Linearity，积分非线性）**：

INL 是 DNL 的累积效果，表示实际输出电压与理想直线（端点拟合）之间的偏差。

$$
\text{INL}(i) = \frac{V_{out}(i) - V_{ideal}(i)}{1\text{ LSB}} = \sum_{j=1}^{i} \text{DNL}(j)
$$

高质量 DAC 通常要求 $|\text{DNL}| < 0.5$ LSB 且 $|\text{INL}| < 1$ LSB。

### Python 模拟非理想 DAC 的 DNL/INL

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def simulate_dac_dnl_inl(n_bits, Vref=5.0, dnl_std=0.15, inl_scale=0.3):
    """模拟带 DNL/INL 误差的非理想 DAC"""
    lsb = Vref / (2**n_bits)
    n_codes = 2**n_bits
    
    # 理想输出
    v_ideal = np.arange(n_codes) * lsb
    
    # 添加随机 DNL（每个台阶独立）
    dnl = np.random.normal(0, dnl_std, n_codes)
    dnl[0] = 0  # 第一个码没有 DNL
    
    # INL 是 DNL 的累积
    inl = np.cumsum(dnl)
    
    # 加一个平滑的正弦型 INL（模拟系统非线性）
    inl += inl_scale * np.sin(2 * np.pi * np.arange(n_codes) / n_codes)
    
    # 实际输出 = 理想 + INL(以LSB为单位)
    v_actual = v_ideal + inl * lsb
    
    return v_ideal, v_actual, dnl, inl

n = 6  # 6位便于可视化
v_ideal, v_actual, dnl, inl = simulate_dac_dnl_inl(n)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 传输特性
ax = axes[0, 0]
ax.plot(v_ideal / (5.0/2**n), v_ideal, 'b-', linewidth=1, label='理想')
ax.plot(v_ideal / (5.0/2**n), v_actual, 'r.', markersize=2, label='实际')
ax.set_xlabel('数字码'); ax.set_ylabel('Vout (V)')
ax.set_title(f'{n}位 DAC 传输特性'); ax.legend(); ax.grid(True, alpha=0.3)

# 误差
ax = axes[0, 1]
error = (v_actual - v_ideal) * 1000  # mV
ax.plot(range(2**n), error, 'm-', linewidth=1)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('数字码'); ax.set_ylabel('误差 (mV)')
ax.set_title('输出电压误差'); ax.grid(True, alpha=0.3)

# DNL
ax = axes[1, 0]
ax.bar(range(2**n), dnl, width=0.8, color='steelblue', edgecolor='none')
ax.axhline(y=1, color='red', linestyle='--', label='+1 LSB (失码界限)')
ax.axhline(y=-1, color='red', linestyle='--', label='-1 LSB (失码界限)')
ax.axhline(y=0.5, color='orange', linestyle=':', label='±0.5 LSB')
ax.axhline(y=-0.5, color='orange', linestyle=':')
ax.set_xlabel('数字码'); ax.set_ylabel('DNL (LSB)')
ax.set_title('微分非线性 (DNL)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# INL
ax = axes[1, 1]
ax.plot(range(2**n), inl, 'g-', linewidth=1.5)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.fill_between(range(2**n), -1, 1, alpha=0.1, color='green', label='±1 LSB')
ax.set_xlabel('数字码'); ax.set_ylabel('INL (LSB)')
ax.set_title('积分非线性 (INL)'); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle(f'{n}位非理想DAC的DNL/INL分析', fontsize=14)
plt.tight_layout()
plt.show()

print(f"DNL 最大值: {np.max(np.abs(dnl)):.3f} LSB")
print(f"INL 最大值: {np.max(np.abs(inl)):.3f} LSB")
print(f"是否单调: {'是' if np.all(np.diff(v_actual) > 0) else '否 (存在失码!)'}")
```

---

## 补充专题三：实际音频信号链中的 ADC 和 DAC

### 你的手机是怎样录音和播放的？

这个流程你每天都在用，但从电路的角度看它极其精妙。

**录音链路（模拟→数字）**：

```
声波 → [麦克风] → 微弱电信号 (mV级)
     → [前置放大器] → 放大到 V 级
     → [抗混叠低通滤波器] → 滤除 >20kHz 分量
     → [Σ-Δ ADC] → 24位/48kHz 数字音频流
     → [数字信号处理器] → 编码为 MP3/AAC
     → 存储到 Flash
```

每一步的限制：
- 麦克风决定了信号的最低噪声水平（本底噪声）。
- 前置放大器引入了热噪声，信噪比（SNR）被限制在约 100-120 dB。
- 抗混叠滤波器必须非常陡峭——在 20 kHz 通带和 24 kHz（奈奎斯特频率的一半=48k/2）阻带之间，只有 4 kHz 的过渡带！这要求高阶滤波器（5-7 阶）。
- Σ-Δ ADC 的过采样率通常为 64-256 倍，对于 48 kHz 输出，内部采样率可能高达 $48\text{k} \times 256 \approx 12.288\text{ MHz}$！

**播放链路（数字→模拟）**：

```
Flash 中的音频文件 → [解码器] → 16/24位 PCM 数据流
    → [数字滤波器/升采样] → 提高采样率
    → [Σ-Δ DAC 或 R-2R DAC] → 阶梯状模拟波形
    → [重建低通滤波器] → 平滑阶梯→连续信号
    → [功率放大器] → 驱动耳机/扬声器
    → [扬声器] → 声波！
```

DAC 之后的"重建滤波器"（Reconstruction Filter）至关重要——它把 DAC 输出的阶梯状波形平滑为连续的模拟信号。没有这个滤波器，你会听到刺耳的"数码声"（高频量化噪声）。

---

## 补充专题四：存储器的层次结构

### 为什么计算机不用一种存储器搞定一切？

原因很简单：**没有一种存储器能同时做到极快、极大容量、极便宜、掉电不丢**。现实中的存储器是在"速度"、"容量"、"成本"三者之间的权衡。现代计算机用**存储层次结构**（Memory Hierarchy）来解决：

```
 容量小                                      容量大
 速度极快                                    速度慢
 极贵                                        极便宜
 
 CPU寄存器 (B级) → L1 Cache (KB级, SRAM) → L2 Cache (MB级, SRAM)
                                              ↓
   机械硬盘 (TB级) ← SSD (GB-TB级, NAND Flash) ← 主内存 (GB级, DRAM)
```

每一层都是下一层的"缓存"：
- **寄存器**：在 CPU 内部，速度=1 个时钟周期，容量=几十到几百字节。
- **L1 Cache**：在 CPU 内部，SRAM，速度=2-4 个周期，容量=32-64 KB。
- **L2 Cache**：在 CPU 内部或极近处，SRAM，速度=10-20 周期，容量=256KB-1MB。
- **L3 Cache**：多核共享，SRAM，速度=30-50 周期，容量=8-32MB。
- **主内存（DRAM）**：速度=50-100 ns（约 200-400 周期），容量=8-64 GB。
- **SSD（NAND Flash）**：速度=10-100 μs（约 10 万周期），容量=256GB-4TB。
- **机械硬盘**：速度=5-10 ms（约 2000 万周期），容量=1-20TB。

从寄存器到机械硬盘，访问时间相差了 **7 个数量级**（从 0.3 ns 到 10 ms）！这就是为什么好的程序要尽量让数据待在 Cache 里——"缓存友好"的代码可以快 100 倍。

这段知识会在后续的计算机体系结构章节中详细展开。现在你只需要记住：**SRAM 做缓存、DRAM 做主存、Flash 做硬盘、ROM 做固件**，每种存储器都有它最适合的位置。

---

**恭喜！** 你完成了数字电路部分的全部 9 章。从最底层的 0 和 1（数制），到逻辑推理（布尔代数），到搭积木（逻辑门），到设计复杂电路（组合+时序），再到让数字和现实世界对话（ADC/DAC）。现在你已经有了看懂任何数字系统的基础能力。

下一章开始进入模拟电路——放大、滤波、反馈。那里是另一个精彩的世界。你会发现，虽然模拟电路里的电压是连续变化而非 0 和 1 的跳变，但分析方法和数字电路有很多相通之处：基尔霍夫定律、节点分析、戴维南等效……这些工具你都已经掌握了。
