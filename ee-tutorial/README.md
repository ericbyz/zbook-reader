# 电子工程教程 — 从电路基础到数电模电

## 概述

本教程是一套**从零开始**的电子工程系统学习资料，覆盖从电路基础理论到数字电路、模拟电路的完整学习路径。教程面向零基础学习者，不需要任何电子学的先修知识——我们会从"电是什么"开始，一步步带你建立完整的电路分析能力。

## 目标读者

- 对电子技术感兴趣的初学者（零基础也能学）
- 计算机/软件工程学生想了解硬件基础
- 想系统梳理电路知识的工程师
- 准备考研或面试需要复习电路/数电/模电的学习者
- 喜欢动手但缺乏理论基础的电子爱好者

## 前置要求

| 要求 | 说明 |
|------|------|
| 数学基础 | 会解一元一次方程、知道坐标系（其他数学在教程中从零讲解）|
| 好奇心 | 对"电是怎么工作的"感兴趣 |
| 学习态度 | 耐心——每个概念都配有详细推导和 Python 代码示例 |

> **不需要**：物理学先修课、微积分、电路基础。这些都会在教程中从零讲起。

## 内容地图

本教程按学习顺序组织，建议按序阅读。每个部分独立成章，也支持按需跳转。

```
电子工程教程 (ee-tutorial)
│
├── 📖 README.md                        ← 你在这里
│
├── 📐 第零部分：数学与物理基础
│   ├── math-physics.md                  物理与数学基础：原子/电荷/单位/方程/微积分直觉
│   ├── math-complex.md                  复数与相量：虚数/极坐标/欧拉公式/相量预览
│   └── math-signals.md                  信号基础：正弦波/频谱/DC vs AC/信号运算
│
├── 🔌 第一部分：电路学 — 直流电路
│   ├── 01-what-is-ee.md                 什么是电气工程？
│   ├── 02-charge-voltage-current.md     电荷、电压与电流
│   ├── 03-resistance-ohm.md             电阻与欧姆定律
│   ├── 04-basic-circuit-elements.md     电路基本元件与拓扑
│   ├── 05-series-parallel.md            串联与并联电路
│   ├── 06-kirchhoff-laws.md             基尔霍夫定律
│   ├── 07-node-mesh-analysis.md         节点电压法与网孔电流法
│   ├── 08-thevenin-norton.md            戴维南与诺顿定理
│   ├── 09-superposition-maxpower.md     叠加定理与最大功率传输
│   ├── 10-capacitors.md                 电容器
│   ├── 11-inductors.md                  电感器
│   ├── 12-first-order-rc.md             RC 电路的充放电
│   ├── 13-first-order-rl.md             RL 电路的动态响应
│   └── 14-second-order-rlc.md           二阶 RLC 电路与振荡
│
├── ⚡ 第二部分：电路学 — 交流电路
│   ├── 15-ac-sinusoidal.md              正弦交流电基础
│   ├── 16-phasor-impedance.md           相量法与阻抗
│   ├── 17-ac-power.md                   交流功率分析
│   └── 18-frequency-resonance.md        频率响应与谐振
│
├── 💾 第三部分：数字电路
│   ├── 19-number-systems.md             数制与编码
│   ├── 20-boolean-algebra.md            布尔代数
│   ├── 21-logic-gates.md               逻辑门电路
│   ├── 22-combinational-design.md       组合逻辑设计
│   ├── 23-arithmetic-circuits.md        算术运算电路
│   ├── 24-latches-flipflops.md          锁存器与触发器
│   ├── 25-counters-registers.md         计数器与寄存器
│   ├── 26-sequential-design.md          时序逻辑设计
│   └── 27-memory-dac-adc.md             存储器与数模转换
│
└── 📻 第四部分：模拟电路
    ├── 28-semiconductor-physics.md      半导体物理基础
    ├── 29-diodes.md                     二极管与整流
    ├── 30-bjt-fundamentals.md           BJT 基础
    ├── 31-bjt-amplifiers.md             BJT 放大电路
    ├── 32-fet.md                        场效应晶体管
    ├── 33-opamp-basics.md               运算放大器基础
    ├── 34-opamp-applications.md         运放应用电路
    ├── 35-power-circuits.md             功率放大与稳压电源
    ├── 36-feedback-stability.md         负反馈与稳定性
    └── 37-practical-design.md           实际电路设计
```

## 教程特色

### 1. 零基础友好
每个概念都从"你每天都能见到的东西"开始引入。电压=水压、电流=水流、电阻=水管阻力……用生活类比建立直觉，再引入严格的数学定义。

### 2. 代码伴随
每个章节都配有 Python 代码（numpy/scipy/matplotlib），用代码模拟电路行为、验证手算结果。不用任何外部 SPICE 软件——纯 Python 就够了。

### 3. 手算为王
这不是"看看就懂"的教程。每章都有详细的手算示例，一步步带你算。电路分析是一项技能，只有动手才能掌握。

### 4. 渐进式结构
遵循「生活例子 → 直观理解 → 形式化定义 → 手算示例 → Python 验证 → 常见误区 → 应用连接」的七步讲解模式。

### 5. 三个完整模块
- **电路学**（14+4 章）：从"电是什么"到能分析任意电路
- **数字电路**（9 章）：从"0 和 1"到设计时序逻辑系统
- **模拟电路**（10 章）：从"硅为什么特殊"到设计实际放大器

## 如何使用本教程

1. **按顺序学**：章节之间有依赖关系，建议从 math-physics.md 开始
2. **动手算**：每个手算示例请拿纸笔跟着算一遍
3. **跑代码**：每个 Python 示例都建议亲自运行、修改参数观察结果
4. **做思考题**：每章末尾有 5-8 道练习题，附完整解答

## 环境准备

```bash
# 推荐使用 conda 创建独立环境
conda create -n ee-tutorial python=3.10
conda activate ee-tutorial

# 安装核心依赖
pip install numpy scipy matplotlib
```

## 推荐学习顺序

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 阶段 0 | 数学物理基础（3 章） | 1 周 |
| 阶段 1 | 直流电路（14 章） | 4-6 周 |
| 阶段 2 | 交流电路（4 章） | 2 周 |
| 阶段 3 | 数字电路（9 章） | 3-4 周 |
| 阶段 4 | 模拟电路（10 章） | 4-6 周 |

**总计：约 14-19 周（稳定节奏）**

---

**准备好了吗？让我们从 [物理与数学基础](./math-physics.md) 开始你的电子工程之旅！**
