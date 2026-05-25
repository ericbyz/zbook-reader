# 概率论：从抛硬币到机器学习

> 你有没有想过：天气预报说"明天 70% 下雨"到底是什么意思？为什么连抛 10 次硬币都是反面也不能说明硬币有问题？医生告诉你"检测阳性"，但你真的得病了吗？**概率论就是回答这些问题的数学语言。**

---

## 为什么概率对机器学习至关重要？

在机器学习的世界里，你拿到的数据永远不完美——传感器有噪声，标注有人为错误，样本只是总体的一小部分。更重要的是，**分类器本质上输出的是概率**（"这张图 85% 是猫"）。如果你不理解概率，你就不知道模型凭什么自信，也不知道该不该信任它的判断。

简单说：**没有概率论，机器学习就是在猜。**

---

## 1. 什么是概率？——从抛硬币说起

### 生活例子

你站在窗前，看到天边乌云翻滚。你心想："今天八成要下雨。"——你刚刚完成了一次概率判断。你不需要知道所有空气分子的运动轨迹，你只需要根据**过去的经验**给出一个**可能性的量化**。

另一个例子：你和朋友打赌抛硬币。公平硬币正反面各 50%——这个"50%"就是概率。

### 直观理解

**概率是一个 0 到 1 之间的数字，用来表示某件事发生的可能性。** 0 = 绝对不可能（比如太阳从西边升起），1 = 必然发生（比如明天太阳照常升起），0.5 = 一半对一半（比如抛一枚公平硬币得到正面）。

你每天都在做概率推理：
- 早上出门前看窗外：乌云密布 → P(下雨) ≈ 0.8 → 带伞
- 打开外卖软件看评分：4.8 分 → P(好吃) ≈ 0.9 → 下单
- 身体不适去体检：检测阳性 → P(真得病\|阳性) = ?  （后面贝叶斯告诉你答案）

### 形式化定义：概率的三条公理

先介绍三个基本概念：

| 术语 | 定义 | 例子（掷骰子） |
|------|------|---------------|
| **样本空间 Ω** | 所有可能结果的集合 | Ω = {1, 2, 3, 4, 5, 6} |
| **事件 A** | 样本空间的子集 | A = {2, 4, 6}（掷出偶数） |
| **概率 P(A)** | 事件 A 发生的可能性 | P(偶数) = ? |

**概率的三条公理**（俄罗斯数学家柯尔莫哥洛夫提出）：

1. **非负性**：任何事件的概率都不小于 0 → P(A) ≥ 0
2. **规范性**：样本空间本身的概率为 1 → P(Ω) = 1（一定会发生某件事）
3. **可加性**：如果 A 和 B 互斥（不可能同时发生），则 P(A∪B) = P(A) + P(B)

> **频率学派 vs 贝叶斯学派**：频率学派说概率是长期频率的极限（"抛无数次硬币，正面比例 → 0.5"）；贝叶斯学派说概率是主观信念的程度（"根据现有信息，我有 60% 的把握认为会下雨"）。ML 中两者并存——最大似然估计是频率派，贝叶斯神经网络是贝叶斯派。

### 手算示例：掷骰子的概率

掷一枚均匀的六面骰子：

- 样本空间 Ω = {1, 2, 3, 4, 5, 6}，共 6 种等可能结果
- P(掷出 1) = 1/6 ≈ 0.167
- P(掷出偶数) = P({2, 4, 6}) = P(2) + P(4) + P(6) = 1/6 + 1/6 + 1/6 = 3/6 = 1/2 = 0.5
- P(掷出 ≥ 5) = P({5, 6}) = 2/6 = 1/3 ≈ 0.333
- P(掷出 7) = 0   （7 不在 Ω 中）
- P(掷出 ≤ 6) = 1   （所有结果都不大于 6）

> **验证公理**：P(1) + P(2) + P(3) + P(4) + P(5) + P(6) = 6 × (1/6) = 1 ✓

### Python 验证：频率趋近概率

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 模拟抛硬币——观察频率如何趋近真值 0.5
n_trials = [10, 50, 100, 500, 1000, 5000, 10000]
for n in n_trials:
    heads = np.random.binomial(n, 0.5)
    freq = heads / n
    print(f"抛 {n:6d} 次 → 正面 {heads:5d} 次 → 频率 {freq:.4f}  (真值 0.5000)")

# 可视化
n_vals = np.arange(1, 2001)
cum_heads = np.cumsum(np.random.binomial(1, 0.5, 2000))
cum_freq = cum_heads / n_vals

plt.figure(figsize=(10, 4))
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.7, label='真实概率 0.5')
plt.plot(n_vals, cum_freq, 'b-', linewidth=0.8, alpha=0.8)
plt.xlabel('抛掷次数')
plt.ylabel('累计正面频率')
plt.title('频率学派的核心思想：次数越多，频率越接近真实概率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 常见误区

| 误区 | 真相 |
|------|------|
| "连抛 5 次反面，第 6 次正面的概率更大" | 错！每次抛硬币独立，P(正面) 始终是 0.5。这是**赌徒谬误**。 |
| "概率 0 就是不可能" | 不一定。一根针落在平面上恰好是某条坐标的概率数学上为 0，但并非不可能。 |
| "天气 30% 下雨 = 30% 地区会下雨" | 错。意思是"在过去的类似天气条件下，有 30% 的日子下了雨"。 |

### ML 应用

| 概念 | ML 场景 |
|------|---------|
| 样本空间 Ω | 所有可能的分类标签 {猫, 狗, 鸟} |
| 事件 | "模型预测为猫的那部分样本" |
| 概率公理 | Softmax 输出各类概率之和必须为 1 |

---

## 2. 条件概率——证据如何改变你的判断

### 生活例子

你和朋友走在街上，看到远处一个身影。你猜：**"是熟人"** 的概率大约 10%（街上人很多）。但当你走近一点，看到对方穿着你朋友常穿的那件花衬衫——此时 P(是熟人\|花衬衫) 瞬间飙升到 80%。

再比如：天气预报说"明天 40% 的概率下雨"。你早上醒来，听到雷声滚滚——现在你心中的下雨概率就远不止 40% 了。**新证据改变了你的概率判断。**

### 直观理解

条件概率 P(A\|B) 描述的是：**在已知 B 发生的条件下，A 发生的概率**。竖线"\|"读作"给定"——P(A\|B) 读作"给定 B 时 A 的概率"。

核心直觉：**B 的发生把样本空间缩小了。** 原来你考虑"所有可能情况"，现在你只需考虑"B 已经发生的那部分"，然后在这部分中看 A 的比例。

### 形式化定义

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

这也能写成乘法规则：$P(A \cap B) = P(A|B) \cdot P(B)$

### 手算示例

**场景**：一个班 40 人，男生 25 人，女生 15 人。男生中戴眼镜的有 10 人，女生中戴眼镜的有 9 人。

|     | 戴眼镜 | 不戴眼镜 | 合计 |
|-----|--------|---------|------|
| 男生 | 10     | 15      | 25   |
| 女生 | 9      | 6       | 15   |
| 合计 | 19     | 21      | 40   |

**问题 1**：随机选一人，戴眼镜的概率是多少？

$$P(\text{戴眼镜}) = 19/40 = 0.475$$

**问题 2**：已知选到的是男生，他戴眼镜的概率是多少？

$$P(\text{戴眼镜}|\text{男生}) = \frac{P(\text{戴眼镜} \cap \text{男生})}{P(\text{男生})} = \frac{10/40}{25/40} = \frac{10}{25} = 0.4$$

**问题 3**：已知选到的人戴眼镜，她是女生的概率是多少？

$$P(\text{女生}|\text{戴眼镜}) = \frac{P(\text{女生} \cap \text{戴眼镜})}{P(\text{戴眼镜})} = \frac{9/40}{19/40} = \frac{9}{19} \approx 0.474$$

> 注意：P(戴眼镜\|男生) = 0.4 ≠ P(男生\|戴眼镜) = 10/19 ≈ 0.526 ——**条件概率不对称**，初学者最容易踩的坑。

### Python 验证

```python
import numpy as np

np.random.seed(123)
N = 40000  # 模拟 40000 人的大数据集

# 按比例生成数据
gender = np.random.choice(['男', '女'], N, p=[25/40, 15/40])
glasses = np.where(
    gender == '男',
    np.random.rand(N) < 10/25,  # P(戴眼镜|男) = 10/25 = 0.4
    np.random.rand(N) < 9/15    # P(戴眼镜|女) = 9/15 = 0.6
)

n_male = (gender == '男').sum()
n_female = (gender == '女').sum()
n_glasses = (glasses == True).sum()
n_male_glasses = ((gender == '男') & (glasses == True)).sum()
n_female_glasses = ((gender == '女') & (glasses == True)).sum()
n_glasses_female = n_female_glasses  # 就是上面这个

print(f"总人数: {N}")
print(f"P(戴眼镜)    = {n_glasses/N:.4f}  (理论值: 0.475)")
print(f"P(戴眼镜|男) = {n_male_glasses/n_male:.4f}  (理论值: 0.400)")
print(f"P(女生|戴眼镜) = {n_female_glasses/n_glasses if n_glasses > 0 else 0:.4f}  (理论值: 0.474)")
```

### 常见误区

| 误区 | 正确理解 |
|------|----------|
| P(A\|B) = P(B\|A) | 完全不等！P(下雨\|乌云) ≠ P(乌云\|下雨) |
| 条件概率一定比原概率大 | 不一定——已知"是周末"，P(加班) 反而更低 |
| P(A\|B) 的定义需要 P(B) > 0 | 如果 P(B) = 0（几乎不可能的事），条件概率无定义 |

### ML 应用

所有**判别模型**的核心：P(y\|x)——给定特征 x 时属于类别 y 的概率。逻辑回归、神经网络分类器、决策树——它们本质上都是在估计条件概率 P(y\|x)。

---

## 3. 贝叶斯定理——把"证据"变成"后验信念"

### 生活例子

你早上起床喉咙疼。你心想："难道感冒了？"但你转念又想到："也可能是昨晚吃火锅太辣了。"

假如感冒的发病率是 5%，而吃火锅喉咙疼的概率是 30%——这还不够精确。你需要一个系统的方法来综合这些信息。**贝叶斯定理就是干这个的。**

### 直观理解

贝叶斯定理告诉你：**拿到新证据后，如何更新你对世界的信念。**

- **先验 P(假设)**：见到证据之前的初始信念（"人群中只有 5% 感冒"）
- **似然 P(证据\|假设)**：如果假设成立，看到证据的可能性（"感冒的人 80% 喉咙疼"）
- **证据 P(证据)**：证据出现的总体概率（"所有人中喉咙疼的比例"）
- **后验 P(假设\|证据)**：见到证据后更新的信念（"喉咙疼的情况下，真感冒的概率"）

> **一句话总结**：后验 ∝ 似然 × 先验。新的信念 = 旧信念 × 新证据的支持力度。

### 形式化定义

从条件概率的乘法规则出发：$P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$，移项即得：

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

其中分母通过**全概率公式**展开：

$$P(B) = \sum_{i} P(B|A_i) \cdot P(A_i)$$

### 手算示例：医疗诊断的"反直觉"

**这是整章最重要的一道手算题，请拿出草稿纸跟着算。**

> 场景：某人去做了一项罕见病筛查。已知：
> - 该病的发病率：P(病) = 0.01  （1%，很罕见）
> - 检测灵敏度（真阳性率）：P(+\|病) = 0.99  （有病的人 99% 检出阳性）
> - 检测特异度（真阴性率）：P(-\|无病) = 0.99  （没病的人 99% 检出阴性），即假阳性率 P(+\|无病) = 0.01

**问题**：如果检测结果是阳性，这个人真正患病的概率是多少？

**大部分人第一反应**："检测准确率 99%，那阳性后得病的概率不就是 99% 左右吗？"

——**错！** 让我们用手算。

**第一步**：写出已知条件。

$$P(\text{病}) = 0.01, \quad P(\text{无病}) = 0.99$$

$$P(+|\text{病}) = 0.99, \quad P(+|\text{无病}) = 0.01$$

**第二步**：用贝叶斯定理。

$$P(\text{病}|+) = \frac{P(+|\text{病}) \cdot P(\text{病})}{P(+)}$$

**第三步**：分母 P(+) 用全概率公式计算。

$$P(+) = P(+|\text{病}) \cdot P(\text{病}) + P(+|\text{无病}) \cdot P(\text{无病})$$

$$= 0.99 \times 0.01 + 0.01 \times 0.99 = 0.0099 + 0.0099 = 0.0198$$

**第四步**：代入贝叶斯公式。

$$P(\text{病}|+) = \frac{0.99 \times 0.01}{0.0198} = \frac{0.0099}{0.0198} = 0.5$$

**结果：只有 50%！** 跟大多数人直觉的 99% 相去甚远。

**为什么这么低？——用"10000 人假想"来解释**：

假设 10000 个人来做检测：

|  | 患病（100人） | 无病（9900人） | 合计 |
|--|--------------|----------------|------|
| 阳性 | 100 × 0.99 = **99** | 9900 × 0.01 = **99** | **198** |
| 阴性 | 1 | 9801 | 9802 |

- 总共 198 人检测阳性，但其中真正患病的只有 99 人
- 99 / 198 = **50%**

**直觉**：因为疾病太罕见了，健康的 9900 人池子太大，哪怕只有 1% 的假阳性率，也会产生 99 个假阳性，追平了 99 个真阳性（99 个真阳性来自 100 个病人 × 99% 灵敏度）。

这就是著名的**基础率谬误（Base Rate Fallacy）**——忽略先验概率低的后果。

### Python 验证

```python
import numpy as np

np.random.seed(42)
N = 1_000_000  # 100 万人

# 生成真实患病状态：1% 的人患病
has_disease = np.random.rand(N) < 0.01

# 生成检测结果
test_result = np.where(
    has_disease,
    np.random.rand(N) < 0.99,    # 真阳性率 99%
    np.random.rand(N) < 0.01     # 假阳性率 1%
)

positive = test_result.sum()
true_positive = (has_disease & test_result).sum()

print(f"=== 蒙特卡洛模拟 (N={N:,}) ===")
print(f"检测阳性总人数: {positive:,}")
print(f"真阳性人数:     {true_positive:,}")
print(f"假阳性人数:     {positive - true_positive:,}")
print(f"P(病|+) = {true_positive}/{positive} = {true_positive/positive:.4f}")
print(f"(理论值: 0.5)")

# 连检两次都是阳性呢？
# 第一次检测后，后验概率 50% 成为新的"先验"
p_post = 0.5
p_second = (0.99 * p_post) / (0.99 * p_post + 0.01 * (1 - p_post))
print(f"\n连续两次阳性 → P(病|+) = {p_second:.4f}")
```

### 常见误区

| 误区 | 正确理解 |
|------|----------|
| 后验概率 = 检测准确率 | 大错！取决于先验。发病率 0.1%，准确率 99% → 后验仅 ~9% |
| 一次阳性 = 确诊 | 罕见病可能需要多次阳性才能确认（见上面的"连续两次"结果） |
| 先验不重要，数据多了就不需要先验 | 数据少时先验至关重要；数据极多时先验的影响才会被"洗掉" |

### ML 应用

- **朴素贝叶斯分类器**：P(类别\|词) ∝ P(词\|类别) × P(类别)，垃圾邮件/情感分析的经典基线
- **贝叶斯优化**：用先验（高斯过程）+ 观测更新后验来选择下一个超参数
- **贝叶斯神经网络**：权重的先验分布 + 数据 → 后验分布 → 不仅给出预测，还给出预测的不确定性

---

## 4. 随机变量——把"不确定"变成"数字"

### 生活例子

你去食堂打饭，阿姨给的菜量时多时少——这个"菜的重量"就是一个随机变量。明天股市的涨跌幅也是一个随机变量。你下周考试能考多少分——也是一个随机变量。

总之，**只要一个量在结果出来之前不确定，它就是随机变量。**

### 直观理解

**随机变量就是一个函数：它把随机实验结果映射成一个实数。** 它不是"变量"（跟编程里的变量不一样），而是一个**标签系统**——给每种可能结果贴上一个数字标签。

| 类型 | 取值特点 | 描述工具 | 例子 |
|------|----------|----------|------|
| 离散随机变量 | 取值可一一列举 | PMF（概率质量函数） | 骰子点数、图片类别编号 |
| 连续随机变量 | 取值充满一个区间 | PDF（概率密度函数） | 身高、温度、股票收益率 |

用数学符号：X 代表随机变量，而 x 代表 X 取到的具体值——P(X = 3) 意思是"随机变量 X 取值 3 的概率"。

### ML 应用

| ML 场景 | 对应的随机变量 |
|---------|---------------|
| 分类任务 | Y ∈ {0, 1, ..., K-1}（类别标签） |
| 回归任务 | Y ∈ ℝ（预测值） |
| 图像生成 | Z ~ N(0,I)（VAE/GAN 的潜变量） |
| 强化学习 | R_t（第 t 步的奖励）、S_t（第 t 步的状态） |

---

## 5. PMF、PDF、CDF——三种描述概率分布的工具

### 生活例子

想一想这两个问题：（1）"骰子掷出 3 的概率是多少？"（2）"你的身高恰好是 170.000... cm 的概率是多少？"

第一个问题好回答：P(X=3) = 1/6。但第二个问题——地球上所有身高恰好等于 170.000... cm（无限个小数位）的人，概率接近于 0。**对于连续数值，问"恰好等于某值"没有意义。** 更有意义的问题是："身高在 169.5 ~ 170.5 cm 之间的概率是多少？"——这要求你用**区间**来思考。

### 直观理解

三种描述方式的角色：

- **PMF（概率质量函数）**：给离散随机变量的每个取值分配一个"概率质量块"。所有质量块加起来 = 1。
- **PDF（概率密度函数）**：一条连续曲线，曲线越高 = 数据越密集，但**密度本身不是概率**。曲线下面积 = 概率。
- **CDF（累积分布函数）**：F(x) = P(X ≤ x)，一条从 0 到 1 的单调上升曲线。回答"小于等于某值的概率是多少"最直观。

### 形式化定义

**离散 - PMF**：$f(x) = P(X = x)$，满足 $\sum_x f(x) = 1$

**连续 - PDF**：$f(x) \geq 0$，$P(a \leq X \leq b) = \int_a^b f(x)\,dx$，满足 $\int_{-\infty}^{\infty} f(x)\,dx = 1$

**CDF**：$F(x) = P(X \leq x)$
- 离散：$F(x) = \sum_{t \leq x} P(X = t)$
- 连续：$F(x) = \int_{-\infty}^x f(t)\,dt$

### 手算示例

**公平骰子的 PMF 和 CDF**：

| x | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| PMF: P(X=x) | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |
| CDF: P(X≤x) | 1/6 | 2/6 | 3/6 | 4/6 | 5/6 | 1 |

**标准正态分布 (μ=0, σ=1) 的 PDF 值手算**（用于定性理解）：

$$f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

在 x=0 处：$f(0) = \frac{1}{\sqrt{2\pi}} \approx \frac{1}{2.507} \approx 0.399$

在 x=1 处：$f(1) = \frac{1}{\sqrt{2\pi}} e^{-1/2} \approx 0.399 \times 0.6065 \approx 0.242$

在 x=2 处：$f(2) = \frac{1}{\sqrt{2\pi}} e^{-2} \approx 0.399 \times 0.1353 \approx 0.054$

> 注意：x=0 处 PDF 值约 0.399，这**绝对不是概率**！P(X = 0) = 0（连续变量取单点的概率恒为 0）。

### Python 验证

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 离散：二项分布 PMF (n=10, p=0.3)
x = np.arange(0, 11)
axes[0, 0].stem(x, stats.binom.pmf(x, 10, 0.3))
axes[0, 0].set_title('PMF：二项分布 B(10, 0.3)')
axes[0, 0].set_ylabel('P(X=x) — 每个"柱子"就是概率')

# 连续：高斯分布 PDF（多条对比）
x_cont = np.linspace(-4, 4, 300)
for mu, sigma, ls, lab in [(-1, 0.5, '--', 'μ=-1,σ=0.5'), (0, 1, '-', 'μ=0,σ=1(标准)'), (2, 1.5, '-.', 'μ=2,σ=1.5')]:
    axes[0, 1].plot(x_cont, stats.norm.pdf(x_cont, mu, sigma), ls, label=lab)
axes[0, 1].set_title('PDF：高斯分布')
axes[0, 1].set_ylabel('密度 — 注意：这不是概率！')
axes[0, 1].legend(fontsize=8)

# 离散：二项分布 CDF
axes[1, 0].step(x, stats.binom.cdf(x, 10, 0.3), where='post')
axes[1, 0].set_title('CDF：二项分布 B(10, 0.3)')
axes[1, 0].set_ylabel('P(X ≤ x) — 累积概率')

# 连续：高斯分布 CDF
axes[1, 1].plot(x_cont, stats.norm.cdf(x_cont))
axes[1, 1].axhline(y=0.975, color='r', ls=':', alpha=0.7)
axes[1, 1].axvline(x=1.96, color='r', ls=':', alpha=0.7)
axes[1, 1].set_title('CDF：标准正态分布\n(F(1.96) ≈ 0.975)')

plt.tight_layout()
plt.show()
```

### 常见误区

| 误区 | 真相 |
|------|------|
| "PDF 值就是概率" | 大错！PDF 值可以超过 1。只有曲线下面积才是概率。 |
| "连续变量取某值的概率 > 0" | 单点概率恒为 0。只有区间才有正概率。 |
| "PMF 和 PDF 是一个东西" | PMF 的值就是概率，PDF 的值只是密度——完全不同的量。 |

---

## 6. 期望与方差——"平均"和"离散度"的数学描述

### 生活例子

你跟朋友玩一个游戏：掷一枚均匀骰子，掷出几点就赢几块钱。那么你平均每次能赢多少钱？"3.5 块钱"——这个 3.5 就是**期望**。

但是如果改为"掷出 6 点赢 100 块钱，否则赢 0 块钱"——期望变了吗？E = 100 × (1/6) + 0 × (5/6) ≈ 16.67 元。虽然期望变高了，但波动也大了——大部分时候颗粒无收，偶尔暴富。这种"波动性"就是**方差**要衡量的。

### 直观理解

- **期望（Expected Value）**：如果你把实验重复无数次，结果的平均值。不是"最可能的值"——它是**概率加权的平均**。
- **方差（Variance）**：衡量结果偏离期望的程度。大方差 = 结果极度不确定 = 高风险。
- **标准差（Standard Deviation）**：方差的平方根，量纲和原始数据一致，更直观。

### 形式化定义

$$E[X] = \sum_x x \cdot P(X=x) \quad \text{或} \quad \int x \cdot f(x)\,dx$$

$$Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

> $\boxed{E[X^2] - (E[X])^2}$ 是手算方差的利器——不用先减再平方，直接算两个期望。

### 手算示例

**场景 1：均匀骰子的期望和方差（手算）**

掷一枚公平六面骰子：$P(X = k) = 1/6$，k = 1, 2, ..., 6

**期望**：

$$E[X] = 1 \times \frac{1}{6} + 2 \times \frac{1}{6} + 3 \times \frac{1}{6} + 4 \times \frac{1}{6} + 5 \times \frac{1}{6} + 6 \times \frac{1}{6}$$

$$= \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5$$

**方差**（用捷径公式 $E[X^2] - (E[X])^2$）：

先算 $E[X^2]$：

$$E[X^2] = 1^2 \times \frac{1}{6} + 2^2 \times \frac{1}{6} + 3^2 \times \frac{1}{6} + 4^2 \times \frac{1}{6} + 5^2 \times \frac{1}{6} + 6^2 \times \frac{1}{6}$$

$$= \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} \approx 15.167$$

$$Var(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - (3.5)^2 = \frac{91}{6} - \frac{49}{4}$$

通分：$= \frac{182}{12} - \frac{147}{12} = \frac{35}{12} \approx 2.917$

标准差：$\sigma = \sqrt{35/12} \approx 1.708$

> 验证：对于 1~6 的均匀离散分布，计算方差的标准公式恰好是 $(6^2 - 1)/12 = 35/12$ ✓

**场景 2：抛硬币的期望和方差**

设 X = 1 正面，X = 0 反面，P(X=1) = p = 0.5

$$E[X] = 1 \times 0.5 + 0 \times 0.5 = 0.5$$

$$E[X^2] = 1^2 \times 0.5 + 0^2 \times 0.5 = 0.5$$

$$Var(X) = 0.5 - 0.5^2 = 0.5 - 0.25 = 0.25$$

标准差：$\sigma = 0.5$

### 期望的线性性质

**这是概率论中最有用的性质之一**：

$$E[aX + bY] = a \cdot E[X] + b \cdot E[Y]$$

**它不要求 X 和 Y 独立！** 无论它们有什么关系，期望的线性性始终成立。

手算验证（小数据集）：

| 观测 | X | Y |
|------|---|---|
| 1    | 2 | 5 |
| 2    | 4 | 3 |
| 3    | 6 | 7 |

$E[X] = (2+4+6)/3 = 4$，$E[Y] = (5+3+7)/3 = 5$

设 Z = 2X + 3Y：Z 值为 19, 17, 33

$E[Z] = (19+17+33)/3 = 23$

$2 \cdot E[X] + 3 \cdot E[Y] = 2 \times 4 + 3 \times 5 = 8 + 15 = 23$ ✓

### Python 验证

```python
import numpy as np

np.random.seed(42)

# 骰子模拟验证期望和方差
dice = np.random.randint(1, 7, 1_000_000)
print(f"=== 骰子的期望与方差 ===")
print(f"理论 E[X]   = 3.5")
print(f"模拟均值    = {dice.mean():.4f}")
print(f"理论 Var(X) = 35/12 ≈ {35/12:.4f}")
print(f"模拟方差    = {dice.var(ddof=0):.4f}  (ddof=0 即总体方差)")
print(f"理论 σ      = {np.sqrt(35/12):.4f}")
print(f"模拟 σ      = {dice.std(ddof=0):.4f}")

# 期望线性性验证（任意分布的 X 和 Y）
print(f"\n=== 期望的线性性质：E[aX+bY] = aE[X] + bE[Y] ===")
X = np.random.exponential(2, 100_000)
Y = np.random.poisson(3, 100_000)
for a, b in [(2, 3), (-1, 0.5), (10, -5)]:
    left = (a * X + b * Y).mean()
    right = a * X.mean() + b * Y.mean()
    print(f"  a={a:3.0f}, b={b:5.1f}: E[aX+bY]={left:.6f} ≈ aE[X]+bE[Y]={right:.6f}  误差={abs(left-right):.2e}")
```

### 常见误区

| 误区 | 真相 |
|------|------|
| "期望就是最可能的值" | 不。骰子每个面等可能，没有"最可能"，但期望是 3.5。 |
| "方差大 = 数据分散" | 散的是相对于**均值**。一组远离均值的数据方差才大。 |
| "Var(aX) = a·Var(X)" | 错！$Var(aX) = a^2 \cdot Var(X)$，平方关系。 |
| "E[XY] = E[X]E[Y] 永远成立" | 只有在 X 和 Y **独立**时才成立。 |

### ML 应用

| 概念 | ML 场景 |
|------|---------|
| 期望 | 损失函数 → $E_{(x,y)\sim D}[\text{Loss}(f(x), y)]$ 就是期望风险 |
| 方差 | Batch Normalization——用 batch 的均值和方差对激活值做标准化 |
| 线性性 | 梯度下降中 mini-batch 的梯度是真实梯度的无偏估计（依赖于期望的线性性） |

---

## 7. 伯努利分布——最简单的概率分布

### 生活例子

任何"非黑即白"的一次性实验：抛一次硬币（正/反）、今天是否下雨（是/否）、病人是否康复（是/否）、用户是否点击广告（点/不点）。

### 直观理解

伯努利分布只描述一件事：做一次实验，成功（1）的概率是 p，失败（0）的概率是 1-p。它是最简单的分布，也是所有二分类问题的数学基础。

### 形式化定义

$$X \sim \text{Bernoulli}(p)$$

$$P(X=1) = p, \quad P(X=0) = 1 - p$$

$$E[X] = p, \quad Var(X) = p(1-p)$$

### 手算期望和方差

**期望**：$E[X] = 1 \times p + 0 \times (1-p) = p$

**方差**：
$E[X^2] = 1^2 \times p + 0^2 \times (1-p) = p$
$Var(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$

> 有趣发现：当 p = 0.5 时方差最大（0.25），当 p → 0 或 p → 1 时方差趋于 0——**确定性越高，方差越小**。

### Python 验证

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

p = 0.3
samples = np.random.binomial(1, p, 10000)

print(f"=== 伯努利分布 Bernoulli(p={p}) ===")
print(f"理论 E[X]   = {p}")
print(f"模拟均值    = {samples.mean():.4f}")
print(f"理论 Var(X) = {p*(1-p):.4f}")
print(f"模拟方差    = {samples.var(ddof=0):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# PMF
xs = [0, 1]
ax1.bar(xs, [1-p, p], width=0.3, color=['#3498db', '#e74c3c'], edgecolor='black')
ax1.set_xticks(xs)
ax1.set_xlabel('X')
ax1.set_ylabel('P(X=x)')
ax1.set_title(f'Bernoulli PMF (p={p})')
for i, v in enumerate([1-p, p]):
    ax1.text(xs[i], v + 0.02, f'{v:.1f}', ha='center', fontsize=11)

# 不同 p 的方差
ps = np.linspace(0, 1, 200)
ax2.plot(ps, ps * (1 - ps), 'b-', linewidth=2)
ax2.set_xlabel('p')
ax2.set_ylabel('Var(X) = p(1-p)')
ax2.set_title('方差随 p 的变化')
ax2.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='p=0.5 方差最大')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 常见误区

| 误区 | 真相 |
|------|------|
| "一次伯努利实验就够了" | 一次实验只有 0 或 1，概率需要通过大量重复来体现。 |
| "p 是固定参数" | 在频率学派中是这样。在贝叶斯学派中 p 本身也是一个随机变量（服从 Beta 分布）。 |

### ML 应用

- **逻辑回归**的输出层——对每个样本，预测值 $\hat{y}$ 就是伯努利分布的参数 p
- **二分类问题的 Cross-Entropy Loss**——本质上是在拟合伯努利分布的参数

---

## 8. 二项分布——多次独立试验的总成功数

### 生活例子

你不是只抛一次硬币，而是连续抛了 10 次。"这 10 次中恰好有 k 次正面"的概率——这就是二项分布。又比如：你给 100 个客户发营销邮件，每个客户独立点击的概率是 5%，"有 8 个客户点击"的概率就是二项分布。

### 直观理解

做 n 次独立的伯努利试验，每次成功概率都是 p。**二项分布描述了"n 次实验中成功次数的分布"。**

它的 PMF 公式实际上就是组合数学：在 n 次试验中选出 k 次成功的位置，然后乘以 k 次成功和 n-k 次失败的概率。

### 形式化定义

$$X \sim \text{Binomial}(n, p)$$

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, ..., n$$

$$E[X] = np, \quad Var(X) = np(1-p)$$

### 手算示例

**场景**：投篮命中率 p = 0.6，投 n = 5 次。计算各种命中次数的概率。

**先回忆组合数**：$\binom{n}{k} = \frac{n!}{k!(n-k)!}$

| 命中 k | $\binom{5}{k}$  | $p^k$ | $(1-p)^{n-k}$ | P(X=k) |
|--------|-----------------|-------|---------------|--------|
| 0 | 1                | 1.0   | $0.4^5 = 0.01024$ | 0.01024 |
| 1 | 5                | 0.6   | $0.4^4 = 0.0256$  | 5 × 0.6 × 0.0256 = 0.07680 |
| 2 | 10               | 0.36  | $0.4^3 = 0.064$   | 10 × 0.36 × 0.064 = 0.23040 |
| 3 | 10               | 0.216 | $0.4^2 = 0.16$    | 10 × 0.216 × 0.16 = 0.34560 |
| 4 | 5                | 0.1296| $0.4^1 = 0.4$     | 5 × 0.1296 × 0.4 = 0.25920 |
| 5 | 1                | 0.07776| $0.4^0 = 1.0$    | 0.07776 |

验证总和：0.01024 + 0.07680 + 0.23040 + 0.34560 + 0.25920 + 0.07776 = 1.00000 ✓

期望：E[X] = 5 × 0.6 = 3（平均命中 3 次）

方差：Var(X) = 5 × 0.6 × 0.4 = 1.2

标准差：$\sigma = \sqrt{1.2} \approx 1.095$

> 最可能的结果是 k=3（概率 0.3456），但偏离 3 到 k=2 或 k=4 的概率也不小——**不要把"期望"和"确定结果"混为一谈**。

### Python 验证 + 画图

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n, p = 10, 0.3
k_vals = np.arange(0, n + 1)
pmf_vals = stats.binom.pmf(k_vals, n, p)

# 手算对照
from math import comb
hand_calc = [comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(n+1)]

print(f"=== 二项分布 Binomial(n={n}, p={p}) ===")
print(f"理论 E[X]   = {n*p}")
print(f"理论 Var(X) = {n*p*(1-p):.4f}")
print(f"\n{'k':>3} {'scipy PMF':>10} {'手算 PMF':>10} {'差值':>10}")
for k in range(n+1):
    print(f"{k:3d} {pmf_vals[k]:10.6f} {hand_calc[k]:10.6f} {abs(pmf_vals[k]-hand_calc[k]):10.2e}")

# 模拟验证
samples = np.random.binomial(n, p, 100000)
print(f"\n模拟 E[X]   = {samples.mean():.4f}")
print(f"模拟 Var(X) = {samples.var(ddof=0):.4f}")

# 画 PMF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(k_vals, pmf_vals, color='steelblue', edgecolor='black', alpha=0.8)
ax1.set_xlabel('k (成功次数)')
ax1.set_ylabel('P(X=k)')
ax1.set_title(f'二项分布 PMF: Binomial(n={n}, p={p})')
ax1.set_xticks(k_vals)
for k, v in zip(k_vals, pmf_vals):
    if v > 0.01:
        ax1.text(k, v + 0.005, f'{v:.3f}', ha='center', fontsize=7)

# 不同 n 的概率形状对比
for n_vals, style in [(5, 'o-'), (20, 's-'), (50, '^-')]:
    ks = np.arange(0, n_vals + 1)
    probs = stats.binom.pmf(ks, n_vals, p)
    ax2.plot(ks, probs, style, label=f'n={n_vals}', alpha=0.7, markersize=3)
ax2.set_xlabel('k')
ax2.set_ylabel('P(X=k)')
ax2.set_title(f'p={p} 不变，n 增大时 PMF 变宽')
ax2.legend()

plt.tight_layout()
plt.show()
```

### 常见误区

| 误区 | 真相 |
|------|------|
| 二项分布只适用于"成功/失败"场景 | 任何可计数的事件次数都行——网站访问量、电话接听数 |
| n 大时二项分布接近均匀 | 不对，n 大 p≠0.5 时接近泊松分布，np很大时接近正态分布 |
| 期望 np 就是最可能结果 | 只有 p=0.5 时对称。p=0.6 时可能结果偏向右侧 |

### ML 应用

- **集成学习**：Bagging 中每个基学习器的准确率 → 投票结果的分布就是二项分布
- **Dropout**：每次以概率 p 保留神经元 → 可以看作伯努利过程
- **A/B 测试**的统计学基础——转化率就是二项分布参数 p

---

## 9. 多项分布——从二分类到多分类

### 生活例子

你做一个在线调查："最喜欢的颜色是？"选项有红、蓝、绿、其他 4 种。假设每种选择的比例分别是 30%, 25%, 25%, 20%。你收集了 100 份答卷，各种颜色的得票数分布就是多项分布。

掷一个 K 面的骰子 n 次（每次不是 2 个结果而是 K 个结果），记录每面出现的次数。

### 形式化定义

$$X \sim \text{Multinomial}(n, [p_1, p_2, ..., p_K]), \quad \sum_{i=1}^K p_i = 1, \quad \sum_{i=1}^K x_i = n$$

$$P(X_1=x_1, ..., X_K=x_K) = \frac{n!}{x_1!\,x_2!\,...\,x_K!} \; p_1^{x_1} p_2^{x_2} ... p_K^{x_K}$$

### 手算示例

**场景**：掷一枚不均匀的三面骰子，p = [0.2, 0.3, 0.5]。掷 5 次，求"(面1出1次, 面2出1次, 面3出3次)"的概率。

$$P(1,1,3) = \frac{5!}{1! \cdot 1! \cdot 3!} \times 0.2^1 \times 0.3^1 \times 0.5^3$$

分子：$5! = 120$，分母：$1! \times 1! \times 3! = 6$，组合系数 = $120 / 6 = 20$

概率：$20 \times 0.2 \times 0.3 \times 0.125 = 20 \times 0.0075 = 0.15$

| 分布 | 类别数 | n 的含义 | 输出形式 | 关系 |
|------|--------|----------|----------|------|
| Bernoulli | K=2 | 1 次实验 | 0 或 1 | 多项分布的特例 |
| Binomial | K=2 | n 次实验 | 成功总次数 | n 个 Bernoulli 之和 |
| Multinomial | K≥2 | n 次实验 | K 维计数向量 | Bernoulli/Binomial 的推广 |

### Python 验证 + 画图

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n_sims = 50000
p = [0.2, 0.3, 0.5]
n_trials = 5

samples = np.random.multinomial(n_trials, p, n_sims)
print(f"=== 多项分布 Multinomial(n={n_trials}, p={p}) ===")
print(f"各维度期望: n×p = {[n_trials*pi for pi in p]}")
print(f"各维度模拟均值: {samples.mean(axis=0)}")
print(f"各维度方差 (理论 np(1-p)): {[n_trials*pi*(1-pi) for pi in p]}")
print(f"各维度模拟方差: {samples.var(axis=0, ddof=0)}")

# 可视化：模拟结果的分布
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, ax in enumerate(axes):
    counts = samples[:, i]
    unique, freq = np.unique(counts, return_counts=True)
    ax.bar(unique, freq / n_sims, color=['#e74c3c','#3498db','#2ecc71'][i],
           alpha=0.7, edgecolor='black')
    ax.set_title(f'维度 {i+1} 的边际分布\n(p={p[i]}, 期望={n_trials*p[i]:.1f})')
    ax.set_xlabel(f'X_{i+1}')
    ax.set_ylabel('频率')

plt.suptitle('多项分布各维度的边缘分布', fontsize=13)
plt.tight_layout()
plt.show()

# 手算验证特定案例
from math import factorial
x = np.array([1, 1, 3])
coeff = factorial(n_trials) / (factorial(x[0])*factorial(x[1])*factorial(x[2]))
prob = coeff * (p[0]**x[0]) * (p[1]**x[1]) * (p[2]**x[2])
print(f"\nP([1,1,3]) 手算: {prob:.4f}")
print(f"P([1,1,3]) scipy: {stats.multinomial.pmf(x, n_trials, p):.4f}")
```

### ML 应用

- **多分类问题**：逻辑回归的多类版本（Softmax 回归）——输出 K 个类别的概率之和为 1
- **文本分类**：文档中各个词的词频分布
- **LDA 主题模型**：文档-主题分布和主题-词分布都是多项分布

---

## 10. 泊松分布——描述"稀有事"发生次数

### 生活例子

你在一家便利店收银，想知道"下一个小时大概会来几位顾客"。平均每小时来 3 位——但实际可能是 0 位、1 位甚至 8 位。这种**单位时间内随机事件发生次数**的分布，就是泊松分布。

其他例子：某路口一天的事故数、一个客服每小时接到的电话数、一页书上的错别字数。

### 直观理解

泊松分布只有一个参数 λ（lambda）——**单位时间/空间内事件的平均发生次数**。λ 越大，分布越像正态分布；λ 越小，分布越像指数衰减的形状。

**泊松分布是二项分布的极限**：当二项分布 n → ∞, p → 0，且 n·p = λ 保持不变时，二项分布趋近泊松分布。这就是为什么泊松适合描述"小概率事件在大样本中的发生次数"。

### 形式化定义

$$X \sim \text{Poisson}(\lambda)$$

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, ...$$

$$E[X] = \lambda, \quad Var(X) = \lambda$$

> 注意：期望 = 方差 = λ——这是泊松分布的标志性特征。如果数据的均值和方差相差很大，说明不适合用泊松模型。

### 手算示例

**场景**：某咖啡店每小时平均来 3 位顾客（λ = 3）。计算接下来一小时不同顾客数的概率。

| k | λᵏ | e^{-λ} | k! | P(X=k) = λᵏ · e^{-λ} / k! |
|---|-----|--------|-----|---------------------------|
| 0 | 1   | 0.04979 | 1   | 0.04979 |
| 1 | 3   | 0.04979 | 1   | 3 × 0.04979 = 0.14936 |
| 2 | 9   | 0.04979 | 2   | 9 × 0.04979 / 2 = 0.22404 |
| 3 | 27  | 0.04979 | 6   | 27 × 0.04979 / 6 = 0.22404 |
| 4 | 81  | 0.04979 | 24  | 81 × 0.04979 / 24 = 0.16803 |
| 5 | 243 | 0.04979 | 120 | 243 × 0.04979 / 120 = 0.10082 |
| 6 | 729 | 0.04979 | 720 | 729 × 0.04979 / 720 = 0.05041 |

E[X] ≈ 0×0.05 + 1×0.15 + 2×0.22 + 3×0.22 + 4×0.17 + 5×0.10 + 6×0.05 ≈ 3.0 ✓

> 最可能值 k=2 和 k=3（概率约 22.4%），接近 λ。λ=3 时约 5% 的概率没人来，约 10% 概率来 5 人。

### Python 验证 + 画图

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import exp, factorial as math_fact

lambd = 3
k_max = 15
k_vals = np.arange(0, k_max + 1)

# 手算 + scipy 对照
print(f"=== 泊松分布 Poisson(λ={lambd}) ===")
print(f"{'k':>3} {'手算 P(X=k)':>14} {'scipy':>14} {'差值':>10}")
for k in range(k_max + 1):
    hand = (lambd**k) * np.exp(-lambd) / math_fact(k)
    scipy_val = stats.poisson.pmf(k, lambd)
    print(f"{k:3d} {hand:14.6f} {scipy_val:14.6f} {abs(hand-scipy_val):10.2e}")

# 模拟
samples = np.random.poisson(lambd, 100000)
print(f"\n理论 E[X]   = {lambd}")
print(f"模拟均值    = {samples.mean():.4f}")
print(f"理论 Var(X) = {lambd}")
print(f"模拟方差    = {samples.var(ddof=0):.4f}")

# 画 PMF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

pmf = stats.poisson.pmf(k_vals, lambd)
ax1.bar(k_vals, pmf, color='#9b59b6', edgecolor='black', alpha=0.8)
ax1.set_xlabel('k')
ax1.set_ylabel('P(X=k)')
ax1.set_title(f'泊松分布 PMF: Poisson(λ={lambd})')
ax1.set_xticks(k_vals)

# 不同 λ 对比
lambdas = [1, 3, 6, 10]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for lam, c in zip(lambdas, colors):
    ks = np.arange(0, 25)
    ax2.plot(ks, stats.poisson.pmf(ks, lam), 'o-', color=c, label=f'λ={lam}', markersize=3)
ax2.set_xlabel('k')
ax2.set_ylabel('P(X=k)')
ax2.set_title('不同 λ 的泊松分布形状：λ 越大越像正态')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 常见误区

| 误区 | 真相 |
|------|------|
| 泊松只适用于"罕见事件" | λ 很大也适用。当 λ > 20，泊松非常接近正态分布。 |
| 事件必须完全随机 | 需要独立性和平稳性。如果事件之间有传染性（来了一个客人带动一群），则不适用。 |
| 期望 = 方差 = λ 总是成立 | 对理论分布成立，但实际数据中两者常不相等（过度离散/欠离散）。 |

### ML 应用

- **计数回归**：预测"用户访问次数""订单数"等非负整数值
- **推荐系统**：用户交互次数的建模
- **自然语言处理**：词频建模

---

## 11. 高斯分布——无处不在的"钟形曲线"

### 生活例子

量一下你全班同学的身高——你会发现大部分人的身高集中在均值附近，极高和极矮的人很少。画出频数直方图，形状就像一口钟。人类身高、测量误差、IQ 分数——数不尽的自然现象都近似服从高斯分布（正态分布）。

### 直观理解

**为什么高斯分布如此普遍？** 答案在**中心极限定理**里——大量独立微小因素的叠加，结果趋近正态。你的身高由几百个基因 + 营养 + 环境共同决定，每个因素贡献一小部分，加在一起就是正态。

高斯分布由两个参数决定：
- **μ（均值）**：钟形曲线的中心位置
- **σ（标准差）**：钟形曲线的"胖瘦"——σ 越大，曲线越矮胖

**68-95-99.7 法则（经验法则）**：
- 68% 的数据落在 μ ± 1σ 内
- 95% 的数据落在 μ ± 2σ 内
- 99.7% 的数据落在 μ ± 3σ 内

### 形式化定义

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}$$

$$E[X] = \mu, \quad Var(X) = \sigma^2$$

### 手算 68-95-99.7 法则验证（标准正态 N(0,1)）

这个法则没法纯手算精确积分，但我们可以定性理解：

$f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ 中：
- 在 μ=0 处 PDF 最大，远离 0 后指数衰减极快
- σ=1 意味着拐点在 x = ±1 处（PDF 曲线从"加速下降"变成"减速下降"的地方）

**手算几个关键点的密度**：

| x | x² | e^{-x²/2} | f(x) ≈ |
|---|-----|-----------|--------|
| 0 | 0   | 1         | 0.399 |
| 1 | 1   | 0.607     | 0.242 |
| 2 | 4   | 0.135     | 0.054 |
| 3 | 9   | 0.011     | 0.004 |

> 到了 3σ 处，密度已经降到峰值的大约 1%，直观解释了 99.7% 落在 ±3σ 内。

### Python 验证 + 画图

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 生成各类高斯分布
x = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 图1: 相同 μ，不同 σ
for sigma, ls, label in [(0.5, '--', 'σ=0.5 (窄)'), (1, '-', 'σ=1 (标准)'), (2, '-.', 'σ=2 (宽)')]:
    axes[0].plot(x, stats.norm.pdf(x, 0, sigma), ls, linewidth=2, label=label)
axes[0].set_title('均值 μ=0 相同，标准差不同')
axes[0].set_xlabel('x')
axes[0].set_ylabel('密度 f(x)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 68-95-99.7 法则可视化
x_fill = np.linspace(-4, 4, 400)
pdf = stats.norm.pdf(x_fill)
axes[1].plot(x_fill, pdf, 'b-', linewidth=2)

fills = [
    (1, 0.68, '#3498db', '68%: ±1σ'),
    (2, 0.95, '#2ecc71', '95%: ±2σ'),
    (3, 0.997, '#e74c3c', '99.7%: ±3σ'),
]
for k, _, color, label in fills:
    mask = (x_fill >= -k) & (x_fill <= k)
    axes[1].fill_between(x_fill[mask], pdf[mask], alpha=0.3, color=color, label=label)
    actual = stats.norm.cdf(k) - stats.norm.cdf(-k)
    print(f"±{k}σ: 理论 {_:.1%}, 精确 {actual:.4f}")

axes[1].set_title('68-95-99.7 法则')
axes[1].set_xlabel('x (以 σ 为单位)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 验证 68-95-99.7
print(f"\n=== 经验法则精确验证 ===")
for k in [1, 2, 3]:
    prob = stats.norm.cdf(k) - stats.norm.cdf(-k)
    print(f"P(μ - {k}σ < X < μ + {k}σ) = {prob:.4f} = {prob*100:.2f}%")
```

### 常见误区

| 误区 | 真相 |
|------|------|
| "自然界的一切都是正态的" | 不。收入分布是右偏的，股价收益率有厚尾——别盲目假设正态。 |
| "数据量大就一定是正态" | CLT 说的是**样本均值**的分布趋近正态，不是原始数据。 |
| "正态分布适合所有数据" | 对有明显偏态、厚尾或有界的变量，先做变换（如对数变换）再建模。 |

### ML 应用

| 场景 | 高斯分布的作用 |
|------|---------------|
| 线性回归 | 误差项假设 ε ~ N(0, σ²) |
| 高斯朴素贝叶斯 | 假设每类特征服从高斯分布 |
| VAE | 潜变量 z ~ N(0, I)——最常用的先验 |
| GMM（高斯混合模型） | 用多个高斯叠加建模复杂分布 |
| Batch Normalization | 把每层激活值标准化到 ~N(0,1) |
| 权重初始化 | He/Glorot 初始化基于正态分布 |

---

## 12. 联合概率、边缘概率、条件概率——一张表说透

### 生活例子

你同时关心两件事：今天的温度和是否下雨。人们常说"又闷又热容易下雨"——你直觉上在做双重判断。联合概率 P(高温, 下雨) 就是"高温和下雨同时发生"的概率。

### 直观理解

画一张 2×2 的表格，一切都清楚了：

|     | Y = 晴天 | Y = 下雨 | **边缘 P(X)** |
|-----|----------|---------|---------------|
| **X = 高温** | 0.30 | 0.05 | **0.35** |
| **X = 低温** | 0.20 | 0.45 | **0.65** |
| **边缘 P(Y)** | **0.50** | **0.50** | **1.00** |

- **联合概率 P(X, Y)**：表格中间 4 个格子——两个变量的值同时发生的概率
- **边缘概率**：把行或列加起来，得到一个变量的"整体分布"，不管另一个变量
- **条件概率 P(Y\|X=高温)**：只看"高温"那行，然后重新归一化使和为 1

核心公式——**全概率公式（边缘化）**：

$$P(Y) = \sum_x P(X=x, Y) \quad \text{或} \quad P(Y) = \sum_x P(Y|X=x)P(X=x)$$

### 手算示例

用上面那张表来算：

**从联合概率推算边缘概率**：
- P(高温) = 0.30 + 0.05 = 0.35 ✓
- P(下雨) = 0.05 + 0.45 = 0.50 ✓

**从联合概率推算条件概率**：
$$P(\text{下雨}|\text{高温}) = \frac{P(\text{高温}, \text{下雨})}{P(\text{高温})} = \frac{0.05}{0.35} \approx 0.143$$

$$P(\text{晴天}|\text{低温}) = \frac{P(\text{低温}, \text{晴天})}{P(\text{低温})} = \frac{0.20}{0.65} \approx 0.308$$

$$P(\text{高温}|\text{下雨}) = \frac{P(\text{高温}, \text{下雨})}{P(\text{下雨})} = \frac{0.05}{0.50} = 0.10$$

> 注意：P(下雨\|高温) ≈ 14.3%，但 P(高温\|下雨) = 10%——再次强调条件概率不对称。

### Python 验证

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100000

# 按联合概率表生成数据
temp = np.random.choice(['高温', '低温'], n, p=[0.35, 0.65])
rain = np.where(
    temp == '高温',
    np.random.rand(n) < 0.05/0.35,   # P(下雨|高温) = 0.05/0.35 ≈ 0.143
    np.random.rand(n) < 0.45/0.65    # P(下雨|低温) = 0.45/0.65 ≈ 0.692
)

# 构建联合概率表
print("=== 联合概率表（模拟）===")
rows = ['高温', '低温']
cols = ['晴天', '下雨']
print(f"{'':>6} {'晴天':>8} {'下雨':>8} {'边缘P(温度)':>11}")
for r in rows:
    vals = []
    for c in [False, True]:
        p = ((temp == r) & (rain == c)).sum() / n
        vals.append(p)
    marginal = vals[0] + vals[1]
    print(f"{r:>6} {vals[0]:8.4f} {vals[1]:8.4f} {marginal:11.4f}")

marginal_rain = []
for c in [False, True]:
    marginal_rain.append(((rain == c)).sum() / n)
print(f"{'边缘':>6} {marginal_rain[0]:8.4f} {marginal_rain[1]:8.4f} {marginal_rain[0]+marginal_rain[1]:11.4f}")

# 条件概率验证
p_rain_given_hot = ((temp == '高温') & (rain == True)).sum() / (temp == '高温').sum()
p_hot_given_rain = ((temp == '高温') & (rain == True)).sum() / (rain == True).sum()
print(f"\nP(下雨|高温)     = {p_rain_given_hot:.4f}  (理论: 0.1429)")
print(f"P(高温|下雨)     = {p_hot_given_rain:.4f}  (理论: 0.1000)")

# 全概率公式验证：P(下雨) = P(下雨|高温)P(高温) + P(下雨|低温)P(低温)
p_rain_total = p_rain_given_hot * 0.35 + (1-0.308) * 0.65
print(f"全概率 P(下雨)   = {p_rain_total:.4f}  (理论: 0.5000)")
```

### ML 应用

| 概念 | 对应 ML |
|------|---------|
| 联合概率 P(x, y) | **生成模型**（如朴素贝叶斯、GAN）——建模数据如何产生 |
| 条件概率 P(y\|x) | **判别模型**（如逻辑回归、神经网络）——直接预测决策边界 |
| 边缘概率 P(x) | **无监督学习**（如密度估计）——只关心数据本身的分布 |

---

## 13. 协方差与相关系数——两个变量是"好朋友"还是"冤家"？

### 生活例子

"身高高的人通常体重也大"——这是**正相关**。"运动时间越长，体重越低"——**负相关**。"你的鞋码和你的数学成绩"——**没有关系**。

协方差和相关系数就是把这种"同增同减"或"此增彼减"关系量化的工具。

### 直观理解

- **协方差 Cov(X, Y)**：正数 = 同向变动，负数 = 反向变动，接近 0 = 没关系。但有个问题——协方差的数值受量纲影响。如果 X 的单位从"米"变成"厘米"，协方差会放大 100 倍，但两个变量之间的关系并没有变。
- **相关系数 ρ**：把协方差除以两个标准差，结果永远在 [-1, 1] 之间——消除了量纲，让你可以公平地比较不同变量对之间的关系强度。

**ρ = 1**：完全正线性相关 → 散点图是一条斜向上的直线
**ρ = 0**：没有线性关系 → 但仍可能有非线性关系！
**ρ = -1**：完全负线性相关 → 散点图是一条斜向下的直线

### 形式化定义

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y} \in [-1, 1]$$

### 手算示例

有 4 个数据点的小样本：

| 编号 | X (学习时间/小时) | Y (考试分数) |
|------|-------------------|-------------|
| 1    | 2                 | 65          |
| 2    | 3                 | 70          |
| 3    | 5                 | 80          |
| 4    | 6                 | 85          |

**步骤 1**：算均值
$$E[X] = \frac{2+3+5+6}{4} = \frac{16}{4} = 4$$
$$E[Y] = \frac{65+70+80+85}{4} = \frac{300}{4} = 75$$

**步骤 2**：算 $E[XY]$
$$E[XY] = \frac{2\times65 + 3\times70 + 5\times80 + 6\times85}{4} = \frac{130 + 210 + 400 + 510}{4} = \frac{1250}{4} = 312.5$$

**步骤 3**：算协方差
$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 312.5 - 4 \times 75 = 312.5 - 300 = 12.5$$

**步骤 4**：算两个标准差
$$E[X^2] = \frac{4+9+25+36}{4} = \frac{74}{4} = 18.5, \quad \sigma_X = \sqrt{18.5 - 16} = \sqrt{2.5} \approx 1.581$$

$$E[Y^2] = \frac{4225+4900+6400+7225}{4} = \frac{22750}{4} = 5687.5, \quad \sigma_Y = \sqrt{5687.5 - 5625} = \sqrt{62.5} \approx 7.906$$

**步骤 5**：算相关系数
$$\rho = \frac{12.5}{1.581 \times 7.906} = \frac{12.5}{12.5} = 1.0$$

> 哈哈，正好是 ρ = 1.0！因为我们人为构造了完美线性的数据（分数 ≈ 5×小时 + 55）。现实中不会这么完美。

### Python 验证 + 画图

```python
import numpy as np
import matplotlib.pyplot as plt

# 手算数据验证
X = np.array([2, 3, 5, 6])
Y = np.array([65, 70, 80, 85])

print(f"=== 协方差和相关系数 手算验证 ===")
print(f"E[X]      = {X.mean()}")
print(f"E[Y]      = {Y.mean()}")
print(f"E[XY]     = {(X*Y).mean()}")
print(f"Cov 手算  = {(X*Y).mean() - X.mean()*Y.mean()}")
print(f"Cov numpy = {np.cov(X, Y, ddof=0)[0,1]}")
print(f"ρ 手算    = {(np.cov(X, Y, ddof=0)[0,1]) / (X.std(ddof=0)*Y.std(ddof=0))}")
print(f"ρ numpy   = {np.corrcoef(X, Y)[0,1]}")

# 三种典型关系的可视化
np.random.seed(42)
n = 300
x = np.random.randn(n)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 正相关 ρ≈0.8
y_pos = 0.8 * x + 0.6 * np.random.randn(n)
r_pos = np.corrcoef(x, y_pos)[0, 1]
axes[0].scatter(x, y_pos, s=10, alpha=0.6, c='#3498db')
axes[0].set_title(f'正相关: ρ = {r_pos:.3f}')
axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
axes[0].grid(True, alpha=0.3)

# 负相关 ρ≈-0.8
y_neg = -0.8 * x + 0.6 * np.random.randn(n)
r_neg = np.corrcoef(x, y_neg)[0, 1]
axes[1].scatter(x, y_neg, s=10, alpha=0.6, c='#e74c3c')
axes[1].set_title(f'负相关: ρ = {r_neg:.3f}')
axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
axes[1].grid(True, alpha=0.3)

# 无相关（非线性也算"无相关"？）
y_quad = x**2 + 0.3 * np.random.randn(n)
r_quad = np.corrcoef(x, y_quad)[0, 1]
axes[2].scatter(x, y_quad, s=10, alpha=0.6, c='#2ecc71')
axes[2].set_title(f'非线性关系: ρ = {r_quad:.3f}\n(注意：低 ρ≠没关系！)')
axes[2].set_xlabel('X'); axes[2].set_ylabel('Y')
axes[2].grid(True, alpha=0.3)

plt.suptitle('相关系数 ρ 解释不同类型的变量关系', fontsize=14)
plt.tight_layout()
plt.show()
```

### 常见误区

| 误区 | 真相 |
|------|------|
| "ρ=0 就是没任何关系" | ρ=0 只表示**没有线性关系**。上面的抛物线关系 ρ≈0，但 Y 完全由 X 决定！ |
| "相关性 = 因果性" | 冰淇淋销量和溺水人数正相关——不是因为冰淇淋导致溺水，而是夏天两者都增多。 |
| "协方差值本身有意义" | 协方差值受单位影响，换算到相关系数才能用来比较不同变量对。 |

### ML 应用

- **PCA 降维**：PCA 的核心 = 协方差矩阵的特征值分解——找方差最大的方向
- **特征选择**：高相关性的特征之间信息冗余，可以删除一部分
- **异常检测**：马氏距离利用协方差矩阵来衡量观测的异常程度

---

## 14. 大数定律——为什么样本够大就可靠

### 生活例子

保险公司凭什么敢承保？因为他们知道——虽然单个客户出不出险无法预测，但当客户数量达到几十万时，总体出险比例会非常稳定也接近真实概率。这就是**大数定律**在保护保险公司。

开赌场的也一样：轮盘赌单局的输赢是随机的，但夜以继日地赌，只要赔率对他稍微有利，他绝对稳赚。

### 直观理解

**大数定律（弱大数定律）**：样本量 n 越来越大时，样本均值趋近于总体均值。

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu \quad \text{当 } n \to \infty$$

> 这是"频率学派"概率观的数学基础——概率就是长期频率的极限。

### Python + 画图

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 5000

# 从一个很"偏"的分布中抽样 —— 指数分布
true_mean = 1.0  # Exp(1) 的真实均值
samples = np.random.exponential(1, n)

# 累计均值序列
cum_mean = np.cumsum(samples) / np.arange(1, n+1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(1, n+1), cum_mean, 'b-', linewidth=0.6, alpha=0.8)
ax.axhline(y=true_mean, color='r', linestyle='--', linewidth=2, label=f'真实均值 μ = {true_mean}')
ax.fill_between(range(1, n+1), cum_mean - 1.0/np.sqrt(np.arange(1, n+1)),
                 cum_mean + 1.0/np.sqrt(np.arange(1, n+1)),
                 alpha=0.15, color='blue', label='±1/√n 带')
ax.set_xlabel('样本量 n', fontsize=12)
ax.set_ylabel('样本均值', fontsize=12)
ax.set_title('大数定律：样本均值随 n 增大收敛到真实均值', fontsize=14)
ax.legend(fontsize=10)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
plt.show()

# 不同初值的情况，看看波动
print("n 在不同阶段下均值的表现:")
for n_check in [10, 50, 200, 1000, 5000]:
    print(f"  n={n_check:5d}: 均值={cum_mean[n_check-1]:.4f}  误差={abs(cum_mean[n_check-1]-true_mean):.4f}")
```

### ML 应用

- **模型评估**：测试集越大，评估指标（准确率、F1）越可靠
- **交叉验证**：K-Fold CV 中多次评估平均 → 大数定律保证稳定的估计
- **梯度下降**：Mini-batch 梯度是整体梯度的一个样本均值近似

---

## 15. 中心极限定理——为什么世界是正态的

### 生活例子

你站在一片麦田前，每株麦子的高度有些差异——但如果你随机拔 30 株测平均高度，重复这个过程 1000 次，这 1000 个样本均值的分布会呈现漂亮的钟形——**不管原始麦子高度的分布长什么样。**

### 直观理解

**中心极限定理（CLT）** 是概率论中最令人惊叹的结论之一：

> 从**任意分布**中独立抽样（期望 μ，方差 σ²），当样本量 n 足够大时，样本均值的分布逼近正态分布 $\mathcal{N}(\mu, \sigma^2/n)$。

注意两个要点：
1. **原始分布可以任意形状**——均匀、指数、二项……甚至你在沙滩上画一个怪异的轮廓，只要均值方差存在就行。
2. **逼近的是均值的分布，不是数据本身的分布**——原始数据不需要是正态的。

这就是为什么正态分布在统计学和机器学习中如此普遍——很多统计量（均值、回归系数……）都是大量独立观测的加权和或函数，所以它们的分布自然逼近正态。

### Python 模拟（核心验证！）

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n_samples = 10000  # 重复抽样次数
sample_size = 30   # 每组样本量

# 三种"完全不正态"的分布
dists = {
    '均匀分布 U(0,1)':       lambda s: np.random.rand(s),
    '指数分布 Exp(1)':       lambda s: np.random.exponential(1, s),
    '二项分布 B(1, 0.2)':    lambda s: np.random.binomial(1, 0.2, s),
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for j, (name, sampler) in enumerate(dists.items()):
    # 原始分布
    raw = sampler(10000)
    axes[0, j].hist(raw, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0, j].set_title(f'原始分布：{name}', fontsize=11)
    axes[0, j].set_ylabel('密度')

    # 样本均值的分布
    means = np.array([sampler(sample_size).mean() for _ in range(n_samples)])
    axes[1, j].hist(means, bins=50, density=True, alpha=0.7, color='darkorange',
                    edgecolor='white', label=f'样本均值分布\n(n={sample_size})')
    axes[1, j].set_title(f'样本均值的分布（n={sample_size}）', fontsize=11)
    axes[1, j].set_xlabel('样本均值')
    axes[1, j].set_ylabel('密度')

    # 叠加理论正态曲线
    mu, sigma = means.mean(), means.std()
    x_curve = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    axes[1, j].plot(x_curve, stats.norm.pdf(x_curve, mu, sigma), 'r-', linewidth=2, label='理论正态')
    axes[1, j].legend(fontsize=8)

    print(f"{name}: CLT 预测 σ_means = σ/√n = {raw.std():.3f}/{np.sqrt(sample_size):.3f} "
          f"= {raw.std()/np.sqrt(sample_size):.4f}, 实际 = {means.std():.4f}")

plt.suptitle('中心极限定理：无论原始分布如何，均值的分布逼近正态', fontsize=14)
plt.tight_layout()
plt.show()
```

### ML 应用

- **置信区间**：模型性能评估中，±1.96·SE 的 95% 置信区间依赖 CLT
- **A/B 测试**：两组用户转化率之差是否显著 → 基于正态近似计算 p 值
- **Bagging**：多个基模型预测的平均 → 方差按 1/n 缩减（由 CLT 保证）
- **Mini-batch 梯度下降**：batch 梯度 ≈ 整体梯度 + 正态噪声（近似）

---

## 16. 蒙特卡洛方法——用随机性解决确定性问题

### 生活例子

你想知道一个不规则湖泊的面积。你不去解复杂的积分方程，而是——在湖周围画一个矩形，然后随机往矩形里撒 10000 颗豆子。数一数落在湖里的豆子有多少：如果 3142 颗落在湖里，那湖面积 ≈ 矩形面积 × 0.3142。

这就是蒙特卡洛。**用随机采样来近似计算确定性量。** 听起来像个玩笑？实际上这是现代科学计算中最重要的工具之一。

### 直观理解

蒙特卡洛的核心公式：

$$E[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \quad x_i \sim P(X)$$

任何可写成期望形式的东西，都可以用抽样来近似。误差随 √N 减小——每次要把精度提高 10 倍，需要 100 倍的样本。

### 手算：蒙特卡洛估算 π

在 [0,1]×[0,1] 的正方形内，画一个半径为 1 的 1/4 圆。向正方形内随机撒点：
- 落在圆内的概率 = (1/4 圆面积) / (正方形面积) = (π/4) / 1 = π/4
- 所以 π = 4 × (圆内点数 / 总点数)

### Python 模拟

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def estimate_pi(n):
    x, y = np.random.rand(n), np.random.rand(n)
    inside = (x**2 + y**2) <= 1.0
    pi_est = 4 * inside.sum() / n
    return pi_est, x, y, inside

print("=== 蒙特卡洛估算 π ===")
print(f"{'n':>10} {'估算值':>10} {'误差':>10}")
print("-" * 35)
for n in [100, 1000, 10000, 100000, 1000000]:
    pi_est, _, _, _ = estimate_pi(n)
    print(f"{n:10d} {pi_est:10.6f} {abs(pi_est - np.pi):10.6f}")

# 可视化
n_vis = 3000
pi_est, x, y, inside = estimate_pi(n_vis)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].scatter(x[inside], y[inside], s=1, c='#e74c3c', alpha=0.6)
axes[0].scatter(x[~inside], y[~inside], s=1, c='#3498db', alpha=0.6)
axes[0].add_patch(plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2))
axes[0].set_aspect('equal')
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
axes[0].set_title(f'蒙特卡洛撒点 (n={n_vis})\n红点=圆内, 蓝点=圆外, π ≈ {pi_est:.4f}')

# 收敛曲线
n_vals = np.logspace(1, 5, 200).astype(int)
pis = []
for n in n_vals:
    pi_e, _, _, _ = estimate_pi(n)
    pis.append(pi_e)
axes[1].plot(n_vals, pis, 'b-', linewidth=0.8)
axes[1].axhline(y=np.pi, color='r', linestyle='--', linewidth=1.5, label=f'π = {np.pi:.6f}')
axes[1].fill_between(n_vals, np.pi - 1.96/np.sqrt(n_vals), np.pi + 1.96/np.sqrt(n_vals),
                     alpha=0.15, label='95% 理论误差带')
axes[1].set_xscale('log')
axes[1].set_xlabel('n (样本数)')
axes[1].set_ylabel('π 估计值')
axes[1].set_title('收敛曲线：样本越多，估计越精确')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### ML 应用

| 场景 | 蒙特卡洛用法 |
|------|-------------|
| **强化学习** | MC 方法用轨迹的回报平均值估计状态价值 $V(s) \approx \frac{1}{N}\sum G_t$ |
| **贝叶斯推断** | MCMC 从难以解析的后验分布 P(θ\|D) 中采样 |
| **Dropout 不确定性** | MC Dropout——测试时多次随机 Dropout，预测方差 = 模型不确定性 |
| **生成模型** | GAN 从随机噪声 z ~ N(0,I) 采样生成新样本 |

---

## 17. 共轭先验——让贝叶斯计算有解析解

### 生活例子

你刚搬到一个新城市，想知道附近外卖的整体好评率。你没有数据——最朴素的想法是"好坏各半"（Beta(1,1) 先验）。看了 3 条评论全是好评——你的判断更新成了"大概 75% 好评"（后验 Beta(4,1)）。看了 100 条评论，80 条好评——你高度自信"好评率就是 80% 左右"（后验 Beta(81,21)）。

每次新数据来了，你只需**更新两个参数**，不需要重新看所有历史数据——这就是共轭先验的魔力。

### 直观理解

贝叶斯定理的核心计算是：后验 ∝ 似然 × 先验。一般后验没有解析解。但如果**先验和后验属于同一个分布族**——这个先验就叫做似然函数的**共轭先验**。

**Beta-Binomial 共轭**（最经典的例子）：
- 数据（似然）：抛 n 次硬币，k 次正面 → Binomial
- 先验：硬币正面概率 θ ~ Beta(α, β)
- 后验：θ\|数据 ~ Beta(α + k, β + (n - k))

先验参数 (α, β) 可以理解为**伪计数**：α = "之前看到的正面次数"，β = "之前看到的反面次数"。后验 = 伪计数 + 真实计数。直观到惊人！

### Python 演示

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
true_p = 0.7  # 真实好评率

# 先验：均匀分布 Beta(1,1)——"什么都不假设"
alpha_prior, beta_prior = 1, 1

# 逐批获取数据
data_sizes = [0, 3, 10, 30, 100, 500]
theta_grid = np.linspace(0.001, 0.999, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

cum_k, cum_n = 0, 0

for idx, size in enumerate(data_sizes):
    if size > 0:
        new_data = np.random.binomial(1, true_p, size)
        cum_k += new_data.sum()
        cum_n += size

    alpha_post = alpha_prior + cum_k
    beta_post = beta_prior + cum_n - cum_k

    posterior = stats.beta.pdf(theta_grid, alpha_post, beta_post)
    axes[idx].plot(theta_grid, posterior, 'b-', linewidth=2)
    axes[idx].axvline(x=true_p, color='r', linestyle='--', alpha=0.7,
                      label=f'真实 p={true_p}')
    axes[idx].set_title(f'n={cum_n}, k={cum_k}\n'
                        f'Beta({alpha_post},{beta_post}), '
                        f'均值={alpha_post/(alpha_post+beta_post):.3f}')
    axes[idx].set_xlabel('θ (好评率)')
    axes[idx].set_ylabel('后验密度')
    axes[idx].legend(fontsize=8)
    axes[idx].set_xlim(0, 1)

plt.suptitle('Beta-Binomial 共轭：数据越多，后验越集中在真实值附近', fontsize=14)
plt.tight_layout()
plt.show()

# 后验参数的完整演变
print(f"{'n':>6} {'k':>6} {'后验参数':>15} {'后验均值':>10} {'后验std':>10}")
print("-" * 55)
cum_k, cum_n = 0, 0
for size in [0, 3, 10, 30, 100, 500]:
    if size > 0:
        new_data = np.random.binomial(1, true_p, size)
        cum_k += new_data.sum()
        cum_n += size
    a = alpha_prior + cum_k
    b = beta_prior + cum_n - cum_k
    mean = a / (a + b)
    std = np.sqrt(a * b / ((a + b)**2 * (a + b + 1)))
    print(f"{cum_n:6d} {cum_k:6d} {'Beta('+str(a)+','+str(b)+')':>15} {mean:10.4f} {std:10.4f}")
```

### 常见共轭先验速查

| 似然（数据分布） | 共轭先验 | 后验 | ML 应用 |
|:--|:--|:--|:--|
| 二项 Binomial | Beta | Beta | A/B 测试、CTR 估计 |
| 泊松 Poisson | Gamma | Gamma | 事件率建模 |
| 高斯（已知 σ²） | 高斯 | 高斯 | 贝叶斯线性回归 |
| 多项 Multinomial | Dirichlet | Dirichlet | LDA 主题模型 |
| 高斯（已知 μ） | 逆 Gamma | 逆 Gamma | 方差估计 |

---

## 18. 多维高斯分布——当世界不止一个变量

### 生活例子

你给一个人做体检，同时测量身高和体重。单看身高是一个正态分布，单看体重也是——但身高和体重之间有正相关。要同时描述这两个变量以及它们之间的关联，你需要**二维高斯分布**。

### 直观理解

一维 → 多维的推广：
- μ（均值标量）→ **μ**（均值向量，d × 1）
- σ²（方差标量）→ **Σ**（协方差矩阵，d × d）

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

协方差矩阵 Σ 决定了分布的**形状**和**方向**：
- **对角元素** = 每个维度的方差（椭圆轴长）
- **非对角元素** = 协方差（椭圆的倾斜方向）

**对 Σ 做特征值分解**：Σ = QΛQᵀ → Λ 的每个特征值是椭圆半轴的平方 → Q 的列向量是椭圆的朝向。

### Python 可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_2d_gaussian(mu, cov, ax, title):
    rv = stats.multivariate_normal(mu, cov)
    grid = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(grid, grid)
    pos = np.dstack((X, Y))
    Z = rv.pdf(pos)
    ax.contour(X, Y, Z, levels=8, colors='blue', alpha=0.6)
    ax.contourf(X, Y, Z, levels=30, cmap='Blues', alpha=0.3)
    samples = rv.rvs(1500, random_state=42)
    ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.15, c='navy')
    # 特征向量
    eigvals, eigvecs = np.linalg.eigh(cov)
    for i in range(2):
        vec = eigvecs[:, i] * np.sqrt(eigvals[i]) * 2.5
        ax.annotate('', xy=mu + vec, xytext=mu,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.plot(*mu, 'r*', markersize=12)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
mu = np.array([0, 0])

cov1 = [[1, 0], [0, 1]]
plot_2d_gaussian(mu, cov1, axes[0],
    '圆形：独立且等方差\nΣ = [[1,0],[0,1]]')

cov2 = [[0.5, 0], [0, 3]]
plot_2d_gaussian(mu, cov2, axes[1],
    '轴对齐椭圆：独立不等方差\nΣ = [[0.5,0],[0,3]]')

cov3 = [[1.5, 1.2], [1.2, 1.5]]
plot_2d_gaussian(mu, cov3, axes[2],
    '旋转椭圆：两维正相关\nΣ = [[1.5,1.2],[1.2,1.5]]')

plt.suptitle('协方差矩阵决定二元高斯的形状（红箭头=特征向量×√特征值）', fontsize=13)
plt.tight_layout()
plt.show()

# 特征值分解揭示几何意义
print("=== 协方差矩阵特征值分解 ===")
for name, cov in [('圆形', cov1), ('轴对齐椭圆', cov2), ('旋转椭圆', cov3)]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    print(f"\n{name}: Σ =\n{np.array(cov)}")
    print(f"  特征值(半轴²): {eigvals}")
    print(f"  主方向角度: {angle:.1f}°")
```

### ML 应用

- **高斯混合模型（GMM）**：多个多维高斯叠加 = 任意复杂分布
- **异常检测**：样本在估计的高斯分布下概率密度过低 → 异常
- **VAE**：潜变量 z ~ N(0, I)，各维度的标准正态先验
- **Kalman 滤波**：状态估计中的不确定性以多维高斯形式传播

---

## 19. 概率不等式——在不知道具体分布时给出上界

### 生活例子

你当班主任。你知道全班这次考试的平均分是 72 分。你在开班会前想评估：有多少同学考了 90 分以上？你不知道具体分数分布——但你可以给出一个**上界**："最多不超过 72/90 = 80% 的同学考了 90 分以上"。

如果还知道标准差是 10 分，你能得到一个更紧的上界。

### 直观理解

概率不等式帮你**在最坏情况下量化风险**。即使你完全不知道分布的形状（只需要均值、方差等），也能给出概率的上界。ML 中的泛化误差界、算法收敛性证明都建立在这些不等式上。

### 马尔可夫不等式（Markov's Inequality）

条件：只要求 X ≥ 0 且期望有限。

$$P(X \geq a) \leq \frac{E[X]}{a}$$

直觉：如果平均分 72 分，那 ≥90 分的人数最多 80%（72/90=0.8）。

### 切比雪夫不等式（Chebyshev's Inequality）

条件：需要知道期望 μ 和方差 σ²。

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

直觉：偏离均值超过 2σ 的数据最多 25%；超过 3σ 最多 ~11.1%。

### Python 验证

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 500_000

# 分别在三种分布上验证切比雪夫不等式
print("=== 切比雪夫不等式：P(|X-μ| ≥ kσ) ≤ 1/k² ===\n")
print(f"{'分布':>18} {'k':>4} {'实际概率':>10} {'1/k² 上界':>10} {'上界/实际':>10}")
print("-" * 60)

for name, data in [
    ('均匀 U(0,1)', np.random.rand(N)),
    ('指数 Exp(1)', np.random.exponential(1, N)),
    ('正态 N(0,1)', np.random.randn(N)),
    ('泊松 Poi(5)', np.random.poisson(5, N)),
]:
    mu, sigma = data.mean(), data.std()
    for k in [2, 3, 4]:
        actual = np.mean(np.abs(data - mu) >= k * sigma)
        bound = 1.0 / k**2
        ratio = bound / actual if actual > 0 else float('inf')
        print(f"{name:>18} {k:4d} {actual:10.6f} {bound:10.6f} {ratio:10.1f}x")
    print()

# 可视化
fig, ax = plt.subplots(figsize=(8, 5))
k_vals = np.arange(1.5, 5.1, 0.1)
for name, data in [('均匀', np.random.rand(N)), ('指数', np.random.exponential(1, N)),
                    ('正态', np.random.randn(N))]:
    mu, sigma = data.mean(), data.std()
    actuals = [np.mean(np.abs(data - mu) >= k * sigma) for k in k_vals]
    ax.plot(k_vals, actuals, linewidth=2, label=f'{name} (实际)')
ax.plot(k_vals, 1.0 / k_vals**2, 'k--', linewidth=2.5, label='切比雪夫上界 1/k²')
ax.set_xlabel('k (σ 的倍数)'); ax.set_ylabel('P(|X-μ| ≥ kσ)')
ax.set_title('切比雪夫不等式：实际概率 vs. 理论上界')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 0.5)
plt.show()

# 关键洞察
print("关键洞察：正态分布 k=3 时实际概率仅 ~0.27%，但切比雪夫给到 11.1%——松了 ~40 倍。")
print("因为切比雪夫不利用分布形状的任何信息。信息越多，界越紧。")
```

### 常见误区

| 误区 | 真相 |
|------|------|
| 上界 = 真实概率 | 上界是"最坏情况"，通常比真实概率松很多倍。 |
| 切比雪夫只能用于正态 | 它对任何有方差有限分布都成立——这正是它的价值。 |

### ML 应用

- **PAC 学习理论**：泛化误差的上界证明
- **Hoeffding 不等式**：有界随机变量均值的集中不等式（比切比雪夫更紧）
- **Bandit 算法（UCB）**：置信区间上界的选择策略

---

## 20. 思考题（10 道，含完整解答）

### 题 1

你抛两枚公平硬币。求：
- (a) 两枚都是正面的概率
- (b) 至少一枚正面的概率
- (c) 已知第一枚是正面，第二枚也是正面的概率

<details>
<summary>点击查看解答</summary>

样本空间：Ω = {(正,正), (正,反), (反,正), (反,反)}，每个概率 1/4。

(a) P(两个正面) = P({(正,正)}) = 1/4

(b) P(至少一个正面) = 1 - P({(反,反)}) = 1 - 1/4 = 3/4

(c) 已知第一枚正面 → 条件在 {(正,正), (正,反)} 上 → P(第二枚正面\|第一枚正面) = (1/4) / (2/4) = 1/2

> 验证：第一枚和第二枚独立，所以条件概率等于无条件概率 1/2。
</details>

### 题 2

某罕见病发病率为 0.5%。某项检测的真阳性率为 98%，假阳性率为 3%。如果检测阳性，真正得病的概率是多少？请用贝叶斯公式手算。

<details>
<summary>点击查看解答</summary>

P(病) = 0.005，P(+|病) = 0.98，P(+|无病) = 0.03

$$P(病|+) = \frac{0.98 \times 0.005}{0.98 \times 0.005 + 0.03 \times 0.995}$$

$$= \frac{0.0049}{0.0049 + 0.02985} = \frac{0.0049}{0.03475} \approx 0.141$$

即约 **14.1%**。虽然比上一节的 50% 高（因为假阳性率 3% 在分母中权重变大了，但发病率更低），但仍然远低于直觉猜测的 98%。

> 多次阳性后：第二次阳性后 P(病) 约 84.5%；第三次阳性后约 99.5%。
</details>

### 题 3

设随机变量 X 的概率质量函数为：P(X=0)=0.2, P(X=1)=0.3, P(X=2)=0.4, P(X=3)=0.1。手算 E[X] 和 Var(X)。

<details>
<summary>点击查看解答</summary>

$$E[X] = 0 \times 0.2 + 1 \times 0.3 + 2 \times 0.4 + 3 \times 0.1 = 0 + 0.3 + 0.8 + 0.3 = 1.4$$

$$E[X^2] = 0^2 \times 0.2 + 1^2 \times 0.3 + 2^2 \times 0.4 + 3^2 \times 0.1 = 0 + 0.3 + 1.6 + 0.9 = 2.8$$

$$Var(X) = E[X^2] - (E[X])^2 = 2.8 - 1.4^2 = 2.8 - 1.96 = 0.84$$

$$\sigma = \sqrt{0.84} \approx 0.917$$
</details>

### 题 4

小明投篮命中率 80%，独立投 5 次。求恰好命中 3 次的概率（手算，并写出二项公式）。

<details>
<summary>点击查看解答</summary>

X ~ Binomial(n=5, p=0.8)

$$P(X=3) = \binom{5}{3} \cdot 0.8^3 \cdot 0.2^2$$

$$= 10 \times 0.512 \times 0.04 = 10 \times 0.02048 = 0.2048$$

> 组合系数 $\binom{5}{3} = \frac{5!}{3!2!} = \frac{120}{6 \times 2} = 10$

期望命中次数：E[X] = 5 × 0.8 = 4 次。
</details>

### 题 5

某咖啡店平均每小时来 4 位顾客（λ=4）。求接下来一小时：(a) 没有顾客的概率；(b) 恰好 2 位顾客的概率；(c) 至少 1 位顾客的概率（手算）。

<details>
<summary>点击查看解答</summary>

$P(X=k) = \frac{4^k \cdot e^{-4}}{k!}$，$e^{-4} \approx 0.01832$

(a) $P(X=0) = \frac{4^0 \cdot e^{-4}}{0!} = e^{-4} \approx 0.0183$

(b) $P(X=2) = \frac{4^2 \cdot e^{-4}}{2!} = \frac{16 \cdot 0.01832}{2} \approx 0.1465$

(c) $P(X \geq 1) = 1 - P(X=0) = 1 - e^{-4} \approx 0.9817$

> 每小时 4 位均值，但 98% 的时间至少来 1 人。如果 λ=1，则大约 63% 的时间至少来 1 人。
</details>

### 题 6

有数据点：(1, 2), (2, 3), (3, 7), (4, 8)。手算协方差和相关系数。

<details>
<summary>点击查看解答</summary>

E[X] = (1+2+3+4)/4 = 2.5
E[Y] = (2+3+7+8)/4 = 5.0

E[XY] = (1×2 + 2×3 + 3×7 + 4×8)/4 = (2+6+21+32)/4 = 61/4 = 15.25

Cov = E[XY] - E[X]E[Y] = 15.25 - 2.5×5 = 15.25 - 12.5 = 2.75

$$E[X^2] = (1+4+9+16)/4 = 30/4 = 7.5, \quad \sigma_X = \sqrt{7.5 - 6.25} = \sqrt{1.25} \approx 1.118$$

$$E[Y^2] = (4+9+49+64)/4 = 126/4 = 31.5, \quad \sigma_Y = \sqrt{31.5 - 25} = \sqrt{6.5} \approx 2.550$$

ρ = 2.75 / (1.118 × 2.550) = 2.75 / 2.851 ≈ **0.965**

> 极强的正相关——X 和 Y 几乎在一条直线上。
</details>

### 题 7

解释为什么"ρ = 0 不代表两个变量独立"，并举例。

<details>
<summary>点击查看解答</summary>

ρ = 0 只表示**没有线性关系**。考虑 Y = X²，X ~ U(-1, 1)：

- X 对称分布在 0 两侧，所以 Cov(X, X²) = E[X³] - E[X]E[X²] = 0 - 0×E[X²] = 0
- 但是 Y 完全由 X 决定——给定 X，Y 的值是确定的！

经典的反例就是上面的抛物线关系。还有圆形分布（如 X = cos(θ), Y = sin(θ), θ ~ U(0, 2π)）——Cov(X, Y) = 0 但 X² + Y² = 1，完全相关。

**启示**：相关系数只衡量线性关系。要做完整的依赖分析，还需要考察秩相关（Spearman）、互信息等非线性度量。
</details>

### 题 8

小明每天投 50 个三分球，命中率稳定在 40%。他一天的总命中数 X 服从什么分布？如果连续 30 天记录每天的命中数，这些日命中数的样本均值的分布大约是什么？请分别给出其参数。

<details>
<summary>点击查看解答</summary>

**日命中数 X**：~ Binomial(n=50, p=0.4)，E[X] = 20，Var(X) = 50×0.4×0.6 = 12

**30 天日均命中数 $\bar{X}$**：根据 CLT，$\bar{X} \approx \mathcal{N}(\mu, \sigma^2/n)$

$\mu = 20$，$\sigma^2/n = 12/30 = 0.4$

所以 $\bar{X} \approx \mathcal{N}(20, 0.4)$，即均值 20，标准差 $\sqrt{0.4} \approx 0.632$。

> 即使原始二项分布不完全对称（p ≠ 0.5），样本量 30 已经使得样本均值的分布近似正态。
</details>

### 题 9

有一个 [0,2]×[0,2] 的正方形，内切一个半径为 1 的圆（圆心在 (1,1)）。写出蒙特卡洛算法估算圆面积（进而估算 π）的步骤，并说明为什么有效。

<details>
<summary>点击查看解答</summary>

**步骤**：
1. 在 [0,2]×[0,2] 内随机均匀撒 N 个点 (x_i, y_i)
2. 统计满足 (x_i - 1)² + (y_i - 1)² ≤ 1 的点数 M
3. 圆面积估计 = (M/N) × 4（正方形面积），π = 圆面积 / r² = 圆面积 / 1 = 圆面积

**为什么有效**：大数定律保证 M/N → P(点在圆内) = 圆面积 / 正方形面积 = π/4。所以 4M/N → π。
</details>

### 题 10

解释贝叶斯学派和频率学派在以下两个问题上的分歧：
- (a) "明天北京下雨的概率"
- (b) "公平硬币抛 100 次后正面的概率"

<details>
<summary>点击查看解答</summary>

**(a) 明天北京下雨的概率**：

- 频率学派：没法说——明天只有一次，不存在"长期频率"的概念。或者他们把它解读为"在历史上类似天气条件下，第二天有 30% 的日子下雨"。
- 贝叶斯学派：坦率承认这是主观信念。"根据现有信息（卫星云图、温度、湿度），我评估下雨的概率是 30%。"这个概率可以在拿到新信息时更新。

**(b) 公平硬币抛 100 次后正面的概率**：

- 频率学派："公平"意味着 p = 0.5。抛 100 次后，正面概率仍是 0.5。他们不把 p 当作随机变量。
- 贝叶斯学派：把 p 当作随机变量，用 Beta 分布建模。抛前可能是 Beta(1,1)（均匀先验），抛 100 次后观察到 k 次正面 → 后验 Beta(1+k, 1+100-k)。贝叶斯会说："硬币的正面概率大约在 k/100 左右，但有一定不确定区间。"

**核心分歧**：频率学派认为参数是固定的未知量，贝叶斯学派认为参数是随机变量、其不确定性用概率分布描述。
</details>

---

## 总结：概率论在 ML 中的全景地图

| 概率概念 | ML 算法/技术 | 解决什么问题 |
|----------|-------------|-------------|
| 概率公理 | Softmax 归一化 | 保证输出是合法概率分布 |
| 条件概率 | 所有判别模型 P(y\|x) | 给定输入预测输出 |
| 贝叶斯定理 | 朴素贝叶斯、贝叶斯优化 | 基于证据更新信念 |
| 伯努利分布 | 逻辑回归 (二元交叉熵) | 二分类问题 |
| 二项分布 | 集成投票的分布 | 多模型的多数投票行为 |
| 多项分布 | Softmax 回归 | 多分类问题 |
| 高斯分布 | 高斯朴素贝叶斯、VAE、GMM | 连续特征/潜变量建模 |
| 泊松分布 | 计数回归 | 预测"次数"类目标 |
| 期望 | 损失函数 = 期望风险 | 定义优化目标 |
| 方差/协方差 | PCA 降维、特征选择 | 去冗余、降维 |
| 大数定律 | 交叉验证、模型评估 | 保证评估稳定可靠 |
| 中心极限定理 | 置信区间、A/B 测试 | 量化不确定性 |
| 蒙特卡洛 | MCMC、强化学习、MC Dropout | 近似不可解积分 |
| 共轭先验 | 贝叶斯参数估计 | 在线更新、解析后验 |
| 多维高斯 | GMM、异常检测、Kalman | 多变量联合建模 |
| 概率不等式 | PAC 学习、泛化误差 | 理论保证最坏情况 |

---

## 下一步

概率论为你建立了一个处理不确定性的框架。但还有一个问题没解决：**从有限样本中推断总体特征时，有多可靠？** 这就需要进入 [统计学](./math-statistics.md)——学习参数估计、假设检验和置信区间。
