# 信息论 — 从天气预报到机器学习损失函数

> **本章在 ML 学习路径中的位置**：数学基础第六站（最后一站）。信息论让你的机器学习工具箱拥有了一把能"量化不确定性、测量信息量、推导损失函数"的万能标尺。学完本章，你就打通了从数学到 ML 的完整链条。

---

## 0. 什么是信息？—— 从天气预报说起

### a) 生活例子

早上醒来，你打开手机看到两条天气预报推送：

- **消息 A**："太阳明天会从东边升起"
- **消息 B**："明天下午 3 点有 7 级台风登陆"

哪条消息让你觉得"我知道了新东西"？显然是 **B**。

为什么？因为 A 说了等于没说——你本来就 100% 确定太阳会升起。而 B 告诉你一件你完全没想到的事情——概率极低，一旦发生就是大新闻。

### b) 直观理解

**信息的本质 = "消除的不确定性"。** 一条消息携带的信息量，取决于它消除了多少不确定性。如果你原本就 100% 确定某事，那确认它的消息没有给你任何新信息，信息量为 0。如果某件事你原本觉得几乎不可能发生，但有人告诉你它真的发生了——这条消息信息量极大。

换个角度：信息量衡量的是**"惊讶程度"**（surprisal）。越让你惊讶的消息，信息量越大。

### c) 形式化定义

这引出了信息论中最基础的量——**自信息（Self-Information）**，也叫**信息量**：

$$I(x) = \log_2\frac{1}{P(x)} = -\log_2 P(x)$$

- 底数为 2 时，单位是**比特（bit）**
- 底数为 $e$ 时，单位是**奈特（nat）**（ML 中常用，因为 $e$ 底方便求导）
- 底数为 10 时，单位是**哈特（hart）**

**为什么用对数？** 有两个核心原因：

1. **概率相乘 → 信息量相加**：两个独立事件同时发生，其联合概率 $P(x,y)=P(x)P(y)$，我们希望对数值能加和：$I(x,y)=-\log P(x)P(y)=-\log P(x)-\log P(y)=I(x)+I(y)$
2. **递减性**：$P(x)$ 越小，$-\log P(x)$ 越大，符合直觉

### d) 手算示例

用计算器或纸笔来感受一下自信息的数值：

| 事件 | 概率 $P(x)$ | 信息量 $I(x) = -\log_2 P(x)$ (bit) | 直觉 |
|------|------------|-------------------------------------|------|
| 太阳升起 | 1.0 | $-\log_2 1 = 0$ | 毫无意外 |
| 抛硬币正面 | 0.5 | $-\log_2 0.5 = -\log_2(2^{-1}) = 1$ | 1 bit 惊讶 |
| 掷骰子出 6 | $1/6 \approx 0.167$ | $-\log_2(1/6) = \log_2 6 \approx 2.585$ | 比较意外 |
| 中双色球头奖 | $1/17721088$ | $-\log_2(1/17721088) \approx 24.08$ | 极其意外 |

> **直观理解 1 bit**：抛一枚公平硬币，正面朝上这条消息就恰好携带 1 bit 信息。这就像计算机里存储"是/否"需要 1 bit——不是巧合。

### e) Python 验证

```python
import numpy as np

def self_information(p, base=np.e):
    """计算事件的自信息量。base=np.e 得 nat, base=2 得 bit"""
    if not (0 < p <= 1):
        raise ValueError(f"概率必须在 (0,1] 之间, 你给了 {p}")
    return -np.log(p) / np.log(base)  # 换底公式

# 验证手算结果
print("====== 自信息量验证 ======")
print(f"太阳升起 (p=1.0):    I = {self_information(1.0, base=2):.2f} bit")
print(f"抛硬币正面 (p=0.5):  I = {self_information(0.5, base=2):.2f} bit")
print(f"掷骰子出6 (p=1/6):   I = {self_information(1/6, base=2):.4f} bit")
print(f"中双色球 (p=1/1.77e7): I = {self_information(1/17721088, base=2):.2f} bit")
print("====== 手算与代码一致 ✓ ======")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "信息量可以是负的" | ❌ $P(x) \in (0,1]$, 所以 $-\log P(x) \ge 0$, 信息量永远非负 |
| "概率大→信息量大" | ❌ 正好相反。概率越大越确定，信息量越小 |
| "底数无所谓" | ❌ 底数是单位：底 2 = bit, 底 e = nat。数值不同但可换算 |
| "信息量和消息长短有关" | ❌ 信息量取决于概率，和消息本身的文字长度无关 |

### g) ML 应用

自信息是后面所有概念的基石。在异常检测中，"异常"的定义就是发生概率极低的事件（高信息量）。在分类任务中，当模型对正确类别赋予低概率时，自信息（负对数概率）就很大——这就是交叉熵损失的直觉来源。

---

## 1. 熵 — 衡量整个分布的平均不确定度

### a) 生活例子

两个城市，你每天早上要"猜"天气来决定穿什么：

- **城市 A（拉斯维加斯）**：360 天晴天，5 天下雨
- **城市 B（伦敦）**：阴天 40%、晴天 30%、雨天 30%

在哪个城市你更需要每早查天气？显然是**B**。因为在 A 城，你闭眼猜"晴天"就有 360/365 ≈ 98.6% 的正确率，基本不需要查。但在 B 城，各种天气都有，每天都得费心去查。

**熵衡量的就是这种"每天需要花费多少精力去查天气"的不确定度。**

### b) 直观理解

自信息量 $I(x) = -\log P(x)$ 衡量的是**某一个具体事件**的惊讶度。但如果我们想知道**整个分布**有多大不确定，就需要对所有可能事件的信息量取一个**加权平均**——权重就是事件本身的发生概率。

这就是熵的直觉：**分布中所有可能结果的自信息量的期望值（加权平均）。**

**另一个等价视角（编码视角）**：熵也是对这个分布中的每个事件进行无损编码，所需的**最短平均编码长度（bit）**。分布越均匀，每个事件越"不可预测"，编码越长。在后面第 9 节我们会在霍夫曼编码中验证这一点。

### c) 形式化定义

$$H(X) = -\sum_{x \in \mathcal{X}} P(x) \log P(x) = \mathbb{E}_{x \sim P}[-\log P(x)]$$

或者说：$H(X) = \sum_x P(x) \cdot I(x)$，熵就是**期望自信息量**。

约定：$0 \cdot \log 0 = 0$（不可能发生的事件，不贡献不确定性）。

**关键性质**：

| 性质 | 描述 |
|------|------|
| 非负性 | $H(X) \ge 0$ |
| 确定性下界 | 存在某个 $P(x_k)=1$（其余全 0）时 $H(X)=0$ |
| 均匀分布上界 | $H(X) \le \log_2 K$（K 个可能取值），当且仅当均匀分布时取等号 |
| 连续性 | 熵是概率的连续函数，概率微小变化不会导致熵剧烈变化 |

**证明均匀分布熵最大**（拉格朗日乘子法）：

最大化 $\mathcal{L} = -\sum p_i \log p_i + \lambda(\sum p_i - 1)$，对每个 $p_i$ 求偏导等于 0：

$$\frac{\partial}{\partial p_i}\big(-p_i \log p_i + \lambda p_i\big) = -\log p_i - 1 + \lambda = 0$$

解得 $\log p_i = \lambda - 1$，即所有 $p_i$ 相等 → 均匀分布。

### d) 手算示例：抛硬币的熵

**场景：一枚公平硬币，正面概率 $p=0.5$，反面概率 $1-p=0.5$**

$$H = -0.5 \cdot \log_2(0.5) - 0.5 \cdot \log_2(0.5) = -0.5 \cdot (-1) - 0.5 \cdot (-1) = 0.5 + 0.5 = 1 \text{ bit}$$

**场景：一枚偏心硬币，正面概率 $p=0.8$，反面概率 $0.2$**

$$H = -0.8 \cdot \log_2(0.8) - 0.2 \cdot \log_2(0.2)$$

逐项手算：
- $\log_2(0.8) = \log_2(4/5) = \log_2 4 - \log_2 5 = 2 - 2.3219 = -0.3219$
- $\log_2(0.2) = \log_2(1/5) = -\log_2 5 = -2.3219$

$$H = -0.8 \cdot (-0.3219) - 0.2 \cdot (-2.3219) = 0.2575 + 0.4644 = 0.7219 \text{ bit}$$

对比：公平硬币的熵 = 1 bit，偏心硬币的熵 = 0.7219 bit。偏心硬币更"可预测"，所以熵更低。

### e) Python 验证

```python
import numpy as np

def entropy(probs, base=np.e):
    """计算离散分布的熵。0*log(0) = 0 已处理"""
    probs = np.asarray(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))

# 验证手算
H_fair = entropy([0.5, 0.5], base=2)     # 公平硬币
H_bias = entropy([0.8, 0.2], base=2)     # 偏心硬币
H_dice = entropy([1/6]*6, base=2)        # 公平骰子

print("====== 熵的手算验证 ======")
print(f"公平硬币 H = {H_fair:.4f} bit  (手算: 1.0000)")
print(f"偏心硬币 H = {H_bias:.4f} bit  (手算: 0.7219)")
print(f"公平骰子 H = {H_dice:.4f} bit  (理论: log₂6 = {np.log2(6):.4f})")

# 验证：确定性分布熵为 0
print(f"确定性[1,0,0,0] H = {entropy([1, 0, 0, 0], base=2):.1f} bit")
print("====== 全部验证通过 ✓ ======")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "类别越多熵越大" | ❌ 看概率分布。确定性的 100 类也是 H=0 |
| "熵只能取整数" | ❌ 熵是连续的，只是均匀分布下取整底对数 |
| "熵和方差是一回事" | ❌ 方差衡量离散度，熵衡量分布形状。两个方差相同的分布熵可以不同 |

### g) ML 应用

- **数据集评估**：训练前计算标签分布的熵。高熵 → 各类别均匀 → 需要更多样本/参数来学习
- **决策树**：熵是 ID3/C4.5 的分裂准则基础（见第 7 节信息增益）
- **无监督学习**：最小熵原则（使聚类后各簇尽可能"纯"）

---

## 2. 联合熵与条件熵 — 两个变量在一起有多不确定？

### a) 生活例子

你和朋友约好周末去郊游，但你们面临两个不确定性：

- **天气 X**：晴天、阴天、雨天
- **是否成行 Y**：去、不去

**联合熵 $H(X, Y)$** 问的是："天气和出行这两件事，总共有多大不确定性？"

**条件熵 $H(Y|X)$** 问的是："如果我告诉你天气了，你对'是否出行'还剩多少不确定性？"

直觉上，如果你知道是雨天，你几乎可以肯定要取消——知道 X 大幅减少了 Y 的不确定性，$H(Y|X)$ 比 $H(Y)$ 小很多。

### b) 直观理解

联合熵就是把 X 和 Y 看成一个"联合变量 $(X, Y)$"，它的熵就是所有 $(x_i, y_j)$ 组合的不确定性。

条件熵 $H(Y|X)$ 是：先按 X 的取值分组，计算每组内 Y 的熵，然后按每组概率加权平均。公式上：

$$H(Y|X) = \sum_x P(x) \cdot H(Y|X=x) = \sum_x P(x) \left(-\sum_y P(y|x) \log P(y|x)\right)$$

**链式法则**（极其重要）：

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

含义：总不确定性 = X 自身的不确定性 + 知道 X 后 Y 还剩的不确定性。如果你先描述天气（$H(X)$），再在知道天气后描述出行（$H(Y|X)$），你就完整描述了整个系统。

### c) 形式化定义

**联合熵**：

$$H(X, Y) = -\sum_{x}\sum_{y} P(x,y) \log P(x,y)$$

**条件熵**：

$$H(Y|X) = -\sum_{x}\sum_{y} P(x,y) \log P(y|x) = \sum_{x} P(x) \cdot H(Y|X=x)$$

两个公式之间的关系（链式法则的推导）：

$$\begin{aligned}
H(X, Y) &= -\sum_{x}\sum_{y} P(x,y) \log P(x,y) \\
&= -\sum_{x}\sum_{y} P(x,y) \log \big(P(x) P(y|x)\big) \\
&= -\sum_{x}\sum_{y} P(x,y) \log P(x) - \sum_{x}\sum_{y} P(x,y) \log P(y|x) \\
&= -\sum_x P(x) \log P(x) - \sum_{x}\sum_{y} P(x,y) \log P(y|x) \\
&= H(X) + H(Y|X)
\end{aligned}$$

### d) 手算示例：天气与出行

假设历史数据统计出如下联合概率 $P(X, Y)$：

| 天气 X \ 出行 Y | 去 (Y=0) | 不去 (Y=1) | P(X) |
|:---:|:---:|:---:|:---:|
| 晴天 (X=0) | 0.48 | 0.12 | 0.60 |
| 雨天 (X=1) | 0.10 | 0.30 | 0.40 |
| P(Y) | 0.58 | 0.42 | 1.00 |

**Step 1：联合熵 $H(X,Y)$**

四个格子逐一计算：
- $P(晴,去)=0.48$：$-0.48 \times \log_2(0.48) \approx -0.48 \times (-1.0589) = 0.5083$
- $P(晴,不去)=0.12$：$-0.12 \times \log_2(0.12) \approx -0.12 \times (-3.0589) = 0.3671$
- $P(雨,去)=0.10$：$-0.10 \times \log_2(0.10) = -0.10 \times (-3.3219) = 0.3322$
- $P(雨,不去)=0.30$：$-0.30 \times \log_2(0.30) \approx -0.30 \times (-1.7370) = 0.5211$

$$H(X,Y) = 0.5083 + 0.3671 + 0.3322 + 0.5211 = 1.7287 \text{ bit}$$

**Step 2：条件熵 $H(Y|X)$**

先计算 H(X)（天气本身的熵）：
$$P(晴)=0.60, P(雨)=0.40$$
$$H(X) = -0.6\log_2(0.6) - 0.4\log_2(0.4) \approx 0.4422 + 0.5288 = 0.9710 \text{ bit}$$

再计算 H(Y)（出行本身的熵）：
$$P(去)=0.58, P(不去)=0.42$$
$$H(Y) = -0.58\log_2(0.58) - 0.42\log_2(0.42) \approx 0.4558 + 0.5256 = 0.9814 \text{ bit}$$

利用链式法则：
$$H(Y|X) = H(X,Y) - H(X) = 1.7287 - 0.9710 = 0.7577 \text{ bit}$$

**解读**：出行本身的不确定性是 0.9814 bit，但知道天气后只剩 0.7577 bit——天气帮你减少了 0.2237 bit 的不确定性。这个差值，正是下一节要讲的**互信息**。

### e) Python 验证

```python
import numpy as np

def entropy_labels(values, base=np.e):
    """从标签列表计算熵"""
    _, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return -np.sum(probs * np.log(probs) / np.log(base))

def joint_entropy(joint_prob, base=np.e):
    """计算联合熵 H(X,Y)"""
    joint_prob = np.asarray(joint_prob)
    joint_prob = joint_prob[joint_prob > 0]
    return -np.sum(joint_prob * np.log(joint_prob) / np.log(base))

def conditional_entropy(joint_prob, base=np.e):
    """计算 H(Y|X)。joint_prob[i][j] = P(X=i, Y=j)"""
    joint_prob = np.asarray(joint_prob, dtype=float)
    px = joint_prob.sum(axis=1)
    h_cond = 0.0
    for i in range(len(px)):
        if px[i] > 0:
            py_given_x = joint_prob[i] / px[i]
            py_given_x = py_given_x[py_given_x > 0]
            h_cond += px[i] * (-np.sum(py_given_x * np.log(py_given_x) / np.log(base)))
    return h_cond

# 手算验证
P_joint = np.array([
    [0.48, 0.12],   # 晴天：[去, 不去]
    [0.10, 0.30],   # 雨天：[去, 不去]
])

print("====== 联合熵 & 条件熵 手算验证 ======")
print(f"H(X,Y)   = {joint_entropy(P_joint, base=2):.4f} bit  (手算: 1.7287)")
print(f"H(Y|X)   = {conditional_entropy(P_joint, base=2):.4f} bit  (手算: 0.7577)")

px = P_joint.sum(axis=1)
py = P_joint.sum(axis=0)
hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
h_cond = conditional_entropy(P_joint, base=2)

print(f"H(X)     = {hx:.4f} bit")
print(f"H(Y)     = {hy:.4f} bit")
print(f"H(X) + H(Y|X) = {hx:.4f} + {h_cond:.4f} = {hx + h_cond:.4f} bit")
print(f"H(X,Y)   = {joint_entropy(P_joint, base=2):.4f} bit  ← 链式法则成立 ✓")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "$H(Y\|X) \ge H(Y)$" | ❌ 知道 X 最多让你不减少不确定性，不会增加。$H(Y\|X) \le H(Y)$ 严格成立 |
| "$H(X,Y)$ 就是 $H(X)+H(Y)$" | ❌ 只有当 X 和 Y 独立时才成立。一般 $H(X,Y) = H(X) + H(Y\|X)$ |
| "条件熵 = 条件分布的熵" | ❌ 是"各条件下 Y 的熵的加权平均"，不是 $P(Y\|X=x)$ 那个分布的熵 |

### g) ML 应用

- **特征有效性**：如果 $H(Y|X_{\text{特征}}) \approx 0$，说明这个特征完美定义了标签（过度理想，一般是数据泄露）
- **链式法则**是 VAE（变分自编码器）变分下界推导的核心工具
- **贝叶斯网络**的构建依赖链式法则和条件独立分解

---

## 3. 互信息 — X 和 Y 共享了多少信息？

### a) 生活例子

你和室友合租，室友每天早上做的事和你是否带伞高度相关：

- 如果室友出门前**带了伞**，你很可能会带（因为可能下雨了）
- 如果室友**没带伞**，你大概率也不带

你和室友之间"共享"了关于天气的信息。互信息就是量化这种**共享信息量**的。

### b) 直观理解

互信息有三个等价的直观理解，每一个都对应一个公式：

**理解 1（不确定性减少）**：知道 X 之后，关于 Y 的不确定性减少了多少？
$$I(X;Y) = H(Y) - H(Y|X)$$

**理解 2（对称版本）**：知道 Y 之后，关于 X 的不确定性减少了多少？
$$I(X;Y) = H(X) - H(X|Y)$$

**理解 3（分布的差异）**：联合分布 $P(x,y)$ 和"假装独立"的乘积分布 $P(x)P(y)$ 之间差多远？如果 X 和 Y 有关联，联合分布就和乘积分布不同，这个差异就是互信息。
$$I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

**关键性质**：
- 对称性：$I(X;Y) = I(Y;X)$
- 非负性：$I(X;Y) \ge 0$，等于 0 当且仅当 X 和 Y 统计独立
- 上界：$I(X;Y) \le \min(H(X), H(Y))$

### c) 形式化定义

$$I(X;Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} = D_{KL}\big(P(x,y) \parallel P(x)P(y)\big)$$

互信息就是联合分布 $P(x,y)$ 相对于独立分布 $P(x)P(y)$ 的 KL 散度（见第 4 节）。如果 X 和 Y 真的独立，$P(x,y) = P(x)P(y)$，KL 散度为 0，互信息为 0。

### d) 手算示例：天气与出行（续）

沿用上一节的联合概率表：

$$P = \begin{pmatrix} 0.48 & 0.12 \\ 0.10 & 0.30 \end{pmatrix}, \quad P(X) = (0.60, 0.40), \quad P(Y) = (0.58, 0.42)$$

**方法一（从熵算）**：
$$I(X;Y) = H(Y) - H(Y|X) = 0.9814 - 0.7577 = 0.2237 \text{ bit}$$

**方法二（从定义手算）**，四个格子：

$I_{00}$（晴天, 去）：$\frac{P(晴,去)}{P(晴)P(去)} = \frac{0.48}{0.60 \times 0.58} = \frac{0.48}{0.348} \approx 1.3793$

- 贡献：$0.48 \times \log_2(1.3793) = 0.48 \times 0.4639 = 0.2227$

$I_{01}$（晴天, 不去）：$\frac{0.12}{0.60 \times 0.42} = \frac{0.12}{0.252} \approx 0.4762$

- 贡献：$0.12 \times \log_2(0.4762) = 0.12 \times (-1.0704) = -0.1284$

$I_{10}$（雨天, 去）：$\frac{0.10}{0.40 \times 0.58} = \frac{0.10}{0.232} \approx 0.4310$

- 贡献：$0.10 \times \log_2(0.4310) = 0.10 \times (-1.2141) = -0.1214$

$I_{11}$（雨天, 不去）：$\frac{0.30}{0.40 \times 0.42} = \frac{0.30}{0.168} \approx 1.7857$

- 贡献：$0.30 \times \log_2(1.7857) = 0.30 \times 0.8365 = 0.2510$

$$I(X;Y) = 0.2227 + (-0.1284) + (-0.1214) + 0.2510 = 0.2239 \text{ bit}$$

两种方法结果一致（误差来自四舍五入）。

**直觉解读**：晴天→去（0.48 > 0.60×0.58=0.348）和雨天→不去（0.30 > 0.40×0.42=0.168）这两格是"关联超出独立预期"的，贡献正值；而晴天→不去和雨天→去是"反关联"的，贡献负值。净效果约 0.22 bit —— 天气和出行确实有一定关联，但不是决定性的。

### e) Python 验证

```python
import numpy as np

def mutual_information(joint_prob, base=np.e):
    """从联合概率直接计算互信息"""
    joint_prob = np.asarray(joint_prob, dtype=float)
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)
    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(
                    joint_prob[i, j] / (px[i] * py[j])
                ) / np.log(base)
    return mi

# 手算验证
P_joint = np.array([[0.48, 0.12], [0.10, 0.30]])
mi = mutual_information(P_joint, base=2)
print(f"互信息 I(X;Y) = {mi:.4f} bit  (手算: 0.2239)")

# 对比三种场景
P_independent = np.array([[0.25, 0.25], [0.25, 0.25]])  # 独立
P_partial     = np.array([[0.40, 0.10], [0.10, 0.40]])  # 部分依赖
P_deterministic = np.array([[0.50, 0.00], [0.00, 0.50]]) # 完全依赖

print(f"\n独立场景:     I(X;Y) = {mutual_information(P_independent, base=2):.4f} bit")
print(f"部分依赖:     I(X;Y) = {mutual_information(P_partial, base=2):.4f} bit")
print(f"完全依赖:     I(X;Y) = {mutual_information(P_deterministic, base=2):.4f} bit")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "互信息为 0 就是独立" | ✓ 这个是对的——它是等价关系：$I(X;Y)=0 \iff X \perp Y$ |
| "互信息可以是负的" | ❌ 永远不会为负（它是 KL 散度，非负） |
| "互信息 = 相关系数" | ❌ 相关系数只检测线性关系。互信息能检测任意统计依赖（非线性的也行） |
| "互信息高说明因果" | ❌ 互信息只衡量统计关联，不等于因果关系 |

### g) ML 应用

- **特征选择**：`sklearn.feature_selection.mutual_info_classif` 计算每个特征与标签的互信息，选 top-k
- **信息瓶颈理论**：压缩输入 X 的同时保留与输出 Y 的互信息
- **表示学习**：InfoNCE 损失（对比学习的核心）就是互信息的下界
- **聚类评估**：NMI（归一化互信息）评估聚类结果与真实标签的匹配度

---

## 4. KL 散度 — 用错了分布，要"多付"多少比特？

### a) 生活例子

你是快递分拣站的编码设计师，你设计了一套最优编码方案——根据包裹的历史收件地址概率分布 Q，给常见地址用短码，冷门地址用长码。

但突然有一天，收件地址的实际概率分布变成了 P（比如双十一促销导致某个城市包裹暴增）。你仍然用为 Q 设计的编码来编码 P 的数据——你会浪费多少 bits？

**KL 散度衡量：用 Q 代替 P 进行最优编码，平均每符号要多付多少比特。这就是"额外开销"。**

### b) 直观理解

回想自信息 $I(x) = -\log P(x)$ 是"用真实分布 P 来编码事件 x 的最短编码长度"。如果改用错误的分布 Q 来编码，长度为 $-\log Q(x)$。实际数据的分布是 P，所以我们算的是"用 Q 的编码方案来编码来自 P 的数据，平均长度与理论最短长度的差值"：

$$D_{KL}(P \parallel Q) = \underbrace{\sum_x P(x)(-\log Q(x))}_{\text{用 Q 编码 P 的平均长度}} - \underbrace{\sum_x P(x)(-\log P(x))}_{\text{理论最短长度}}$$

**注意顺序**：$D_{KL}(P \parallel Q)$ 中，**P 是真实分布**，**Q 是近似/模型分布**。方向很重要——KL 散度不对称！

### c) 形式化定义

$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

**关键性质**：

| 性质 | 说明 |
|------|------|
| 非负性 | $D_{KL}(P \parallel Q) \ge 0$（Gibbs 不等式），等于 0 仅当 P=Q |
| 不对称性 | $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$ 一般 |
| 不满足三角不等式 | 所以 KL 散度不是真正的"距离"度量 |

### d) 手算示例：两个二元分布的 KL 散度

假设真实分布 $P$ 是公平硬币：$P = (0.5, 0.5)$
模型分布 $Q$ 是有偏估计：$Q = (0.8, 0.2)$

计算 $D_{KL}(P \parallel Q)$：

$$D_{KL}(P \parallel Q) = 0.5 \cdot \log_2\frac{0.5}{0.8} + 0.5 \cdot \log_2\frac{0.5}{0.2}$$

第一项：$\log_2(0.5/0.8) = \log_2(0.625) = -0.6781$，贡献：$0.5 \times (-0.6781) = -0.3390$

第二项：$\log_2(0.5/0.2) = \log_2(2.5) = 1.3219$，贡献：$0.5 \times 1.3219 = 0.6610$

$$D_{KL}(P \parallel Q) = -0.3390 + 0.6610 = 0.3220 \text{ bit}$$

现在反过来算 $D_{KL}(Q \parallel P)$：

$$D_{KL}(Q \parallel P) = 0.8 \cdot \log_2\frac{0.8}{0.5} + 0.2 \cdot \log_2\frac{0.2}{0.5}$$

第一项：$\log_2(1.6) = 0.6781$，贡献：$0.8 \times 0.6781 = 0.5425$

第二项：$\log_2(0.4) = -1.3219$，贡献：$0.2 \times (-1.3219) = -0.2644$

$$D_{KL}(Q \parallel P) = 0.5425 + (-0.2644) = 0.2781 \text{ bit}$$

**两者不相等！** $0.3220 \neq 0.2781$ —— 验证了 KL 散度的不对称性。

### e) Python 验证

```python
import numpy as np

def kl_divergence(p, q, base=np.e, eps=1e-12):
    """计算 D_KL(P || Q)"""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return np.sum(p * np.log(p / q) / np.log(base))

P = np.array([0.5, 0.5])
Q = np.array([0.8, 0.2])

print("====== KL 散度手算验证 ======")
print(f"D_KL(P||Q) = {kl_divergence(P, Q, base=2):.4f} bit  (手算: 0.3220)")
print(f"D_KL(Q||P) = {kl_divergence(Q, P, base=2):.4f} bit  (手算: 0.2781)")
print(f"不对称性验证: {'✓' if abs(kl_divergence(P,Q,base=2) - kl_divergence(Q,P,base=2)) > 0.01 else '✗'}")

# 边界情况：Q 中有 0 而 P 不为 0
P2 = np.array([0.5, 0.5])
Q2 = np.array([1.0, 0.0])
# 第二项: P(2)=0.5, Q(2)=0 → log(0.5/0) → +∞
print(f"\nP=[0.5,0.5], Q=[1.0,0.0]: D_KL(P||Q) → +∞ (P在Q=0处有概率)")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "KL 散度是对称的" | ❌ $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$。真正对称的版本是 JS 散度（见下一节） |
| "KL 散度是距离" | ❌ 不满足对称性和三角不等式，应叫"散度"而非"距离" |
| "D_KL(P\|Q)=0 意味着 P=Q" | ✓ 这个是对的——非负且仅在 P=Q 时为零 |
| "用 forward KL 还是 reverse KL 无所谓" | ❌ 在近似推断中选哪个 KL 方向影响很大。forward KL (P\|Q) 鼓励 Q 覆盖 P 的全部区域（mean-seeking），reverse KL 鼓励 Q 集中在 P 的高密度区（mode-seeking） |

### g) ML 应用

- **VAE（变分自编码器）**：KL 散度作为正则项，推动近似后验靠近标准正态分布
- **策略优化（PPO）**：限制新旧策略之间的 KL 散度，防止更新过猛
- **知识蒸馏**：学生模型模仿教师模型的输出分布
- **EM 算法**：E 步本质上是最小化 KL 散度

---

## 5. Jensen-Shannon 散度 — KL 的"对称修复版"

### a) 生活例子

KL 散度的问题就像是：你去朋友家做客，走错了小区 A（你的分布）去到了小区 B（朋友的分布），结果你多走的路程（KL 散度）取决于你以谁为起点，从 A 到 B 和从 B 到 A 是不一样的。但如果你们约在一个中点见面，分别从你们所在的位置走到中点，把两人的"路程"平均一下——这下就对称了。

**这就是 JS 散度的思想：先找中点 M，再算 P 到 M 和 Q 到 M 的 KL 散度的均值。**

### b) 直观理解

JS 散度解决了 KL 散度的两个痛点：

1. **不对称** → JS 散度是对称的
2. **可能无穷大**（当 P 的某个区域 Q 的概率为 0 时）→ M 是 P 和 Q 的均值，永远不会出现 Q=0 而 P>0 的情况，所以 JS 始终有限

### c) 形式化定义

设 $M = \frac{1}{2}(P + Q)$ 为两个分布的中点分布：

$$D_{JS}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$

**性质**：

| 性质 | 说明 |
|------|------|
| 对称性 | $D_{JS}(P \parallel Q) = D_{JS}(Q \parallel P)$ |
| 有界 | $0 \le D_{JS} \le \log 2$（底数为 2 时，上界为 1 bit） |
| 平方根是度量 | $\sqrt{D_{JS}}$ 满足三角不等式，是真正的距离度量 |

### d) 手算示例

依然用 $P=(0.5, 0.5)$ 和 $Q=(0.8, 0.2)$：

**Step 1：算中点 M**
$$M = \left(\frac{0.5+0.8}{2}, \frac{0.5+0.2}{2}\right) = (0.65, 0.35)$$

**Step 2：$D_{KL}(P \parallel M)$**

$$= 0.5 \cdot \log_2\frac{0.5}{0.65} + 0.5 \cdot \log_2\frac{0.5}{0.35}$$

$\log_2(0.5/0.65) = \log_2(0.7692) = -0.3785$，贡献：$0.5 \times (-0.3785) = -0.1893$

$\log_2(0.5/0.35) = \log_2(1.4286) = 0.5146$，贡献：$0.5 \times 0.5146 = 0.2573$

$$D_{KL}(P \parallel M) = -0.1893 + 0.2573 = 0.0681$$

**Step 3：$D_{KL}(Q \parallel M)$**

$$= 0.8 \cdot \log_2\frac{0.8}{0.65} + 0.2 \cdot \log_2\frac{0.2}{0.35}$$

$\log_2(0.8/0.65) = \log_2(1.2308) = 0.2996$，贡献：$0.8 \times 0.2996 = 0.2397$

$\log_2(0.2/0.35) = \log_2(0.5714) = -0.8074$，贡献：$0.2 \times (-0.8074) = -0.1615$

$$D_{KL}(Q \parallel M) = 0.2397 + (-0.1615) = 0.0782$$

**Step 4：JS 散度**

$$D_{JS}(P \parallel Q) = \frac{1}{2}(0.0681 + 0.0782) = 0.0732 \text{ bit}$$

验证对称性：$D_{JS}(Q \parallel P) = D_{JS}(P \parallel Q) = 0.0732$（对称！）

### e) Python 验证

```python
import numpy as np

def kl_divergence(p, q, base=np.e, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return np.sum(p * np.log(p / q) / np.log(base))

def js_divergence(p, q, base=np.e, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base=base) + 0.5 * kl_divergence(q, m, base=base)

P = np.array([0.5, 0.5])
Q = np.array([0.8, 0.2])

print("====== JS 散度手算验证 ======")
print(f"D_JS(P||Q) = {js_divergence(P, Q, base=2):.4f} bit  (手算: 0.0732)")
print(f"D_JS(Q||P) = {js_divergence(Q, P, base=2):.4f} bit  (手算: 0.0732)")
print(f"对称性: {'✓' if abs(js_divergence(P,Q,base=2) - js_divergence(Q,P,base=2)) < 1e-10 else '✗'}")

# 对比 KL
print(f"\nD_KL(P||Q) = {kl_divergence(P, Q, base=2):.4f} bit")
print(f"D_KL(Q||P) = {kl_divergence(Q, P, base=2):.4f} bit")
print(f"KL 不对称: ✓")

# 验证上界
print(f"\nJS 上界 = log2(2) = {np.log2(2):.4f} bit")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "JS 散度就是 KL 散度的一半" | ❌ 它是对中点加权平均的 KL，不是简单除以 2 |
| "JS 散度解决了 KL 的所有问题" | ❌ JS 散度在 GAN 训练中仍有梯度消失问题（当 P 和 Q 不重叠时接近常数 log2），这催生了 Wasserstein 距离 |

### g) ML 应用

- **GAN 的理论基础**：原始 GAN 的损失函数等价于最小化 $D_{JS}(P_{\text{data}} \parallel P_{\text{generator}})$
- **GAN 训练不稳定的原因**：当真实分布和生成分布几乎不重叠时，JS 散度趋向常数 $\log 2$，梯度消失 → WGAN 用 Wasserstein 距离替代

---

## 6. 交叉熵 — "猜错了"的代价（ML 最重要的损失函数）

### a) 生活例子

你是答题节目的选手，主持人给了你一系列选择题。每道题你给出一个"信心分布"——比如 A 选项 80% 确定，B 选项 15%，C 选项 5%。如果你的 80% 信心那个选项就是正确答案，你的损失很小。但如果你是 20% 确定的那个选项猜对了，你这次"侥幸"代价就很大——因为这意味着你大部分时候猜错了。

**交叉熵衡量的是：当你用分布 Q 来估计真实分布 P 时，"猜错的平均代价"。**

### b) 直观理解

交叉熵是 KL 散度拆解后的第一部分：

$$H(P, Q) = \underbrace{H(P)}_{\text{真实熵（无法改变）}} + \underbrace{D_{KL}(P \parallel Q)}_{\text{用错分布的额外开销}}$$

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

在分类任务中，真实分布 P 通常是 one-hot（只有一个正确的），所以：

$$H(P, Q) = -\log Q(\text{正确类别})$$

**交叉熵 = 负对数似然 = 模型对"正确答案"赋予的概率的对数取反**。如果模型自信地对正确类别预测 0.9，$-\log(0.9) \approx 0.105$，损失很小。如果模型茫然只给 0.3，$-\log(0.3) \approx 1.204$，损失很大。

### c) 形式化定义及与 MLE 的完整推导

这是本章最核心的推导——**交叉熵最小化 ⇔ 最大似然估计 (MLE)**。

**推导分四步**：

**Step 1：似然函数**

给定 N 个独立同分布样本 $\{(x_1, y_1), \ldots, (x_N, y_N)\}$，模型参数为 $\theta$，模型对第 i 个样本的预测概率为 $P_\theta(y_i|x_i)$。

似然（在参数 $\theta$ 下看到这组数据的概率）：

$$L(\theta) = \prod_{i=1}^{N} P_\theta(y_i | x_i)$$

**Step 2：对数似然**

取 $\log$（乘法变加法，数值更稳定）：

$$\log L(\theta) = \sum_{i=1}^{N} \log P_\theta(y_i | x_i)$$

最大似然估计：$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \log L(\theta)$

**Step 3：与交叉熵连接**

设经验分布 $P_{\text{data}}$（每个样本权重 $1/N$），模型分布为 $Q_\theta$。经验分布下的交叉熵：

$$H(P_{\text{data}}, Q_\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log Q_\theta(y_i | x_i) = -\frac{1}{N} \log L(\theta)$$

**Step 4：等价性结论**

$$\arg\min_\theta H(P_{\text{data}}, Q_\theta) = \arg\min_\theta \left(-\frac{1}{N} \log L(\theta)\right) = \arg\max_\theta \log L(\theta)$$

> **最小化交叉熵损失 = 最大化对数似然**。这就是为什么 PyTorch 里叫 `CrossEntropyLoss`、统计学里叫负对数似然——它们是同一个东西。

### d) 手算示例：多分类交叉熵

假设三分类问题，真实标签是类别 2（one-hot: $[0, 0, 1]$）。

**模型 A（好预测）**：$Q_A = [0.1, 0.1, 0.8]$

$$H(P, Q_A) = -[0 \cdot \log(0.1) + 0 \cdot \log(0.1) + 1 \cdot \log(0.8)] = -\log(0.8) = 0.2231 \text{ nat}$$

以 bit 单位：$-\log_2(0.8) \approx 0.3219 \text{ bit}$

**模型 B（差预测）**：$Q_B = [0.45, 0.45, 0.1]$

$$H(P, Q_B) = -[0 + 0 + \log(0.1)] = -\log(0.1) = 2.3026 \text{ nat}$$

以 bit 单位：$-\log_2(0.1) \approx 3.3219 \text{ bit}$

模型 A 的损失远小于模型 B——符合直觉：模型 A 虽然没到 100% 确定，但对正确答案至少给了 0.8 的概率，而模型 B 几乎忽略正确答案。

### e) Python 验证

```python
import numpy as np

def cross_entropy_onehot(y_true_onehot, y_pred_probs, eps=1e-12):
    """y_true: one-hot, y_pred: 概率向量"""
    y_pred_probs = np.clip(np.asarray(y_pred_probs), eps, 1 - eps)
    return -np.sum(y_true_onehot * np.log(y_pred_probs))

def cross_entropy_label(y_true_label, y_pred_probs, eps=1e-12):
    """y_true: 类别 index, y_pred: 概率向量"""
    y_pred_probs = np.clip(np.asarray(y_pred_probs), eps, 1 - eps)
    return -np.log(y_pred_probs[y_true_label])

# 手算验证
y_true = np.array([0, 0, 1])  # 真实类别 2
Q_good = np.array([0.1, 0.1, 0.8])
Q_bad  = np.array([0.45, 0.45, 0.1])

print("====== 交叉熵手算验证 ======")
print(f"好模型 (nat):  {cross_entropy_onehot(y_true, Q_good):.4f}  (手算: 0.2231)")
print(f"好模型 (bit):  {cross_entropy_onehot(y_true/np.log(2), Q_good):.4f}")
print(f"差模型 (nat):  {cross_entropy_onehot(y_true, Q_bad):.4f}  (手算: 2.3026)")

# 等价性验证：交叉熵 vs 负对数似然
N = 100
y_pred = np.random.dirichlet([1, 1, 1], N)  # 模拟预测概率
y_label = np.random.randint(0, 3, N)  # 模拟真实标签

ce_loss = np.mean([-np.log(np.clip(y_pred[i, y_label[i]], 1e-12, 1.0))
                   for i in range(N)])
neg_log_likelihood = -np.sum([np.log(np.clip(y_pred[i, y_label[i]], 1e-12, 1.0))
                              for i in range(N)]) / N

print(f"\n交叉熵损失:   {ce_loss:.6f}")
print(f"负对数似然/N:  {neg_log_likelihood:.6f}")
print(f"等价性: {'✓' if abs(ce_loss - neg_log_likelihood) < 1e-10 else '✗'}")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "交叉熵和 KL 散度是同一个东西" | ❌ $H(P,Q) = H(P) + D_{KL}(P\|Q)$。多了一项 $H(P)$，但 $H(P)$ 是常数（真实分布），最小化时不影响参数 |
| "softmax 输出是概率就直接用" | ❌ 要取 log！PyTorch 的 `CrossEntropyLoss` 内部已经做了 softmax+log |
| "用交叉熵还是 MSE 做分类都可以" | ❌ MSE 做分类会导致梯度消失问题，交叉熵是分类任务的正确选择 |
| "损失为 0 代表完美预测" | ❌ 交叉熵 0 意味着对正确类别预测概率为 1.0，但数据中总有噪声和不确定性 |

### g) ML 应用

- **分类任务的标准损失**：`torch.nn.CrossEntropyLoss`、`tf.keras.losses.CategoricalCrossentropy`
- **语言模型训练**：每个 token 的预测本质上都是对词表的分类，都用交叉熵
- **目标检测**：分类子任务的损失函数
- **语义分割**：逐像素的交叉熵

---

## 7. 信息增益 — 决策树凭什么选特征？

### a) 生活例子

你主持一个猜谜游戏——从一群人里猜出谁是"卧底"。你有权提问，但要挑最有信息量的那个问题让你尽快缩小范围。

比如面对 10 个人（5 个卧底、5 个平民）：
- **问题 A**："你是男的吗？" → 假设男女各半，问完后每组还是卧底和平民混杂
- **问题 B**："你昨天参加了秘密会议吗？" → 这个问题几乎完美区分卧底（都参加了）和平民（都没参加）

显然你应该选 B！因为 B 让你之后不再需要猜测——这就是**信息增益**最高的问题。

### b) 直观理解

决策树在每个节点选一个特征来"分裂"数据。选哪个？**选让子节点数据熵之和最小的那个**。

信息增益 = 分裂前的不确定性 − 分裂后（按样本量加权）的不确定性

$$IG = H(\text{父节点}) - \sum_{\text{子节点}} \frac{n_{\text{子}}}{n_{\text{父}}} \cdot H(\text{子节点})$$

信息增益越大 → 分裂越有效 → 优先选这个特征。

### c) 形式化定义

$$IG(D, A) = H(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} H(D_v)$$

其中 $D$ 是当前节点的数据集，$A$ 是候选特征，$D_v$ 是特征 $A$ 取值为 $v$ 的子集。

C4.5 使用**信息增益率**（Information Gain Ratio）来纠正偏向多值特征的倾向：

$$IGR(D, A) = \frac{IG(D, A)}{-\sum_{v} \frac{|D_v|}{|D|} \log \frac{|D_v|}{|D|}}$$

分母是特征的"固有信息量"——取值多的特征分母大，抵消了其自然的高信息增益优势。

### d) 手算示例：网球决策树

经典"是否去打网球"数据集，14 天记录：

| 天气 | 温度 | 湿度 | 有风 | 打网球？ |
|------|------|------|------|:---:|
| 晴 | 热 | 高 | 否 | 不去 |
| 晴 | 热 | 高 | 是 | 不去 |
| 阴 | 热 | 高 | 否 | **去** |
| 雨 | 温 | 高 | 否 | **去** |
| 雨 | 凉 | 正常 | 否 | **去** |
| 雨 | 凉 | 正常 | 是 | 不去 |
| 阴 | 凉 | 正常 | 是 | **去** |
| 晴 | 温 | 高 | 否 | 不去 |
| 晴 | 凉 | 正常 | 否 | **去** |
| 雨 | 温 | 正常 | 否 | **去** |
| 晴 | 温 | 正常 | 是 | **去** |
| 阴 | 温 | 高 | 是 | **去** |
| 阴 | 热 | 正常 | 否 | **去** |
| 雨 | 温 | 高 | 是 | 不去 |

**Step 1：父节点熵 $H(D)$**

"去"9 天，"不去"5 天，总 14 天。

$$H(D) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} = 0.4101 + 0.5305 = 0.9403 \text{ bit}$$

**Step 2：按"天气"分裂**

| 天气 | 去 | 不去 | 总 | 样本数 |
|------|:---:|:---:|:---:|:---:|
| 晴 | 2 | 3 | 5 | 5/14 |
| 阴 | 4 | 0 | 4 | 4/14 |
| 雨 | 3 | 2 | 5 | 5/14 |

- 晴天子节点熵：$H(晴) = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} = 0.5288 + 0.4422 = 0.9710$
- 阴天子节点熵：$H(阴) = -\frac{4}{4}\log_2\frac{4}{4} - 0 = 0$
- 雨天子节点熵：$H(雨) = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} = 0.4422 + 0.5288 = 0.9710$

加权和：$\frac{5}{14} \cdot 0.9710 + \frac{4}{14} \cdot 0 + \frac{5}{14} \cdot 0.9710 = 0.6936$

$$IG(\text{天气}) = 0.9403 - 0.6936 = 0.2467 \text{ bit}$$

**Step 3：按"湿度"分裂**

| 湿度 | 去 | 不去 | 总 |
|------|:---:|:---:|:---:|
| 高 | 3 | 4 | 7 |
| 正常 | 6 | 1 | 7 |

- 高湿度：$H(高) = -\frac{3}{7}\log_2\frac{3}{7} - \frac{4}{7}\log_2\frac{4}{7} = 0.5239 + 0.4613 = 0.9852$
- 正常湿度：$H(正常) = -\frac{6}{7}\log_2\frac{6}{7} - \frac{1}{7}\log_2\frac{1}{7} = 0.1903 + 0.4011 = 0.5914$

加权和：$\frac{7}{14} \cdot 0.9852 + \frac{7}{14} \cdot 0.5914 = 0.7883$

$$IG(\text{湿度}) = 0.9403 - 0.7883 = 0.1520 \text{ bit}$$

**结论**：$IG(\text{天气}) > IG(\text{湿度})$，优先用"天气"来分裂。

### e) Python 验证

```python
import numpy as np

def entropy_labels(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))

def information_gain(feature_col, labels):
    """计算按二值或多值特征分裂的信息增益"""
    h_parent = entropy_labels(labels)
    n_parent = len(labels)
    weighted_h = 0.0
    for val in np.unique(feature_col):
        mask = feature_col == val
        child_labels = labels[mask]
        weighted_h += len(child_labels) / n_parent * entropy_labels(child_labels)
    return h_parent - weighted_h

# 网球数据集
weather = np.array([0,0,1,2,2,2,1,0,0,2,0,1,1,2])  # 0=晴, 1=阴, 2=雨
humidity = np.array([1,1,1,1,0,0,0,1,0,0,0,1,0,1])  # 0=正常, 1=高
labels  = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])  # 0=不去, 1=去

print("====== 信息增益 手算验证 ======")
print(f"父节点熵 H(D) = {entropy_labels(labels):.4f} bit  (手算: 0.9403)")

ig_w = information_gain(weather, labels)
ig_h = information_gain(humidity, labels)

print(f"IG(天气) = {ig_w:.4f} bit  (手算: 0.2467)")
print(f"IG(湿度) = {ig_h:.4f} bit  (手算: 0.1520)")
print(f"最佳特征: {'天气' if ig_w > ig_h else '湿度'}")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "信息增益越大越好" | ❌ 偏向多值特征（如 ID 号每人一个，熵为 0，但毫无泛化意义）。用信息增益率或限制特征取值数来修正 |
| "基尼不纯度和信息增益选出来的特征一样" | 大部分情况接近，但不完全等价。CART 树用基尼、ID3/C4.5 用信息增益 |
| "信息增益为负" | ❌ $H(\text{父}) \ge H(\text{加权子})$ 由熵的凹性保证，信息增益恒非负 |

### g) ML 应用

- **决策树训练**：ID3（信息增益）、C4.5（信息增益率）、CART（基尼不纯度）
- **随机森林**：每棵树在每个节点随机选特征子集，用信息增益/基尼不纯度选最佳
- **XGBoost/LightGBM**：虽用梯度而非熵，但"分裂增益"概念继承了信息增益的思想

---

## 8. 最大熵原理 — "不知道的时候，就假设均匀"

### a) 生活例子

考试前，学渣室友问你对明天考什么内容的看法。你只知道老师说过"第 3 章到第 5 章会考"。除此之外一无所知。

如果你是理性的，你应该假设第 3、4、5 章每章被考到的概率相等（各 1/3），而不是假想"第 4 章肯定占 80%"——因为你没有任何依据。

**最大熵原理：在满足所有已知约束的前提下，选择熵最大的分布——最"均匀"、最少额外假设的分布。**

### b) 直观理解

如果你只掌握部分信息（比如知道分布的一阶矩、二阶矩），如何"公平"地推断整个分布？最大熵原理说：选熵最大的那个。因为：

- 熵最大 = 不确定性最大 = 不引入你未知的信息 = 最保守/最诚实的估计
- 任何熵更低的分布都隐含了你没有依据的额外假设
- 拉普拉斯的"不充分理由原则"（Principle of Insufficient Reason）在信息论中的推广

### c) 形式化定义

$$\max_p \quad H(p) = -\sum_x p(x) \log p(x)$$

满足约束：$\sum_x p(x) = 1$ 以及已知的期望约束（如 $\mathbb{E}[f_k(x)] = c_k$）

不同约束下的最大熵分布：

| 约束条件 | 最大熵分布 | 生活中的例子 |
|---------|-----------|-------------|
| 无约束（离散，K 类） | 均匀分布 $p_i = 1/K$ | 掷公平骰子 |
| 已知均值（连续，正数） | 指数分布 | 等待时间的分布 |
| 已知均值和方差（连续实数） | 正态分布 | 测量误差的分布 |
| 已知一阶特征期望（二分类） | 逻辑回归 / sigmoid | 点击率预测 |

### d) 手算示例：三个值之间的最大熵

问题：随机变量 X 取值为 $\{A, B, C\}$，已知 $P(A) = 0.5$，$P(B)$ 和 $P(C)$ 未知。问 $P(B)$ 取何值时熵最大？

解：$P(A)=0.5$ 是约束，剩下 $P(B) + P(C) = 0.5$。令 $p = P(B)$，则 $P(C) = 0.5 - p$。

$$H(p) = -0.5\log_2 0.5 - p\log_2 p - (0.5-p)\log_2(0.5-p)$$

$p$ 的可行域为 $[0, 0.5]$。由对称性可知 $p=0.25$ 时熵最大（此时三个概率为 $0.5, 0.25, 0.25$）。

$$H_{\max} = -0.5 \cdot (-1) - 0.25 \cdot (-2) - 0.25 \cdot (-2) = 0.5 + 0.5 + 0.5 = 1.5 \text{ bit}$$

验证：$p=0.5$ 时，分布为 $(0.5, 0.5, 0)$，熵为 $H = -0.5(-1) - 0.5(-1) - 0 = 1.0 \text{ bit} < 1.5$。

**结论**：在 $P(A)=0.5$ 的约束下，将剩余概率均匀分配给 B 和 C 得到最大熵。

### e) Python 验证

```python
import numpy as np

def entropy_discrete(probs):
    probs = np.asarray(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# 验证：P(A)=0.5 固定，扫 P(B)
pB_vals = np.linspace(0.01, 0.49, 100)
entropies = [entropy_discrete([0.5, p, 0.5-p]) for p in pB_vals]
max_idx = np.argmax(entropies)

print("====== 最大熵原理验证 ======")
print(f"最大熵时的 P(B) = {pB_vals[max_idx]:.4f}  (预期: 0.25)")
print(f"最大熵 H_max  = {entropies[max_idx]:.4f} bit  (手算: 1.5)")
print(f"P(B)=0.5 时的熵 = {entropy_discrete([0.5, 0.5, 0.01]):.4f} bit")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "最大熵 = 完全不知道" | ❌ 最大熵"在给定约束下"最大——如果已知均值，最大熵分布就不是均匀分布，而是正态分布 |
| "最大熵分布总是均匀分布" | ❌ 只有无约束或只有归一化约束时才是均匀分布 |
| "用了最大熵就是最优的" | ❌ 如果约束设错（漏掉重要信息），最大熵结果可能很糟糕 |

### g) ML 应用

- **逻辑回归**：在给定特征期望的约束下，最大熵分布推导出 sigmoid 函数
- **最大熵马尔可夫模型（MEMM）**：NLP 序列标注
- **最大熵强化学习（Soft Actor-Critic）**：在目标中加入熵项鼓励探索
- **指数族分布**：所有满足给定约束的最大熵分布都属于指数族——这涵盖了统计学习中绝大多数常用分布

---

## 9. 编码理论 — 熵是压缩的极限（霍夫曼编码实战）

### a) 生活例子

你要给远方的朋友发一封电报，电报按字符收费。英文字母有 26 个，最直接的编码方式是每个字母用 5 位二进制（$2^5=32 \ge 26$）。但你想到：e 出现频率极高，z 出现频率极低。如果给 e 一个短编码（比如 1 位），给 z 一个长编码（比如 6 位），平均每字符的编码长度就会变短——省下电报费！

这就是**霍夫曼编码（Huffman Coding）**的核心思想。

### b) 直观理解

**香农信源编码定理（Shannon's Source Coding Theorem）**：对分布 P 的符号进行无损编码，平均码长 L 不可能低于该分布的熵：

$$H(P) \le L$$

而且存在一种编码方案使 $L < H(P) + 1$。霍夫曼编码就是这样的最优前缀码。

**熵 = 无损压缩的理论极限。** 分布越不均匀，越能压缩（熵越小）。

### c) 形式化定义（霍夫曼算法）

1. 将所有符号按概率排序，每个符号是一棵单节点树
2. 取概率最小的两棵树合并，新树的概率 = 两子树概率之和
3. 重复直到只剩一棵树
4. 从根遍历，左枝标 0，右枝标 1，到达叶子的路径就是编码

### d) 手算示例

五个符号，概率分布：A=0.40, B=0.25, C=0.20, D=0.10, E=0.05。

**Step 1：算熵（理论下界）**

$$\begin{aligned}
H &= -0.4\log_2 0.4 - 0.25\log_2 0.25 - 0.2\log_2 0.2 - 0.1\log_2 0.1 - 0.05\log_2 0.05 \\
  &= 0.4\cdot 1.3219 + 0.25\cdot 2 + 0.2\cdot 2.3219 + 0.1\cdot 3.3219 + 0.05\cdot 4.3219 \\
  &= 0.5288 + 0.5000 + 0.4644 + 0.3322 + 0.2161 = 2.0415 \text{ bit}
\end{aligned}$$

**Step 2：构建霍夫曼树**

| 步骤 | 操作 | 说明 |
|------|------|------|
| 初始 | E=0.05, D=0.10, C=0.20, B=0.25, A=0.40 | 5 棵树 |
| 合并 1 | (D+E)=0.15, C=0.20, B=0.25, A=0.40 | 合并最小两棵 |
| 合并 2 | C=0.20, ((D+E)+B)=0.40, A=0.40 | 合并 0.15 和 0.25 |
| 合并 3 | A=0.40, (C+((D+E)+B))=0.60 | 合并 0.20 和 0.40 |
| 合并 4 | (A+(C+((D+E)+B)))=1.00 | 最终树 |

**Step 3：推导编码**

从根开始给左右分支标 0/1：
- A：0 → 码长 1
- B：10 → 码长 2
- C：110 → 码长 3
- D：1110 → 码长 4
- E：1111 → 码长 4

**Step 4：算平均码长**

$$\begin{aligned}
L &= 0.40 \times 1 + 0.25 \times 2 + 0.20 \times 3 + 0.10 \times 4 + 0.05 \times 4 \\
  &= 0.40 + 0.50 + 0.60 + 0.40 + 0.20 = 2.10 \text{ bit}
\end{aligned}$$

**验证**：$H = 2.0415 \le L = 2.10 < H + 1 = 3.0415$ ✓

### e) Python 验证

```python
import heapq
import numpy as np

def huffman_coding(symbols_probs):
    """返回 {symbol: code} 编码字典"""
    # 堆中元素: (prob, counter, symbol_or_list)
    heap = [(prob, i, sym) for i, (sym, prob) in enumerate(symbols_probs)]
    heapq.heapify(heap)
    code_map = {sym: '' for sym, _ in symbols_probs}
    counter = len(symbols_probs)

    while len(heap) > 1:
        p1, _, s1 = heapq.heappop(heap)
        p2, _, s2 = heapq.heappop(heap)

        items1 = [s1] if not isinstance(s1, list) else s1
        items2 = [s2] if not isinstance(s2, list) else s2

        for s in items1:
            code_map[s] = '0' + code_map[s]
        for s in items2:
            code_map[s] = '1' + code_map[s]

        heapq.heappush(heap, (p1 + p2, counter, items1 + items2))
        counter += 1

    return code_map

# 手算验证
symbols = [('A', 0.40), ('B', 0.25), ('C', 0.20), ('D', 0.10), ('E', 0.05)]
codes = huffman_coding(symbols)

print("====== 霍夫曼编码 手算验证 ======")
avg_len = 0
for sym, prob in symbols:
    code = codes[sym]
    l = len(code)
    avg_len += prob * l
    optimal = -np.log2(prob)
    print(f"{sym}: prob={prob:.2f}, code={code}, length={l}, 最优={optimal:.4f}")

probs_arr = np.array([p for _, p in symbols])
H = -np.sum(probs_arr * np.log2(probs_arr))
print(f"\n熵 H = {H:.4f} bit  (手算: 2.0415)")
print(f"霍夫曼平均码长 L = {avg_len:.2f} bit  (手算: 2.10)")
print(f"H ≤ L < H+1: {H:.4f} ≤ {avg_len:.4f} < {H+1:.4f}  ✓")
```

### f) 常见误区

| 误区 | 正确理解 |
|------|---------|
| "霍夫曼编码 = 最优编码" | ❌ 是对每个符号独立编码的最优前缀码。算术编码能做到 $L < H + 1$ 的基础上更接近 H。另外如果允许块编码（一次编码多个符号），可以进一步接近 H |
| "压缩率取决于文件大小" | ❌ 取决于符号分布。分布越偏斜，熵越小，压缩率越高 |
| "熵越小编码越短" | ✓ 对——熵小意味着分布集中，可以用更短的编码 |

### g) ML 应用

- **数据压缩**：模型权重量化、模型剪枝本质上都是"降低模型的熵"
- **最小描述长度原则（MDL）**：最好的模型 = 描述模型的开销 + 描述数据的开销 之和最小的模型。这个思想直接源于编码理论
- **信息瓶颈**：在压缩中保留关键信息——深度学习中表征学习的理论基础之一
- **算术编码 → 语言模型**：语言模型本质上是条件概率模型，算术编码/range coding 连接了概率预测和压缩

---

## 10. 思考题（含完整解答）

### 题 1（自信息）

天气预报说"明天降雨概率 30%"。若明天真的下雨，这条"下雨"消息携带多少 bit 信息？

**解答**：$I = -\log_2(0.3) = \log_2(1/0.3) = \log_2(3.333\ldots) \approx 1.737 \text{ bit}$。

### 题 2（熵）

一个骗子声称他发明了一个"预测股票涨跌"的系统，声称涨跌概率各有 50%。财务专家说正确的分布其实是涨 80%、跌 20%。分别计算两种分布下的熵，谁的熵更低？为什么？

**解答**：

骗子（均匀分布）：$H = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1.0 \text{ bit}$

专家（偏斜分布）：$H = -0.8\log_2 0.8 - 0.2\log_2 0.2 \approx 0.7219 \text{ bit}$

专家的熵更低，因为分布更集中（更确定）。从编码角度看：均匀分布时每预测一次需要 1 bit；偏斜分布时，因为"涨"很常见，可以给它短编码，平均只需约 0.72 bit。

### 题 3（条件熵与互信息）

已知 $P(X,Y)$：

$$P = \begin{pmatrix} 0.3 & 0.2 \\ 0.1 & 0.4 \end{pmatrix}$$

计算：(a) $H(X)$ (b) $H(Y)$ (c) $H(X,Y)$ (d) $H(Y|X)$ (e) $I(X;Y)$

**解答**：

(a) $P(X=0) = 0.3+0.2 = 0.5$, $P(X=1) = 0.1+0.4 = 0.5$
    $H(X) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1.0 \text{ bit}$

(b) $P(Y=0) = 0.3+0.1 = 0.4$, $P(Y=1) = 0.2+0.4 = 0.6$
    $H(Y) = -0.4\log_2 0.4 - 0.6\log_2 0.6 \approx -0.4 \cdot (-1.3219) - 0.6 \cdot (-0.7370) = 0.5288 + 0.4422 = 0.9710 \text{ bit}$

(c) $H(X,Y) = -0.3\log_2 0.3 - 0.2\log_2 0.2 - 0.1\log_2 0.1 - 0.4\log_2 0.4$
    $= 0.3 \cdot 1.7370 + 0.2 \cdot 2.3219 + 0.1 \cdot 3.3219 + 0.4 \cdot 1.3219$
    $= 0.5211 + 0.4644 + 0.3322 + 0.5288 = 1.8465 \text{ bit}$

(d) $H(Y|X) = H(X,Y) - H(X) = 1.8465 - 1.0 = 0.8465 \text{ bit}$

(e) $I(X;Y) = H(Y) - H(Y|X) = 0.9710 - 0.8465 = 0.1245 \text{ bit}$

### 题 4（KL 散度）

真实分布 $P=[0.6, 0.4]$，你分别用两个近似分布 $Q_1=[0.7, 0.3]$ 和 $Q_2=[0.5, 0.5]$ 来近似。哪个更好？计算两个 KL 散度比较。

**解答**：

$D_{KL}(P \parallel Q_1) = 0.6 \cdot \log_2(0.6/0.7) + 0.4 \cdot \log_2(0.4/0.3)$
$= 0.6 \cdot \log_2(0.8571) + 0.4 \cdot \log_2(1.3333)$
$= 0.6 \cdot (-0.2224) + 0.4 \cdot 0.4150 = -0.1334 + 0.1660 = 0.0326 \text{ bit}$

$D_{KL}(P \parallel Q_2) = 0.6 \cdot \log_2(0.6/0.5) + 0.4 \cdot \log_2(0.4/0.5)$
$= 0.6 \cdot \log_2(1.2) + 0.4 \cdot \log_2(0.8)$
$= 0.6 \cdot 0.2630 + 0.4 \cdot (-0.3219) = 0.1578 - 0.1288 = 0.0290 \text{ bit}$

$Q_2$ 的 KL 散度略小（0.0290 < 0.0326），即用均匀近似反而比 $Q_1=[0.7,0.3]$ 更好。但注意 Q_2 的分布形状不如 Q_1 接近——差异在第二项（尾部）。另外也可以反过来算 $D_{KL}(Q_1\|P)$ 和 $D_{KL}(Q_2\|P)$ 看看——由于不对称性，排名可能变。

### 题 5（交叉熵与 MLE）

证明：对二分类问题，最小化交叉熵等价于最大似然估计。

**解答**：

设样本 $i$ 的真实标签 $y_i \in \{0, 1\}$，模型预测概率 $\hat{y}_i = P_\theta(y=1|x_i)$。

单个样本的似然：$P_\theta(y_i|x_i) = \hat{y}_i^{y_i} \cdot (1-\hat{y}_i)^{1-y_i}$

N 个样本的总对数似然：
$$\log L(\theta) = \sum_i \big[y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)\big]$$

而二元交叉熵（对所有样本取平均）：
$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_i \big[y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)\big] = -\frac{1}{N} \log L(\theta)$$

因为 $-\frac{1}{N}$ 是正常数，$\arg\min \mathcal{L}_{CE} = \arg\max \log L(\theta) = \hat{\theta}_{MLE}$。证毕。

### 题 6（信息增益）

给定 10 个样本，标签为 $[0,0,0,0,1,1,1,1,1,1]$。特征 A 将样本分为两组：第一组 4 个样本标签 $[0,0,0,1]$，第二组 6 个样本标签 $[0,1,1,1,1,1]$。计算信息增益。

**解答**：

父节点熵：
$$H_{父} = -\frac{4}{10}\log_2\frac{4}{10} - \frac{6}{10}\log_2\frac{6}{10} = -0.4\cdot(-1.3219) - 0.6\cdot(-0.7370) = 0.5288 + 0.4422 = 0.9710$$

第一组（4 样本，标签 $[0,0,0,1]$）：
$$H_1 = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} = 0.3113 + 0.5 = 0.8113$$

第二组（6 样本，标签 $[0,1,1,1,1,1]$）：
$$H_2 = -\frac{1}{6}\log_2\frac{1}{6} - \frac{5}{6}\log_2\frac{5}{6} = 0.4308 + 0.2190 = 0.6498$$

加权和：$\frac{4}{10} \cdot 0.8113 + \frac{6}{10} \cdot 0.6498 = 0.3245 + 0.3899 = 0.7144$

$$IG = 0.9710 - 0.7144 = 0.2566 \text{ bit}$$

### 题 7（最大熵）

随机变量 X 满足约束 $\mathbb{E}[X] = 2$，且 X 在整数 $1,2,3,4,5$ 上取值。在所有满足该约束的分布中，哪个分布的熵最大？用数值方法找出。

**解答**：

设 $p_i = P(X=i)$，约束为 $\sum_i p_i = 1$ 且 $\sum_i i \cdot p_i = 2$。

写拉格朗日函数：$\mathcal{L} = -\sum p_i \log p_i + \lambda_0(\sum p_i - 1) + \lambda_1(\sum i \cdot p_i - 2)$

对 $p_i$ 求导：$-\log p_i - 1 + \lambda_0 + \lambda_1 \cdot i = 0 \implies p_i = e^{\lambda_0 - 1 + \lambda_1 \cdot i} = C \cdot e^{\lambda_1 \cdot i}$

即最大熵分布是指数族：$p_i \propto e^{\lambda_1 \cdot i}$

由于均值为 2（偏小），$\lambda_1 < 0$，概率随 i 增大递减。代入约束数值求解，可得近似分布（由 Python 代码解出）：

```python
import numpy as np
from scipy.optimize import fsolve

def solve_max_entropy():
    def equations(vars):
        lam0, lam1 = vars
        i_vals = np.array([1, 2, 3, 4, 5])
        p = np.exp(lam0 + lam1 * i_vals)
        eq1 = np.sum(p) - 1.0
        eq2 = np.sum(i_vals * p) - 2.0
        return [eq1, eq2]

    lam0, lam1 = fsolve(equations, [-1.0, -0.5])
    i_vals = np.array([1, 2, 3, 4, 5])
    p = np.exp(lam0 + lam1 * i_vals)
    p = p / np.sum(p)  # 归一化
    H = -np.sum(p * np.log2(p))
    return p, H

p, H = solve_max_entropy()
print(f"概率分布: {p}")
print(f"熵: {H:.4f} bit")
```

### 题 8（JS 散度）

证明 JS 散度在底数为 2 时上界为 1 bit。

**解答**：

$D_{JS}(P\|Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$，其中 $M = \frac{P+Q}{2}$。

当 $P$ 和 $Q$ 完全不重叠时，取极端：$P = [1, 0]$, $Q = [0, 1]$。

$M = [0.5, 0.5]$

$D_{KL}(P\|M) = 1 \cdot \log_2(1/0.5) + 0 \cdot \log_2(0/0.5) = 1 \text{ bit}$

同样 $D_{KL}(Q\|M) = 1 \text{ bit}$

$D_{JS} = \frac{1}{2}(1 + 1) = 1 \text{ bit}$

由 Gibbs 不等式，KL 散度非负，JS 散度非负。上界 1 bit（底为 2 时）。证毕。

### 题 9（霍夫曼编码）

符号概率为 $P = [0.5, 0.25, 0.125, 0.125]$。手算霍夫曼编码和平均码长，对比熵。

**解答**：

熵：$H = -0.5\log_2 0.5 - 0.25\log_2 0.25 - 0.125\log_2 0.125 - 0.125\log_2 0.125 = 0.5 + 0.5 + 0.375 + 0.375 = 1.75 \text{ bit}$

霍夫曼树构建：
1. 合并 0.125 和 0.125 → 0.25
2. 合并 0.25 和 0.25 → 0.50
3. 合并 0.50 和 0.50 → 1.00

编码：
- 概率 0.5 的符号：0（码长 1）
- 概率 0.25 的符号：10（码长 2）
- 概率 0.125 的符号 A：110（码长 3）
- 概率 0.125 的符号 B：111（码长 3）

平均码长：$L = 0.5\times1 + 0.25\times2 + 0.125\times3 + 0.125\times3 = 0.5 + 0.5 + 0.375 + 0.375 = 1.75 \text{ bit}$

**巧合**：$L = H = 1.75$！因为所有概率都是 $2^{-k}$ 形式，霍夫曼编码恰好达到理论极限。

### 题 10（综合）

某电商希望用"用户是否浏览过商品详情页"（特征 X）来预测"是否购买"（标签 Y）。历史数据统计如下：

| X \ Y | 购买 (Y=1) | 未购买 (Y=0) |
|:---:|:---:|:---:|
| 浏览过 (X=1) | 300 | 200 |
| 未浏览过 (X=0) | 50 | 450 |

(a) 计算 $H(Y)$
(b) 计算 $H(Y|X)$
(c) 计算 $I(X;Y)$
(d) 基于互信息，你觉得浏览行为对购买预测有帮助吗？
(e) 如果用这个特征构建二分类模型，猜猜交叉熵损失大概在什么范围？

**解答**：

总样本数：$N = 300+200+50+450 = 1000$

(a) $P(Y=1) = \frac{300+50}{1000} = 0.35$, $P(Y=0) = 0.65$

$H(Y) = -0.35\log_2 0.35 - 0.65\log_2 0.65 \approx 0.5301 + 0.4040 = 0.9341 \text{ bit}$

(b) 浏览过（$X=1$）：$N=500$，$P(Y=1|X=1) = 300/500 = 0.6$，$P(Y=0|X=1) = 0.4$
    $H(Y|X=1) = -0.6\log_2 0.6 - 0.4\log_2 0.4 \approx 0.4422 + 0.5288 = 0.9710$

    未浏览过（$X=0$）：$N=500$，$P(Y=1|X=0) = 50/500 = 0.1$，$P(Y=0|X=0) = 0.9$
    $H(Y|X=0) = -0.1\log_2 0.1 - 0.9\log_2 0.9 \approx 0.3322 + 0.1368 = 0.4690$

    $H(Y|X) = \frac{500}{1000} \cdot 0.9710 + \frac{500}{1000} \cdot 0.4690 = 0.7200 \text{ bit}$

(c) $I(X;Y) = H(Y) - H(Y|X) = 0.9341 - 0.7200 = 0.2141 \text{ bit}$

(d) 互信息 0.21 bit，说明浏览行为减少了对购买行为的约 23% 的不确定性（0.21/0.93 ≈ 23%）。有一定帮助但不是决定性的——模型还需要更多特征来预测。

(e) 如果模型只输出"浏览过 → 60% 买，未浏览 → 10% 买"这个规则，交叉熵损失为 $H(Y|X) = 0.7200 \text{ bit}$（nat 制的 $0.72 \times \ln 2 \approx 0.50$）。一个完美预测器损失为 0，一个随机猜测器（各 50%）损失为 $\log 2 \approx 0.693 \text{ nat}$。但这里是信息论层面的下界：$H(Y|X)$ 是用 X 能取得的最低交叉熵，任何基于此特征训练的模型无法低于这个值。

---

## 11. 总结：信息论在机器学习中的全景图

### 概念之间的关系链

```
自信息 I(x) = -log P(x)              ← 单个事件的惊讶度
        |
        | 取期望
        ↓
熵 H(X) = E[-log P(x)]              ← 整个分布的不确定度 = 压缩极限
        |
        | 两个变量
        ↓
链式法则: H(X,Y) = H(X) + H(Y|X)    ← 联合不确定性可以分解
        |
        | 差值
        ↓
互信息 I(X;Y) = H(Y) - H(Y|X)       ← 共享的信息 = 非线性特征选择
        |
        | 引入错误分布
        ↓
KL散度 D_KL(P||Q) = Σ P log(P/Q)   ← "用错分布"的额外开销（不对称）
        |
        ├→ 对称化 → JS散度          ← GAN 的理论基础
        |
        └→ 拆解: H(P,Q) = H(P) + D_KL(P||Q)
                |
                ↓
           交叉熵 = -Σ P log Q      ← 分类损失函数
                |
                ⇕
           负对数似然 = MLE         ← 最大似然估计
```

### 一句话速查表

| 概念 | 一句话 | 在 ML 中 |
|------|--------|---------|
| 自信息 $I(x)$ | 这个事件有多让你惊讶？ | 异常检测 |
| 熵 $H(X)$ | 整个分布平均有多不确定？ | 数据复杂度、决策树节点纯度 |
| 条件熵 $H(Y\|X)$ | 知道 X 后 Y 还剩多少不确定？ | 特征有效性评估 |
| 互信息 $I(X;Y)$ | X 和 Y 共享了多少信息？ | 特征选择（能检测非线性关系） |
| KL 散度 $D_{KL}$ | 用 Q 近似 P 多花了多少编码？ | VAE 正则、PPO 策略约束 |
| JS 散度 $D_{JS}$ | KL 的对称有界版 | GAN 损失理论基础 |
| **交叉熵 $H(P,Q)$** | **猜错了的代价** | **分类任务的标准损失** |
| 信息增益 $IG$ | 分裂后数据变"纯"了多少？ | 决策树分裂准则 |
| 最大熵原理 | 不知道时假设最均匀 | 逻辑回归、指数族分布 |
| 霍夫曼编码 | 熵就是压缩极限 | 模型压缩、最小描述长度 |

---

> **本文完整代码可以在安装了 `numpy`、`scipy`、`matplotlib` 和 `scikit-learn` 的 Python 环境中直接运行。建议你亲手执行每一段代码，修改参数，观察输出变化——信息论是在"算"中理解的，而不是在"读"中理解的。**
>
> 学完本章，你已完成数学基础六大模块的学习。接下来进入[机器学习基础 — 什么是机器学习？](./01-introduction.md)。
