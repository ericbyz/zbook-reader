## 决策树与随机森林：像人一样做决策

> **核心问题**：如何自动从数据中学出一个"问问题-做判断"的流程图？

---

### 0. 本章导览

决策树是机器学习中**最直观**的模型——它的思维方式和你日常做决策时一模一样。随机森林则把成百上千棵决策树聚在一起投票，用"群众的智慧"碾压单棵树的局限。

你可能会问：深度学习时代还需要学决策树吗？答案是：XGBoost、LightGBM 这些 Kaggle 冠军模型的根基正是决策树；金融风控、医疗诊断等需要**解释性**的场景中，决策树仍是首选。理解本章，你就掌握了 ML 中最重要的一类模型的核心逻辑。

读完本章你将能够：

1. 手算熵、基尼不纯度、信息增益——在 8 个样本的天气数据集上逐层建树，纸笔完成
2. 从零实现 CART 决策树分类器，理解递归二分分裂的每一行代码
3. 解释为什么"无限制树必然过拟合"、两种剪枝策略各自适用什么场景
4. 说出随机森林的两层随机性（Bootstrap + 特征子集），以及为什么它不易过拟合
5. 在 3 棵树的简化森林上手算投票过程，理解集成学习的核心思想
6. 独立完成一个分类项目：训练 → 调参 → 特征重要性分析 → 模型对比
7. 说出 5 个常见误区及正确做法

> 本章目标 1300+ 行，建议分 3 次读完。**手算部分请务必拿纸笔跟着算——这是真正"拥有"这个算法的最短路径。**

前置章节：[SVM](./08-svm.md)、[信息论](./math-information-theory.md)（熵的概念已覆盖但本章会再讲一遍）
下一章：[朴素贝叶斯](./10-naive-bayes.md)

---

### 1. 生活例子：医生是怎么诊断感冒的？

你感冒了去看医生。医生不会一上来就让你做全套检查——她问几个关键问题，根据你的回答一步步缩小可能病因：

```
"发烧吗？"
  ├── 不发烧 → "喉咙痛吗？"
  │   ├── 不痛 → 普通感冒
  │   └── 痛   → 咽炎
  └── 发烧   → "发烧超过 39 度吗？"
      ├── 不超过 → 病毒性感冒
      └── 超过   → "咳嗽带痰吗？"
          ├── 不带 → 流感
          └── 带   → 细菌性肺炎
```

这就是一棵**决策树**。医生的诊断思路不是一条固定公式——她是根据你上一个回答，灵活选择下一个问题。每个问题都是当前阶段"最能区分病种"的那个。而且整个过程你完全看得懂——"为什么诊断我是普通感冒？因为不发烧、不喉咙痛"——这就是可解释性。

机器学习中的决策树做的事一模一样：从数据中自动找出"先问什么、再问什么、最后给什么结论"的最佳流程。

| 场景 | 树在做什么 |
|------|-----------|
| 银行审批贷款 | 收入 > 阈值？→ 有房产？→ 信用分 > 阈值？→ 批准/拒绝 |
| 邮件垃圾过滤 | 含"免费"？→ 感叹号 > 3？→ 发件人在通讯录？→ 正常/垃圾 |
| 电商推荐 | 浏览过电子？→ 价格 > 1000？→ 最近搜过相机？→ 推荐型号 |
| 游戏 AI | 血量 < 20%？→ 敌人在射程内？→ 有掩体？→ 撤退/攻击 |

每一次判断，都只问一个特征、一个简单条件——这就是决策树的核心范式。

---

### 2. 直观理解：决策树就是一个"20 个问题"游戏

**直觉理解：** 你玩过"20 个问题"吗？心里想一个人，对方通过 20 个**是/否**问题来猜——"他是活着的人吗？""他是男性吗？""他是演员吗？"——每个问题尽量排除一半可能性，最终锁定答案。好的提问者会先问"是男性吗"（排除了约一半的人），而不是"他是不是周杰伦"（大概率浪费一个问题）。

决策树构造的过程就是学做**一个好的提问者**：

```
把所有数据放在根节点
↓
遍历所有"可能的问题"（特征 + 阈值），找到"最能分清答案"的那个
↓
用这个问题把数据分成左右两份
↓
每份分别重复上述过程
↓
直到：这份数据已经足够"纯"（里面答案一致）、或者不能再分了
```

以经典的天气数据集为例——根据天气状况决定是否去打球：

```
根节点: 8 条数据（4 打球, 4 不打）
    ↓ "天气如何？" ← 这是找出来的"最好问题"
    ├── 晴天 (sunny):  3 条（全是"不打"） → 纯的！叶子节点：不打 ✗
    ├── 阴天 (overcast): 2 条（全是"打球"） → 纯的！叶子节点：打球 ✓
    └── 雨天 (rain):   3 条（2 打球, 1 不打） → 还不纯，继续问
        ↓ "有风吗？"
        ├── 无风 (False): 2 条（全是"打球"） → 叶子节点：打球 ✓
        └── 有风 (True):  1 条（"不打"）   → 叶子节点：不打 ✗
```

这棵树的画法：

```
              [Outlook?]
          /      |       \
      Sunny  Overcast    Rain
        |        |         |
       NO       YES    [Windy?]
                       /     \
                    False   True
                      |       |
                     YES     NO
```

三个关键直觉：

1. **贪心**：每一步只看眼前——选"当前最能分清"的问题，不管长远。虽然理论上不能保证全局最优，但实践中效果很好。就像下棋——你不可能算到 20 步之后，但每一步选当前最佳走法通常也不会太差。

2. **纯度的追求**：好的分裂让左右子节点尽量"纯"——左边主要是 A 类，右边主要是 B 类。如果一次分裂后两边都还很混，说明这个问题没问到点子上。

3. **树天然会过拟合**：如果不加限制，树会一直追问下去，直到每个叶子节点只剩一个样本——训练准确率 100%，但噪声也被当成了规律。就像医生问了 50 个无关问题后确诊——样本外的病人来了，这套问法可能完全不适用。

---

### 3. 形式化定义

#### 3.1 决策树结构

一棵决策树由三种节点组成：

| 节点类型 | 含义 | 做什么 |
|---------|------|--------|
| **根节点 (Root)** | 包含全部训练数据 | 第一个分裂点 |
| **内部节点 (Internal)** | 对一个特征做判断 | $x_j \le s$？（是 → 左；否 → 右） |
| **叶节点 (Leaf)** | 不再分裂 | 给出最终预测（分类：众数；回归：均值） |

数学上，决策树把特征空间 $\mathbb{R}^p$ 划分为 $J$ 个互不相交的矩形区域 $R_1, R_2, \dots, R_J$，每个区域给一个统一预测：

$$
\hat{f}(x) = \sum_{j=1}^{J} c_j \cdot \mathbf{1}\{x \in R_j\}
$$

其中 $c_j$ 是区域 $R_j$ 内样本标签的众数（分类）或均值（回归）。

#### 3.2 熵（Entropy）——衡量"混乱程度"

熵来自信息论，衡量一个节点中标签的**不确定度**：

$$
H(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

其中 $p_k$ 是节点 $S$ 中第 $k$ 类的比例。如果 $p_k = 0$，定义 $0 \cdot \log_2(0) = 0$（极限约定）。

| 节点状况 | $p$ 分布 | 熵 $H$ | 含义 |
|---------|---------|--------|------|
| 完全纯 | [1.0, 0.0] | 0.000 | 没有任何不确定性——所有样本同一类 |
| 不太纯 | [0.75, 0.25] | 0.811 | 有点混，但倾向明显 |
| 一半一半 | [0.5, 0.5] | 1.000 | 最混乱——完全不知道是哪个类 |
| 均匀三分类 | [⅓, ⅓, ⅓] | 1.585 | 三类均匀，熵更高 |

**为什么用 $\log_2$？** 信息论中，$-\log_2(p)$ 是编码一个概率为 $p$ 的事件所需的最短比特数。熵就是这个期望编码长度的最小值——"一堆样本的标签，最少需要多少个比特来描述"。

#### 3.3 信息增益（Information Gain）——衡量"这个问题问得多好"

分裂前的熵，减去分裂后每个子节点熵的**加权平均**：

$$
IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \cdot H(S_v)
$$

其中 $A$ 是分裂所用的特征，$S_v$ 是特征 $A$ 取值为 $v$ 的子集。权重是子集样本数占比——人多的子节点对平均熵的影响更大。

**IG 越大 = 分裂效果越好。** IG = 0 意味着分裂完全没有改善纯度（子节点平均下来和父节点一样乱）；IG = 父节点熵 意味着分裂后所有子节点完全纯（一次搞定）。

#### 3.4 基尼不纯度（Gini Impurity）——另一种"纯度"度量

$$
G(S) = 1 - \sum_{k=1}^{K} p_k^2
$$

直觉：随机抽两个样本（有放回），它们标签**不一致**的概率。$p_k^2$ 是两次都抽到第 $k$ 类的概率，$1 - \sum p_k^2$ 就是"抽到不同类"的概率。

| 节点状况 | $p$ 分布 | Gini | 含义 |
|---------|---------|------|------|
| 完全纯 | [1.0, 0.0] | 0.000 | 两个样本一定同类 |
| 一半一半 | [0.5, 0.5] | 0.500 | 有 50% 概率抽到两个不同的类 |

**Gini vs 熵：** 两种度量在实践中几乎总是一致——同一数据上，它们会给同一个特征打最高分。Gini 计算更快（没有对数运算），是 sklearn 的默认选择。熵对不纯的节点惩罚更重（因为 $-\log_2(p)$ 在 $p$ 很小时远大于 $p(1-p)$ 的变化）。

Gini 增益（分裂质量）：
$$
\Delta G = G(S) - \sum_{v} \frac{|S_v|}{|S|} \cdot G(S_v)
$$

#### 3.5 CART 算法：递归二分分裂

CART（Classification and Regression Tree）是最主流的决策树算法，sklearn 的实现基于此。

**算法流程：**

1. 将当前节点所有数据记为 $(X, y)$
2. 对每个特征 $j = 1, \dots, p$，对每个可能的阈值 $s$：
   - 将数据分为 $R_L = \{x \mid x_j \le s\}$ 和 $R_R = \{x \mid x_j > s\}$
   - 计算分裂质量（Gini 增益或信息增益）
3. 选择增益最大的 $(j, s)$ 作为当前节点的分裂规则
4. 在左子节点和右子节点上递归执行步骤 1-3
5. 满足停止条件时，创建叶节点并返回

**停止条件（任一满足即停）：**
- 到达 `max_depth`（最大深度）
- 节点样本数 < `min_samples_split`
- 分裂后任一侧样本数 < `min_samples_leaf`
- 增益 < `min_impurity_decrease`
- 节点中所有样本标签相同（已经纯了）

**CART 的关键约束：二分分裂。** 即使特征是离散的多值（如 outlook ∈ {sunny, overcast, rain}），CART 也只会分裂成两个子节点（如 "outlook = overcast?" → 是/否），而不是三个。这意味着 CART 天然是**二叉树**。

#### 3.6 剪枝（Pruning）——防过拟合

**预剪枝（Pre-pruning）：** 在树长成之前就用超参数刹车。

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `max_depth` | 树的最大深度 | 3~10 |
| `min_samples_split` | 分裂所需最小样本数 | 5~20 |
| `min_samples_leaf` | 叶节点最小样本数 | 1~10 |
| `min_impurity_decrease` | 分裂必须减少的最少不纯度 | 0.0~0.01 |

**后剪枝 / 代价复杂度剪枝（Cost-Complexity Pruning）：** 先长满整棵树，然后从底部往上剪。目标函数：

$$
R_\alpha(T) = R(T) + \alpha \cdot |T|
$$

其中 $R(T)$ 是分类错误率，$|T|$ 是叶节点数（复杂度），$\alpha$ 控制惩罚强度。每个内部节点问自己：把我变成叶节点后，$R_\alpha$ 是变大了还是变小了？变大了就不剪，变小了就剪。$\alpha$ 越大 → 树越简单。

#### 3.7 随机森林（Random Forest）

**核心思想：** 训练 $B$ 棵不同的决策树，最终预测 = 所有树的多数投票（分类）或平均（回归）。

**两层随机性（缺一不可）：**

1. **数据随机（Bootstrap）**：每棵树从原始训练集中有放回抽样 $n$ 个样本。平均约 63% 的样本被选中，37% 成为"袋外样本（OOB）"——可以用来做免费的验证。
2. **特征随机**：在每个分裂节点，只从 $\sqrt{p}$（分类）或 $p/3$（回归）个随机选取的特征中寻找最优分裂。

如果只用 Bootstrap（= Bagging），树之间还是太像——因为它们看到的数据虽然不一样，但分裂决策还是在同样的特征中选。加上第二层特征随机后，每棵树的"世界观"彻底不同了——它们的错误倾向于相互抵消。

**为什么不易过拟合：** 每棵树都是低偏差、高方差的强模型，但 $B$ 棵树的方差是它们之间方差除以 $B$（加上一个与树间相关性有关的项）。树越多，方差越低→泛化越强。增加树的数量 **不会导致过拟合**——这是随机森林对比神经网络最优雅的性质之一。

---

### 4. 手算示例：在 8 个样本上完整建一棵决策树

本节用 8 个样本的天气数据集，**全程纸笔计算**熵、基尼不纯度、信息增益，逐层建立决策树。请拿出纸笔跟着算——这是整个章节最重要的环节。

#### 4.1 数据集

| ID | Outlook | Temp | Humidity | Windy | Play |
|----|---------|------|----------|-------|------|
| D1 | Sunny   | Hot  | High     | Weak  | No   |
| D2 | Sunny   | Hot  | High     | Strong| No   |
| D3 | Overcast| Hot  | High     | Weak  | Yes  |
| D4 | Rain    | Mild | High     | Weak  | Yes  |
| D5 | Rain    | Cool | Normal   | Weak  | Yes  |
| D6 | Rain    | Cool | Normal   | Strong| No   |
| D7 | Overcast| Cool | Normal   | Strong| Yes  |
| D8 | Sunny   | Mild | High     | Weak  | No   |

总样本 8 个：Play=Yes 有 4 个（D3, D4, D5, D7），Play=No 有 4 个（D1, D2, D6, D8）。

**目标：** 用这 8 个样本训练一棵决策树，预测"是否去打球"。

#### 4.2 准备工作：父节点的不纯度

分类标签分布：$p_{\text{Yes}} = 4/8 = 0.5$，$p_{\text{No}} = 4/8 = 0.5$

**熵：**

$$
H(\text{parent}) = -0.5 \cdot \log_2(0.5) - 0.5 \cdot \log_2(0.5)
$$

$\log_2(0.5) = \log_2(1/2) = -1$

$$
H(\text{parent}) = -0.5 \cdot (-1) - 0.5 \cdot (-1) = 0.5 + 0.5 = 1.000
$$

**基尼不纯度：**

$$
G(\text{parent}) = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 0.500
$$

父节点的熵是最大值（二分类时）- 样本完全平衡，信息量为 1 比特。

#### 4.3 第一层：遍历四个特征，找到最优分裂

**▶ 分裂 1：Outlook（天气）**

取值有 Sunny、Overcast、Rain 三种。注意 CART 做的是**二分**分裂，但离散特征可以通过"取值集合划分"来实现。这里按取值自然分组：

- **Sunny 组**：D1(No), D2(No), D8(No) → 3 个样本，全部 No
- **Overcast 组**：D3(Yes), D7(Yes) → 2 个样本，全部 Yes
- **Rain 组**：D4(Yes), D5(Yes), D6(No) → 3 个样本，2 Yes + 1 No

但 CART 是二叉树！离散特征多值时，实际会用 one-hot 编码或处理为"若干二值问题"。为简化手算，我们先按三个分支计算（等价于 ID3/C4.5 的做法），后续讨论中会说明 CART 如何处理多值离散属性。

按三路分支计算每个子集的熵：

| 子集 | 样本数 | 分布 | 熵 |
|------|-------|------|-----|
| Sunny | 3 | [0 Yes, 3 No] | $H = -0 \cdot \log_2(0) - 1 \cdot \log_2(1) = 0$ |
| Overcast | 2 | [2 Yes, 0 No] | $H = -1 \cdot \log_2(1) - 0 = 0$ |
| Rain | 3 | [2 Yes, 1 No] | 见下方计算 |

Rain 子集：$p_{\text{Yes}} = 2/3$，$p_{\text{No}} = 1/3$

$$
\begin{aligned}
H(\text{Rain}) &= -\frac{2}{3} \log_2\left(\frac{2}{3}\right) - \frac{1}{3} \log_2\left(\frac{1}{3}\right) \\
&= -\frac{2}{3} \cdot (-0.585) - \frac{1}{3} \cdot (-1.585) \\
&= 0.390 + 0.528 = 0.918
\end{aligned}
$$

加权平均熵：

$$
\begin{aligned}
H_{\text{avg}}(\text{Outlook}) &= \frac{3}{8} \cdot 0 + \frac{2}{8} \cdot 0 + \frac{3}{8} \cdot 0.918 \\
&= 0 + 0 + 0.344 = 0.344
\end{aligned}
$$

**信息增益：**

$$
IG(\text{Outlook}) = 1.000 - 0.344 = \boxed{0.656}
$$

Gini 增益：

| 子集 | Gini |
|------|------|
| Sunny | $1 - (0^2 + 1^2) = 0$ |
| Overcast | $1 - (1^2 + 0^2) = 0$ |
| Rain | $1 - \left(\frac{2}{3}\right)^2 - \left(\frac{1}{3}\right)^2 = 1 - \frac{4}{9} - \frac{1}{9} = \frac{4}{9} = 0.444$ |

$$
G_{\text{avg}}(\text{Outlook}) = \frac{3}{8}\cdot 0 + \frac{2}{8}\cdot 0 + \frac{3}{8}\cdot 0.444 = 0.167
$$

$$
\Delta G(\text{Outlook}) = 0.500 - 0.167 = \boxed{0.333}
$$

**▶ 分裂 2：Temperature（气温）**

| 子集 | 样本 | 分布 | 熵 | Gini |
|------|------|------|-----|------|
| Hot | D1(No), D2(No), D3(Yes) | [1Y, 2N] | 0.918 | $1 - \frac{1}{9} - \frac{4}{9} = 0.444$ |
| Mild | D4(Yes), D8(No) | [1Y, 1N] | 1.000 | $1 - 0.25 - 0.25 = 0.500$ |
| Cool | D5(Yes), D6(No), D7(Yes) | [2Y, 1N] | 0.918 | 0.444 |

$$
\begin{aligned}
H_{\text{avg}}(\text{Temp}) &= \frac{3}{8}\cdot 0.918 + \frac{2}{8}\cdot 1.000 + \frac{3}{8}\cdot 0.918 \\
&= 0.344 + 0.250 + 0.344 = 0.938
\end{aligned}
$$

$$
IG(\text{Temperature}) = 1.000 - 0.938 = \boxed{0.062}
$$

**▶ 分裂 3：Humidity（湿度）**

| 子集 | 样本 | 分布 | 熵 | Gini |
|------|------|------|-----|------|
| High | D1, D2, D3, D4, D8 | [2Y, 3N] | 0.971 | 0.480 |
| Normal | D5, D6, D7 | [2Y, 1N] | 0.918 | 0.444 |

High 子集（5 samples, 2Y+3N）：
$$
\begin{aligned}
H(\text{High}) &= -\frac{2}{5}\log_2\left(\frac{2}{5}\right) - \frac{3}{5}\log_2\left(\frac{3}{5}\right) \\
&= -0.4 \cdot (-1.322) - 0.6 \cdot (-0.737) \\
&= 0.529 + 0.442 = 0.971
\end{aligned}
$$

$$
\begin{aligned}
H_{\text{avg}}(\text{Humidity}) &= \frac{5}{8}\cdot 0.971 + \frac{3}{8}\cdot 0.918 \\
&= 0.607 + 0.344 = 0.951
\end{aligned}
$$

$$
IG(\text{Humidity}) = 1.000 - 0.951 = \boxed{0.049}
$$

**▶ 分裂 4：Windy（有风）**

| 子集 | 样本 | 分布 | 熵 | Gini |
|------|------|------|-----|------|
| Weak | D1, D3, D4, D5, D8 | [3Y, 2N] | 0.971 | 0.480 |
| Strong | D2, D6, D7 | [1Y, 2N] | 0.918 | 0.444 |

$$
H_{\text{avg}}(\text{Windy}) = \frac{5}{8}\cdot 0.971 + \frac{3}{8}\cdot 0.918 = 0.951
$$

$$
IG(\text{Windy}) = 1.000 - 0.951 = \boxed{0.049}
$$

**▶ 第一层汇总：各特征的信息增益**

| 特征 | 加权子节点熵 | 信息增益 IG | 基尼增益 |
|------|-------------|------------|---------|
| **Outlook** | 0.344 | **0.656 ✓** | **0.333 ✓** |
| Temperature | 0.938 | 0.062 | 0.042 |
| Humidity | 0.951 | 0.049 | 0.033 |
| Windy | 0.951 | 0.049 | 0.033 |

**Outlook 赢了！** 两个指标都选它。IG = 0.656，是最接近"一次搞定"（IG = 1.0）的选择。

分裂后的树结构：
```
         [Outlook?]
      /      |       \
  Sunny   Overcast    Rain
  (3N)    (2Y)       (2Y+1N)
```

Sunny 分支全是 No → 纯的，设为叶子节点 "No"
Overcast 分支全是 Yes → 纯的，设为叶子节点 "Yes"
Rain 分支还不纯 → 需要继续分裂！

#### 4.4 第二层：对 Rain 子集（D4, D5, D6）继续分裂

Rain 子集：D4(Yes, Mild, High, Weak), D5(Yes, Cool, Normal, Weak), D6(No, Cool, Normal, Strong)

$H(\text{Rain}) = 0.918$

**▶ 分裂 Rain 子集：按 Temperature**

| 子集 | 样本 | 分布 | 熵 |
|------|------|------|-----|
| Mild | D4(Yes) | [1Y] | 0 |
| Cool | D5(Yes), D6(No) | [1Y, 1N] | 1.000 |

$$
H_{\text{avg}} = \frac{1}{3}\cdot 0 + \frac{2}{3}\cdot 1.000 = 0.667
$$

$$
IG(\text{Temp}|Rain) = 0.918 - 0.667 = 0.251
$$

**▶ 分裂 Rain 子集：按 Humidity**

| 子集 | 样本 | 分布 | 熵 |
|------|------|------|-----|
| High | D4(Yes) | [1Y] | 0 |
| Normal | D5(Yes), D6(No) | [1Y, 1N] | 1.000 |

$$
IG(\text{Humidity}|Rain) = 0.918 - 0.667 = 0.251
$$

**▶ 分裂 Rain 子集：按 Windy**

| 子集 | 样本 | 分布 | 熵 |
|------|------|------|-----|
| Weak | D4(Yes), D5(Yes) | [2Y] | 0 |
| Strong | D6(No) | [1N] | 0 |

$$
IG(\text{Windy}|Rain) = 0.918 - 0 = \boxed{0.918}
$$

**Windy 赢了！** 信息增益 = 0.918，是**完美分裂**——两个子节点都完全纯净。

```
Rain 分支:
       [Windy?]
      /        \
   Weak       Strong
  (D4,D5→Yes)  (D6→No)
     YES          NO
```

#### 4.5 最终决策树

```
              [Outlook?]
          /      |        \
      Sunny   Overcast    Rain
        |        |          |
       NO       YES     [Windy?]
                        /       \
                     Weak     Strong
                       |         |
                      YES        NO
```

**验证：对训练集预测**——每条数据沿着树走一遍：

| ID | 天气 | 有风? | 路径 | 预测 | 实际 |
|----|------|-------|------|------|------|
| D1 | Sunny | — | Sunny → NO | NO | NO ✓ |
| D2 | Sunny | — | Sunny → NO | NO | NO ✓ |
| D3 | Overcast | — | Overcast → YES | YES | YES ✓ |
| D4 | Rain | Weak | Rain → Windy=Weak → YES | YES | YES ✓ |
| D5 | Rain | Weak | Rain → Windy=Weak → YES | YES | YES ✓ |
| D6 | Rain | Strong | Rain → Windy=Strong → NO | NO | NO ✓ |
| D7 | Overcast | — | Overcast → YES | YES | YES ✓ |
| D8 | Sunny | — | Sunny → NO | NO | NO ✓ |

训练准确率 **100%**。这棵树用 3 个内部节点就完美记住了 8 个样本的规律——而且每个节点的判断都有直观的业务解释。

#### 4.6 补充：信息增益率（C4.5 改进）——为什么温度不是好特征

你可能注意到 Temperature 的 IG (0.062) 比 Humidity/Windy 的 IG (0.049) 略高。但你直觉上可能觉得"温度"并不是一个很好的分裂依据——因为 Hot 和 Cool 的分布几乎完全对称。C4.5 算法引入**信息增益率（Gain Ratio）**来纠正信息增益对多值特征的偏好：

$$
\text{GainRatio}(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}
$$

其中 SplitInfo 是"这个特征自身的信息量"：

$$
SplitInfo(S, A) = -\sum_{v} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)
$$

手算各特征的 SplitInfo 和 Gain Ratio：

| 特征 | IG | SplitInfo | Gain Ratio |
|------|-----|-----------|------------|
| Outlook | 0.656 | $-\frac{3}{8}\log_2\frac{3}{8} - \frac{2}{8}\log_2\frac{2}{8} - \frac{3}{8}\log_2\frac{3}{8} = 1.561$ | 0.420 |
| Temperature | 0.062 | $-\frac{3}{8}\log_2\frac{3}{8} - \frac{2}{8}\log_2\frac{2}{8} - \frac{3}{8}\log_2\frac{3}{8} = 1.561$ | 0.040 |
| Humidity | 0.049 | $-\frac{5}{8}\log_2\frac{5}{8} - \frac{3}{8}\log_2\frac{3}{8} = 0.954$ | 0.051 |
| Windy | 0.049 | $-\frac{5}{8}\log_2\frac{5}{8} - \frac{3}{8}\log_2\frac{3}{8} = 0.954$ | 0.051 |

Outlook 仍然稳居第一（Gain Ratio = 0.420），但 Humidity 和 Windy 通过更简单的分裂结构（SplitInfo 更小），在增益率上反超了 Temperature。C4.5 因此比 ID3（纯信息增益）更不容易偏向"分支多的特征"。

> **为什么 sklearn 用 CART（Gini）而不是 C4.5？** CART 的二分约束天然避免了多值偏好问题——你分得再多也只是一刀切成两半。Gini + 二分 = 简洁、高效、够用。

#### 4.7 补充：CART 二叉树如何处理多值离散特征

上面的 Outlook 分裂产生了三个分支，但 CART 要求**二叉树**。CART 的处理方式是——对于有 $m$ 个取值的离散特征，遍历所有 $2^{m-1}-1$ 种二划分。

Outlook 有 3 个取值（Sunny, Overcast, Rain），可能的二划分：
- {Sunny} vs {Overcast, Rain}
- {Overcast} vs {Sunny, Rain}
- {Rain} vs {Sunny, Overcast}

分别计算这三种划分的信息增益，选最大的。课后思考题 Q3 带你手算这个。实际应用中，如果离散值太多（$m$ 很大），计算量爆炸——所以 sklearn 建议对高基数离散特征做 One-Hot 编码，将其转为多个二值特征。

---

### 5. 手算示例：3 棵树的随机森林投票

有了单棵决策树的基础，现在看随机森林——把 3 棵"不一样的树"聚在一起投票。

#### 5.1 生成 3 个 Bootstrap 样本

从原始 8 个样本中有放回抽样 8 个（每次抽后放回）：

**Bootstrap 1**（随机种子 seed=1）：
D1, D1, D2, D3, D4, D5, D7, D8
→ D6 没被抽到（OOB），D1 出现了两次

**Bootstrap 2**（seed=2）：
D2, D3, D4, D4, D5, D6, D6, D7
→ D1, D8 没被抽到

**Bootstrap 3**（seed=3）：
D1, D2, D3, D3, D5, D6, D7, D7
→ D4, D8 没被抽到

#### 5.2 每棵树随机选特征 + 训练

分类问题，$p = 4$ 个特征，每棵树随机选 $m = \lfloor\sqrt{4}\rfloor = 2$ 个特征。

**Tree 1** — 可用特征：[Outlook, Temperature]，Bootstrap 1 数据

| D1(重复) | Sunny | Hot  | No  |
| D1(重复) | Sunny | Hot  | No  |
| D2       | Sunny | Hot  | No  |
| D3       | Overcast | Hot  | Yes |
| D4       | Rain  | Mild | Yes |
| D5       | Rain  | Cool | Yes |
| D7       | Overcast | Cool | Yes |
| D8       | Sunny | Mild | No  |

标签分布：4 Yes (D3,D4,D5,D7), 4 No (D1×2,D2,D8)

在 Outlook 上：Sunny(3N), Overcast(2Y), Rain(2Y) → 中等纯
在 Temperature 上：Hot(2N+1Y), Mild(1Y+1N), Cool(2Y)

选择最佳分裂 → 假设选了 Outlook（因为是 Bootstrap 1 的最佳特征）。Tree 1 训练完成。

**Tree 2** — 可用特征：[Humidity, Windy]，Bootstrap 2 数据

| D2       | High  | Strong | No  |
| D3       | High  | Weak   | Yes |
| D4(重复) | High  | Weak   | Yes |
| D4(重复) | High  | Weak   | Yes |
| D5       | Normal| Weak   | Yes |
| D6(重复) | Normal| Strong | No  |
| D6(重复) | Normal| Strong | No  |
| D7       | Normal| Strong | Yes |

标签：4 Yes (D3,D4×2,D5,D7), 4 No (D2,D6×2)

最佳分裂 → 选了 Windy（因为两种湿度下标签都混杂，Windy 更好区分）。Tree 2 训练完成。

**Tree 3** — 可用特征：[Outlook, Windy]，Bootstrap 3 数据

| D1       | Sunny  | Weak   | No  |
| D2       | Sunny  | Strong | No  |
| D3(重复) | Overcast | Weak | Yes |
| D3(重复) | Overcast | Weak | Yes |
| D5       | Rain   | Weak   | Yes |
| D6       | Rain   | Strong | No  |
| D7(重复) | Overcast | Strong | Yes |
| D7(重复) | Overcast | Strong | Yes |

标签：5 Yes (D3×2,D5,D7×2), 3 No (D1,D2,D6)

Outlook 仍然是最强特征。Tree 3 训练完成。

#### 5.3 投票：对一个新样本做预测

新的一天：**Sunny, Mild, Normal, Weak**——到底去不去打球？

分别问 3 棵树：

| 树 | 看到的特征 | 预测 | 推理 |
|----|-----------|------|------|
| Tree 1 | Outlook=Sunny, Temp=Mild | **NO** | Sunny 分支全是 No（看 D1, D2, D8） |
| Tree 2 | Humidity=Normal, Windy=Weak | **YES** | Normal+Weak → D5 是 Yes，D6 是 No 但有风；Weak 下多数是 Yes |
| Tree 3 | Outlook=Sunny, Windy=Weak | **NO** | Outlook=Sunny → No（D1, D2 都是 No） |

投票结果：NO = 2 票，YES = 1 票 → **最终预测：NO（不去打球）**

> 注意这 3 棵树判断依据不同——Tree 1 和 Tree 3 看到了 Outlook，Tree 2 没看到 Outlook 但看到了 Humidity 和 Windy。Tree 2 是少数派，但正是因为这种"不同视角"的存在，当多数派在某类样本上集体犯错时，少数派可能把结果拉回正确方向——这就是集成的力量。

#### 5.4 更严谨的验证：OOB 误差

Tree 1 的 OOB 样本是 D6（Play=No），Tree 3 的 OOB 样本是 D4（Play=Yes）。用没参与训练的样本做验证：

| 样本 | 参与训练的树 | 预测 | 结果 |
|------|------------|------|------|
| D6（No） | Tree 2, Tree 3 | Tree 2 → ?, Tree 3 → NO | 多数判 No ✓ |
| D4（Yes） | Tree 1, Tree 2 | Tree 1 → ?, Tree 2 → YES | 多数判 Yes ✓ |

OOB 误差是随机森林内置的免费验证——不需要单独留验证集。

---

### 6. Python 实现

#### 6.1 从零手写 CART 决策树分类器

```python
import numpy as np
from collections import Counter

class SimpleDecisionTree:
    """从零手写 CART 决策树分类器"""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion  # 'gini' or 'entropy'
        self.tree_ = None

    def _impurity(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        if self.criterion == 'gini':
            return 1 - np.sum(probs ** 2)
        else:
            return -np.sum(probs * np.log2(probs + 1e-12))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = -1
        best = None
        n = len(y)
        parent_imp = self._impurity(y)

        for j in range(X.shape[1]):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                n_left, n_right = left_mask.sum(), right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                imp_left = self._impurity(y[left_mask])
                imp_right = self._impurity(y[right_mask])
                weighted = (n_left / n) * imp_left + (n_right / n) * imp_right
                gain = parent_imp - weighted

                if gain > best_gain:
                    best_gain = gain
                    best = {
                        'feature': j, 'threshold': t,
                        'left_idx': np.where(left_mask)[0],
                        'right_idx': np.where(right_mask)[0]
                    }
        return best, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        n = len(y)
        counter = Counter(y)

        if (len(counter) == 1 or n < self.min_samples_split
                or depth >= self.max_depth):
            return {'type': 'leaf', 'class': counter.most_common(1)[0][0]}

        best, gain = self._best_split(X, y)

        if best is None or gain <= 0:
            return {'type': 'leaf', 'class': counter.most_common(1)[0][0]}

        left_tree = self._build_tree(
            X[best['left_idx']], y[best['left_idx']], depth + 1)
        right_tree = self._build_tree(
            X[best['right_idx']], y[best['right_idx']], depth + 1)

        return {
            'type': 'node',
            'feature': best['feature'],
            'threshold': best['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: dict):
        if node['type'] == 'leaf':
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ============ 测试：在手算的天气数据上验证 ============
# 编码：Sunny=0, Overcast=1, Rain=2; Hot=0, Mild=1, Cool=2;
#       High=0, Normal=1; Weak=0, Strong=1; No=0, Yes=1
X_weather = np.array([
    [0, 0, 0, 0],  # D1: Sunny, Hot,  High, Weak  → No
    [0, 0, 0, 1],  # D2: Sunny, Hot,  High, Strong → No
    [1, 0, 0, 0],  # D3: Overcast, Hot,  High, Weak  → Yes
    [2, 1, 0, 0],  # D4: Rain, Mild, High, Weak  → Yes
    [2, 2, 1, 0],  # D5: Rain, Cool, Normal, Weak → Yes
    [2, 2, 1, 1],  # D6: Rain, Cool, Normal, Strong → No
    [1, 2, 1, 1],  # D7: Overcast, Cool, Normal, Strong → Yes
    [0, 1, 0, 0],  # D8: Sunny, Mild, High, Weak  → No
])
y_weather = np.array([0, 0, 1, 1, 1, 0, 1, 0])

tree_weather = SimpleDecisionTree(max_depth=3, criterion='entropy')
tree_weather.fit(X_weather, y_weather)
print(f"手写树 - 天气数据准确率: {tree_weather.score(X_weather, y_weather):.1%}")
print(f"树结构: {tree_weather.tree_}")
```

**输出示例：**
```
手写树 - 天气数据准确率: 100.0%
```

#### 6.2 sklearn 快速对比：决策树 vs 随机森林 vs 单棵树

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine

wine = load_wine()
X_w, y_w = wine.data, wine.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X_w, y_w, test_size=0.3, stratify=y_w, random_state=42)

models = {
    '决策树 (默认)':
        DecisionTreeClassifier(random_state=42),
    '决策树 (max_depth=4)':
        DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42),
    '随机森林 (100棵)':
        RandomForestClassifier(n_estimators=100, random_state=42),
    '随机森林 (200棵+调参)':
        RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                               random_state=42),
}

print(f"{'模型':<25} {'训练':>8} {'测试':>8} {'5折CV':>8}")
print("-" * 52)
for name, model in models.items():
    model.fit(X_tr, y_tr)
    train_acc = model.score(X_tr, y_tr)
    test_acc = model.score(X_te, y_te)
    cv = cross_val_score(model, X_w, y_w, cv=5).mean()
    print(f"{name:<25} {train_acc:>8.4f} {test_acc:>8.4f} {cv:>8.4f}")
```

**输出示例：**
```
模型                        训练     测试    5折CV
----------------------------------------------------
决策树 (默认)                1.0000   0.9074   0.8821
决策树 (max_depth=4)         0.9677   0.9259   0.8989
随机森林 (100棵)              1.0000   0.9815   0.9769
随机森林 (200棵+调参)          0.9919   0.9815   0.9769
```

默认决策树训练准确率 100% 但测试只有 90%——**过拟合**。限制深度后测试提升。随机森林训练和测试都高，CV 也很稳定。

#### 6.3 过拟合与剪枝效果

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

np.random.seed(42)
X_m, y_m = make_moons(n_samples=200, noise=0.3, random_state=42)
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_m, y_m, test_size=0.3, random_state=42)

depths = range(1, 16)
train_scores, test_scores = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_tr_m, y_tr_m)
    train_scores.append(tree.score(X_tr_m, y_tr_m))
    test_scores.append(tree.score(X_te_m, y_te_m))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, 'o-', label='训练准确率', linewidth=2)
plt.plot(depths, test_scores, 's-', label='测试准确率', linewidth=2)
plt.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='最佳深度 ≈ 4')
plt.xlabel('max_depth'); plt.ylabel('准确率')
plt.title('决策树：深度越大，过拟合越严重')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# 三种策略对比
tree_full = DecisionTreeClassifier(random_state=42).fit(X_tr_m, y_tr_m)
tree_pre = DecisionTreeClassifier(
    max_depth=4, min_samples_split=10, random_state=42).fit(X_tr_m, y_tr_m)

# 后剪枝: 找最佳 ccp_alpha
path = tree_full.cost_complexity_pruning_path(X_tr_m, y_tr_m)
alphas = path.ccp_alphas[:min(30, len(path.ccp_alphas))]
cv_scores = []
for a in alphas:
    t = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
    cv_scores.append(cross_val_score(t, X_tr_m, y_tr_m, cv=5).mean())
best_alpha = alphas[np.argmax(cv_scores)]
tree_post = DecisionTreeClassifier(
    random_state=42, ccp_alpha=best_alpha).fit(X_tr_m, y_tr_m)

print(f"{'模型':<12} {'训练':>8} {'测试':>8} {'叶节点':>6}")
print("-" * 38)
for name, t in [('无限制', tree_full), ('预剪枝', tree_pre), ('后剪枝', tree_post)]:
    print(f"{name:<12} {t.score(X_tr_m, y_tr_m):>8.3f} "
          f"{t.score(X_te_m, y_te_m):>8.3f} {t.get_n_leaves():>6}")
```

**输出示例：**
```
模型           训练     测试   叶节点
--------------------------------------
无限制          1.000    0.833     28
预剪枝          0.907    0.883      7
后剪枝          0.921    0.883      8
```

无限制树 28 个叶节点——几乎每个叶子对应几个训练样本。剪枝后 7-8 个叶子，泛化更强。

#### 6.4 简化版随机森林：手写 vs sklearn

```python
class SimpleRandomForest:
    """超简化随机森林——用于理解原理，不用于生产"""

    def __init__(self, n_trees: int = 20, max_depth: int = 10,
                 max_features: str = 'sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees_ = []
        self.feature_subsets_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        n_features = int(np.sqrt(p)) if self.max_features == 'sqrt' else p

        for _ in range(self.n_trees):
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            feat_subset = np.random.choice(
                p, size=n_features, replace=False)
            self.feature_subsets_.append(feat_subset)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, random_state=42)
            tree.fit(X_boot[:, feat_subset], y_boot)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((X.shape[0], self.n_trees), dtype=int)
        for i, (tree, fs) in enumerate(
                zip(self.trees_, self.feature_subsets_)):
            preds[:, i] = tree.predict(X[:, fs])
        return np.array([np.bincount(p).argmax() for p in preds])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


rf_simple = SimpleRandomForest(n_trees=30, max_depth=8)
rf_simple.fit(X_tr_m, y_tr_m)

tree_single = DecisionTreeClassifier(
    max_depth=8, random_state=42).fit(X_tr_m, y_tr_m)
rf_sklearn = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr_m, y_tr_m)

print(f"单棵决策树      训练: {tree_single.score(X_tr_m, y_tr_m):.3f}  "
      f"测试: {tree_single.score(X_te_m, y_te_m):.3f}")
print(f"手写随机森林     训练: {rf_simple.score(X_tr_m, y_tr_m):.3f}  "
      f"测试: {rf_simple.score(X_te_m, y_te_m):.3f}")
print(f"sklearn随机森林  训练: {rf_sklearn.score(X_tr_m, y_tr_m):.3f}  "
      f"测试: {rf_sklearn.score(X_te_m, y_te_m):.3f}")
```

#### 6.5 特征重要性

```python
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_w, y_w)

mdi_imp = rf.feature_importances_
perm_imp = permutation_importance(
    rf, X_w, y_w, n_repeats=10, random_state=42, n_jobs=-1)

print("特征重要性 Top 5 (Wine 数据集):")
print(f"{'特征名':<20s} {'MDI':>8s} {'置换':>8s}")
print("-" * 40)
for idx in np.argsort(mdi_imp)[::-1][:5]:
    print(f"{wine.feature_names[idx]:<20s} {mdi_imp[idx]:>8.4f} "
          f"{perm_imp.importances_mean[idx]:>8.4f}")
```

**输出示例：**
```
特征重要性 Top 5 (Wine 数据集):
特征名                     MDI      置换
----------------------------------------
proline                 0.1854   0.1534
flavanoids              0.1744   0.1237
color_intensity         0.1392   0.0825
od280/od315_of_diluted_wines 0.1238   0.0743
alcohol                 0.0976   0.0682
```

---

#### 6.6 完整的 sklearn 实战：GridSearchCV 调参 + 模型持久化

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import load_breast_cancer
import joblib

data = load_breast_cancer()
X, y = data.data, data.target

# ===== 决策树 GridSearch =====
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 3, 5, 10],
    'criterion': ['gini', 'entropy'],
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X, y)

print("=" * 60)
print("决策树 GridSearchCV 最佳结果")
print("=" * 60)
print(f"最佳参数: {dt_grid.best_params_}")
print(f"最佳 CV 准确率: {dt_grid.best_score_:.4f}")

# ===== 随机森林 GridSearch =====
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 12, None],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2', None],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X, y)

print("\n" + "=" * 60)
print("随机森林 GridSearchCV 最佳结果")
print("=" * 60)
print(f"最佳参数: {rf_grid.best_params_}")
print(f"最佳 CV 准确率: {rf_grid.best_score_:.4f}")

# ===== 最终模型对比 =====
best_dt = dt_grid.best_estimator_
best_rf = rf_grid.best_estimator_

print("\n" + "=" * 60)
print("最终模型 5 折 CV 对比")
print("=" * 60)
dt_cv = cross_val_score(best_dt, X, y, cv=5)
rf_cv = cross_val_score(best_rf, X, y, cv=5)
print(f"最佳决策树   CV: {dt_cv.mean():.4f} (±{dt_cv.std():.4f})")
print(f"最佳随机森林 CV: {rf_cv.mean():.4f} (±{rf_cv.std():.4f})")

# 特征重要性报告
print("\n" + "=" * 60)
print("随机森林特征重要性 Top 10")
print("=" * 60)
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
for rank, idx in enumerate(indices[:10], 1):
    print(f"  {rank:>2}. {data.feature_names[idx]:<30s} {importances[idx]:.4f}")

# 模型持久化（可选）
# joblib.dump(best_rf, 'rf_breast_cancer.pkl')
# best_rf_loaded = joblib.load('rf_breast_cancer.pkl')
```

**输出示例：**
```
============================================================
决策树 GridSearchCV 最佳结果
============================================================
最佳参数: {'criterion': 'gini', 'max_depth': 5,
           'min_samples_leaf': 3, 'min_samples_split': 5}
最佳 CV 准确率: 0.9349

============================================================
随机森林 GridSearchCV 最佳结果
============================================================
最佳参数: {'max_depth': 8, 'max_features': 'sqrt',
           'min_samples_leaf': 1, 'n_estimators': 200}
最佳 CV 准确率: 0.9648

============================================================
最终模型 5 折 CV 对比
============================================================
最佳决策树   CV: 0.9349 (±0.0231)
最佳随机森林 CV: 0.9648 (±0.0156)
```

> **实战教训：** GridSearchCV 搜索空间不宜太大——上面的 4×4×4×2=128 组合已经需要 128×5=640 次拟合。对于随机森林的 3×4×3×3=108 组合，每次拟合 200 棵树，总计算量很大。实际工作中先用粗粒度搜索（如 max_depth=3/7/15），锁定范围后再细调。

---

### 7. 常见误区

**误区 1："熵和 Gini 哪个好？我得仔细选。"**

实践中两者差距极小——99% 的情况下选同一个最佳分裂。Gini 快（没对数），是 sklearn 默认。熵对"不纯"惩罚更重——如果你特别在意节点的纯度、或者数据极度不平衡，可以试试熵。**不要在这个超参上花太多时间。**

**误区 2："决策树不用归一化，太好了！"**

部分正确——树的分类只需比较阈值，不依赖距离度量，确实不需要归一化。但**如果你用 `ccp_alpha` 做后剪枝，错误率对不同量纲的特征可能敏感**（虽然在分类中影响很小）。此外，如果你的 pipeline 中混合了需要归一化的模型，仍然要做。

**误区 3："随机森林不会过拟合，所以我用 10000 棵树。"**

"不易过拟合"不意味着"绝对不过拟合"。如果单棵树的深度不受限制（`max_depth=None`），每棵树都深度过拟合，那么 10000 棵过拟合树的投票结果还是过拟合的——它们只是过拟合在不同的数据子集上，投票只是把这些过拟合"平均"了一下。真正防止过拟合的是**限制单棵树的深度**，而不是树的数量。

**误区 4："特征重要性高 = 特征对预测贡献大。"**

MDI 特征重要性有已知偏差：
1. 偏好高基数特征（取值多的特征有更多分裂机会）
2. 偏好连续特征（连续特征的阈值选择更丰富）
3. 在有相关特征时会"分摊"重要性——如果两个特征高度相关，它们各自的重要性看起来都不高，但去掉任意一个都会导致模型变差

**更可靠的做法是置换重要性**——打乱特征后看准确率下降多少。如果两者结论一致，可以放心使用。

**误区 5："决策树可解释，所以我一定知道模型在做什么。"**

浅树（深度 ≤ 4）确实一目了然。但深度 10 的树有上百个叶子节点——"可解释"不等于"易理解"。如果追求极致可解释性，应配合剪枝保持树在 3-5 层以内。对于需要解释的场景，浅决策树 > 深决策树 > 随机森林 > 神经网络。

---

### 8. ML 应用

#### 8.1 决策树的真实应用场景

| 领域 | 应用 | 为什么用决策树 |
|------|------|---------------|
| **金融风控** | 贷款审批、信用卡欺诈检测 | 需要向监管解释"为什么拒绝这个申请" |
| **医疗诊断** | 症状 → 疾病分类 | 诊断路径与医生思维一致，可审核 |
| **客户分群** | 用户画像、流失预测 | 输出可读的分群规则，业务方直接使用 |
| **故障诊断** | 工业设备、网络故障定位 | 故障排除树本身就是决策树 |
| **推荐系统** | 特征组合探索、冷启动 | 快速验证哪些特征交互有信号 |

#### 8.2 可视化决策树——把模型变成可沟通的流程图

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 在 Wine 数据上训练一棵浅树用于展示
tree_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_viz.fit(X_w, y_w)

plt.figure(figsize=(18, 8))
plot_tree(tree_viz, feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True, rounded=True, fontsize=9,
          impurity=True, proportion=True)
plt.title('Wine 数据集的决策树（深度=3）', fontsize=14)
plt.tight_layout()
plt.show()

# 打印文本版决策规则（更轻量）
from sklearn.tree import export_text
tree_rules = export_text(tree_viz, feature_names=wine.feature_names)
print(tree_rules)
```

**`plot_tree` 参数速查：**
- `filled=True`：按类别给节点染色，一眼看出哪个类占优
- `impurity=True`：显示每个节点的 Gini/熵值
- `proportion=True`：显示比例而非绝对数量
- `rounded=True`：圆角节点，好看一些

生产环境中，把 `plot_tree` 输出的 SVG/PNG 放进模型报告，是对非技术同事最有说服力的沟通材料——"你看，模型判断这瓶酒是 Class 0，因为它 color_intensity ≤ 3.4 且 proline ≤ 755"。

#### 8.3 随机森林的工程实践

- **基线模型首选**：拿到一个新的表格数据分类任务，先跑一遍随机森林。它几乎不需要特征工程、不需要调参，出来的结果就是强基线——如果后面的复杂模型超不过这个基线，说明你的特征可能有问题。
- **特征筛选**：用特征重要性剔除无关特征，减少后续模型的输入维度。常见做法是取重要性 Top 80%，丢弃底部 20%。
- **缺失值填充**：随机森林可以用 `IterativeImputer` 来填充缺失值——用其他特征预测缺失的那个，效果通常优于均值/中位数填充。
- **异常检测**：随机森林的 Proximity Matrix（样本对样本的距离矩阵，基于两样本落在同一叶节点的频率）可以识别离群样本。
- **概率校准**：`RandomForestClassifier` 的 `predict_proba` 虽然粗糙（基于各棵树投票比例），但在实践中作为排序分数是有效的。如果需要校准的概率，用 `CalibratedClassifierCV` 包装一层。

#### 8.4 从决策树到梯度提升

决策树是催化剂——它催生了两个统治表格数据的算法家族：

- **Bagging（并行）**：随机森林 → ExtraTrees → 更多的树 + 更多的随机性
- **Boosting（串行）**：AdaBoost → GBDT → XGBoost → LightGBM → CatBoost

两者核心差异：

| 维度 | Bagging (随机森林) | Boosting (GBDT/XGBoost) |
|------|-------------------|------------------------|
| 训练方式 | 各棵树并行、独立训练 | 串行——每棵树修正前一棵的残差 |
| 偏差 | 高（每棵树是弱模型） | 低（每棵树都在逼近真实值） |
| 方差 | 低（投票平均） | 较高（模型强，容易过拟合） |
| 过拟合趋势 | 不易过拟合 | 容易过拟合——需早停 + 低学习率 |
| 训练速度 | 快（天然并行） | 慢（必须串行） |
| 性能天花板 | 中高 | **极高**（大量 Kaggle 冠军方案） |

理解决策树，就理解了这些 Kaggle 冠军模型的核心构建块。第 12 章会详细介绍集成学习的完整谱系。

#### 8.5 超参数调优的实战优先级

**决策树（从最重要到最不重要）：**

1. **`max_depth`**（防过拟合第一防线）：从 3 开始，每次翻倍（3, 6, 12, 24, None），取验证曲线拐点
2. **`min_samples_leaf`**（第二防线）：对噪声数据设到 3~5，可以显著改善泛化
3. **`min_samples_split`**（第三防线）：一般设为 `min_samples_leaf * 2`
4. **`criterion`**：gini 默认即可，几乎不用调

**随机森林（从最重要到最不重要）：**

1. **`n_estimators`**（先调到稳定）：100 → 200 → 300，找到性能不再提升的点后固定
2. **`max_depth`**（防过拟合）：同决策树，但可以设得稍保守（因为树多）
3. **`max_features`**（控制树间相关性）：分类用 sqrt，回归用 p/3，一般不用改
4. **`min_samples_leaf`**（对大数据集）：调大可以减少训练时间，同时改善泛化

**经验法则：**
- `n_estimators` 越大越好，但边际收益递减。100→200 提升可能 1-2%，200→500 提升通常 <0.5%。
- 如果训练/测试准确率差超过 5%，优先降 `max_depth` 或增大 `min_samples_leaf`
- 如果两者都低（欠拟合），先增大 `max_depth` 再增 `n_estimators`

```python
# 快速评估 n_estimators 的影响
for n in [10, 50, 100, 200, 400]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_w, y_w, cv=5)
    print(f"n_estimators={n:>3}  CV 准确率: {scores.mean():.4f} (±{scores.std():.4f})")
```

---

### 9. 思考题

以下 10 道题由浅入深，覆盖理论手算、直观理解、实践误区。建议每题先自己思考 2 分钟再对答案。

---

**Q1：一个节点中有 10 个样本，全部是同一个类。它的熵和基尼不纯度各是多少？如果样本均分到 5 个类呢？**

<details>
<summary>点击查看解答</summary>

全同类情况：
- $p_1 = 1.0$，其余 $p_k = 0$
- 熵：$H = -1.0 \cdot \log_2(1.0) = 0$（约定 $0 \cdot \log_2(0) = 0$）
- 基尼：$G = 1 - 1^2 = 0$

均分 5 类情况：$p_k = 0.2$ ∀k
- 熵：$H = -5 \cdot (0.2 \cdot \log_2(0.2)) = -5 \cdot 0.2 \cdot (-2.322) = 2.322$（最大值）
- 基尼：$G = 1 - 5 \cdot 0.04 = 1 - 0.20 = 0.80$

两种度量在"纯"和"最混"两端一致，但中间区域的数值尺度不同。熵的数值范围更大——这也是为什么它来自信息论（用比特度量不确定性，天然就是 log 刻度）。
</details>

---

**Q2：在手算示例中，为什么 Humidity 和 Windy 的信息增益完全相同（0.049）？这是巧合还是有深层原因？**

<details>
<summary>点击查看解答</summary>

恰好是巧合——两个特征在当前数据集上都把 8 个样本分成了 5 和 3 两组，且两组的标签分布恰好对称（High: 3No+2Yes, Normal: 2Yes+1No vs Weak: 3Yes+2No, Strong: 1Yes+2No），导致加权平均熵相同。

在另一个样本量更大的数据集中，Humidity 和 Windy 的信息增益通常不同。此处的巧合是因为数据太少（8 个样本）、取值组合恰好对称。

这也揭示了一个重要事实：**小数据集上手算出来的"最佳特征"可能是随机波动造成的。** 真实应用中必须用足够的样本 + 交叉验证来确认特征的有效性。
</details>

---

**Q3：CART 做二叉树时，Outlook 有三种取值（Sunny, Overcast, Rain）。请手算 {Sunny, Overcast} vs {Rain} 这一划分的信息增益。和章节中三路分支的 0.656 有什么不同？**

<details>
<summary>点击查看解答</summary>

划分 {Sunny, Overcast} vs {Rain}：

- 左 {Sunny, Overcast}：D1(No), D2(No), D3(Yes), D7(Yes), D8(No) → [2Y, 3N]，$H = 0.971$
- 右 {Rain}：D4(Yes), D5(Yes), D6(No) → [2Y, 1N]，$H = 0.918$

加权平均：$\frac{5}{8} \cdot 0.971 + \frac{3}{8} \cdot 0.918 = 0.607 + 0.344 = 0.951$

$IG = 1.000 - 0.951 = \boxed{0.049}$

远小于三路分支的 0.656！

另外两种划分：
- {Sunny} vs {Overcast, Rain}：左 3N(H=0)，右 4Y+1N(H=0.722)，IG = 1 - (3/8·0 + 5/8·0.722) = 0.549
- {Overcast} vs {Sunny, Rain}：左 2Y(H=0)，右 2Y+4N(H=0.918)，IG = 1 - (2/8·0 + 6/8·0.918) = 0.312

最优二划分是 {Sunny} vs {Overcast, Rain}，IG = 0.549。虽然比三路分支的 0.656 低，但它是 CART 二值化约束下的最优解。
</details>

---

**Q4：如果父节点是纯的（IG = 0 或 Gini = 0），后续子节点还能继续吗？为什么"纯节点"是最理想的停止条件？**

<details>
<summary>点击查看解答</summary>

不能，也不需要。算法逻辑中 `len(counter) == 1` 直接触发停止——当前节点所有样本是同一类，再分裂没有任何意义：无论怎么分，子节点也都是纯的（或空的），不纯度不会进一步下降。

"纯节点"是最理想的叶节点——它给出的是 100% 确定的预测。从信息论的角度，纯节点的熵为 0，意味着不需要任何额外的比特来描述标签——数据已经被完美分类。

在实际应用中，追求让所有叶子节点都纯会导致严重的过拟合（记住噪音）。所以通常在纯节点条件之外，还会加上 `min_samples_leaf`、`max_depth` 等约束——宁可让某些叶节点"不太纯但泛化好"，也不要"全纯但过拟合"。
</details>

---

**Q5：一个数据集中有 100 条数据（50 正 + 50 负），但某特征 A 可以完美分裂：左边 50 全是正类，右边 50 全是负类。这种情况下的信息增益是多少？是否有问题？**

<details>
<summary>点击查看解答</summary>

父节点熵 = 1.0。分裂后左 H=0，右 H=0，加权平均 = 0。IG = 1.0 = 最大值。

**问题：** 在真实世界的 50/50 随机数据上，某个特征能完美把所有正类和负类恰好分开——这太"完美"了，几乎不可能是真实规律。这种情况下：

1. **数据泄露（Data Leakage）**：特征 A 可能就是标签本身（或直接从标签推导出来的）。例如你用"是否付款"来预测"是否购买"——因果倒置。
2. **过拟合的信号**：在小数据上，纯靠运气也可能出现某个特征"看起来"完美分类。
3. **信息增益没有上限**：IG = 1.0 意味着一个特征就解决了问题——但模型对你来说已经没价值了（特征 A = 答案）。

实践中应检查：这个"完美特征"是否在真实场景中可获取？是否和标签有直接的因果关系？
</details>

---

**Q6：你正在用随机森林做分类，测试准确率 85%。把树从 100 棵增加到 500 棵，准确率变成 85.1%。值得加到 2000 棵吗？为什么？**

<details>
<summary>点击查看解答</summary>

**不值得。** 收益递减：100→500 增加了 400 棵树，只提升了 0.1%。500→2000 大概率进一步趋近于零提升（甚至可能因数值精度问题略微波动）。

随机森林的错误率随树数量的增加呈 **O(1/√B)** 的衰减趋势——初期每加一棵树都有明显改善，但收益递减非常快。实践中：
- 100~200 棵树：对于大多数问题已经足够
- 300~500 棵树：如果数据很复杂或对稳定性有极致要求
- 1000+ 棵树：只有在竞赛冲榜时才有意义

每增加一棵树都有计算成本（训练 + 预测 + 存储）。与其加树，不如调其他参数（max_depth、max_features、min_samples_leaf）或用 GridSearchCV 系统地搜索。
</details>

---

**Q7：对于有 10000 个特征的高维数据（如基因表达数据），`max_features='sqrt'` 会选多少个特征？如果某特征只在 1% 的样本上与标签相关，它有多大机会被选中？这个机制有什么问题？**

<details>
<summary>点击查看解答</summary>

$m = \lfloor\sqrt{10000}\rfloor = 100$ 个特征。

每个分裂节点，一个特定弱特征被选中的概率 = $100 / 10000 = 1\%$。一棵树有多个分裂节点（假设平均 50 个），该特征在这棵树中至少出现一次的概率 ≈ $1 - (1 - 0.01)^{50} = 1 - 0.99^{50} \approx 39.5\%$。

看起来不低，但**问题在于**：在 100 棵树中，这个特征出现的位置、使用的阈值各不相同。弱信号可能被更强的特征完全淹没——"每次都被选，但每次都不被选为最佳分裂"。

此外，sqrt 策略在高维稀疏数据中会严重偏向**高方差特征**（取值多的特征在随机子集中更容易偶然出现好的分裂）。对于基因数据等 p ≫ n 场景，建议：
1. 先用方差过滤 / 单变量检验做特征初筛
2. 增大 `max_features`（如设为 `p/3` 或直接用 `1.0`）
3. 考虑用 GBDT（XGBoost/LightGBM）代替随机森林——它们天然处理稀疏高维数据
</details>

---

**Q8：手算的 3 棵树随机森林中，Tree 2 没有看到 Outlook 特征。如果真实的"打球决策"完全由 Outlook 决定（不看 Outlook 就没法准确判断），Tree 2 是不是"噪音制造者"？它拖累了整体性能吗？**

<details>
<summary>点击查看解答</summary>

在"Outlook 决定一切"这个极端假设下——是的，Tree 2 的投票可能是随机猜测，会拖低准确率。

但这正是随机森林设计中一个重要的 trade-off：**特征随机性降低了树间的相关性（降低方差），但代价是每棵树的偏差可能增大**——看不够全部特征，单棵树的准确率比全特征树低。

随机森林成立的前提是：**大多数特征多少有些信号，而且信号之间有冗余**。在真实数据中，几乎不存在"100% 由一个特征决定"的情况——Outlook 决定打球的 70%，但 Humidity、Windy 各贡献 10-15%。砍掉 Outlook，其他特征合起来还能做到 60-70% 的准确率。

如果在你的领域确实存在"少数几个特征极度重要"：
- 调大 `max_features`（如用 `p` 或 `p * 0.5`）
- 或者换用 **ExtraTrees**——它比随机森林更随机（连阈值都是随机选的），但树更多（500+），在特征极度不均衡时反而更鲁棒
</details>

---

**Q9：如果数据有 95% 是 A 类、5% 是 B 类，决策树会怎么表现？你该怎么调整？**

<details>
<summary>点击查看解答</summary>

极其不平衡时，决策树可能会建出一棵"永远预测 A"的树——因为即便少数样本是 B，把 B 分错对总不纯度的影响太小，分裂的增益可能不如不分裂。

举例：99 个 A + 1 个 B → Gini = $1 - (0.99^2 + 0.01^2) ≈ 0.0198$。任何分裂最多也只能降到 0——增益上限只有 0.02，算法可能直接判定"不值得分裂"。

**应对策略：**
1. **`class_weight='balanced'`**：sklearn 自动给少数类更大的权重——对于少数类 B 的错误，Gini/熵计算时放大惩罚
2. **调大 `min_samples_leaf`**：防止叶子节点太小——单样本叶子在训练时看起来很纯，其实是过拟合噪音
3. **采样策略**：上采样少数类（SMOTE）或下采样多数类
4. **对评估指标不过分关注准确率**：在不平衡数据上，准确率是误导性的——应该看 F1、AUC-ROC、Precision-Recall

随机森林在不平衡数据上通常比单棵决策树好——Bootstrap 采样让某些树看到更多的少数类样本（因为采样引入了随机性，少数类被过采样的树会更多关注 B）。
</details>

---

**Q10：请不查资料说出：熵的公式 $H = -\sum p_k \log_2(p_k)$ 中，负号是干什么的？为什么用 $\log_2$ 而不是 $\ln$ 或 $\log_{10}$？如果所有分类器都用 Gini 不纯度，世界会有什么不同？**

<details>
<summary>点击查看解答</summary>

**负号的来源：** $p_k \in (0, 1]$，$\log_2(p_k) \le 0$。$p_k \cdot \log_2(p_k)$ 总是负数或零。加负号后熵变成正数——熵是"信息量的期望"，信息量 $-\log_2(p)$ 本身是正的（小概率事件包含更多惊喜 → 更多信息）。所以熵 = 每个类别的概率 × 该类别出现时的信息量 的加权和。

**为什么是 $\log_2$：** 信息论的比特（bit）——1 比特就是一个"是/否"问题能消除的不确定性。$\log_2$ 让熵的单位是比特，物理意义明确。换成 $\ln$ 只是单位变成 nat（自然单位），换底可以互相转换：$\ln(p) = \log_2(p) \cdot \ln(2)$。由于决策树比的是相对大小（谁的 IG 更大），用什么底数**完全不影响排序**——$-\sum p_k \ln(p_k)$ 给的信息增益排名和 $-\sum p_k \log_2(p_k)$ 完全一样（只是数值乘了个常数 $\ln 2$）。

**如果全世界都用 Gini：**
- 工程角度：计算快了一些（没对数），决策树的训练时间略微缩短。对于 GB 级数据，这可能是分钟级的差异。
- 理论角度：信息论和 ML 之间的桥梁断了一点点——熵连接着交叉熵、KL 散度、互信息等一整套信息论工具。Gini 虽然也能推广（有一个叫"Tsallis entropy"的家族包含 Gini 作为特例），但远不如 Shannon 熵的生态丰富。
- 实践角度：**没什么实质性不同**——两个指标 99% 的时候选同一个分裂点。就像用英里还是公里——不影响你到达目的地。
</details>

---

决策树和随机森林只是集成学习的序幕。下一章是 [朴素贝叶斯](./10-naive-bayes.md)——一个基于概率的"天真"分类器。然后再下一章，我们将面对 Kaggle 的真正统治者：[集成方法](./12-ensemble-methods.md)——XGBoost、LightGBM 和 CatBoost 会把你现在学的一切推到极致。
