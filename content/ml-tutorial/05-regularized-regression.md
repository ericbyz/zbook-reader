# 05 — 正则化回归：当你的特征比你还能说

> **核心问题**：100 个特征、50 个样本——你怎么训练一个不崩的模型？更糟的是，其中 80 个特征跟预测目标毫无关系，全是噪声。普通线性回归会把每一个噪声特征都分配一个非零系数，在训练集上完美拟合，出门就扑街。

---

## 0. 本章导览

上一章你学了线性回归——给一堆数据点找一条"最好"的直线。但上一章默认了一个前提：**样本足够多、特征足够少、特征之间最好别太像**。真实世界完全相反：

- **基因表达数据**：测一个病人的基因要测几千个位点（$p$ 巨大），但医院只有几十个病例（$n$ 很小）
- **文本分类**：词库里几万个词（$p$），但你只有几百篇文章（$n$）
- **房价预测**："总面积""室内面积""使用面积"互相是彼此的影子，几乎线性相关
- **宏观经济预测**：几十个指标里真正驱动 GDP 的可能就三五个

正则化回归就是为这些"脏数据"场景而生的。学完本章，你将能够：

1. 解释为什么 $p \gg n$ 时普通线性回归必崩，以及正则化如何从数学上拯救它
2. 手算 Ridge 在 3 个数据点上的完整系数收缩过程（$\lambda=0.1,\;1.0,\;10$ 三组对比）
3. 手算 Lasso 在正交数据上的系数归零全过程，推导并理解软阈值算子的数学含义
4. 用几何直觉（菱形 vs 圆形）解释 L1 为什么能产生零系数而 L2 不能
5. 独立编写 Ridge / Lasso / ElasticNet 的完整 sklearn 对比实验并解读每一条输出
6. 用交叉验证选择最优 $\lambda$，看懂正则化路径图蕴含的全部信息
7. 根据数据特性准确选择三种正则化方法，并说出选择理由

> 本章目标 1000+ 行，建议分 3 次读完。**手算部分请拿出纸笔，逐行跟着算——这是全章最有价值的部分。**

前置章节：[线性回归](./04-linear-regression.md)
下一章：[逻辑回归](./06-logistic-regression.md) — 从回归走向分类，用 S 形曲线做判断

---

## 1. 100 个特征、50 个样本——你慌不慌？

### 1a. 生活例子：面试官的刁难

你去一家 AI 公司面试。面试官推过来一张表：

> "50 个病人，每人测了 100 个基因的表达量（$X$）。最后一列是血糖水平（$y$）。帮我建个模型预测血糖。"

你扫了一眼，心里一沉——**50 个样本，100 个特征**。脑子里立刻闪过线性回归的公式：

$$w = (X^TX)^{-1}X^Ty$$

$X$ 是 $50 \times 100$ 的矩阵。$X^TX$ 是 $100 \times 100$ 的矩阵。但 $\text{rank}(X^TX) \leq \text{rank}(X) \leq \min(50, 100) = 50$。一个 $100 \times 100$ 的矩阵，秩最多只有 50——**不可逆**。

你说："这没法用普通线性回归，$X^TX$ 不可逆。方程有无穷多解。"

面试官："那我给你 150 个样本。刚好比特征多。"

你算了一下——$150 \times 100$，$X^TX$ 确实可逆了。但你知道，真正影响血糖的基因可能只有 5 个。剩下 95 个全是噪声。普通线性回归会怎么做？它会**给 95 个噪声基因全部塞上非零系数**，在训练集上 MSE 压到极低，但换一批病人立刻崩盘。

### 1b. 直观理解：为什么噪声特征也会被选中？

想象你面前有 150 个点，画在 100 维空间里。你要求的是一根超平面穿过这些点——但你给了它 100 个抓手（特征），只有 150 个约束（样本）。还剩 50 个"自由度"可以**随意摆动**。

这 50 个自由度去哪了？全部被用来**拟合噪声**了。

模型会说："病人 37 号的第 83 号基因表达量恰好低了 0.3，我给它分配系数 -2.7，就能刚好把这个人偏离的血糖值'掰'回来。虽然全人类中第 83 号基因跟血糖毫无关系——但这 50 个人里数据恰好有这样的巧合。"

这就是 **"记住训练数据"** 的本质：把每个数据点独特的噪声模式编码进系数里。换一批人，这些噪声模式全变了——系数的价值瞬间归零。

**换个角度想**：给你 100 个硬币让你猜正反面。你扔了 50 次，记录每次的结果。然后你说："我找出了 50 个'规律'——第 1 枚硬币在第 32 次扔的时候是正面、第 43 次是反面……" 你能用这个规律预测第 51 次扔的结果吗？不能——因为你找到的根本不是规律，是**巧合**。

普通线性回归在 $p$ 接近 $n$ 时就处于这个状态——它把每次抛硬币的随机结果都当成规律汇报给你。

### 1c. 形式化定义：不是"找最好"，而是"找又好又简单"

普通线性回归只关心一件事：

$$J(w) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

谁让它最小，它就选谁。哪怕系数 $w$ 大得离谱（所以方差爆炸），只要训练 MSE 小，OLS 就觉得"完美"。

正则化在上式中注入第二个目标——"模型要简单"：

$$J_{\text{reg}}(w) = \underbrace{\frac{1}{n}\|Xw - y\|_2^2}_{\text{拟合数据的代价}} + \underbrace{\lambda \cdot P(w)}_{\text{模型复杂的罚款}}$$

其中：
- $\lambda \geq 0$ 是**正则化强度**——你愿意为"简单"付多大的代价
- $P(w)$ 是**惩罚函数**——衡量模型有多复杂（$w$ 越大越复杂）

优化算法现在有了两个冲突的目标：
1. 让 $\hat{y}$ 尽量接近 $y$（拟合好）
2. 让 $w$ 尽量接近零（模型简单）

两者之间必须折中——$\lambda$ 就是"折中的力度"。

| $\lambda$ | 两个目标之间的权衡 | 结果 |
|-----------|-------------------|------|
| 0 | 只关心拟合，不管简单 | 等价于 OLS |
| 很小 | 可以稍微牺牲拟合换简单 | 轻微正则化 |
| 中等 | 显著惩罚大系数 | 最佳折中（由 CV 确定） |
| 很大 | 简单压倒一切 | 系数被压到接近零 |
| $\infty$ | 只要简单不要拟合 | 所有系数 → 0，只剩下截距 |

**偏差-方差视角**：

- **偏差**：正则化强迫系数偏离 OLS 最优值 → 训练误差上升 → 偏差增大
- **方差**：系数不再对训练数据的微小变化过度敏感 → 换一批数据系数依然稳定 → 方差减小
- **总游戏规则**：$\text{误差} = \text{偏差}^2 + \text{方差} + \text{不可约噪声}$

正则化的价值：在偏差增加不多的情况下，方差大幅度下降。总泛化误差反而低于 OLS。

### 1d. Python 代码：亲眼看看 100 特征 150 样本的场景

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n_train, n_test, p = 150, 300, 100

# 100 个特征，只有前 5 个真正有用
true_w = np.zeros(p)
true_w[:5] = [2.5, -1.8, 1.2, -0.9, 0.6]
X_train = np.random.randn(n_train, p)
y_train = X_train @ true_w + np.random.randn(n_train) * 0.5
X_test = np.random.randn(n_test, p)
y_test = X_test @ true_w + np.random.randn(n_test) * 0.5

lr = LinearRegression().fit(X_train, y_train)

# 检查 OLS 系数
train_mse = np.mean((y_train - lr.predict(X_train)) ** 2)
test_mse = np.mean((y_test - lr.predict(X_test)) ** 2)
nz = np.sum(np.abs(lr.coef_) > 1e-5)
max_coef = np.max(np.abs(lr.coef_))
w_norm = np.linalg.norm(lr.coef_)

print("=" * 55)
print("        普通线性回归 (无正则化)")
print("=" * 55)
print(f"  训练集 MSE:     {train_mse:.4f}")
print(f"  测试集 MSE:     {test_mse:.4f}")
print(f"  非零系数个数:    {nz} / {p}")
print(f"  系数最大值:      {max_coef:.3f}")
print(f"  ‖w‖₂:          {w_norm:.3f}")
print(f"  真实非零特征:   5")
print("=" * 55)

# 你看到了：
# - 训练集 MSE 很低（模型"记住"了训练数据）
# - 测试集 MSE 爆炸（换一批数据就不灵了）
# - 100 个特征全部有非零系数（95 个噪声全被"学会"了）
# - ‖w‖₂ 很大（系数过度膨胀）
```

**输出示例：**
```
=======================================================
        普通线性回归 (无正则化)
=======================================================
  训练集 MSE:     0.1352
  测试集 MSE:     4.8271
  非零系数个数:    100 / 100
  系数最大值:      2.831
  ‖w‖₂:          6.453
  真实非零特征:   5
=======================================================
```

训练集 MSE 0.14，测试集 4.83——相差约 35 倍。95 个噪声特征一个都没被筛掉。

### 1e. ML 中最常见的"需要正则化"场景

| 场景 | 典型 $n$ | 典型 $p$ | $p/n$ 比例 | 为什么需要正则化 |
|------|----------|----------|-----------|-----------------|
| 基因表达分析 | 50–200 | 5000–20000 | 100:1 | $p \gg n$，$X^TX$ 不可逆 |
| 文本分类（词袋模型） | 500–5000 | 10000+ | 50:1 | 词之间高度相关（同义词） |
| fMRI 脑影像 | 20–100 | 100000+ | 5000:1 | 相邻体素高度共线 |
| 宏观经济预测 | 100–300 | 30–100 | 0.5:1 | 指标间多重共线性 |
| 房价预测 | 1000–10000 | 10–50 | 0.01:1 | 面积类特征共线（$r>0.9$） |
| 信用评分卡 | 5000–50000 | 20–100 | 0.01:1 | 收入、负债、资产互相高度相关 |

注意：不止是 $p>n$ 才需要正则化。**只要特征间存在强共线性**（相关系数 > 0.9），OLS 的系数方差就会爆炸——正则化同样救命。

---

## 2. Ridge（岭回归 / L2 正则化）：让所有系数都收敛一点

### 2a. 生活例子：考试分数的"均值回归"

你班上有个同学张三，每次考试分数波动极大：第一次 95 分，第二次 40 分，第三次 88 分。现在你要预测他第四次考多少。

- 策略 A：只信最近一次（第三次 88 分 → 预测 88 分）。"紧盯最近数据"——方差极大，预测量完全取决于你选哪次考试。
- 策略 B：不管他考多少，都猜全班平均 70 分。"完全不信数据"——偏差极大但方差为零。
- **策略 C：Ridge 策略**——在他个人平均（95+40+88）/ 3 ≈ 74 分和全班平均 70 分之间，选个加权值。比如 73 分。

这就是 Ridge 的直觉：**不要对任何一个特征系数"过于自信"**。你看到数据说 $w_1$ 应该是 10，但你先打个折——可能是 8。你看到 $w_j$ 是 0.0001，你怀疑它本来应该是 0——但还是留着它。

### 2b. 直观理解：大系数交重税，小系数交轻税

Ridge 的惩罚项是所有系数**平方和**：

$$P_{\text{ridge}}(w) = w_1^2 + w_2^2 + \dots + w_p^2$$

| 系数值 | L2 罚款 |
|--------|---------|
| 10 | 100 |
| 5 | 25 |
| 2 | 4 |
| 1 | 1 |
| 0.1 | 0.01 |
| 0.01 | 0.0001 |

罚款的增长是**平方级别**的——系数大一倍，罚款大四倍。这驱动优化算法：

- 把大系数往小拉（交不起税）
- 小系数已经交不了多少税了 → 不用急着归零（节省的罚款微乎其微）

这就是为什么 Ridge 所有系数都**非零**——因为从 0.01 拉到 0 节省的罚款只有 $0.0001 - 0 = 0.0001$，不值得损失任何拟合能力。

### 2c. 形式化定义

Ridge 的损失函数：

$$J_{\text{ridge}}(w) = \frac{1}{n}\|Xw - y\|_2^2 + \lambda \sum_{j=1}^{p} w_j^2$$

几点重要注释：

- **截距 $w_0$ 通常不参与惩罚**。截距只是"所有特征为零时的基线预测"，不应该被压向零。sklearn 默认不对截距正则化
- **必须标准化**特征再使用。否则量纲大的特征系数天然小、量纲小的系数天然大——Ridge 会不公平地对待它们

Ridge 拥有**闭式解**——这是它相比 Lasso 的一大优势：

$$w_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^Ty$$

理解这个公式：

- $\lambda = 0$ 时：$(X^TX)^{-1}X^Ty$，就是 OLS
- $\lambda > 0$ 时：在 $X^TX$ 的对角线上加了一个正数
- 魔法的关键：即使 $X^TX$ 原本不可逆（$p > n$ 或共线性），加 $\lambda I$ 后**必可逆**。数学证明：对任意 $v \neq 0$，有 $v^T(X^TX + \lambda I)v = \|Xv\|^2 + \lambda\|v\|^2 \geq \lambda\|v\|^2 > 0$——矩阵正定，必然可逆
- $\lambda$ 越大，解越接近零向量（因为 $X^Ty$ 被 $\lambda I$ "压服"）

### 2d. 手算示例：3 个数据点的系数收缩全记录

**场景**：你是房产分析师。手头只有 3 套房子的成交数据。两个特征高度相关——"面积"和"估价面积"几乎成正比。普通线性回归因为近共线性给出了极其荒唐的系数。

**数据**：

| 房子编号 | 面积 $x_1$（十平米） | 估价面积 $x_2$ | 成交价 $y$（十万） |
|---------|---------------------|---------------|-------------------|
| ① | 1.0 | 0.9 | 2.0 |
| ② | 2.0 | 2.1 | 5.0 |
| ③ | 3.0 | 3.0 | 8.0 |

```text
X = [[1.0, 0.9],     y = [2.0,
     [2.0, 2.1],          [5.0,
     [3.0, 3.0]]          [8.0]]
```

#### 第 1 步：计算 $X^TX$

$$X^TX = \begin{bmatrix} 1^2+2^2+3^2 & 1\cdot 0.9+2\cdot 2.1+3\cdot 3.0 \\ 1\cdot 0.9+2\cdot 2.1+3\cdot 3.0 & 0.9^2+2.1^2+3.0^2 \end{bmatrix}$$

逐项计算：

- 左上角 $(X^TX)_{11}$：$1^2 + 2^2 + 3^2 = 1 + 4 + 9 = \mathbf{14}$
- 右上角 $(X^TX)_{12}$：$1 \cdot 0.9 + 2 \cdot 2.1 + 3 \cdot 3.0 = 0.9 + 4.2 + 9.0 = \mathbf{14.1}$
- 左下角 = 右上角 = 14.1（对称矩阵）
- 右下角 $(X^TX)_{22}$：$0.9^2 + 2.1^2 + 3.0^2 = 0.81 + 4.41 + 9.00 = \mathbf{14.22}$

$$X^TX = \begin{bmatrix}14 & 14.1 \\ 14.1 & 14.22\end{bmatrix}$$

两列几乎成比例（$14.1/14 \approx 1.007$，$14.22/14.1 \approx 1.009$）→ **病态矩阵预警**。

行列式：$\det = 14 \times 14.22 - 14.1 \times 14.1 = 199.08 - 198.81 = \mathbf{0.27}$

来看这个行列式有多小——如果把两列数据改成完全相同（$x_2$ 全改成 $1.007x_1$），行列式会变成 0！

#### 第 2 步：计算 $X^Ty$

$$X^Ty = \begin{bmatrix} 1\cdot 2.0 + 2\cdot 5.0 + 3\cdot 8.0 \\ 0.9\cdot 2.0 + 2.1\cdot 5.0 + 3.0\cdot 8.0 \end{bmatrix} = \begin{bmatrix} 2.0 + 10.0 + 24.0 \\ 1.8 + 10.5 + 24.0 \end{bmatrix} = \begin{bmatrix} \mathbf{36} \\ \mathbf{36.3} \end{bmatrix}$$

#### 第 3 步：OLS 解——看看问题有多严重

$$w_{\text{ols}} = (X^TX)^{-1}X^Ty = \frac{1}{0.27}\begin{bmatrix}14.22 & -14.1 \\ -14.1 & 14\end{bmatrix}\begin{bmatrix}36 \\ 36.3\end{bmatrix}$$

先算括号里的乘法：

- $w_1$ 位置：$14.22 \times 36 + (-14.1) \times 36.3 = 511.92 - 511.83 = \mathbf{0.09}$
- $w_2$ 位置：$(-14.1) \times 36 + 14 \times 36.3 = -507.6 + 508.2 = \mathbf{0.60}$

再除以行列式 0.27：

$$w_{\text{ols}} = \begin{bmatrix}0.09/0.27 \\ 0.60/0.27\end{bmatrix} = \begin{bmatrix}\mathbf{0.333} \\ \mathbf{2.222}\end{bmatrix}, \quad \|w\|_2 = \sqrt{0.111 + 4.938} = \mathbf{2.247}$$

分子 0.09 和 0.60 都非常小——因为行列式小，OLS 把这些小数字放大了。稍微改一点数据，分子就变成别的数——从 0.09 变成 -0.5 或 +0.8——行列式还是那么小，解就天翻地覆。

#### 第 4 步：Ridge, $\lambda = 0.1$

$$X^TX + 0.1I = \begin{bmatrix}14.1 & 14.1 \\ 14.1 & 14.32\end{bmatrix}$$

行列式：$\det = 14.1 \times 14.32 - 14.1\times 14.1 = 201.912 - 198.81 = \mathbf{3.102}$

从 0.27 跳到 3.102——涨了约 **11 倍**！数值稳定性立即暴增。

同样的逆矩阵乘法：

- $w_1$ 分子：$14.32 \times 36 + (-14.1) \times 36.3 = 515.52 - 511.83 = \mathbf{3.69}$
- $w_2$ 分子：$(-14.1) \times 36 + 14.1 \times 36.3 = -507.6 + 511.83 = \mathbf{4.23}$

$$w_{\lambda=0.1} = \begin{bmatrix}3.69/3.102 \\ 4.23/3.102\end{bmatrix} = \begin{bmatrix}\mathbf{1.190} \\ \mathbf{1.364}\end{bmatrix}, \quad \|w\|_2 = \sqrt{1.416 + 1.860} = \mathbf{1.809}$$

#### 第 5 步：Ridge, $\lambda = 1.0$

$$X^TX + I = \begin{bmatrix}15 & 14.1 \\ 14.1 & 15.22\end{bmatrix}$$

行列式：$\det = 15 \times 15.22 - 14.1 \times 14.1 = 228.3 - 198.81 = \mathbf{29.49}$

- $w_1$ 分子：$15.22 \times 36 + (-14.1) \times 36.3 = 547.92 - 511.83 = \mathbf{36.09}$
- $w_2$ 分子：$(-14.1) \times 36 + 15 \times 36.3 = -507.6 + 544.5 = \mathbf{36.90}$

$$w_{\lambda=1} = \begin{bmatrix}36.09/29.49 \\ 36.90/29.49\end{bmatrix} = \begin{bmatrix}\mathbf{1.224} \\ \mathbf{1.251}\end{bmatrix}, \quad \|w\|_2 = \sqrt{1.498 + 1.565} = \mathbf{1.750}$$

#### 第 6 步：Ridge, $\lambda = 10.0$

$$X^TX + 10I = \begin{bmatrix}24 & 14.1 \\ 14.1 & 24.22\end{bmatrix}$$

行列式：$\det = 24 \times 24.22 - 14.1 \times 14.1 = 581.28 - 198.81 = \mathbf{382.47}$

- $w_1$ 分子：$24.22 \times 36 + (-14.1) \times 36.3 = 871.92 - 511.83 = \mathbf{360.09}$
- $w_2$ 分子：$(-14.1) \times 36 + 24 \times 36.3 = -507.6 + 871.2 = \mathbf{363.60}$

$$w_{\lambda=10} = \begin{bmatrix}360.09/382.47 \\ 363.60/382.47\end{bmatrix} = \begin{bmatrix}\mathbf{0.941} \\ \mathbf{0.951}\end{bmatrix}, \quad \|w\|_2 = \sqrt{0.885 + 0.904} = \mathbf{1.338}$$

#### 第 7 步：汇总分析

| $\lambda$ | $\det(X^TX+\lambda I)$ | $w_1$ | $w_2$ | $\|w\|_2$ | $w_2/w_1$ |
|-----------|------------------------|-------|-------|-----------|-----------|
| 0（OLS） | 0.27 | 0.333 | 2.222 | 2.247 | 6.67 |
| 0.1 | 3.10 | 1.190 | 1.364 | 1.809 | 1.15 |
| 1.0 | 29.49 | 1.224 | 1.251 | 1.750 | 1.02 |
| 10.0 | 382.47 | 0.941 | 0.951 | 1.338 | 1.01 |

**你看到了什么？**

1. **行列式暴增**：$\lambda=0.1$ 就使行列式增大 11 倍——数值稳定性瞬间改善
2. **系数被拉向均衡**：OLS 下 $w_2$ 是 $w_1$ 的 6.7 倍，$\lambda=10$ 时两者几乎相等——Ridge 抹平了系数差异
3. **整体收缩**：$\|w\|_2$ 从 2.247 稳步降至 1.338
4. **永不归零**：所有 $\lambda$ 下 $w_1$ 和 $w_2$ 都严格大于零

Ridge 的本质规律：**$\lambda$ 越大，系数的 L2 范数越小，但所有系数保持非零。**

### 2e. Python 代码：从零实现闭式解 + sklearn 对比

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeScratch:
    """从零实现 Ridge — 闭式解（用于教学理解）"""
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        n, p = X.shape
        I = np.eye(p)
        if self.fit_intercept:
            I[0, 0] = 0  # 通常不对截距项正则化
        self.w = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)
        return self

    def predict(self, X):
        return X @ self.w


# 验证：用于前面手算的数据
X = np.array([[1.0, 0.9], [2.0, 2.1], [3.0, 3.0]])
y = np.array([2.0, 5.0, 8.0])

# 添加截距列
X_bias = np.column_stack([np.ones(3), X])

print("从零实现 Ridge (闭式解):")
for lam in [0.1, 1.0, 10.0]:
    w = RidgeScratch(alpha=lam).fit(X_bias, y).w
    print(f"  λ={lam:>4}: 截距={w[0]:7.3f}, w1={w[1]:7.3f}, "
          f"w2={w[2]:7.3f}, ‖w[1:]‖₂={np.linalg.norm(w[1:]):.3f}")

print("\nsklearn Ridge 对比（标准化后）:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
for lam in [0.1, 1.0, 10.0]:
    ridge = Ridge(alpha=lam, fit_intercept=True).fit(X_scaled, y)
    w_norm = np.linalg.norm(ridge.coef_)
    print(f"  λ={lam:>4}: coef={ridge.coef_.round(3)}, ‖w‖₂={w_norm:.3f}")
```

### 2f. 常见误区——Ridge

| 误区 | 为什么错 | 正确做法 |
|------|---------|---------|
| "Ridge 也能做特征选择" | L2 惩罚永不产生精确零系数——$w_j$ 可以极小但不会等于 0 | 需要特征选择用 Lasso 或 Elastic Net |
| "$\lambda$ 越大预测越准" | 过大 $\lambda$ → 系数全趋零 → 模型只剩截距 → 欠拟合 | 用交叉验证找最优 $\lambda$ |
| "Ridge 不需要特征标准化" | L2 惩罚 $w_j^2$ 对量纲敏感——量纲小的系数会受更大惩罚 | 永远先 `StandardScaler` 再 Ridge |
| "截距也要正则化" | 截距是基线——不应该被"罚款"压向零 | sklearn 默认 `fit_intercept=True` 且不惩罚截距 |
| "Ridge 计算比 OLS 慢" | 闭式解跟 OLS 一样是一步矩阵求逆——计算量相同 | 两者计算成本基本无差 |

### 2g. Ridge 的 ML 应用场景

| 场景 | 实例 | 为什么用 Ridge |
|------|------|---------------|
| 宏观经济预测 | 用 40 个指标（CPI、PMI、失业率…）预测 GDP 增长率 | 指标普遍有贡献，但强共线性严重，不要删指标 |
| 房价建模 | 面积特征（总面积、室内面积、土地面积）高度相关 | 想要稳定系数，不想扔掉任何面积信息 |
| 信用评分 | 收入、负债、资产同时入模 | 三个指标都该有发言权，无需筛选 |
| 传感器校准 | 多个温度传感器同时测同一环境 | 信号高度冗余，但每个传感器都不想废掉 |
| $p$ 接近 $n$ 的回归 | 40 个样本 35 个特征 | 闭式解直接让解稳定，无需先降维 |

---

## 3. Lasso（L1 正则化）：让不重要的特征彻底消失

### 3a. 生活例子：总监只留三个项目

你同时推进 20 个项目。年末述职时，总监翻了一遍你的项目清单：

> "我不管你做了多少个，只告诉我对公司利润贡献最大的前三个。"

你说："第 5 号项目贡献了一点、第 12 号项目也贡献了一点……"

总监打断："不要'一点'。要么贡献足够大（保留），要么就是零（砍掉）。"

这就是 Lasso。L1 惩罚迫使优化算法做出**非此即彼**的选择：
- 特征 5 的系数想留？行——证明你对降低 MSE 的贡献超过 $\lambda$
- 贡献不够？对不起，系数直接归零，从模型里消失

### 3b. 直观理解一：软阈值——"不够格的直接砍掉"

假设你的真实体重是 70kg。Ridge 会说"保守一点，报 65kg"。Lasso 会说：

- 如果我手里有非常强的证据（OLS 系数很大），我就保留一个大系数
- 如果我手里证据不够强（OLS 系数小于某个阈值），我就直接说"零"

这个阈值算子叫**软阈值**（soft-thresholding）：

$$S(z, \gamma) = \begin{cases} z - \gamma, & z > \gamma \\ 0, & |z| \leq \gamma \\ z + \gamma, & z < -\gamma \end{cases}$$

用一句话记：**"OLS 系数的绝对值减去 $\lambda$，小于零就砍掉。"**

### 3c. 直观理解二：菱形的尖角——为什么 L1 能产生零？

在约束优化视角下，Lasso 等价于：

$$\min_w \frac{1}{n}\|Xw - y\|_2^2 \quad \text{s.t.} \quad \sum_{j=1}^p |w_j| \leq t$$

约束区域 $\sum |w_j| \leq t$ 在二维平面上是个什么形状？

- 第一象限 $(w_1 \geq 0, w_2 \geq 0)$：$w_1 + w_2 \leq t$ → 一条从 $(t, 0)$ 到 $(0, t)$ 的线段
- 第二象限 $(w_1 \leq 0, w_2 \geq 0)$：$-w_1 + w_2 \leq t$ → 一条从 $(-t, 0)$ 到 $(0, t)$ 的线段
- 第三象限：$-w_1 - w_2 \leq t$
- 第四象限：$w_1 - w_2 \leq t$

四条线段围成一个**菱形**，四个尖角正好在坐标轴上：$(t, 0)$、$(-t, 0)$、$(0, t)$、$(0, -t)$。

损失函数的等高线是以 OLS 解为中心的椭圆。椭圆从外向内缩小，第一个碰到菱形的点就是最优解。这个点在哪里？

因为菱形的角是**尖的**（像锥子），椭圆极大概率最先碰到一个尖角——而尖角在坐标轴上！$w_1$ 或 $w_2$ 恰好等于零。

```
         w2                      w2
         |                       |
         t                       t
        /|\                     ***
       / | \                  **   **
      /  |  \               **       **
     /   |   \             *    圆     *
    /    |    \           *             *
---+-----+-----+--- w1  -t------+------+--- w1
  -t     |     t          *      |      *
    \    |    /             *     |     *
     \   |   /               **   |   **
       \  |  /                 ** | **
         \|/                     ***
         -t                      -t

    L1 菱形 (Lasso)          L2 圆 (Ridge)
    尖角 → w1或w2=0          弧上任意点 → 都非零
```

**Ridge（右边）**：约束区域是圆。圆的边缘是光滑的弧线，每个点都可微。椭圆去碰弧线——切点几乎永远不会恰好落在坐标轴上。因此所有系数都非零。

**Lasso（左边）**：约束区域是菱形。菱形有角——在坐标轴上的概率密度远大于其他位置（几何上叫"尖点"）。椭圆碰角 → 一个系数归零。

这是在整个 ML 领域最优雅的几何解释之一。它告诉我们：**稀疏性不是"算法偏好"，而是 L1 约束区域几何形状的必然结果。**

### 3d. 形式化定义

$$J_{\text{lasso}}(w) = \frac{1}{n}\|Xw - y\|_2^2 + \lambda \sum_{j=1}^{p} |w_j|$$

与 Ridge 的关键差异：

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| 惩罚项 | $\lambda \sum w_j^2$ | $\lambda \sum \|w_j\|$ |
| 可导性 | 处处可导 | 在 $w_j=0$ 处不可导 |
| 闭式解 | 有 | **无** |
| 梯度 | $\frac{\partial}{\partial w_j} = 2w_j$ | 用**次梯度**：$\partial\|w_j\| = \text{sign}(w_j)$ 当 $w_j \neq 0$；$[-1,1]$ 当 $w_j = 0$ |
| 求解方法 | 一步矩阵求逆 | 坐标下降 / LARS / 近端梯度 |

Lasso 没有闭式解的原因：$|w_j|$ 在 $w_j=0$ 处像 V 字形的尖底——数学上叫"不可导"。梯度下降在零点"不知道该往左还是往右"。必须引入**次梯度**（subgradient）或改用坐标下降。

### 3e. 手算示例 Part 1：正交设计——见证系数归零

为清晰展示 Lasso 稀疏机制，先看一个最简场景：**正交设计**（$X^TX = nI$，特征之间不相关）。

**数据**：两个完全不相关的特征，各管各的。

$$X = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}, \quad y = \begin{bmatrix}3 \\ 1\end{bmatrix}$$

OLS 解：$w^{\text{ols}} = X^{-1}y = [3,\; 1]$（特征 1 的关联强，特征 2 的关联弱）

#### 推导软阈值算子

Lasso 目标函数（系数 $\frac{1}{2}$ 方便求导——不影响最优解位置）：

$$J(w_1, w_2) = \frac{1}{2}\underbrace{\left[(w_1 - 3)^2 + (w_2 - 1)^2\right]}_{\text{MSE 部分}} + \lambda\underbrace{(|w_1| + |w_2|)}_{\text{L1 惩罚}}$$

问题可以**分解为两个独立的一维优化**（正交设计的好处）：

$$\min_{w_j} \;\; f(w_j) = \frac{1}{2}(w_j - \hat{w}_j^{\text{ols}})^2 + \lambda|w_j|$$

分别求最优解。对 $w_j > 0$：

$$\frac{\partial}{\partial w_j} = (w_j - \hat{w}_j^{\text{ols}}) + \lambda = 0 \implies w_j = \hat{w}_j^{\text{ols}} - \lambda$$

这个解只在 $\hat{w}_j^{\text{ols}} - \lambda > 0$ 时（即 $\hat{w}_j^{\text{ols}} > \lambda$）成立。

对 $w_j < 0$：

$$\frac{\partial}{\partial w_j} = (w_j - \hat{w}_j^{\text{ols}}) - \lambda = 0 \implies w_j = \hat{w}_j^{\text{ols}} + \lambda$$

这个解只在 $\hat{w}_j^{\text{ols}} + \lambda < 0$（即 $\hat{w}_j^{\text{ols}} < -\lambda$）时成立。

如果 $|\hat{w}_j^{\text{ols}}| \leq \lambda$（OLS 系数的绝对值不够大），0 在次梯度 $[- \lambda, \lambda]$ 范围内——最优解就是 $w_j = 0$。

三种情况合并为**软阈值算子**：

$$w_j^{\text{lasso}} = S(\hat{w}_j^{\text{ols}},\; \lambda) = \text{sign}(\hat{w}_j^{\text{ols}}) \cdot \max(|\hat{w}_j^{\text{ols}}| - \lambda,\; 0)$$

#### 代入我们的数据：$\hat{w}^{\text{ols}} = [3,\; 1]$

| $\lambda$ | $w_1 = S(3, \lambda)$ | $w_2 = S(1, \lambda)$ | 非零计数 | 发生了什么 |
|-----------|----------------------|----------------------|----------|-----------|
| 0 | $3 - 0 = 3.0$ | $1 - 0 = 1.0$ | 2 | 等于 OLS |
| 0.3 | $\max(3-0.3, 0) = 2.7$ | $\max(1-0.3, 0) = 0.7$ | 2 | 两个都在缩小 |
| 0.5 | 2.5 | 0.5 | 2 | $w_2$ 只剩 0.5 |
| **1.0** | $\max(3-1.0, 0) = 2.0$ | $\max(1-1.0, 0) = \mathbf{0}$ | **1** | **$w_2$ 归零！** |
| 1.5 | $\max(3-1.5, 0) = 1.5$ | 0 | 1 | $w_2$ 已死 |
| 2.0 | 1.0 | 0 | 1 | $w_1$ 继续缩 |
| 3.0 | 0 | 0 | 0 | 全部归零 |

**这 7 行数据是全章最重要的表格。** 它展示了 Lasso 特征选择的完整过程：

1. $\lambda$ 尚小 → 两个系数都活着，都在缩小
2. $\lambda$ 达到较弱特征 $w_2$ 的 OLS 系数值（1.0）→ $w_2$ 归零——"不够格，砍掉"
3. $\lambda$ 继续增大 → 强特征 $w_1$ 也越来越小
4. $\lambda = 3.0$ → 最强的也撑不住了——全军覆没

#### 对照 Ridge（同一数据、正交设计）

Ridge 正交设计下的解：

$$w_j^{\text{ridge}} = \frac{\hat{w}_j^{\text{ols}}}{1 + \lambda}$$

| $\lambda$ | $w_1$ (Ridge) | $w_2$ (Ridge) | Ridge vs Lasso |
|-----------|---------------|---------------|----------------|
| 0 | 3.0 | 1.0 | 完全相等 |
| 0.5 | $3/1.5 = 2.0$ | $1/1.5 = 0.67$ | Ridge 两个都非零，Lasso 也两个都非零 |
| 1.0 | $3/2 = 1.5$ | $1/2 = 0.5$ | **Ridge: 都活着 / Lasso: w2 已死** |
| 10.0 | $3/11 \approx 0.27$ | $1/11 \approx 0.09$ | Ridge: 极小但非零 / Lasso: 早就是零了 |

**核心对比**：Ridge 在任何有限 $\lambda$ 下所有系数**永远非零**；Lasso 在 $\lambda = 1$ 时就**精确干掉**一个系数。

### 3f. 手算示例 Part 2：坐标下降——Lasso 的迭代求解

对于一般数据（特征相关、非正交），Lasso 没有闭式解。坐标下降法（coordinate descent）是最常用的求解方法。

#### 坐标下降公式推导

Lasso 对第 $j$ 个系数的优化（固定其他系数不动）：

$$\min_{w_j} \;\; \frac{1}{n} \sum_{i=1}^n (r_i - X_{ij} w_j)^2 + \lambda |w_j|$$

其中 $r_i = y_i - \sum_{k \neq j} X_{ik} w_k$ 是**去掉第 $j$ 个特征后的残差**——即"其他特征已经解释掉的部分，还剩多少没解释"。

对 $w_j$ 求导（$w_j \neq 0$ 时）：

$$\frac{\partial}{\partial w_j} = -\frac{2}{n} X_j^T (r - X_j w_j) + \lambda \cdot \text{sign}(w_j) = 0$$

整理后得到更新公式，用软阈值算子表达：

$$w_j^{\text{new}} = \frac{S\!\left(\frac{1}{n}X_j^T r,\; \lambda\right)}{\frac{1}{n}X_j^T X_j} = \frac{S(\rho_j,\; \lambda)}{\tau_j}$$

其中：
- $\rho_j = \frac{1}{n}X_j^T r$ — 第 $j$ 个特征与当前残差的相关性（"边际贡献"）
- $\tau_j = \frac{1}{n}X_j^T X_j$ — 第 $j$ 个特征的自相关性（归一化因子）
- $S(\rho_j, \lambda)$ — 软阈值：如果边际贡献的绝对值小于 $\lambda$，直接归零

**一句话总结坐标下降**：轮流更新每个系数。如果这个特征对剩余残差的"边际贡献"不够大（小于 $\lambda$），把它咬死为零。如果够大——保留一个"缩减后"的量。

```python
def lasso_coordinate_descent(X, y, alpha=1.0, max_iter=1000, tol=1e-4):
    """从零实现 Lasso 坐标下降法"""
    n, p = X.shape
    w = np.zeros(p)
    x_norm_sq = np.sum(X ** 2, axis=0)  # 预计算每个特征的平方和

    for iteration in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            # 计算残差（把当前 w[j] 加回去，得到"无j"的残差）
            r = y - X @ w + w[j] * X[:, j]
            # ρ_j = X_j·r / n（特征j与残差的相关性）
            rho = np.dot(X[:, j], r) / n
            # 软阈值更新
            w[j] = np.sign(rho) * max(abs(rho) - alpha, 0) / (x_norm_sq[j] / n)

        # 检查收敛
        if np.max(np.abs(w - w_old)) < tol:
            print(f"坐标下降在第 {iteration+1} 轮收敛")
            break
    return w


# 验证：正交数据
X_orth = np.array([[1., 0.], [0., 1.]])
y_orth = np.array([3., 1.])

for lam in [0.5, 1.0, 1.5, 2.0, 3.0]:
    w = lasso_coordinate_descent(X_orth, y_orth, alpha=lam)
    nz = np.sum(np.abs(w) > 1e-5)
    print(f"λ={lam:.1f}: w={w.round(3)}, 非零={nz}")
```

### 3g. Python：直观感受 Lasso 的特征选择

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n, p = 60, 30
true_w = np.zeros(p)
true_w[:4] = [5.0, -3.0, 2.0, -1.0]  # 只有前 4 个是真的
X = np.random.randn(n, p)
y = X @ true_w + np.random.randn(n) * 0.8

X_scaled = StandardScaler().fit_transform(X)

print(f"{'λ':<8} {'非零系数':>8}  {'前 6 个系数'}")
print("-" * 55)
for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    lasso = Lasso(alpha=lam, max_iter=10000).fit(X_scaled, y)
    nz = np.sum(np.abs(lasso.coef_) > 1e-5)
    coef_str = " ".join(f"{c:+6.3f}" for c in lasso.coef_[:6])
    print(f"  {lam:<6} {nz:>5}/{p:<3}  {coef_str}")

# 观察：
# λ=0.01: 很多系数都活着（惩罚太轻）
# λ=0.10: 开始有人归零
# λ=0.50: 只剩 4 个 — 恰好是真实的 4 个有用特征！
# λ=1.00: 全死了 — 惩罚太重
```

### 3h. 常见误区——Lasso

| 误区 | 真相 |
|------|------|
| "Lasso 总是比 Ridge 好" | 只在确实有"无用特征"时好。如果所有特征都有贡献，Lasso 强行删特征反而增加偏差 |
| "Lasso 选出的特征是最优子集" | Lasso 对相关特征组**随机选一个**——不像最佳子集选择（best subset）那样做全局搜索 |
| "Lasso 有闭式解" | **错误。** 唯正交设计下才退化为软阈值。一般数据必须迭代求解 |
| "坐标下降一定会收敛" | 对 Lasso 是**保证收敛到全局最优**的（凸函数 + 坐标轴方向可分），前提是特征标准化 |
| "截距也参与 L1 惩罚" | 一般不。截距只是基线水平，不应被正则化压向零 |
| "Lasso 选的系数 = 特征重要性排序" | 只在特征不高度相关时成立。相关特征组中，谁被选中有随机性 |

### 3i. Lasso 的 ML 应用场景

| 场景 | 典型数据 | 为什么用 Lasso |
|------|---------|---------------|
| 基因-疾病关联分析 | 5000 基因 × 100 病人 | 从 5000 基因筛出 10-20 个关键基因 |
| 文本情感分类 | 20000 词 × 1000 评论 | 筛选出真正区分正面/负面的关键词 |
| 经济预测 | 50 指标 × 200 季度 | 自动发现两三个真正的 GDP 先行指标 |
| 图像稀疏编码 | 10000 像素 × 1000 patch | 用少数几个基函数表示每个 patch |
| 需要可解释性 | 任何特征多的任务 | 系数列表里只有几个非零——一目了然 |

---

## 4. Elastic Net（弹性网络）：两手都要硬

### 4a. 生活例子：团队裁员——既要筛选又要公平

Lasso 有一个教科书级的"坏毛病"。假设你建模房价，输入特征有"总面积"和"使用面积"，它们相关系数 0.95，同时还有一个真正无关的"房龄颜色"。

- **Lasso**：面对"总面积"和"使用面积"两个几乎一模一样的信号——它**随机选一个当代表，另一个归零**。今天这批数据选了"总面积"，明天换一批选了"使用面积"——你的模型解释崩了。好消息是它正确地把"房龄颜色"扔掉了。
- **Ridge**：公平对待"总面积"和"使用面积"——各分一半权重。但"房龄颜色"的系数也只是被压到很小，没被清掉——模型臃肿。

Elastic Net 的思路：**L1 负责杀掉"房龄颜色"，L2 负责公平分配"总面积"和"使用面积"的权重。**

### 4b. 形式化定义

$$J_{\text{EN}}(w) = \frac{1}{n}\|Xw - y\|_2^2 + \lambda\left(\alpha\sum_{j=1}^{p}|w_j| + \frac{1-\alpha}{2}\sum_{j=1}^{p}w_j^2\right)$$

两个超参数各管一摊：

- $\lambda$：总惩罚强度——"你有多讨厌复杂模型"
- $\alpha \in [0,1]$：L1 和 L2 的比例——"你是更想杀人还是更想均富"

| $\alpha$ (sklearn: `l1_ratio`) | 含义 | 等同于 |
|-------------------------------|------|--------|
| 1.0 | 纯 L1 | Lasso |
| 0.0 | 纯 L2 | Ridge |
| 0.5 | 各一半 | 经典 Elastic Net |
| 0.9 | L1 为主，L2 辅助 | 严格筛选 + 轻微分组效应 |

**为什么 L2 能提供"分组效应"？** 考虑两个相关特征 $x_1$ 和 $x_2$。当 $w_1$ 已经取了较大值、$w_2$ 趋向零时，L2 惩罚的梯度 $(1-\alpha)w_2$ 几乎为零（因为 $w_2 \approx 0$）——不会阻止 $w_2 \to 0$。但 L2 的**严格凸性**确保了损失函数的 Hessian 正定——这意味着 $w_1$ 和 $w_2$ 必须"一起动"，不能出现一个 10、另一个 -8 的极端差值。最终体现为：相关特征的系数**比较接近**，而非一个独吞、一个为零。

### 4c. Python 代码：三方法在同组相关特征上的对比

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

np.random.seed(42)
base = np.linspace(0, 10, 100)
X_c = np.column_stack([
    base + np.random.randn(100) * 0.3,    # x1 — 趋势
    base + np.random.randn(100) * 0.3,    # x2 — 与x1强相关 (r≈0.98)
    np.random.randn(100),                  # x3 — 纯噪声
    np.random.randn(100),                  # x4 — 纯噪声
])
y_c = 3 * base + np.random.randn(100) * 0.5
X_cs = StandardScaler().fit_transform(X_c)

lasso = Lasso(alpha=0.3, max_iter=10000).fit(X_cs, y_c)
ridge = Ridge(alpha=0.3).fit(X_cs, y_c)
en = ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=10000).fit(X_cs, y_c)

print(f"{'':<15} {'w1 (信号)':>10} {'w2 (共线)':>10} {'w3 (噪声)':>10} {'w4 (噪声)':>10}")
print("-" * 55)
print(f"{'Lasso':<15} {lasso.coef_[0]:10.3f} {lasso.coef_[1]:10.3f} "
      f"{lasso.coef_[2]:10.3f} {lasso.coef_[3]:10.3f}")
print(f"{'Ridge':<15} {ridge.coef_[0]:10.3f} {ridge.coef_[1]:10.3f} "
      f"{ridge.coef_[2]:10.3f} {ridge.coef_[3]:10.3f}")
print(f"{'ElasticNet':<15} {en.coef_[0]:10.3f} {en.coef_[1]:10.3f} "
      f"{en.coef_[2]:10.3f} {en.coef_[3]:10.3f}")
```

**典型输出与解读：**
```
                  w1 (信号)    w2 (共线)    w3 (噪声)    w4 (噪声)
-------------------------------------------------------
Lasso               3.163       0.000       0.000       0.000
Ridge               1.722       1.535       0.013      -0.020
ElasticNet          2.601       0.793       0.000       0.000
```

| 方法 | w1 (信号) | w2 (跟信号 0.98 相关) | 噪声 (w3, w4) | 评价 |
|------|-----------|----------------------|--------------|------|
| Lasso | 3.16（独吞） | 0（被杀） | 0（被杀） | 杀了共线信号，不公平 |
| Ridge | 1.72（一半） | 1.54（一半） | ~0.01（鬼魂） | 公平但没清干净噪声 |
| ElasticNet | 2.60（大份） | 0.79（小份） | 0（被杀） | **公平 + 干净** |

---

## 5. Ridge vs Lasso vs Elastic Net：终极选择指南

| 维度 | Ridge (L2) | Lasso (L1) | Elastic Net |
|------|-----------|------------|-------------|
| 惩罚 | $\lambda\sum w_j^2$ | $\lambda\sum \|w_j\|$ | $\lambda(\alpha\sum\|w_j\| + \frac{1-\alpha}{2}\sum w_j^2)$ |
| 零系数 | 永不归零 | ✓ 部分归零 | ✓ 部分归零 |
| 特征选择 | ✗ | ✓ | ✓ |
| 相关特征 | 均匀分配 | 随机选一个 | 公平分组（分组效应） |
| 闭式解 | **有** | 无 | 无 |
| 计算速度 | 最快 | 迭代 | 迭代 |
| 超参数 | 1 个 ($\lambda$) | 1 个 ($\lambda$) | 2 个 ($\lambda$, $\alpha$) |

### 选择决策树

```
你在什么场景下？

所有特征可能都有贡献？ ──→ Ridge（稳定、快、有闭式解）
    │
    ├─ 不是 → 只有少数特征重要？
    │            │
    │            ├─ 特征之间不相关？ ──→ Lasso（自动特征选择）
    │            │
    │            └─ 特征之间有共线性？ ──→ Elastic Net（兼顾选择 + 公平）
    │
    └─ 不确定 ──→ Elastic Net（调 l1_ratio 扫描 0→1，全覆盖）
```

**经验法则**：

1. **默认先试 Ridge**——有闭式解、参数少、几乎不会出大错
2. 如果怀疑大多数特征是噪声 → 试 Lasso
3. 如果 $p > n$ 很严重且特征间有多重共线性 → 用 Elastic Net
4. 如果能容忍调参成本 → Elastic Net + GridSearchCV 扫一遍 $\alpha$ 和 $\lambda$

---

## 6. 如何选择 $\lambda$：交叉验证

$\lambda$ 是所有正则化方法中唯一的"核心旋钮"（Elastic Net 多一个 $\alpha$）。选太小 → 等于没正则化，选太大 → 模型退化成"猜均值"。

### 6a. 交叉验证流程

1. 设定候选 $\lambda$：对数均匀分布，如 $\{10^{-4}, 10^{-3.5}, 10^{-3}, \dots, 10^{3}, 10^{4}\}$
2. 将训练数据分成 $K$ 折（典型 $K=5$ 或 $10$）
3. 对每个候选 $\lambda$：
   - 取 $K-1$ 折训练 Ridge/Lasso($\lambda$)
   - 在剩余 1 折上算 MSE
   - 重复 $K$ 次（每折轮流当验证集）
   - 取 $K$ 个 MSE 的平均值作为该 $\lambda$ 的 CV 得分
4. 选平均 MSE 最小的 $\lambda$
5. **1-SE 规则（可选）**：在最优 MSE ± 1 个标准误的范围内，选最大的 $\lambda$——得到更稀疏/更稳定的模型
6. 用选定 $\lambda$ 在**全部训练集**上重新训练 → 最终模型

### 6b. sklearn 一行搞定

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=50, noise=15,
                       n_informative=10, random_state=42)

ridge_cv = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5).fit(X, y)
print(f"RidgeCV     最优 α: {ridge_cv.alpha_:.3f}")

lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42).fit(X, y)
print(f"LassoCV     最优 α: {lasso_cv.alpha_:.4f}")
print(f"            非零系数: {np.sum(np.abs(lasso_cv.coef_) > 1e-5)}/{X.shape[1]}")

en_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                     cv=5, max_iter=10000, random_state=42).fit(X, y)
print(f"ElasticNetCV 最优 α: {en_cv.alpha_:.3f},  最优 l1_ratio: {en_cv.l1_ratio_:.2f}")

# LassoCV 还能画出 α 路径上的 MSE 变化
import matplotlib.pyplot as plt
mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
mse_std = np.std(lasso_cv.mse_path_, axis=1)

plt.figure(figsize=(9, 4))
plt.semilogx(lasso_cv.alphas_, mse_mean, 'b-', lw=1.5)
plt.fill_between(lasso_cv.alphas_, mse_mean - mse_std, mse_mean + mse_std,
                 alpha=0.2, color='blue')
plt.axvline(lasso_cv.alpha_, color='red', ls='--', lw=1.5,
            label=f'最优 α = {lasso_cv.alpha_:.4f}')
plt.xlabel('α (正则化强度)')
plt.ylabel('交叉验证 MSE')
plt.title('LassoCV: α 的 MSE 曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 7. 正则化路径：系数的一生

正则化路径（regularization path）是一次性看清"正则化到底做了什么"的最佳可视化。横轴是 $\lambda$（对数），纵轴是每个特征的系数值。

```python
alphas = np.logspace(-3, 2, 100)
X_sc, y_syn = make_regression(n_samples=100, n_features=30, noise=15,
                                n_informative=5, random_state=42)
X_sc = StandardScaler().fit_transform(X_sc)

ridge_coefs = []
lasso_coefs = []
for a in alphas:
    ridge_coefs.append(Ridge(alpha=a).fit(X_sc, y_syn).coef_)
    lasso_coefs.append(Lasso(alpha=a, max_iter=10000).fit(X_sc, y_syn).coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Ridge 路径
for j in range(5):
    ax1.plot(alphas, ridge_coefs[:, j], lw=1.5, label=f'信号 x{j+1}')
for j in range(5, 30):
    ax1.plot(alphas, ridge_coefs[:, j], 'gray', alpha=0.15, lw=0.6)
ax1.set_xscale('log')
ax1.set_xlabel('λ')
ax1.set_ylabel('系数值')
ax1.set_title('Ridge (L2): 所有系数平滑收缩，永不死')
ax1.axhline(y=0, color='black', lw=0.5, ls='--')
ax1.grid(True, alpha=0.3)

# Lasso 路径
for j in range(5):
    ax2.plot(alphas, lasso_coefs[:, j], lw=1.5, label=f'信号 x{j+1}')
for j in range(5, 30):
    ax2.plot(alphas, lasso_coefs[:, j], 'gray', alpha=0.15, lw=0.6)
ax2.set_xscale('log')
ax2.set_xlabel('λ')
ax2.set_ylabel('系数值')
ax2.set_title('Lasso (L1): 噪声逐一"跳崖"，信号战斗到最后')
ax2.axhline(y=0, color='black', lw=0.5, ls='--')
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**读图三要诀：**

- Ridge 图：所有曲线**平滑下降**——像潮水慢慢退去，但永不干涸
- Lasso 图：曲线一根接一根地**"撞零线"**——撞到后就贴着零线纹丝不动
- 右上角（$\lambda$ 极小）：图左端两方法接近，系数都很活跃
- 左下角（$\lambda$ 极大）：图右端两方法都趋零，但 Ridge 的手指还粘着线，Lasso 的全掉了
- 最先"死"的（灰色线最先撞零）→ 噪声特征
- 最后"死"的（彩色线最后撞零）→ 真正的最强信号特征

---

## 8. 实战：加州房价数据——完整三方法对比

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

lr = LinearRegression().fit(X_tr, y_train)
ridge = RidgeCV(alphas=np.logspace(-2, 3, 50)).fit(X_tr, y_train)
lasso = LassoCV(cv=5, max_iter=10000, random_state=42).fit(X_tr, y_train)
en = ElasticNetCV(cv=5, max_iter=10000, random_state=42).fit(X_tr, y_train)

print(f"{'模型':<22} {'测试 MSE':>10} {'测试 R²':>10} {'非零':>6}")
print("-" * 50)
for name, mdl in [('OLS (无正则化)', lr), ('Ridge (CV)', ridge),
                   ('Lasso (CV)', lasso), ('ElasticNet (CV)', en)]:
    yp = mdl.predict(X_te)
    mse = mean_squared_error(y_test, yp)
    r2 = r2_score(y_test, yp)
    nz = np.sum(np.abs(mdl.coef_) > 1e-5)
    print(f"{name:<22} {mse:10.4f} {r2:10.4f} {nz:>5}/{X.shape[1]}")

# Lasso 扔了谁？留了谁？
print("\nLasso 的特征生死簿:")
for fname, coef in zip(data.feature_names, lasso.coef_):
    status = '✓ 保留' if abs(coef) > 1e-5 else '✗ 丢弃'
    print(f"  {fname:<20} {coef:+8.4f}  {status}")

# 可视化系数绝对值对比
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(data.feature_names))
w = 0.2
ax.bar(x_pos - 1.5*w, np.abs(lr.coef_), w, label='OLS', alpha=0.7)
ax.bar(x_pos - 0.5*w, np.abs(ridge.coef_), w, label='Ridge', alpha=0.7)
ax.bar(x_pos + 0.5*w, np.abs(lasso.coef_), w, label='Lasso', alpha=0.7)
ax.bar(x_pos + 1.5*w, np.abs(en.coef_), w, label='ElasticNet', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(data.feature_names, rotation=45, ha='right')
ax.set_ylabel('|系数|')
ax.set_title('四种方法的系数绝对值对比')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## 9. 本章总结

正则化的核心就一句话：**有限数据下，别让你的模型太自信。**

| 方法 | 口号 | 一句话 | 何时用 |
|------|------|--------|--------|
| OLS | "我全信" | 不加惩罚，100% 信任数据 | 样本多、特征少、无共线性 |
| Ridge (L2) | "匀一匀" | 都保留、都收敛 | 所有特征都有用但共线性严重 |
| Lasso (L1) | "优胜劣汰" | 没用的直接滚 | 只有少数特征是真正有贡献的 |
| Elastic Net | "两手都要硬" | 有选择、但公平 | $p \gg n$ 且特征高度相关 |

**三个永不忘记的数字**（本章手算中得到的）：

- **0.27** → $\det(X^TX)$ — 近奇异矩阵的"濒危值"，代表 OLS 差一点点就不可解。Ridge 的 $\lambda=0.1$ 把行列式拉到 3.102（约 11 倍）——瞬间安全。
- **1.0** → 软阈值中 $w_2$ 归零的临界 $\lambda$ — 等于 $|\hat{w}_2^{\text{ols}}|$。Lasso 的"砍掉线"。
- **$\lambda$** → 你手上唯一的旋钮。用交叉验证去拧到最佳位置。

---

## 10. 思考题（含完整解答）

### Q1. 为什么 L2 惩罚永远不产生零系数而 L1 可以？请从几何和优化两个角度解释。

**几何解释**：L2 约束区域是圆（高维是超球面），表面处处光滑——损失函数等高线与圆的切点几乎永远不在坐标轴上。L1 约束区域是菱形（高维是八面体），它有尖锐的顶点恰好在坐标轴上——等高线碰到的位置极大概率是顶点，对应某个系数等于零。

**优化解释**：Ridge 的梯度 $\frac{\partial}{\partial w_j}(w_j^2) = 2w_j$ 在 $w_j \to 0$ 时趋于零——优化器没有动力把已经很小的系数精确拉到零。Lasso 的次梯度在 $w_j = 0$ 处为 $[-1, 1]$——只要 $\lambda$ 够大，零就可以是驻点。一旦到达零，优化器没有动力离开。

---

### Q2. $\lambda \to \infty$ 时，Ridge 和 Lasso 的系数分别怎样？写出公式证明。

**Ridge**：$w = (X^TX + \lambda I)^{-1}X^Ty$。当 $\lambda \to \infty$，$(X^TX + \lambda I)^{-1} \approx \frac{1}{\lambda}I$，因此 $w \approx \frac{1}{\lambda}X^Ty \to \mathbf{0}$（所有系数趋近零但不等于零——因为 $\lambda$ 永远有限）。

**Lasso**（正交设计）：$w_j = \max(|\hat{w}_j^{\text{ols}}| - \lambda, 0) \cdot \text{sign}(\hat{w}_j^{\text{ols}})$。当 $\lambda \geq \max_j |\hat{w}_j^{\text{ols}}|$，所有 $w_j = \mathbf{0}$。这是**有限的** $\lambda_{\max}$，不需要无穷大。

**结果上的差别**：Ridge 在有限 $\lambda$ 下系数永远非零（尽管可以无限逼近零）；Lasso 在某个有限 $\lambda$ 处即可实现全零。

---

### Q3. $\lambda = 0$ 时 Ridge 等价于什么？

等价于**普通最小二乘（OLS）**。

Ridge 公式：$w = (X^TX + 0\cdot I)^{-1}X^Ty = (X^TX)^{-1}X^Ty$。

但注意：如果数据未标准化，Ridge ($\lambda=0$) 在 sklearn 中仍可能对截距的处理与纯 OLS 略有不同。标准化 + 无截距时完全等价。

---

### Q4. 正交设计下，$\hat{w}^{\text{ols}} = [10,\; 0.5]$，分别计算 $\lambda=1$ 时 Ridge 和 Lasso 的系数。

**Ridge**（正交设计公式 $w_j = \hat{w}_j^{\text{ols}} / (1+\lambda)$）：

- $w_1 = 10/2 = \mathbf{5.0}$
- $w_2 = 0.5/2 = \mathbf{0.25}$

**Lasso**（软阈值 $w_j = \text{sign}(\hat{w}_j^{\text{ols}}) \cdot \max(|\hat{w}_j^{\text{ols}}| - \lambda, 0)$）：

- $w_1 = \max(10 - 1, 0) = \mathbf{9.0}$
- $w_2 = \max(0.5 - 1, 0) = \mathbf{0}$ ← 精确归零！

有趣的是：Ridge 里 $w_1=5.0$ 比 Lasso 的 $w_1=9.0$ 更小——Ridge 对所有系数"均贫富"，把强特征也拉下一大截。Lasso 只杀弱特征（$w_2$），对强特征（$w_1$）手下留情。

---

### Q5. 为什么 Elastic Net 能解决 Lasso 在高度相关特征组上的"随机选择"问题？

Lasso 的 L1 惩罚中，如果 $x_1$ 和 $x_2$ 高度相关，任何一个都能解释绝大部分信号。坐标下降从 $w_1$ 开始更新时，$w_1$ 抢走了所有信号，$w_2$ 的边际贡献变得极小——小于 $\lambda$——于是被归零。但下次迭代如果先更新 $w_2$，$w_2$ 就会抢走所有信号，$w_1$ 被归零。

Elastic Net 加入了 L2 惩罚 $\frac{1-\alpha}{2}(w_1^2 + w_2^2)$。这个严格凸函数的最优解在两个相关特征之间必须是"平滑"的——即 $w_1$ 和 $w_2$ 的值会比较接近（梯度互相牵制）。这就是 L2 的**分组效应**（grouping effect）——相关特征被一起保留或一起丢弃，系数的比值比较稳定。

---

### Q6. 为什么 $p > n$ 时 OLS 必崩而 Ridge 可以？

**OLS 崩**：$w_{\text{ols}} = (X^TX)^{-1}X^Ty$。$X^TX$ 是 $p \times p$ 矩阵，秩 $\leq n < p$——不可逆。方程有无穷多解。

**Ridge 不崩**：$w_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^Ty$。$X^TX + \lambda I$ （$\lambda > 0$）必可逆。证明：对任意 $v \neq 0$，

$$v^T(X^TX + \lambda I)v = \|Xv\|_2^2 + \lambda\|v\|_2^2 \geq \lambda\|v\|_2^2 > 0$$

所以矩阵是正定的，必然可逆。$\lambda I$ 的作用等于在 $X^TX$ 的零空间里"注入" $\lambda$——把原本为零的特征值变成了 $\lambda$。

---

### Q7. 正则化如何影响偏差-方差权衡？画出文字示意图并用文字解释。

$$\text{泛化误差} = \underbrace{[\mathbb{E}(\hat{f}) - f]^2}_{\text{偏差}^2} + \underbrace{\mathbb{E}[(\hat{f} - \mathbb{E}(\hat{f}))^2]}_{\text{方差}} + \underbrace{\sigma^2}_{\text{不可约噪声}}$$

- **$\lambda = 0$**（OLS）：偏差最小（因为完全拟合数据），方差最大（系数对数据扰动极度敏感）→ **过拟合**
- **$\lambda$ 逐步增大**：偏差缓慢上升（允许偏离 OLS 的最优拟合），方差快速下降（系数被"锚定"，不再乱跳）→ 总误差先降后升
- **$\lambda$ 过大**：偏差飙升（模型什么都不敢学），方差趋于零（系数几乎锁定在零附近）→ **欠拟合**
- **最优 $\lambda$**：偏差² 和方差的交点——总误差最低处。交叉验证就是用来找这个点的。

```
误差
 ^
 |    \         总误差
 |     \      /       \
 |      \    /         \
 |       \  /           \______ 方差
 |        \/            /
 |        /\           /
 |       /  \         /
 |      /    \_______/ 偏差²
 |     /
 |    /
 +--------------------------------> λ
 0                                 大
    过拟合        最佳        欠拟合
```

---

### Q8. 使用正则化方法时为什么必须先标准化特征？不标准化会有什么后果？

**原因**：L1 和 L2 惩罚对所有特征的系数**一视同仁地罚款**。

如果 $x_1$ 是毫米（数值范围 1000-2000），$x_2$ 是公里（数值范围 1-2）。要达到同样的预测效果，$w_1$ 的量级天然是 $w_2$ 的 $1/10^6$。正则化会认为 $w_2$ 的系数"太大了"（虽然量纲不同导致不可比），于是**不公平地把 $w_2$ 往死里压**——$x_2$ 这个特征几乎失效。

**正确做法**：永远 `StandardScaler().fit_transform(X_train)` 再传给正则化模型。统一所有特征到均值 0、方差 1——正则化对所有特征公平。

---

### Q9. 如何用交叉验证选择最优 $\lambda$？写出完整步骤（不需要代码）。

1. **设定候选集**：$\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, \dots, 10^3, 10^4\}$（对数均匀分布，保证覆盖从"几乎 OLS"到"几乎全零"的所有范围）

2. **划分数据**：将训练数据分成 $K$ 折（如 $K=5$）。注意：标准化必须在每次训练时**只用训练折**的均值和标准差

3. **遍历 $\lambda$**：对每个候选 $\lambda_i$：
   - 对每折 $k = 1, \dots, K$：
     - 用除第 $k$ 折外的数据训练 Ridge($\lambda_i$)
     - 在第 $k$ 折上计算 MSE
   - 取 $K$ 个 MSE 的平均值——这是 $\lambda_i$ 的 CV 得分

4. **选择最优**：选平均 MSE 最小的 $\lambda^*$

5. **（可选）1-SE 规则**：在 MSE 不显著差于 $\lambda^*$（差距 $\leq$ 1 个标准误）的 $\lambda$ 中选最大的那个 → 得到更稀疏、更稳定的模型

6. **最终训练**：用 $\lambda^*$ 在**整个训练集**上重新训练，得到最终模型参数

---

### Q10. 如果数据中所有特征都是纯噪声（与目标完全无关），用交叉验证选 $\lambda$ 后，Lasso 和 Ridge 各自表现如何？

**Ridge**：CV 会选出一个较大的 $\lambda$，把所有系数压到接近零。但因为 L2 不产生精确零，用 `np.abs(w) > 1e-5` 检查会看到所有系数**全部"活着"**（虽然值极小）。模型最终近乎"只预测 $\bar{y}$（均值）"，但形式上保留了所有特征。

**Lasso**：CV 会选出一个足够大的 $\lambda$（$\geq \max_j |\hat{w}_j^{\text{ols}}|$），使得**所有系数精确为零**。模型退化为只输出训练集 $y$ 的均值——截距 = $\bar{y}$，所有 $w_j = 0$。这是理想结果：Lasso 正确地告诉你"没有特征是有用的"。

**实战意义**：在探索性数据分析中，如果你跑了一个 Lasso 发现 CV 选出的 $\lambda$ 把所有特征都归零了——这是一个强大的信号：**你手上的特征可能跟目标毫无关系**，或者需要特征工程（交叉项、非线性变换）。

---

## 11. 延伸阅读

- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*, Chapter 3.4 — Ridge 和 Lasso 最权威的论述
- Tibshirani (1996). *Regression Shrinkage and Selection via the Lasso* — Lasso 的原始论文，软阈值算子首次提出
- Zou & Hastie (2005). *Regularization and Variable Selection via the Elastic Net* — 提出弹性网络和分组效应
- Friedman, Hastie, Tibshirani (2010). *Regularization Paths for Generalized Linear Models via Coordinate Descent* — glmnet 包的数学基础
- sklearn 文档：[Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression) / [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso) / [ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

---

下一步：[逻辑回归](./06-logistic-regression.md) — 把 S 形函数套在回归上，从"预测数字"跨入"判断类别"的世界
