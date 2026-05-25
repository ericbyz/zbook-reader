# 03 — 数据预处理：机器学习的第一步

> **核心问题**：你再好的算法，喂给它垃圾数据，也只能产出垃圾结果。数据预处理就是把"生数据"变成"熟数据"的全过程。

---

## 0. 本章导览

这一章是 ML 实践的第一个分水岭——前面的数学基础告诉你"算法怎么算"，这一章告诉你"数据怎么给"。学过本章后，你将能够：

1. 识别五种缺失值类型，并选择正确的填充策略
2. 手算标准化（μ 和 σ），理解 StandardScaler 内部到底做了什么
3. 为一个包含数值和分类特征的混合数据集手写完整的 X 矩阵变换过程
4. 用 IQR 法检测异常值，并决定删、缩、还是留
5. 正确地做 train/val/test 划分，避免最常见的五种数据泄露
6. 搭建一个防泄露的完整 sklearn Pipeline
7. 用交叉验证可靠地评估模型，不再被一次随机划分误导

> 本章目标 1200+ 行，建议分 2 次读完。所有手算示例请准备好纸笔跟着算。

---

## 1. 垃圾进，垃圾出：为什么数据预处理是 ML 的第一关

### 1a. 生活例子：做菜前的备料

你走进厨房要做一道菜，打开冰箱看到：西兰花蔫了、猪肉快过期了、葱上还带着泥。你会把这些东西直接扔进锅里吗？

不会。你会做这些事：**扔掉烂叶子（删除异常值）、把菜洗干净（处理缺失值）、切成一样大小（特征缩放）、把酱油和盐分开用（分类编码）**。做完这些，食材才能下锅。

数据预处理就是 ML 的"洗菜、切菜"阶段。80% 的时间花在这一步，20% 的时间建模——这不是浪费时间，而是花在最该花的地方。

### 1b. 直观理解：数据里有五种"脏东西"

你打开一份典型的真实数据，会看到什么？

| 脏东西 | 例子 | 不处理直接用会怎样 |
|--------|------|-------------------|
| **缺失值** | 年龄列有 10% 空着 | 大多数算法报错（NaN 无法计算） |
| **量纲悬殊** | 身高 1.7m vs 年收入 200000 元 | KNN、K-Means 被收入主导，身高完全没用 |
| **文字类别** | 城市列写着"北京""上海" | 数学公式只吃数字，不吃汉字 |
| **极端异常值** | 某个用户消费 999999 元 | 线性回归被一个点拉歪整条线 |
| **数据泄露** | 标准化时用了测试集的 μ 和 σ | 你在训练时偷看了答案，模型在真实场景必崩 |

### 1c. 形式化定义

数据预处理是将原始数据矩阵 $\mathbf{D}_{\text{raw}}$ 通过一系列变换函数 $f_1, f_2, \dots, f_n$ 转化为规范化矩阵 $\mathbf{D}_{\text{clean}}$ 的过程：

$$
\mathbf{D}_{\text{clean}} = f_n(\dots f_2(f_1(\mathbf{D}_{\text{raw}}))\dots)
$$

每个 $f_i$ 是一个 $Transformer$：接收数据输入，输出变换后的数据。它们的共同规则是：

- `fit(X_train)`：只在训练集上学习变换参数（如均值 $\mu$、标准差 $\sigma$、类别列表）
- `transform(X_test)`：用学到的参数应用到新数据，绝不对新数据重新学习

这条规则是你理解"数据泄露"的钥匙——后面会反复用到。

### 1d. 手算示例：看一眼脏数据的破坏力

给你一个极小的数据集，感受量纲对距离计算的控制：

```
三套房子的数据：
   面积(m²)    房价(万元)
A:   50         80
B:   70         100
C:   100        300
```

纯按面积：C 和 B 差 30m²，C 和 A 差 50m²。A 和 C 最不像。
纯按房价：C 和 B 差 200 万，C 和 A 差 220 万。结果近似但尺度完全不同。

如果把两个特征直接喂给 KNN（用欧氏距离）：

| 比较 | 面积差 | 房价差 | 欧氏距离 |
|------|--------|--------|---------|
| A vs B | 20 | 20 | $\sqrt{20^2 + 20^2} = 28.3$ |
| A vs C | 50 | 220 | $\sqrt{50^2 + 220^2} = 225.6$ |
| B vs C | 30 | 200 | $\sqrt{30^2 + 200^2} = 202.2$ |

房价差（200-ish）完全压倒了面积差（20-50）。**如果不做缩放，你的 KNN 实际上只看房价，面积等于白给了。**

### 1e. Python 验证

```python
import numpy as np
import pandas as pd

# 构造一份典型的"脏数据"，让你亲眼看看问题
np.random.seed(42)
n = 1000
raw_data = pd.DataFrame({
    'age':        np.random.choice([25, 30, 35, 40, 45, np.nan], n,
                                   p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1]),
    'income':     np.random.exponential(50000, n).astype(int),
    'city':       np.random.choice(['北京', '上海', '广州', '深圳'], n),
    'gender':     np.random.choice(['男', '女', 'unknown'], n,
                                   p=[0.48, 0.48, 0.04]),
    'purchase':   np.random.lognormal(5, 1.5, n),
})
raw_data.loc[::50, 'income'] *= 100

print(f"形状: {raw_data.shape}")
print(f"\n缺失值统计:\n{raw_data.isnull().sum()}")
print(f"\n数值范围:\n{raw_data.describe().loc[['min','max','mean']].round(1)}")
print(f"\n城市唯一值: {raw_data['city'].unique()}")
print(f"性别唯一值: {raw_data['gender'].unique()}")
```

输出会告诉你：
- `age` 有约 10% 的缺失值
- `income` 的 max 是 min 的上万倍
- `city` 和 `gender` 是文字，模型没法直接算

### 1f. 常见误区

**误区：「我用的 XGBoost，它自动处理缺失值，不需要预处理」**

❌ 半错。XGBoost/LightGBM 确实有内置的缺失值处理，但只限于训练阶段——你部署时如果有新的缺失模式，后果不确定。而且，树模型只解决了缺失值和异常值的问题，不解决量纲缩放（虽然树不需要）和分类编码（你仍然需要把"北京"转成数字）。预处理是全流程的需要，不是某个算法的需要。

**误区：「预处理就是随便填个均值、标准化一下就行了」**

❌ 每个预处理决策都在做假设。你填均值，就是假设数据是正态的；你删缺失行，就是假设缺失是随机的。这些假设如果不成立，你引入了偏差而不自知。

### 1g. ML 应用连接

预处理的质量是一个 ML 项目的天花板。一个常见的经验数字：**在 Kaggle 竞赛中，特征工程和预处理带来的提升往往超过换模型。** 你花三天时间把数据弄干净、特征弄对，比调三天超参数有用得多。

---

## 2. 处理缺失值：数据中的"洞"

### 2a. 生活例子：填一张缺了信息的身高表

你是体育老师，手里有一张学生体质表：

| 姓名 | 身高(cm) | 体重(kg) | 跳远(m) |
|------|----------|----------|---------|
| 张三 | 170      | 65       | 2.1     |
| 李四 | 165      | ?        | 1.8     |
| 王五 | ?        | 70       | 2.3     |
| 赵六 | 180      | 80       | ?       |

你现在要统计"体能综合分 = 身高 × 0.3 + 体重 × 0.3 + 跳远 × 0.4"。你怎么处理问号？

- **直接删除？** 李四、王五、赵六全删，只剩张三一行——数据损失了 75%。
- **填平均值？** 已知身高平均 (170+165+180)/3 = 171.7，给王五填 171.7。但王五如果其实是全班最矮的（所以他不好意思填），你就高估了他。
- **填中位数？** 身高中位数 = 170。如果王五确实是 150cm，你错得少一点（差 20 vs 差 21.7），但本质上还是猜。

没有完美的答案——你只能在"信息损失"和"引入偏差"之间权衡。

### 2b. 直观理解：三种"缺"的原因

缺失背后有三种不同的原因，决定了你该怎么处理：

| 缺失类型 | 英文 | 通俗解释 | 例子 |
|----------|------|----------|------|
| 完全随机缺失 | MCAR | 缺不缺和谁都没关系，纯随机 | 传感器偶然断电，断的时候所有人都一样 |
| 随机缺失 | MAR | 缺和某个**已知**变量有关 | 女性比男性更不爱报体重（你知道了性别就能推测缺失倾向） |
| 非随机缺失 | MNAR | 缺和**缺失值本身**有关 | 高收入者更不爱填收入（越缺的越极端） |

MCAR 最无害：你随便删或填，不引入系统性偏差。
MNAR 最危险：缺失本身就是重要信号——"这个人没填收入"可能意味着他是高收入者。

### 2c. 形式化定义

设 $\mathbf{X}$ 是 $n \times p$ 的完整数据矩阵，$\mathbf{R}$ 是同样形状的指示矩阵：$R_{ij} = 1$ 表示 $X_{ij}$ 缺失，$R_{ij} = 0$ 表示已观测。

| 类型 | 条件 | 处理难度 |
|------|------|----------|
| MCAR | $P(R \mid X) = P(R)$ | 低 |
| MAR | $P(R \mid X) = P(R \mid X_{\text{obs}})$ | 中 |
| MNAR | $P(R \mid X)$ 不能只用 $X_{\text{obs}}$ 解释 | 高 |

### 2d. 手算示例：在小数据上手工填充缺失值

给你一个 4×3 的玩具数据：

```
      f1    f2    f3
A:   10.0  20.0  30.0
B:    8.0   NaN  25.0
C:    NaN  22.0  28.0
D:   12.0  18.0   NaN
```

**策略 1：删除含缺失值的行**

只有 A 是完整的 → 保留 A，删掉 B、C、D。结果：只剩 1 行。数据损失 75%。

**策略 2：均值填充**

- f1 的已知值：10.0, 8.0, 12.0 → 均值 = (10+8+12)/3 = 10.0
- f2 的已知值：20.0, 22.0, 18.0 → 均值 = (20+22+18)/3 = 20.0
- f3 的已知值：30.0, 25.0, 28.0 → 均值 = (30+25+28)/3 = 27.67

填充后：
```
      f1      f2      f3
A:   10.0    20.00   30.00
B:    8.0    20.00   25.00
C:   10.0    22.00   28.00
D:   12.0    18.00   27.67
```

注意：f1 的方差从 2.67 变成了 1.89——填充让数据"收敛"了，低估了原分布的真实方差。

**策略 3：中位数填充**

- f1 已知值排序：8.0, 10.0, 12.0 → 中位数 = 10.0（和均值一样，因为均匀）
- f2 已知值排序：18.0, 20.0, 22.0 → 中位数 = 20.0
- f3 已知值排序：25.0, 28.0, 30.0 → 中位数 = 28.0

f3 的均值是 27.67，中位数是 28.0。如果你的数据有极端值——比如加入一行 E: f3 = 500——均值会被拉到 (30+25+28+500)/4 = 145.75，而中位数只变成 (28+30)/2 = 29.0。**中位数比均值更鲁棒。**

**策略 4：KNN 填充（k=2）**

用 f1 和 f3 找 B 最近的邻居（忽略 NaN 列）。B 的 f1=8.0, f3=25.0：

- 距 A: f1 差 (10-8)/2 = 1, f3 差 (30-25)/5 = 1 → 总 = 2
- 距 C: f1 NaN，跳过；C 只有 f3=28, f1 未知，比较不了
- 距 D: f3 NaN，跳过

只有 A 是有效的邻居 → 用 A 的 f2 = 20.0 来填 B。实际上 KNN 需要在完整数据上做，所以通常先填一个粗略值再迭代。

**策略 5：迭代填充（粗略演示思想）**

1. 先用均值填所有 NaN（得到初始猜测）
2. 把 f2 作为 y，用 f1 和 f3 做线性回归预测 B 的 f2
3. 把 f1 作为 y，用 f2 和 f3 做线性回归预测 C 的 f1
4. 重复直到收敛

（手算太冗长，原理就是"用其他列的信息猜缺失值"）

### 2e. Python 验证

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# 构造含缺失的数据
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'f1': np.random.randn(n) + 10,
    'f2': np.random.randn(n) * 5 + 50,
    'f3': np.random.randn(n) * 2 + 100,
})
mask = np.random.random(df.shape) < 0.2
df_missing = df.mask(mask)
print(f"缺失总数: {df_missing.isnull().sum().sum()} / {df.size}")

# 五种策略对比
strategies = {
    '删除行': df_missing.dropna(),
    '均值填充': SimpleImputer(strategy='mean').fit_transform(df_missing),
    '中位数填充': SimpleImputer(strategy='median').fit_transform(df_missing),
    'KNN填充': KNNImputer(n_neighbors=5).fit_transform(df_missing),
}

for name, arr in strategies.items():
    if isinstance(arr, pd.DataFrame):
        arr = arr.values
    mse = np.mean((arr - df.values) ** 2)
    print(f"{name}: MSE={mse:.4f}")

# MSE 越小，填充越接近真实值
```

典型输出：KNN 填充的 MSE 最小（因为它利用列间相关性），均值和中位数相近，删除行因为样本不同无法直接比较 MSE。

### 2f. 常见误区

**误区：「树模型能自动处理 NaN，所以永远不需要管缺失值」**

❌ 树模型的分裂逻辑确实可以绕过 NaN（XGBoost 自动学习"最佳缺失方向"），但这只解决了训练时的技术问题。如果你的缺失是 MNAR（高价值客户刻意不填收入），缺失本身就是最有用的特征——你应该建一个 `is_missing` 列来捕获这个信号，而不是让树默默地"消化"掉。

**误区：「用 0 填充是最安全的」**

❌ 0 是一个有意义的数值。年龄填 0 是不合理的。均值/中位数填在正常范围内，更不容易产生极端偏差。除非你有明确理由（如"缺失 = 不存在 = 0 次购买"），否则别用 0。

### 2g. 策略选择决策表

| 条件 | 推荐策略 | 理由 |
|------|----------|------|
| 缺失 < 3%，数据量大 | 直接删除 | 几乎不损失信息，最干净 |
| 数值特征、近似正态 | 均值填充 | 快速，对分布中心影响小 |
| 数值特征、有异常值 | 中位数填充 | 不受极端值影响 |
| 特征间有相关性 | KNN 填充 / 迭代填充 | 利用特征关系，填充更准 |
| 分类特征 | 众数填充 / 新建"缺失"类 | 保持类别含义 |
| 树模型（RF, XGBoost） | 不填或简单填 | 树自带缺失处理 |
| MNAR 场景 | 新建 `is_missing` 列 | 缺失本身是信号 |

---

## 3. 特征缩放：把不同量纲拉到同一起跑线

### 3a. 生活例子：算两个人的"综合实力"

你要从两份简历中挑一个人做投资顾问：

| 候选人 | 身高(cm) | 年收入(元) |
|--------|----------|------------|
| 小张   | 175      | 200,000     |
| 小李   | 170      | 300,000     |

如果你直接加总："实力分 = 身高 + 收入"：
- 小张 = 175 + 200,000 = 200,175
- 小李 = 170 + 300,000 = 300,170

收入完全淹没了身高。但如果你把两个特征缩放到同一尺度：
- 身高标准化：假设平均 172.5，标准差 3.5 → 小张 = (175-172.5)/3.5 = 0.71
- 收入标准化：假设平均 250,000，标准差 70,710 → 小张 = (200000-250000)/70710 = -0.71

现在两个特征同等重要，模型才能公平对待。

### 3b. 直观理解：三种缩放方式的话术

| 方法 | 做了什么 | 什么时候用 |
|------|----------|------------|
| **标准化 (Z-score)** | 变成均值为 0、标准差为 1 的分布 | 数据近似正态，大多数 ML 算法 |
| **归一化 (Min-Max)** | 压缩到 [0, 1] 区间 | 需要固定范围（图像像素、神经网络） |
| **鲁棒缩放 (Robust)** | 用中位数和四分位距代替均值和标准差 | 数据有大量极端异常值 |

### 3c. 形式化定义

**标准化 (Standardization)**：

$$
x'_i = \frac{x_i - \mu}{\sigma}, \quad \mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \quad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}
$$

**归一化 (Min-Max Scaling)**：

$$
x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}
$$

**鲁棒缩放 (Robust Scaling)**：

$$
x'_i = \frac{x_i - Q_{50}}{Q_{75} - Q_{25}}
$$

其中 $Q_{25}$、$Q_{50}$、$Q_{75}$ 分别是第 25、50（中位数）、75 百分位数。IQR = $Q_{75} - Q_{25}$。

### 3d. 手算示例：亲手算一遍标准化

**数据集：** `[15, 18, 22, 25, 20]`（5 个人的年龄）

**第 1 步：计算均值 μ**

$$
\mu = \frac{15 + 18 + 22 + 25 + 20}{5} = \frac{100}{5} = 20.0
$$

**第 2 步：计算每个值与均值的差，然后平方**

| $x_i$ | $x_i - \mu$ | $(x_i - \mu)^2$ |
|---------|-------------|-------------------|
| 15      | -5          | 25                |
| 18      | -2          | 4                 |
| 22      | +2          | 4                 |
| 25      | +5          | 25                |
| 20      | 0           | 0                 |
| **和**  | 0           | 58                |

**第 3 步：计算标准差**

$$
\sigma = \sqrt{\frac{58}{5}} = \sqrt{11.6} \approx 3.406
$$

**第 4 步：对每个值做标准化**

| $x_i$ | $x'_i = (x_i - 20) / 3.406$ | 验证 |
|---------|------------------------------|------|
| 15      | (15 - 20) / 3.406 = **-1.468** | |
| 18      | (18 - 20) / 3.406 = **-0.587** | |
| 22      | (22 - 20) / 3.406 = **+0.587** | |
| 25      | (25 - 20) / 3.406 = **+1.468** | |
| 20      | (20 - 20) / 3.406 = **0.000**  | |

**第 5 步：验证均值 = 0，标准差 = 1**

新均值 = (-1.468 - 0.587 + 0.587 + 1.468 + 0) / 5 = 0.000 ✓

新标准差：
$$
\begin{aligned}
\sigma'^2 &= [(-1.468)^2 + (-0.587)^2 + (0.587)^2 + (1.468)^2 + 0^2] / 5 \\
&= [2.155 + 0.345 + 0.345 + 2.155 + 0] / 5 = 5.0 / 5 = 1.0
\end{aligned}
$$

$\sigma' = 1.0$ ✓

**这就是 StandardScaler 在背后做的全部事情。**

再对同一数据做 Min-Max 归一化手算：

$x_{\min} = 15, \quad x_{\max} = 25$

| $x_i$ | $x'_i = (x_i - 15) / (25 - 15)$ |
|---------|----------------------------------|
| 15      | 0.00                             |
| 18      | 0.30                             |
| 22      | 0.70                             |
| 25      | 1.00                             |
| 20      | 0.50                             |

最大值 1.0，最小值 0.0 ✓

### 3e. Python 验证

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 用和手算一样的 5 个年龄值
X = np.array([[15], [18], [22], [25], [20]])

# 标准化
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print(f"标准化后: μ={X_std.mean():.4f}, σ={X_std.std(ddof=0):.4f}")
print(f"值: {X_std.ravel().round(3)}")
# 应该输出: μ=0.0000, σ=1.0000

# 归一化
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)
print(f"\n归一化后: min={X_mm.min():.2f}, max={X_mm.max():.2f}")
print(f"值: {X_mm.ravel().round(2)}")

# 鲁棒缩放（加入极端值后对比）
X_outlier = np.array([[15], [18], [22], [25], [20], [200]])
X_robust = RobustScaler().fit_transform(X_outlier)
X_std_outlier = StandardScaler().fit_transform(X_outlier)
print(f"\n含异常值 (200):")
print(f"Standard 最后一个值: {X_std_outlier[-1][0]:.4f}")
print(f"Robust  最后一个值:    {X_robust[-1][0]:.4f}")
# Robust 对异常值的缩放更温和
```

### 3f. 常见误区

**误区：「所有算法都需要缩放」**

❌ 树模型（决策树、随机森林、GBDT、XGBoost）基于分裂阈值的比较来做决策，`if x > 3.5` 和 `if 2x > 7.0` 在逻辑上完全等价，所以缩放与否不影响树。但基于距离的（KNN、K-Means、SVM）和基于梯度的（线性回归、逻辑回归、神经网络）**必须缩放**。

**误区：「训练集和测试集分开做 StandardScaler」**

❌ 这是最危险的数据泄露之一。正确做法：
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 在训练集上学习 μ,σ 并变换
X_test_scaled  = scaler.transform(X_test)          # 用同一组 μ,σ 变换测试集
```
如果你对测试集也 `.fit_transform()`，测试集的 μ 和 σ 就"泄露"进训练流程了——你在测试时用了你本不该知道的信息。

### 3g. ML 应用连接

特征缩放直接影响梯度下降的收敛速度。不做缩放的梯度下降如下图左——梯度沿着"陡峭"方向剧烈震荡，收敛极慢；做了缩放后如下图右——梯度在各方向均匀，快速收敛到最优解。

这个在后续的线性回归、逻辑回归、神经网络章节中会反复出现。现在只需要记住：**除了树模型，所有模型都该做缩放。**

---

## 4. 处理分类变量：把"名字"翻译成"数字"

### 4a. 生活例子：菜单上的辣度标注

你走进一家餐厅，菜单上每道菜有三种标注：

- 不辣 🌶
- 微辣 🌶🌶
- 特辣 🌶🌶🌶🌶🌶

这里有天然的顺序：不辣 < 微辣 < 特辣。你可以编码成 0、1、2——模型能理解"2 比 0 更辣"。

但如果你把"菜系"（川菜、粤菜、湘菜）也编成 0、1、2，模型就会以为"湘菜(2) > 粤菜(1) > 川菜(0)"——这完全是无中生有。菜系之间没有大小关系。

这就是分类编码的核心原则：**有序的给数字顺序，无序的给独立维度。**

### 4b. 直观理解：三种编码方式

| 编码 | 做了什么 | 例子 | 何时用 |
|------|----------|------|--------|
| Label Encoding | 每类一个整数 0, 1, 2, ... | 北京→0, 上海→1, 广州→2 | 仅适用于**有序**分类 |
| One-Hot Encoding | 每类一个 0/1 列 | 北京→[1,0,0], 上海→[0,1,0] | 适用于**无序**分类 |
| Ordinal Encoding | 你指定顺序 | 高中→0, 本科→1, 硕士→2 | 用于有**已知排序**的分类 |

### 4c. 形式化定义

给定一个有 $k$ 个不同类别值的分类变量 $c \in \{v_1, v_2, \dots, v_k\}$：

**Label Encoding** 是一个双射 $f: \{v_1, \dots, v_k\} \to \{0, 1, \dots, k-1\}$。它假设类别间存在全序关系。

**One-Hot Encoding** 是一个映射 $g: \{v_1, \dots, v_k\} \to \{0, 1\}^k$，其中对于输入 $c = v_j$，输出向量在第 $j$ 位为 1，其余为 0：$g(v_j) = \mathbf{e}_j$（单位基向量）。

**Ordinal Encoding** 是 Label Encoding 的特例——编号顺序由领域知识指定，而不是随机分配。

### 4d. 手算示例：One-Hot 矩阵变换全过程

假设你的数据集有 5 个样本，"城市"和"学历"两列：

```
样本    城市      学历
S1     北京      本科
S2     上海      硕士
S3     广州      博士
S4     北京      硕士
S5     深圳      本科
```

**城市（无序 → One-Hot）：**

城市有 4 种取值：北京、上海、广州、深圳 → 4 列

```
        city_北京  city_上海  city_广州  city_深圳
S1(北京):   1          0          0          0
S2(上海):   0          1          0          0
S3(广州):   0          0          1          0
S4(北京):   1          0          0          0
S5(深圳):   0          0          0          1
```

**学历（有序 → Ordinal）：**

学历顺序：本科 → 硕士 → 博士 → 编码为 0, 1, 2

```
S1(本科): 0
S2(硕士): 1
S3(博士): 2
S4(硕士): 1
S5(本科): 0
```

**最终数值特征矩阵（合并后）：**

```
        city_北京  city_上海  city_广州  city_深圳  education
S1:        1         0         0         0          0
S2:        0         1         0         0          1
S3:        0         0         1         0          2
S4:        1         0         0         0          1
S5:        0         0         0         1          0
```

原始是 5 行 × 2 列的字符串矩阵，最终变成 5 行 × 5 列的数值矩阵——4 列来自 One-Hot（$k-1$ 即可，但这里用了全维度），1 列来自 Ordinal。

> **维数陷阱**：如果去掉一列（如去掉 `city_北京`），可以用剩下的 3 列重新编码，这叫 **dummy encoding**（哑变量编码），可以节省一个维度。大多数情况下推荐用 $k-1$ 列，`OneHotEncoder(drop='first')`。但对树模型来说，保留全 $k$ 列减少了列与列之间的耦合，有时效果更好。

### 4e. Python 验证

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

df = pd.DataFrame({
    'city':      ['北京', '上海', '广州', '北京', '深圳'],
    'education': ['本科', '硕士', '博士', '硕士', '本科'],
})
print("原始数据:\n", df)

# --- Label Encoding (会自动按字母顺序分配，不保证有意义顺序) ---
le = LabelEncoder()
df['city_label'] = le.fit_transform(df['city'])
print(f"\nLabel Encoding 映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"⚠ 上海→4, 北京→0。模型会以为 '上海 > 北京'！")

# --- One-Hot (sparse_output=False 返回密集矩阵) ---
ohe = OneHotEncoder(sparse_output=False)
city_ohe = ohe.fit_transform(df[['city']])
print(f"\nOne-Hot 列名: {ohe.get_feature_names_out(['city'])}")
print(city_ohe)

# --- Ordinal (指定顺序) ---
ordinal = OrdinalEncoder(categories=[['本科', '硕士', '博士']])
df['edu_ordinal'] = ordinal.fit_transform(df[['education']])
print(f"\nOrdinal Encoding: 本科→0, 硕士→1, 博士→2")
print(df[['education', 'edu_ordinal']])
```

### 4f. 常见误区

**误区：「LabelEncoder 可以用于无序分类特征」**

❌ 这是 scikit-learn 初学者最容易犯的错误。`LabelEncoder` 按字母顺序分配 0, 1, 2...，这意味着"上海"变成了 4、"北京"变成了 0。线性模型、神经网络会把这种数值大小当真——以为"上海(4) 在数值上 > 北京(0)"。**LabelEncoder 只能用于目标变量 y 的编码，或确实有序的分类特征。**

**误区：「One-Hot 总是最好的」**

❌ One-Hot 对高基数特征（如"邮政编码"有上万个类别）会生成上万列，导致维度爆炸和过拟合。这种情况用 Target Encoding（用目标变量的均值替换类别）或 Embedding（神经网络中学习低维向量表示）更合适。

### 4g. ML 应用连接

分类编码直接影响线性可解释性。如果你用 Ordinal Encoding 把"北京→0, 上海→1"喂给线性回归，得到的系数对"上海 vs 北京"的比较不是独立的——它受中间类别（如果有的话）的影响。One-Hot 的每个系数有独立的含义：`city_上海` 的系数 = "和基准城市相比，上海对目标变量的平均影响"。

---

## 5. 处理异常值：数据中的"捣乱分子"

### 5a. 生活例子：班级成绩单里的 5 分试卷

全班 40 人考数学，39 人成绩在 60-95 之间，但有一个人考了 5 分。你如果直接算"全班平均分"，这个 5 分会把均值下拉好几档。更可怕的是——老师算期末总评时用这个均值做基线，结果所有人的评分都因为这个 5 分的人偏高了一截。

这就是异常值的影响力：**一个极端点可以扭曲整个统计分析。**

### 5b. 直观理解：两种检测"不合群"的方法

| 方法 | 原理 | 适用场景 | 缺点 |
|------|------|----------|------|
| **Z-score** | 看某个值距均值多少个标准差，> 3 算异常 | 数据近似正态分布 | 均值和标准差本身被异常值影响 |
| **IQR** | 用四分位数：$< Q_1 - 1.5 \times IQR$ 或 $> Q_3 + 1.5 \times IQR$ | 任何分布 | 对正态分布会漏检一些边界值 |

### 5c. 形式化定义

**Z-score 法：**

$$
z_i = \frac{x_i - \mu}{\sigma}
$$

若 $|z_i| > 3$，则 $x_i$ 为异常值。依据：正态分布下，$|z| > 3$ 的概率约 0.27%。

**IQR 法：**

$$
\text{IQR} = Q_{75} - Q_{25}
$$

下界：$Q_{25} - 1.5 \times \text{IQR}$
上界：$Q_{75} + 1.5 \times \text{IQR}$

落在界外的点即为异常值。1.5 是 John Tukey 的经典阈值（对正态分布，约 0.7% 的正常数据会落界外）。

### 5d. 手算示例：IQR 检测

数据：`[60, 72, 75, 78, 80, 82, 85, 88, 90, 95, 5]`（11 个人的分数，最后一个是异常值）

排序：`[5, 60, 72, 75, 78, 80, 82, 85, 88, 90, 95]`

n = 11，位置：
- Q1 位置 = 0.25 × (11+1) = 3 → Q1 = 72
- Q3 位置 = 0.75 × (11+1) = 9 → Q3 = 88

IQR = 88 - 72 = 16

- 下界 = 72 - 1.5 × 16 = 72 - 24 = **48**
- 上界 = 88 + 1.5 × 16 = 88 + 24 = **112**

检测：5 < 48 → **5 是异常值 ✓**。其余值都在 [48, 112] 内。

再用 Z-score 法（因为 5 的存在，μ 和 σ 已被污染）：
- μ = (60+72+75+78+80+82+85+88+90+95+5) / 11 = 810/11 = 73.64
- σ² = [(60-73.64)² + ... + (5-73.64)²] / 11 ≈ 588.96
- σ ≈ 24.27

z(5) = (5 - 73.64) / 24.27 = -2.83

|−2.83| = 2.83 < 3 → Z-score 法**没有**检测出 5 是异常值！

这就是 Z-score 的致命弱点：**异常值污染了 μ 和 σ，让 Z-score 变得不敏感。** IQR 基于百分位数，天然免疫这种污染。

### 5e. Python 验证

```python
import numpy as np
from scipy import stats

data = np.array([60, 72, 75, 78, 80, 82, 85, 88, 90, 95, 5])

# IQR 法
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = np.where((data < lower) | (data > upper))[0]
print(f"IQR 法: 下界={lower:.1f}, 上界={upper:.1f}")
print(f"异常值索引: {outliers_iqr}, 值: {data[outliers_iqr]}")

# Z-score 法
z = np.abs(stats.zscore(data))
outliers_z = np.where(z > 3)[0]
print(f"\nZ-score 法异常值: {data[outliers_z] if len(outliers_z)>0 else '没检测到!'}")

# 处理策略对比
from scipy.stats.mstats import winsorize

print(f"\n处理前: min={data.min()}, max={data.max()}, mean={data.mean():.1f}")
print(f"删除异常: {data[data >= lower][data[data >= lower] <= upper].mean():.1f}")
print(f"中位数替代: {np.median(data):.0f}")

# winsorize: 把最低的 5% 和最高的 5% 按回边界
winsorized = winsorize(data, limits=[0.05, 0.05])
print(f"缩尾后: min={winsorized.min()}, max={winsorized.max()}")
```

### 5f. 常见误区

**误区：「有异常值就应该删掉」**

❌ 异常值有两种：错误（传感器故障、录入失误）和信息（黑天鹅事件、欺诈交易）。前者该删，后者是有价值的数据。2020 年的股票交易量异常值如果删了，你就错过了最重要的市场信号。

**误区：「IQR 的 1.5 倍是普适标准」**

❌ 1.5 是探索性数据分析的默认值，不是法律。对于严格的工业质量控制，可能用 3 倍 IQR；对于宽松的探索，可能用 1 倍。阈值要根据业务调整。

### 5g. ML 应用连接

线性回归的最小二乘法对异常值极其敏感——一个极端点就可以让拟合线大幅度偏转（因为误差是平方的）。解决方案：
- 用 RobustScaler + 缩尾处理
- 换用对异常值鲁棒的模型：Huber 回归、RANSAC 回归、树模型
- 对数变换压缩右偏分布的极端值

---

## 6. 数据划分：Train / Validation / Test Split

### 6a. 生活例子：高考的题目你不能提前做

假设你是老师，下学期要出期末考题。你决定：

- **训练集**：前几年的期末考试卷——让学生反复练习，学习解题模式
- **验证集**：今年的模拟考——看看学生学没学到东西，决定要不要换教学方法
- **测试集**：真正的期末考试——**只能在最后做一次**，代表学生的真实水平

如果你把期末考卷的题提前让学生在练习里做过，那期末成绩就没有任何意义——你只是在测试记忆力，不是测试能力。

同理，**模型在训练时绝不能"偷看"测试数据**——否则你只是在测试它背数据的本领，而不是泛化能力。

### 6b. 直观理解：三个数据集的职责

| 数据集 | 作用 | 看了几次 | 类比 |
|--------|------|----------|------|
| **训练集 (Train)** | 学模型参数（w, b） | 每个 epoch 看 | 平时的作业和练习题 |
| **验证集 (Val)** | 调超参数、选模型、early stopping | 反复看 | 模拟考——根据成绩调整学习策略 |
| **测试集 (Test)** | 最终评估 | **只看一次** | 高考——成绩代表真实水平 |

### 6c. 形式化定义

给定 $n$ 个样本的数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$，划分为三个互不相交的子集：

$$
\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}
$$

且 $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$。

常见划分比例：

| 数据总量 | 推荐 (Train/Val/Test) |
|----------|-----------------------|
| < 1,000 | 60/20/20 |
| 1,000 - 100,000 | 70/15/15 |
| > 100,000 | 80/10/10 |
| 深度学习（百万级） | 98/1/1 |

### 6d. 手算示例：分层抽样 vs 随机抽样

假设你的二分类数据集有 10 个样本：

```
类别 0: 8 个样本  (A1, A2, A3, A4, A5, A6, A7, A8)
类别 1: 2 个样本  (B1, B2)
```

你要拿出 2 个样本做测试集（20%）。

**随机抽样**：纯随机抽 2 个。有可能抽到 (A3, A5)，测试集中没有类别 1 → 你的模型在测试集上根本不需要分辨类别 1，评估不全面。

更坏的情况：抽到 (B1, B2)，测试集全是类别 1 → 你评估的根本是一个不同的分布。

**分层抽样 (Stratified Split)**：按类别比例抽。类别 0 占 80%，测试集中应有 2 × 80% ≈ 1.6 → 抽 1 或 2 个类别 0 样本。实际做法：从 8 个类别 0 中随机抽 2 个，从 2 个类别 1 中随机抽 1 个（round 到最接近的整数）。测试集中类别 0 : 类别 1 = 2:1，近似于原来的 4:1，远比随机抽样更可靠。

### 6e. Python 验证

```python
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(1000) * 0.3 > 0).astype(int)

# 三步划分：先切出测试集，再从剩余中切出验证集
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

print(f"训练集: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"验证集: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
print(f"测试集: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

# 验证分层效果: 类别比例是否一致
for name, yy in [("全部", y), ("训练", y_train),
                 ("验证", y_val), ("测试", y_test)]:
    print(f"{name}: 类别1占比 = {yy.mean():.3f}")

# 时间序列的特殊处理（不能随机打乱！）
date_range = pd.date_range('2023-01-01', periods=365)
ts_data = pd.DataFrame({
    'date':  date_range,
    'value': np.random.randn(365).cumsum() + 100,
})
train = ts_data[ts_data['date'] < '2023-10-01']
val   = ts_data[(ts_data['date'] >= '2023-10-01') & (ts_data['date'] < '2023-12-01')]
test  = ts_data[ts_data['date'] >= '2023-12-01']
print(f"\n时间序列: Train={len(train)}, Val={len(val)}, Test={len(test)}")
print("⚠ 时间序列必须按时间顺序划分，不能用随机 shuffle")
```

### 6f. 常见误区

**误区：「划分前先做特征工程和标准化」**

❌ 这是最严重的数据泄露！如果你在全量数据上做了标准化（计算了整体的 μ 和 σ），再划分，那么训练集的 μ 和 σ 里就包含了测试集的信息。正确流程：**先划分 → 只在训练集上 fit → transform 全部。**

**误区：「测试集太小了，我多评估几次」**

❌ 每多评估一次，你就在用测试集做"调参"——你在往测试集的方向优化。如果你反复调整模型多次，每次都用测试集打分，那你实际上在让测试集充当验证集。真正的测试集只能最后用一次。

### 6g. ML 应用连接

在 Kaggle 比赛中，竞赛方提供的 public leaderboard 其实是一个公开的"测试集"。许多人对着 public LB 反复调参，导致在 private test set（真正的测试集）上大幅下滑——这就是典型的测试集泄露导致的过拟合。你的本地验证集应该模拟这个效果：**不要对着验证集反复调参，用交叉验证来减少调参的偏差。**

---

## 7. 交叉验证：一次划分不够，来多次取平均

### 7a. 生活例子：一次考试不能代表真实水平

你参加了一场数学竞赛，题目难度刚好撞在你的舒适区——你拿了高分，但这不代表你数学真的好。反之，如果题目全考了你的知识盲区——你考砸了，但也不代表你数学真的差。

怎么办？**考多次，取平均**。不同的试卷（不同的 train/val 划分）给你不同的分数，平均值比单次更可靠。

### 7b. 直观理解：K 折交叉验证

把数据均匀切成 K 块（折）。每次拿 K-1 块训练，剩下 1 块验证。轮换 K 次，每次"站出来"验证的是不同的一块。最后取 K 个验证分数的均值和标准差作为最终评价。

K=5 的示意图：

```
折1 [Val] [Train] [Train] [Train] [Train]  →  Score₁
折2 [Train] [Val] [Train] [Train] [Train]  →  Score₂
折3 [Train] [Train] [Val] [Train] [Train]  →  Score₃
折4 [Train] [Train] [Train] [Val] [Train]  →  Score₄
折5 [Train] [Train] [Train] [Train] [Val]  →  Score₅

最终报告: Score = mean(Score₁..₅) ± std(Score₁..₅)
```

### 7c. 形式化定义

K 折交叉验证分数：

$$
\text{CV Score} = \frac{1}{K} \sum_{k=1}^{K} \text{Score}(\hat{f}_{-k}, \mathcal{D}_k)
$$

其中 $\hat{f}_{-k}$ 是在除第 $k$ 折外的数据上训练的模型，$\mathcal{D}_k$ 是第 $k$ 折的数据。

K 的选择：
| K | 适用场景 | 计算量 |
|---|----------|--------|
| 5 或 10 | 大多数情况 | 中等 |
| 2 或 3 | 计算资源极其有限 | 低 |
| n (LOOCV) | 样本量 < 100 | 极高 |

### 7d. 手算示例：3-Fold CV 的全过程

假设有 6 个带标签的样本：

```
样本    x      y (类别)
S1     2.0     0
S2     3.0     0
S3     5.0     1
S4     6.0     1
S5     1.5     0
S6     7.0     1
```

分 3 折（每折 2 个样本），假设按顺序分：

```
Fold 1: S1, S2   (类别 0, 0 — 全是 0!)
Fold 2: S3, S4   (类别 1, 1 — 全是 1!)
Fold 3: S5, S6   (类别 0, 1 — 混合)
```

**第一次迭代：** Fold 1 做验证，Fold 2+3 做训练
- 训练集 = {S3(1), S4(1), S5(0), S6(1)}，3 个 1 和 1 个 0
- 假设我们用一个简单的规则：预测为训练集中多数类（1）
- 验证集 = {S1(0), S2(0)}，两个都是 0
- 模型始终预测 1 → 准确率 = **0/2 = 0.00**

**第二次迭代：** Fold 2 做验证
- 训练集 = {S1(0), S2(0), S5(0), S6(1)}，3 个 0 和 1 个 1
- 多数类规则 → 始终预测 0
- 验证集 = {S3(1), S4(1)}，两个都是 1
- 准确率 = **0/2 = 0.00**

**第三次迭代：** Fold 3 做验证
- 训练集 = {S1(0), S2(0), S3(1), S4(1)}，2 个 0 和 2 个 1 → 平局
- 平局时随便选 0 → 始终预测 0
- 验证集 = {S5(0), S6(1)}
- S5 的预测 0 = 标签 0 ✓，S6 的预测 0 ≠ 标签 1 ✗
- 准确率 = **1/2 = 0.50**

CV 均值为 (0 + 0 + 0.50) / 3 = **0.167**。标准差很大 → 数据太少、划分不稳定。

**如果打乱后用分层 K 折：**

StratifiedKFold 保证每折的类别比例一致（50% 是 0，50% 是 1），每折都有 1 个 0 和 1 个 1：

```
Fold 1: S1(0), S3(1)
Fold 2: S2(0), S5(1)   (假设重排后)
Fold 3: S6(0), S4(1)   (假设重排后)
```

现在每次验证都是 1 个 0 和 1 个 1，评估更稳定，CV 分数更有意义。

### 7e. Python 验证

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression())

# 普通 5 折 CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"5 折 CV 每折: {scores}")
print(f"平均: {scores.mean():.4f} ± {scores.std():.4f}")

# 分层 5 折（推荐）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"分层 5 折: 平均 {strat_scores.mean():.4f} ± {strat_scores.std():.4f}")

# 留一法 (Leave-One-Out) — 用于极小数据集
from sklearn.model_selection import LeaveOneOut
X_small, y_small = X[:30], y[:30]
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X_small, y_small, cv=loo)
print(f"留一法 (n=30): 平均 {loo_scores.mean():.4f} ± {loo_scores.std():.4f}")
```

### 7f. 常见误区

**误区：「CV 就是要 K 越大越好，用 LOOCV 就对了」**

❌ LOOCV 虽然最大化了训练数据使用率，但计算代价是 K 倍，且方差极大（每次只在一个样本上评估，这个样本的随机噪声决定了该折的分数）。K=5 或 K=10 在实践中通常是效率和稳定性的最佳平衡。

**误区：「交叉验证可以防止数据泄露」**

❌ CV 解决的是"评估方差"问题，不是"数据泄露"问题。如果你在 CV 之前先对全量数据做了标准化（计算了整体 μ 和 σ），你的 CV 分数仍然是被污染的数据泄露结果。正确的做法是把预处理放入 Pipeline，让 CV 在每次迭代时只在训练折上 fit。

### 7g. ML 应用连接

`GridSearchCV` 和 `RandomizedSearchCV` 内部就是用交叉验证来选择最佳超参数。它们在训练集的子划分中做 CV，选择表现最好的参数组合，然后用全量训练集以该参数重新训练模型。这种"嵌套 CV"（外层 CV 评估 + 内层 CV 选参）是防止调参数据泄露的标准做法。

---

## 8. 完整 Pipeline + 数据泄露：把一切锁在一起

### 8a. 生活例子：银行的审批流程不能说漏嘴

你是银行信贷审批官。面试流程是：

1. 申请人先填表（原始数据）
2. 工作人员根据所有申请人的情况，确定一个"基准收入和基准年龄"（fit：学 μ 和 σ）
3. 用这个基准给每个人打分（transform）

如果你在第 2 步时，先偷偷看了一眼明天才提交申请的客户信息，然后把这个信息也纳入了基准计算——你就是在作弊。别的申请人今天不可能知道明天会发生什么。你在训练阶段用了未来的信息。

这就是**数据泄露**的实质：**你在 fit 阶段使用了你不应该在训练时访问的数据。**

### 8b. 直观理解：Pipeline 的防泄露机制

`Pipeline` 是一个把多个步骤串起来的容器。关键规则：

- `pipeline.fit(X_train, y_train)` → 每个步骤依次执行 `fit_transform()`，最后一个步骤执行 `fit()`
- `pipeline.predict(X_test)` → 每个步骤只执行 `transform()`（用已学到的参数），最后一个步骤执行 `predict()`

**Pipeline 把"先 fit 再 transform"的逻辑锁死——你不可能对测试集单独 fit。** 所有 `fit` 类操作被隔离在训练阶段。

### 8c. 形式化定义

一个预处理 + 建模 Pipeline 是一个有序序列：

$$
\text{Pipeline} = [T_1 \to T_2 \to \dots \to T_m \to E]
$$

其中 $T_i$ 是 Transformer（`fit_transform`），$E$ 是 Estimator（`fit`/`predict`）。

对于混合特征类型（数值 + 分类），用 `ColumnTransformer` 为不同列指定不同的子 Pipeline：

$$
\text{ColumnTransformer} = [ (name_1, pipeline_1, cols_1), (name_2, pipeline_2, cols_2), \dots ]
$$

### 8d. 手算示例：Pipeline 执行顺序的逐步追踪

假设你有这个 Pipeline：

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression()),
])
```

训练数据（注意第一行 f2 是 NaN）：

```
      f1     f2
A:    10     20
B:     8     NaN     ← 缺失
C:    12     18
D:    15     25
```

**执行 `pipeline.fit(X_train)`：**

1. SimpleImputer.fit(): 在训练数据上计算每列中位数
   - f1 中位数 = (8, 10, 12, 15) → (10+12)/2 = 11
   - f2 中位数 = (18, 20, 25) → 20
2. SimpleImputer.transform(): B 的 f2 从 NaN 变成 20

```
      f1     f2
A:    10     20
B:     8     20    ← 已填充
C:    12     18
D:    15     25
```

3. StandardScaler.fit(): 在填充后的数据上计算 μ 和 σ
   - μ_f1 = (10+8+12+15)/4 = 11.25, σ_f1 ≈ 2.59
   - μ_f2 = (20+20+18+25)/4 = 20.75, σ_f2 ≈ 2.59
4. StandardScaler.transform(): 标准化

```
          f1        f2
A:    (10-11.25)/2.59 = -0.48    (20-20.75)/2.59 = -0.29
B:    (8-11.25)/2.59  = -1.25    (20-20.75)/2.59 = -0.29
C:    (12-11.25)/2.59 =  0.29    (18-20.75)/2.59 = -1.06
D:    (15-11.25)/2.59 =  1.45    (25-20.75)/2.59 =  1.64
```

5. LogisticRegression.fit(): 在标准化后的数据上训练

**执行 `pipeline.predict(X_test)`，假设测试数据是：**

```
      f1     f2
E:    11     NaN
F:    14     22
```

1. SimpleImputer.transform(): 用训练阶段学到的中位数（11, 20）填充
   - E 的 f2 → 20
2. StandardScaler.transform(): 用训练阶段的 μ(11.25, 20.75) 和 σ(2.59, 2.59) 变换
3. LogisticRegression.predict(): 预测

**关键点：测试数据的 f2 中位数可能是 22，但步骤 1 用的仍然是 20（训练时的中位数）。你没有让测试数据"教"Imputer 该填什么——你用的是训练时学到的经验。**

### 8e. Python 验证：完整的防泄露 Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# 构造含缺失值 + 分类 + 异常值的脏数据
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age':        np.where(np.random.random(n) < 0.15, np.nan,
                           np.random.randint(18, 70, n).astype(float)),
    'income':     np.random.exponential(50000, n).astype(int),
    'education':  np.random.choice(['高中', '本科', '硕士', '博士'], n),
    'city':       np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n),
    'hours':      np.random.normal(40, 10, n),
    'experience': np.random.exponential(5, n),
})
df.loc[::30, 'income'] *= 50   # 添加异常值
df['target'] = ((df['income'] > 50000) & (df['age'].fillna(35) > 30)).astype(int)

# 先划分，再处理（防泄露第一步）
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)

# 分类特征类型
num_features   = ['age', 'income', 'hours', 'experience']
ord_features   = ['education']
nom_features   = ['city']

# 数值子流水线：中位数填缺失 + 鲁棒缩放（抗异常值）
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

# 有序分类子流水线
ord_pipe = Pipeline([
    ('ordinal', OrdinalEncoder(categories=[['高中', '本科', '硕士', '博士']])),
])

# 无序分类子流水线：常数值填缺失 + One-Hot
nom_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='未知')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# ColumnTransformer 组装
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('ord', ord_pipe, ord_features),
    ('nom', nom_pipe, nom_features),
])

# 完整 Pipeline: 预处理 + 模型
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression(max_iter=1000, random_state=42)),
])

# 训练 + 评估
full_pipeline.fit(X_train, y_train)
print(f"训练集准确率: {full_pipeline.score(X_train, y_train):.4f}")
print(f"测试集准确率: {full_pipeline.score(X_test, y_test):.4f}")

# 交叉验证确认泛化性能（Pipeline 保证 CV 内部不泄露）
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5)
print(f"5折CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 在新数据上预测——Pipeline 自动执行全部预处理步骤
new_sample = pd.DataFrame({
    'age': [28], 'income': [60000], 'education': ['本科'],
    'city': ['成都'], 'hours': [42], 'experience': [3],
})
pred = full_pipeline.predict(new_sample)
print(f"新样本预测: 类别 {pred[0]}")

# 看一下 Pipeline 内部做了什么（查看中间步骤的输出）
X_train_transformed = preprocessor.fit_transform(X_train)
print(f"\n变换后特征数: {X_train_transformed.shape[1]} "
      f"(原始 {X_train.shape[1]} 列 → 包含 One-Hot 扩展)")
```

### 8f. 常见误区（数据泄露特辑）

**泄露类型 1：在划分前做标准化**

```python
# ❌ 错误——数据泄露!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # 在全部数据上 fit
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

✅ 正确：
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # 只在训练集 fit
X_test  = scaler.transform(X_test)          # 用训练集的参数 transform
```

**泄露类型 2：在填补缺失值时用了全量数据**

```python
# ❌ 在全量数据上算均值，然后填充
all_mean = X.mean()                         # X 包含训练+测试
X_train_filled = X_train.fillna(all_mean)   # 泄露!
```

**泄露类型 3：特征选择用了测试集的信息**

```python
# ❌ 用全量数据做特征选择
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)   # 看了全量 y
```

**泄露类型 4：交叉验证时在数据划分前预处理**

```python
# ❌ CV 外部已经污染了
X_scaled = StandardScaler().fit_transform(X)   # 全量 fit
scores = cross_val_score(model, X_scaled, y)   # CV 看到的已是泄露数据
```

✅ 正确做法：把预处理放在 Pipeline 里，让 Pipeline 参与 CV 的每次迭代。

**泄露类型 5：时间序列用随机划分**

```python
# ❌ 时间序列用 random split —— 你在用"未来的数据"预测"过去"
X_train, X_test, y_train, y_test = train_test_split(ts_X, ts_y, shuffle=True)
```

✅ 按时间切：前 70% = train，中间 15% = val，后 15% = test。

### 8g. ML 应用连接

一个正确搭建的 Pipeline 对象可以直接 `pickle` 序列化保存，然后加载到生产环境中用于预测——**不需要在部署时重新 fit 任何步骤**。这是 ML 工程化的基本要求：

```python
import joblib
joblib.dump(full_pipeline, 'model_pipeline.pkl')

# 三个月后，在新服务器上加载并预测
loaded = joblib.load('model_pipeline.pkl')
prediction = loaded.predict(new_data)   # 一步到位，零泄露
```

---

## 9. 本章思考题（10 题，全解答）

### 题目 1 — 缺失值策略选择

数据集中"年收入"列有 20% 的缺失值。已知高收入人群更倾向于不填收入（MNAR）。你会用什么策略处理？解释理由。

<details>
<summary>解答</summary>

这是 MNAR 场景——缺失本身携带着"高收入"这个信号。

推荐策略：**新建一个 `income_is_missing` 的 0/1 列，然后用中位数填充缺失值。**

理由：
1. `income_is_missing` 列把"是否缺失"这个信号显式化了。后续的树模型或线性模型可以学到"缺失收入 → 通常是高收入 → 可能高消费"这样的模式。
2. 中位数填充比均值好，因为收入分布通常右偏（有极端高收入者）。如果高收入者全不填，剩下的全是中低收入，均值会比中位数更低——中位数更接近中部的真实水平。
3. 不要删除这些行——你删除的正好是你最想知道的那群人（高收入客户）。

不推荐均值填充（会被留下的中低收入拉低），也不推荐 KNN 填充（会基于中低收入的邻居来填高收入，系统性低估）。
</details>

---

### 题目 2 — 手算标准化

用"总体标准差"公式（除以 n，不是 n-1）手工标准化这个数据集：`[3, 7, 7, 9, 14]`。写出每一步，并验证标准化后 μ=0, σ=1。

<details>
<summary>解答</summary>

**步骤 1**：μ = (3+7+7+9+14) / 5 = 40/5 = **8.0**

**步骤 2**：计算方差

| x | x-μ | (x-μ)² |
|---|-----|--------|
| 3 | -5  | 25     |
| 7 | -1  | 1      |
| 7 | -1  | 1      |
| 9 | +1  | 1      |
| 14| +6  | 36     |
| Σ | 0   | 64     |

σ² = 64/5 = 12.8 → σ = √12.8 ≈ **3.578**

**步骤 3**：标准化

| x | z = (x-8)/3.578 |
|---|-----------------|
| 3 | -1.397          |
| 7 | -0.279          |
| 7 | -0.279          |
| 9 | +0.279          |
| 14| +1.677          |

**步骤 4**：验证
- 新均值 = (-1.397 - 0.279 - 0.279 + 0.279 + 1.677) / 5 = 0.001/5 ≈ **0** ✓
- 新方差 = [(-1.397)² + (-0.279)² + (-0.279)² + (0.279)² + (1.677)²] / 5
  = (1.952 + 0.078 + 0.078 + 0.078 + 2.812) / 5 = 4.998/5 ≈ 1.0 → σ ≈ 1.0 ✓

注意：Scikit-learn 的 `StandardScaler` 默认使用 `ddof=0`（除以 n，即总体标准差），和这里一致。而 `np.std(ddof=1)` 是样本标准差（除以 n-1），两者有细微差异。
</details>

---

### 题目 3 — One-Hot 后的特征数

一个数据集有 3 个分类特征：`color: {红, 蓝, 绿}`、`size: {S, M, L, XL}`、`brand: {品牌A, 品牌B}`。如果你对它们做 One-Hot 编码（使用 `drop='first'`），最终会产生多少列？如果不用 `drop='first'` 呢？

<details>
<summary>解答</summary>

One-Hot 编码原理：每个有 k 个类别的特征，产生 k 列（全模式）或 k-1 列（drop first）。

- `color`: 3 个类别 → 3-1 = **2** 列 (or 3 列)
- `size`: 4 个类别 → 4-1 = **3** 列 (or 4 列)
- `brand`: 2 个类别 → 2-1 = **1** 列 (or 2 列)

使用 `drop='first'`：2 + 3 + 1 = **6 列**
不用 `drop='first'`：3 + 4 + 2 = **9 列**

`drop='first'` 节省了 3 列，但对线性模型来说，这 3 列是冗余的——知道前 k-1 列的值，第 k 列就是确定的（全为 0）。**放弃全信息编码（6 列）已经囊括了所有信息，额外的 3 列是线性相关的，会导致多重共线性问题。** 不过对于树模型和正则化线性模型（Lasso/Ridge），全 k 列有时更方便解释——每个系数直接对应一个类别，不需要"相对于参考类"的解读。
</details>

---

### 题目 4 — 标准化 vs 归一化：什么时候用哪个？

你正在做一个信用卡欺诈检测项目（数据有大量交易，少数几笔欺诈）。特征包括"交易金额"（右偏，有极端大额交易）和"交易时间（小时）"（0-23）。你应该用 StandardScaler 还是 MinMaxScaler？为什么？

<details>
<summary>解答</summary>

推荐：**交易金额用 RobustScaler，交易时间用 MinMaxScaler。**

分析：
- **交易金额**：右偏分布 + 有极端值（大额交易）。StandardScaler 的 μ 和 σ 会被极端值拉偏（和本章 5d 的手算例子一样，Z-score 可能检测不到异常值）。RobustScaler 用中位数和 IQR，天然免疫极端值。推荐 RobustScaler。
- **交易时间**：有明确边界 [0, 23]，且 0 点接近 23 点（周期性）。MinMaxScaler 保持边界 [0, 1]，适合这种有界特征。StandardScaler 虽然也能用，但不会利用这个边界信息。

如果交易金额没有极端异常值（已做缩尾处理），StandardScaler 也是可以的，特别是如果你用逻辑回归——标准化后的系数在量纲上可比。

额外提醒：对欺诈检测这种极度不平衡分类任务，缩放之前先划分数据，并且用 `stratify=y` 保证训练/测试集中欺诈样本比例一致。
</details>

---

### 题目 5 — 异常值策略判断

你有一个房价预测数据集。`square_feet`（面积）列中有一个值 = 1 平方英尺。这看起来很不合理——最小的公寓也不至于这么小。你应该怎么处理？

<details>
<summary>解答</summary>

**调查优先于删除。** 1 平方英尺 ≈ 0.093 平方米——这可能是数据录入错误、单位混淆（平方米被当成平方英尺），或者一个特殊类型的房产（停车位、储物间，确实只有 1 平方英尺）。

处理方案（按优先级排序）：

1. **先检查同一行的其他列**：如果这个样本的卧室数 = 0、价格 = $5000，这很可能真的是一个停车位——它是一个有效的数据点，不应删除。你甚至可以利用它：建一个 `is_parking` 特征。

2. **如果是录入错误**（其他列看起来正常，如 3 卧 2 卫但面积 = 1 且价格 = $500,000），可以用中位数填充 `square_feet`，或者用 `square_feet = price / price_per_sqft_median` 反向推算。

3. **如果无法确认原因**：用中位数替换。不要用均值——如果 1 是错误，它把均值拉低了，用被污染的均值来修复它的"错误"不太合理（循环论证）。

4. **绝对不要直接删除这一行**，除非你 100% 确定它是错误且样本数足够多。每一个被删掉的样本都是信息的丢失。
</details>

---

### 题目 6 — 训练/测试划分顺序

下面是一个 ML 新手写的 5 行代码，指出所有数据泄露：

```python
X = df.drop('target', axis=1)
y = df['target']

X = X.fillna(X.median())         # ①
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ②

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)  # ③
```

<details>
<summary>解答</summary>

**泄露 ①**：`X.fillna(X.median())` 在**全量数据**（包含测试集）上计算中位数。测试集的缺失值被测试集自己的中位数填充 → 泄露。应该在划分后用训练集的 median 填充。

**泄露 ②**：`scaler.fit_transform(X)` 同样在全量数据上计算 μ 和 σ。测试集的特征值参与了 μ 和 σ 的计算 → 泄露。

**泄露 ③**：`train_test_split` 放在最后——已经来不及了。前面两步已经把全量数据的信息"混"进了 X_scaled，再划分也无法消除泄露。

另外，没有 `stratify=y`——如果类别不平衡，测试集的类别比例可能偏斜。

**正确顺序**：
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train = X_train.fillna(X_train.median())
X_test  = X_test.fillna(X_train.median())           # 用训练集的中位数
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
```

更好的做法：全部封装在 Pipeline 中。
</details>

---

### 题目 7 — 为什么标准化后 μ=0, σ=1（代数证明）

证明：对于任意一组数 $x_1, x_2, \dots, x_n$，令 $z_i = \frac{x_i - \mu}{\sigma}$（其中 $\mu$ 和 $\sigma$ 是原始数据的均值和标准差），则 $z$ 的均值必为 0，标准差必为 1。

<details>
<summary>解答</summary>

**均值证明：**

$$
\begin{aligned}
\bar{z} &= \frac{1}{n}\sum_{i=1}^{n} z_i = \frac{1}{n}\sum_{i=1}^{n} \frac{x_i - \mu}{\sigma} \\
&= \frac{1}{\sigma} \cdot \frac{1}{n}\left(\sum_{i=1}^{n} x_i - \sum_{i=1}^{n} \mu\right) \\
&= \frac{1}{\sigma} \cdot \frac{1}{n}\left(n\mu - n\mu\right) = \frac{1}{\sigma} \cdot 0 = 0
\end{aligned}
$$

**标准差证明：**

$$
\begin{aligned}
\sigma_z^2 &= \frac{1}{n}\sum_{i=1}^{n} (z_i - \bar{z})^2 = \frac{1}{n}\sum_{i=1}^{n} z_i^2 \quad (\because \bar{z}=0) \\
&= \frac{1}{n}\sum_{i=1}^{n} \left(\frac{x_i - \mu}{\sigma}\right)^2 = \frac{1}{\sigma^2} \cdot \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2 \\
&= \frac{1}{\sigma^2} \cdot \sigma^2 = 1
\end{aligned}
$$

$\therefore \sigma_z = 1$

**关键洞察**：标准化本质上是在原数据上做了一个**线性变换**：$z = \frac{1}{\sigma}x - \frac{\mu}{\sigma}$。这个变换把分布的中心移到 0，并把宽度压缩到单位长度。任何通过线性变换可以修正的问题，都可以用标准化解决。
</details>

---

### 题目 8 — 交叉验证的陷阱

你有一个 100 个样本的数据集，做 10 折交叉验证。每次你都在 90 个样本上训练，10 个样本上验证。但你在这 10 次评估之后，根据结果调整了模型的超参数（比如学习率），然后重新跑了 10 折 CV——反复了 5 轮，每次都报告 CV 均值。这种做法的潜在问题是什么？

<details>
<summary>解答</summary>

你实际上是在**用验证集做训练**——只是比较隐蔽而已。

每调整一次超参数并在同一份数据上重新评估，你就在让超参数往"在这 100 个样本上表现更好"的方向偏。反复 5 轮 × 每次看了 100 个样本 = 你的超参数已经被这 100 个样本充分"优化"了。最终报告的 CV 分数会被高估——你的模型已经适应了这 100 个样本的特点。

**正确做法**：

1. **嵌套交叉验证（Nested CV）**：外层 CV 做评估，内层 CV 做超参数搜索。外层从未见过内层选出来的超参数。
2. **保留一个独立的 hold-out set**：在全部实验之前就先锁死一个测试集（如 15% 的样本），所有调参都只在剩下的 85% 上做。最后只在 hold-out 上评估一次。
3. **承认局限性**：如果你的数据集只有 100 个样本，无论怎么划分，评估的不确定性都很大。CV 的 std 会告诉你这个不确定性——如果 std 很大（如 ±5%），你就不应该对不同模型之间的 ±1% 差异做出确定性判断。
</details>

---

### 题目 9 — ColumnTransformer 的执行逻辑

下面这段代码中，`OrdinalEncoder` 的 `categories` 参数里写了 5 个学历。但数据中只出现了 4 种。测试集中出现了第 5 种（'其他'）。会发生什么？

```python
ord_pipe = Pipeline([
    ('ordinal', OrdinalEncoder(
        categories=[['高中', '本科', '硕士', '博士', '其他']],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )),
])
preprocessor = ColumnTransformer([
    ('ord', ord_pipe, ['education']),
])
```

<details>
<summary>解答</summary>

**训练时**：`categories` 中明确了 5 个类别，但训练数据中只有 4 个（没有'其他'）。`OrdinalEncoder` 会按照你指定的顺序分配 0, 1, 2, 3, 4——即使某个类别没在训练数据中出现，它也会"预留"这个位置。训练时没问题。

**测试时**：如果测试集中出现了'其他'（第 5 个类别），`handle_unknown='use_encoded_value'` 告诉 encoder：遇到未知类别时，用 `unknown_value=-1` 代替。所以'其他'会被编码为 -1。

**为什么这是好的设计？**
- 你提前预留了已知类别体系，即使训练时没出现，测试时出现也不会报错
- `unknown_value=-1` 把"没见过"这个信号显式化了——模型可以学到 -1 = "不认识的学历"
- 相比之下，`OneHotEncoder(handle_unknown='ignore')` 遇到未知类别时会把所有列都设为 0，等价于"这个类别是什么都不选"

**注意**：如果你不写 `categories` 参数，`OrdinalEncoder` 会从训练数据中自动学习类别，训练数据中没出现的类别在测试时会被 `handle_unknown` 处理。
</details>

---

### 题目 10 — 综合诊断：预处理流水线的反模式检查

下面是一个新手的 ML 项目流程。找出至少 5 个错误，并解释每个错误的后果：

```python
# 某数据分析师的"信用卡违约预测"项目
df = pd.read_csv('credit_data.csv')

# 1. 发现有缺失值，直接扔掉所有含 NaN 的行
df = df.dropna()                         # 剩 60% 数据

# 2. 把年龄 < 18 或 > 80 的行删掉（认为是异常值）
df = df[df['age'].between(18, 80)]       # 又少 5%

# 3. 把 'gender' 用 LabelEncoder 转成 0/1
df['gender'] = LabelEncoder().fit_transform(df['gender'])

# 4. 标准化所有数值列
num_cols = ['age', 'income', 'bill_amount', 'pay_amount']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. 划分
X, y = df.drop('default', axis=1), df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6. 训练
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

<details>
<summary>解答</summary>

**错误 1：没有了解缺失原因就删除行**
- 后果：丢失了 40% 的数据。如果缺失是 MNAR（信用差的人更不愿意填信息），你删除的恰好是最容易违约的那批人 → 模型评估失真（看起来准确率高，因为难判的样本都被删了）。

**错误 2：粗暴地用年龄范围删"异常值"**
- 后果：18 岁的借款人可能是真实的（大学生信用卡）。年龄 > 80 的也可能是真实的退休借款人。`between(18, 80)` 这个硬编码阈值没有数据依据。应该用 IQR 或结合业务知识判断。

**错误 3：对二分类 gender 用了 LabelEncoder 而非 OneHotEncoder**
- 后果：LabelEncoder 把 gender 映射成 0 和 1，逻辑回归会以为性别是一个数值特征——"性别每增加 1，违约概率变化 β"。如果只有男/女两种值，最终效果等同于 One-Hot（单列 0/1），但这种做法不透明且有误导性。

**错误 4：标准化在划分之前做（数据泄露！）**
- 后果：`scaler.fit_transform(df[num_cols])` 在整个数据集上计算 μ 和 σ，测试集的信息泄漏进了训练集。训练集里 `age` 的标准化值依赖了测试集的人的年龄。部署到真实场景后，因为没有全量数据的 μ 和 σ，模型表现会下降。

**错误 5：没有做 stratified split**
- 后果：信用卡违约是一个不平衡分类任务（违约通常 < 5%）。如果不加 `stratify=y`，测试集中违约样本的比例可能偏离原始分布，甚至可能一个违约样本都没有——你怎么评估一个"找不出任何违约"的模型？

**错误 6（额外）：没有把预处理封装成 Pipeline**
- 后果：部署时需要手动复现上述所有步骤，极容易出错。而且代码不可追溯——你不知道训练时的 scaler 参数是多少。

**错误 7（额外）：只用 Accuracy 评估不平衡分类任务**
- 后果：如果违约率只有 4%，一个"永远预测不违约"的模型 Accuracy = 96%，但这个模型完全无用。应该用 Precision、Recall、F1、ROC-AUC 等指标（将在第 11 章详细讲解）。
</details>

---

## 10. 本章小结

数据预处理是机器学习中被低估但至关重要的环节。本章的核心信息可以浓缩为三条：

**1. "先划分，再处理"——这是防泄露的铁律。** 任何在划分前对全量数据做的 fit 类操作（计算均值、标准差、中位数、类别列表、特征选择分数）都是数据泄露。正确流程永远是：`split → fit on train → transform all`。

**2. 算法决定了预处理策略。** 树模型不需要缩放、可以容忍缺失值和异常值——但你仍然需要理解你的数据。线性模型需要干净的数值输入——缺失值要填、文字要编码、量纲要一致。了解你用的算法对数据的要求，是选择预处理策略的前提。

**3. Pipeline 是你的安全带。** 把预处理步骤全部装进 `Pipeline`，它自动保证 `fit` 只在训练集上发生，`transform` 用的是同一套参数。将来部署模型，一个 `joblib.load` + `pipeline.predict` 就完成了全部数据变换——不需要手动复现任何步骤。

数据预处理本质上是**消除噪点、统一尺度、保留信号**。你不是在"美化"数据，而是在让算法能聚焦在真正有意义的数据模式上——偏差-方差分解中那不可减少的 $\sigma^2$ 在一定程度上也可以通过更好的数据预处理来压缩（识别并剔除数据噪声）。

---

**下一步：[线性回归](./04-linear-regression.md) — 你的第一个机器学习算法。现在数据已经清洗干净、量纲统一、编码正确，是时候让算法上场了。**
