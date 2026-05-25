## 分类模型评估：准确率不是全部

> **核心问题**：为什么一个"99% 准确率"的模型可能毫无价值？如何正确地衡量一个分类模型的好坏？

面试官问"你的模型准确率是多少"，你骄傲地回答"99%"。然后他微微一笑："哦？那如果我把所有样本都预测为 A 类，准确率也是 99%，你的模型和这个'傻瓜分类器'有什么区别？"你哑口无言——这就是为什么我们需要一套完整的分类评估体系。

读完本章你将能够：
- 从零实现精确率、召回率、F1、ROC-AUC 等所有核心指标
- 在草稿纸上手算混淆矩阵和全部指标，真正理解而非记忆公式
- 理解准确率在什么情况下会"欺骗"你
- 掌握 ROC 曲线和 PR 曲线的区别与适用场景
- 针对具体业务场景选择正确的评估指标

---

### 1. 为什么准确率不够？

#### 直觉理解

假设你是一家银行的信用卡欺诈检测团队。历史数据显示每 10000 笔交易中只有 1 笔是欺诈。你训练了一个分类器，准确率高达 **99.99%**。多么令人振奋的数字——直到你发现它把**每一笔交易都预测为正常**。这个模型给出了 0 个欺诈预警，准确率却几乎是完美的。

这个现象有一个正式的名字——**准确率悖论**（Accuracy Paradox）。

#### 形式化定义

准确率的定义极其简单：

$$\text{Accuracy} = \frac{\text{预测正确的样本数}}{\text{总样本数}}$$

当数据集极度不平衡时——比如 99% 正常、1% 欺诈——一个把所有样本都预测为多数的"懒惰分类器"就能获得 99% 的准确率。**准确率高不等于模型好用**，尤其是在少数类比多数类重要得多的情况下。

```python
import numpy as np

# 模拟不平衡数据：10000样本，100个正例
np.random.seed(42)
y_true = np.zeros(10000, dtype=int)
y_true[:100] = 1              # 前100个是正例
np.random.shuffle(y_true)

# "傻瓜分类器"：全部预测为负例
y_pred_dumb = np.zeros(10000, dtype=int)
acc_dumb = np.mean(y_pred_dumb == y_true)
print(f"傻瓜分类器的准确率: {acc_dumb:.4f}")  # 0.9900
print(f"但一个真正的欺诈都没抓到！")
```

#### 应用连接

| 场景 | 不平衡程度（少数类占比） | 只用准确率会怎样 |
|------|-------------------------|-----------------|
| 信用卡欺诈检测 | ~0.1% | 全部预测正常 → 99.9% 准确率，却漏掉所有欺诈 |
| 罕见病筛查 | ~0.01% | 全部预测阴性 → 99.99% 准确率，无人被确诊 |
| 工业缺陷检测 | ~0.5% | 全部预测合格 → 99.5% 准确率，不良品全部出厂 |

当你关心的那个类别本身就很稀少时，准确率不是一个好指标——你需要的是能**分别衡量正例和负例预测能力**的指标。

---

### 2. 混淆矩阵 (Confusion Matrix)

#### 直觉理解

混淆矩阵不过是一张 **"预测 vs 真实"的交叉表格**，把分类结果拆成了四种情况。为了好记，我们用一个医疗检测的类比：

- 你去医院做体检。检测结果可能是阳性（有病）或阴性（没病），而你可能真的有病或真的没病。
- **TP（真阳性）**：你有病，检测也说你有病 → 正确警报 ✓
- **FP（假阳性）**：你没病，检测却说你有病 → 虚惊一场 ✗
- **FN（假阴性）**：你有病，检测却说没病 → 漏诊！（危险）
- **TN（真阴性）**：你没病，检测也说没病 → 确认安全 ✓

这四种情况填在一张 2×2 的表格里，就是**混淆矩阵**。

#### 形式化定义

|  | 预测为正 | 预测为负 |
|--|---------|---------|
| **实际为正** | TP (True Positive) | FN (False Negative) |
| **实际为负** | FP (False Positive) | TN (True Negative) |

TP 和 TN 是模型做对的部分，FP 和 FN 是模型做错的部分——但**两种错误的代价截然不同**。漏诊（FN）可能致命，虚惊（FP）最多让病人多做一次检查。

---

### 3. 手算混淆矩阵与全部指标（10 个样本）

*这一节是本章的核心。关掉 Python，只用手算，彻底理解每个指标从何而来。*

#### 3.1 原始数据

假设我们有一个二分类器，在 10 个测试样本上给出了预测结果（1 代表阳性/正例，0 代表阴性/负例）：

| 样本编号 | 真实标签 y | 预测标签 ŷ |
|:--------:|:----------:|:----------:|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 1 | 0 |
| 4 | 1 | 1 |
| 5 | 0 | 0 |
| 6 | 0 | 1 |
| 7 | 0 | 0 |
| 8 | 1 | 0 |
| 9 | 0 | 0 |
| 10 | 0 | 1 |

> **在草稿纸上逐行检查**：对比每一行的 y 和 ŷ，判断它属于 TP / FP / FN / TN 中的哪一类。

#### 3.2 逐样本归类

拿出纸笔，跟着做：

- 样本 1：y=1, ŷ=1 → 真实为正，预测为正 → **TP ✓**
- 样本 2：y=1, ŷ=1 → 真实为正，预测为正 → **TP ✓**
- 样本 3：y=1, ŷ=0 → 真实为正，预测为负 → **FN ✗**（漏报！）
- 样本 4：y=1, ŷ=1 → 真实为正，预测为正 → **TP ✓**
- 样本 5：y=0, ŷ=0 → 真实为负，预测为负 → **TN ✓**
- 样本 6：y=0, ŷ=1 → 真实为负，预测为正 → **FP ✗**（误报！）
- 样本 7：y=0, ŷ=0 → 真实为负，预测为负 → **TN ✓**
- 样本 8：y=1, ŷ=0 → 真实为正，预测为负 → **FN ✗**（漏报！）
- 样本 9：y=0, ŷ=0 → 真实为负，预测为负 → **TN ✓**
- 样本 10：y=0, ŷ=1 → 真实为负，预测为正 → **FP ✗**（误报！）

#### 3.3 填入混淆矩阵

汇总计数：

|  | 预测为正 (ŷ=1) | 预测为负 (ŷ=0) | **合计** |
|--|:------------:|:------------:|:------:|
| **实际为正 (y=1)** | TP = **3** | FN = **2** | **5** |
| **实际为负 (y=0)** | FP = **2** | TN = **3** | **5** |
| **合计** | **5** | **5** | **10** |

验证：TP + FP + FN + TN = 3 + 2 + 2 + 3 = 10 ✓（所有样本都恰好被归入一类）

#### 3.4 手算四个核心指标

记牢四个数字：**TP=3, FP=2, FN=2, TN=3**，以下所有指标都从这四个数算出。

---

**① 准确率 (Accuracy)**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{3 + 3}{10} = \frac{6}{10} = 0.60$$

解读：10 个样本中，模型预测对了 6 个。但别急着高兴——看下面的拆分。

---

**② 精确率 (Precision)** — 查准率

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{3}{3 + 2} = \frac{3}{5} = 0.60$$

解读：模型一共说了 5 次"这是正例"，其中 3 次说对了。精确率回答：**模型喊"正"的时候，有多大概率真的是正？**

用警察类比：警察抓了 5 个人（TP+FP=5），其中 3 个是真罪犯（TP=3），2 个是冤枉的（FP=2）。精确率 = 3/5 = 60%。

---

**③ 召回率 (Recall)** — 查全率

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{3}{3 + 2} = \frac{3}{5} = 0.60$$

解读：真实世界中一共有 5 个正例（TP+FN=5），模型找到了其中 3 个（TP=3），漏掉了 2 个（FN=2）。召回率回答：**所有真罪犯中，我们抓住了几个？**

---

**④ F1 Score** — 精确率与召回率的调和平均

$$\text{F1} = 2 \cdot \frac{P \cdot R}{P + R} = 2 \cdot \frac{0.6 \times 0.6}{0.6 + 0.6} = 2 \cdot \frac{0.36}{1.2} = 2 \cdot 0.3 = 0.60$$

在这个例子中 P、R 恰好相等，F1 也等于它们。但换个例子就能看到调和平均的"惩罚效应"：

> 假设 P=0.9, R=0.1 → 算术平均 = 0.5 → 调和平均 F1 = 2×0.09/1.0 = **0.18**
>
> 算术平均允许"一个好、一个烂"得 0.5 分，但调和平均说："一个接近 0，整体就该接近 0"——因为一个几乎抓不到正例的模型不是好模型。

---

**⑤ 特异性 (Specificity)**

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{3}{3 + 2} = \frac{3}{5} = 0.60$$

解读：5 个真正的负例中，3 个被正确识别为负。特异性回答：**健康人中，多少人被正确告知"你没病"？**

---

#### 3.5 汇总

| 指标 | 公式 | 计算过程 | 结果 |
|------|------|---------|:----:|
| Accuracy | (TP+TN) / Total | (3+3) / 10 | **0.60** |
| Precision | TP / (TP+FP) | 3 / (3+2) | **0.60** |
| Recall | TP / (TP+FN) | 3 / (3+2) | **0.60** |
| F1 Score | 2PR / (P+R) | 2×0.6×0.6 / (0.6+0.6) | **0.60** |
| Specificity | TN / (TN+FP) | 3 / (3+2) | **0.60** |

> **关键洞察**：Acc=Pre=Rec=F1=0.6 在这个例子中恰好相等——但这只是巧合。下一节你会看到，当类别不平衡时这些指标会剧烈分化。

#### 3.6 Python 验证

手算完了，用 Python 确认一下：

```python
import numpy as np

y_true = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 0])
y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1])

TP = np.sum((y_true == 1) & (y_pred == 1))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))
TN = np.sum((y_true == 0) & (y_pred == 0))

print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")

Accuracy  = (TP + TN) / len(y_true)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)

print(f"Accuracy  = {Accuracy:.2f}")
print(f"Precision = {Precision:.2f}")
print(f"Recall    = {Recall:.2f}")
print(f"F1        = {F1:.2f}")
```

---

### 4. 核心分类指标（通用篇）

上面的手算掌握了本质，现在回到一般情况。以下 Python 实现用于任意规模的二分类数据。

#### 4.1 混淆矩阵

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 生成模拟的二分类预测结果
np.random.seed(42)
n = 1000
y_true = np.concatenate([np.ones(300), np.zeros(700)]).astype(int)
np.random.shuffle(y_true)

# 模拟一个有偏好的分类器：识别正例还行，但有误报
y_pred = np.zeros(n, dtype=int)
pos_mask = (y_true == 1)
y_pred[pos_mask] = np.random.binomial(1, 0.7, size=pos_mask.sum())   # 召回70%
neg_mask = (y_true == 0)
y_pred[neg_mask] = np.random.binomial(1, 0.05, size=neg_mask.sum())  # 5%误报

# ======= 从零手写混淆矩阵 =======
def confusion_matrix_scratch(y_true, y_pred):
    """从零实现混淆矩阵"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])  # sklearn 用的是 [[TN, FP], [FN, TP]]

cm_scratch = confusion_matrix_scratch(y_true, y_pred)
print("手写混淆矩阵:")
print(f"         预测负  预测正")
print(f"实际负    {cm_scratch[0,0]:5d}  {cm_scratch[0,1]:5d}")
print(f"实际正    {cm_scratch[1,0]:5d}  {cm_scratch[1,1]:5d}")

# sklearn 对比
cm_sklearn = confusion_matrix(y_true, y_pred)
print(f"\nsklearn 混淆矩阵:\n{cm_sklearn}")
```

#### 4.2 从零实现所有核心指标

```python
TP = cm_scratch[1, 1]; FP = cm_scratch[0, 1]
FN = cm_scratch[1, 0]; TN = cm_scratch[0, 0]
total = TP + FP + FN + TN

def accuracy_scratch(tp, tn, total):
    return (tp + tn) / total

def precision_scratch(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_scratch(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_scratch(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def specificity_scratch(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

acc   = accuracy_scratch(TP, TN, total)
prec  = precision_scratch(TP, FP)
rec   = recall_scratch(TP, FN)
f1    = f1_scratch(prec, rec)
spec  = specificity_scratch(TN, FP)

print(f"【手写实现】")
print(f"  Accuracy:    {acc:.4f}")
print(f"  Precision:   {prec:.4f}")
print(f"  Recall:      {rec:.4f}")
print(f"  F1 Score:    {f1:.4f}")
print(f"  Specificity: {spec:.4f}")

# ======= sklearn 对比 =======
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"\n【sklearn 验证】")
print(f"  Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
print(f"  Precision:   {precision_score(y_true, y_pred):.4f}")
print(f"  Recall:      {recall_score(y_true, y_pred):.4f}")
print(f"  F1 Score:    {f1_score(y_true, y_pred):.4f}")
```

#### 直觉对比：精确率 vs 召回率

| 思维模型 | 精确率 (Precision) | 召回率 (Recall) |
|----------|-------------------|-----------------|
| 警察 | 抓的人里，多少是真罪犯 | 所有罪犯里，抓了多少 |
| 电商推荐 | 推荐的商品里，多少被点击 | 用户想买的商品里，多少被推荐了 |
| 搜索 | 返回结果里，多少相关 | 相关结果里，多少被返回 |
| 优化倾向 | 减少虚警 | 减少漏报 |

#### 应用连接

**没有哪个指标在所有场景下都是最好的。** 癌症筛查追求高召回率（宁可吓到健康人也不能漏掉患者），反垃圾邮件追求高精确率（宁可放过垃圾邮件也不能把重要邮件扔到垃圾箱）。**你的指标选择 = 你对这两种错误的相对代价判断**。

---

### 5. ROC 曲线与 AUC

#### 直觉理解

分类器输出的是一个概率分数，默认高于 0.5 判为正、低于 0.5 判为负——但这个 0.5 的阈值是人为设定的。如果把阈值调低（比如大于 0.3 就算正），召回率上升（多抓）但误报也上升；阈值调高则相反。

**ROC 曲线**把所有可能的阈值遍历一遍，画出每个阈值下的**TPR（召回率）**对**FPR（假阳性率 = 1 - 特异性）**。你可以把它理解为一张"分类器的说明书"，告诉你这个模型在各个工作点上的表现。

#### 形式化定义

$$TPR = \frac{TP}{TP + FN} \quad (\text{真正率，即 Recall})$$

$$FPR = \frac{FP}{FP + TN} \quad (\text{假正率，= 1 - Specificity})$$

AUC = Area Under the ROC Curve。AUC 有一个漂亮的概率解释：**随机抽取一个正例和一个负例，AUC 等于分类器给正例的分数高于负例的概率**。AUC = 1 表示完美排序，AUC = 0.5 表示随机猜测，AUC < 0.5 表示模型把正负搞反了。

---

### 6. 手算 ROC：三个阈值下的 TPR 与 FPR

*继续用 10 个样本的例子。这次不仅知道预测标签，还知道每个样本被模型打出的概率分数。*

#### 6.1 带分数的 10 个样本

| 样本编号 | 真实标签 y | 模型打分 (score) |
|:--------:|:----------:|:----------------:|
| 1 | 1 | 0.95 |
| 2 | 1 | 0.85 |
| 3 | 1 | 0.40 |
| 4 | 1 | 0.70 |
| 5 | 0 | 0.10 |
| 6 | 0 | 0.60 |
| 7 | 0 | 0.30 |
| 8 | 1 | 0.20 |
| 9 | 0 | 0.05 |
| 10 | 0 | 0.75 |

> 先做一步：**把分数从高到低排序**，方便后续操作：
>
> | 排名 | 样本 | y | score |
> |:----:|:----:|:-:|:-----:|
> | 1 | 1 | 1 | 0.95 |
> | 2 | 2 | 1 | 0.85 |
> | 3 | 10 | 0 | 0.75 |
> | 4 | 4 | 1 | 0.70 |
> | 5 | 6 | 0 | 0.60 |
> | 6 | 3 | 1 | 0.40 |
> | 7 | 7 | 0 | 0.30 |
> | 8 | 8 | 1 | 0.20 |
> | 9 | 5 | 0 | 0.10 |
> | 10 | 9 | 0 | 0.05 |

先记下两个关键常数：**总正例数 = 5，总负例数 = 5**。

---

#### 6.2 阈值 = 0.80（高阈值，保守策略）

判定规则：score ≥ 0.80 → 预测为正，否则为负。

逐样本判断：

- 样本 1 (score=0.95 ≥ 0.80, y=1) → **TP**
- 样本 2 (score=0.85 ≥ 0.80, y=1) → **TP**
- 样本 10 (score=0.75 < 0.80) → 预测为负，y=0 → TN
- 样本 4 (score=0.70 < 0.80) → 预测为负，y=1 → **FN**
- 样本 6 (score=0.60 < 0.80) → 预测为负，y=0 → TN
- 样本 3 (score=0.40 < 0.80) → 预测为负，y=1 → **FN**
- 样本 7 (score=0.30 < 0.80) → 预测为负，y=0 → TN
- 样本 8 (score=0.20 < 0.80) → 预测为负，y=1 → **FN**
- 样本 5 (score=0.10 < 0.80) → 预测为负，y=0 → TN
- 样本 9 (score=0.05 < 0.80) → 预测为负，y=0 → TN

汇总：TP=2, FP=0, FN=3, TN=5

$$\text{TPR} = \frac{2}{2+3} = \frac{2}{5} = 0.40$$

$$\text{FPR} = \frac{0}{0+5} = \frac{0}{5} = 0.00$$

> ROC 点：**(FPR=0.00, TPR=0.40)**
>
> 解读：阈值很高 → 非常"苛刻"，只抓了 40% 的正例，但一个负例都没冤枉。

---

#### 6.3 阈值 = 0.50（中等阈值，默认策略）

判定规则：score ≥ 0.50 → 预测为正。

| 样本 | score | score ≥ 0.50? | y | 归类 |
|:----:|:-----:|:------------:|:-:|:----:|
| 1 | 0.95 | ✓ | 1 | TP |
| 2 | 0.85 | ✓ | 1 | TP |
| 10 | 0.75 | ✓ | 0 | **FP** |
| 4 | 0.70 | ✓ | 1 | TP |
| 6 | 0.60 | ✓ | 0 | **FP** |
| 3 | 0.40 | ✗ | 1 | FN |
| 7 | 0.30 | ✗ | 0 | TN |
| 8 | 0.20 | ✗ | 1 | FN |
| 5 | 0.10 | ✗ | 0 | TN |
| 9 | 0.05 | ✗ | 0 | TN |

汇总：TP=3, FP=2, FN=2, TN=3

$$\text{TPR} = \frac{3}{3+2} = \frac{3}{5} = 0.60$$

$$\text{FPR} = \frac{2}{2+3} = \frac{2}{5} = 0.40$$

> ROC 点：**(FPR=0.40, TPR=0.60)**
>
> 解读：和 Section 3 中默认预测完全一致。抓了 60% 的正例，但也冤枉了 40% 的负例。

---

#### 6.4 阈值 = 0.25（低阈值，激进策略）

判定规则：score ≥ 0.25 → 预测为正。

| 样本 | score | score ≥ 0.25? | y | 归类 |
|:----:|:-----:|:------------:|:-:|:----:|
| 1 | 0.95 | ✓ | 1 | TP |
| 2 | 0.85 | ✓ | 1 | TP |
| 10 | 0.75 | ✓ | 0 | FP |
| 4 | 0.70 | ✓ | 1 | TP |
| 6 | 0.60 | ✓ | 0 | FP |
| 3 | 0.40 | ✓ | 1 | TP |
| 7 | 0.30 | ✓ | 0 | **FP** |
| 8 | 0.20 | ✗ | 1 | **FN** |
| 5 | 0.10 | ✗ | 0 | TN |
| 9 | 0.05 | ✗ | 0 | TN |

汇总：TP=4, FP=3, FN=1, TN=2

$$\text{TPR} = \frac{4}{4+1} = \frac{4}{5} = 0.80$$

$$\text{FPR} = \frac{3}{3+2} = \frac{3}{5} = 0.60$$

> ROC 点：**(FPR=0.60, TPR=0.80)**
>
> 解读：阈值很低 → 非常"宽松"，抓了 80% 的正例（仅漏掉 1 个），但冤枉了 60% 的负例。

---

#### 6.5 三个点连起来

| 阈值 | 策略 | FPR | TPR | ROC 坐标 |
|:----:|------|:---:|:---:|:--------:|
| 0.80 | 保守（高门槛） | 0.00 | 0.40 | (0.00, 0.40) |
| 0.50 | 默认 | 0.40 | 0.60 | (0.40, 0.60) |
| 0.25 | 激进（低门槛） | 0.60 | 0.80 | (0.60, 0.80) |

> 把它们画在坐标纸上，再加两个端点 **(0, 0)** 和 **(1, 1)**，你就得到了一条 ROC 曲线的**手算草图**。

两点洞察：

1. **阈值越低，FPR 和 TPR 同时升高**——你抓了更多正例，但也误判了更多负例。这就是经典的 tradeoff。
2. **点在对角线 y=x 上方**——说明模型比随机猜测强（随机猜测的 TPR = FPR）。本例中曲线偏高，说明排序能力不错。

---

### 7. ROC 曲线与 AUC（通用篇）

#### 从零实现 ROC 和 AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

# 模拟概率分数（正例分数偏高，但不是完美的）
np.random.seed(42)
y_score = np.where(
    y_true == 1,
    np.random.beta(7, 3, size=n),   # 正例分数偏右
    np.random.beta(3, 7, size=n)    # 负例分数偏左
)

# ======= 从零计算 ROC 曲线 =======
def roc_curve_scratch(y_true, y_score):
    """从零计算 ROC 曲线的 FPR 和 TPR"""
    desc_idx = np.argsort(y_score)[::-1]  # 分数从高到低排序
    y_true_sorted = y_true[desc_idx]

    tpr, fpr = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    tp = fp = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    tpr = np.array([0.0] + tpr)
    fpr = np.array([0.0] + fpr)
    return fpr, tpr

# ======= 从零计算 AUC（梯形积分） =======
def auc_scratch(fpr, tpr):
    return np.trapz(tpr, fpr)

fpr_scratch, tpr_scratch = roc_curve_scratch(y_true, y_score)
auc_val = auc_scratch(fpr_scratch, tpr_scratch)
print(f"手写 AUC: {auc_val:.4f}")

# sklearn 对比
fpr_sk, tpr_sk, _ = roc_curve(y_true, y_score)
auc_sk = roc_auc_score(y_true, y_score)
print(f"sklearn AUC: {auc_sk:.4f}")

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_scratch, tpr_scratch, 'b-', lw=2, label=f'ROC (AUC = {auc_val:.3f})')
plt.plot(fpr_sk, tpr_sk, 'r--', lw=1, label=f'sklearn ROC (AUC = {auc_sk:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='随机猜测 (AUC=0.5)')
plt.xlabel('假正率 (FPR)')
plt.ylabel('真正率 (TPR / Recall)')
plt.title('ROC 曲线 — 手写 vs sklearn')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

#### 应用连接

ROC 对类别平衡不敏感——正负比例变化时 ROC 曲线几乎不变，因此适合评估**类别分布相对均匀**的模型对比。但如果正例极端稀少（< 1%），PR 曲线会是更好的选择。

---

### 8. PR 曲线 (Precision-Recall Curve)

#### 直觉理解

ROC 曲线有个隐藏的盲点：当正例非常稀少时，TN（真负例）数量极其庞大，导致 FPR = FP/(FP+TN) 总是很小，ROC 曲线看起来"很好"——但实际精确率可能不高。PR 曲线直接刻画精确率和召回率的权衡，对不平衡数据更敏感。

#### 形式化定义

PR 曲线遍历所有阈值，画出每个阈值下的 Precision（精确率）对 Recall（召回率）。总结指标是**平均精确率**（Average Precision, AP）。

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# ======= 从零实现 PR 曲线 =======
def pr_curve_scratch(y_true, y_score):
    """从零计算 PR 曲线的 precision 和 recall"""
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]

    precisions, recalls = [], []
    n_pos = np.sum(y_true == 1)
    tp = fp = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / n_pos
        precisions.append(p)
        recalls.append(r)

    return np.array(precisions), np.array(recalls)

prec_scratch, rec_scratch = pr_curve_scratch(y_true, y_score)
ap_val = np.trapz(prec_scratch, rec_scratch)

print(f"手写 AP: {ap_val:.4f}")

prec_sk, rec_sk, _ = precision_recall_curve(y_true, y_score)
ap_sk = average_precision_score(y_true, y_score)
print(f"sklearn AP: {ap_sk:.4f}")

# 绘制 PR 曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(rec_scratch, prec_scratch, 'b-', lw=2, label=f'手写 PR (AP = {ap_val:.3f})')
axes[0].plot(rec_sk, prec_sk, 'r--', lw=1, label=f'sklearn PR (AP = {ap_sk:.3f})')
axes[0].set_xlabel('召回率 (Recall)'); axes[0].set_ylabel('精确率 (Precision)')
axes[0].set_title('PR 曲线 — 手写 vs sklearn')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(fpr_scratch, tpr_scratch, 'b-', lw=2, label=f'ROC (AUC = {auc_val:.3f})')
axes[1].plot(rec_scratch, prec_scratch, 'r-', lw=2, label=f'PR (AP = {ap_val:.3f})')
axes[1].set_xlabel('召回率 / 假正率'); axes[1].set_ylabel('真正率 / 精确率')
axes[1].set_title('ROC vs PR — 同一模型的两个视角')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

#### 应用连接

| 曲线 | 横轴 | 纵轴 | 何时用 |
|------|------|------|--------|
| ROC | FPR = FP/(FP+TN) | TPR = Recall | 类别大致平衡，care about 排序质量 |
| PR | Recall | Precision | 正例极少（<1%），care about 少类的预测质量 |

对于欺诈检测、罕见病筛查等严重不平衡问题，**PR 曲线比 ROC 更能暴露模型的真实质量**。

---

### 9. 多分类评估

#### 直觉理解

当有 3 个或更多类别时，我们需要把多分类"拆解"成多个二分类问题来评估。有三种聚合策略：

- **Macro-average**：每个类别算一个指标，然后直接取平均。小类和大类**权重相同**，不受样本量影响。
- **Micro-average**：把所有类别的 TP、FP、FN 加在一起算一个全局指标。由大类主导。
- **Weighted-average**：Macro 的加权版，按每个类别的样本数加权。

```python
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_multi = rf.predict(X_test)

# ======= 从零实现 macro-average Precision =======
def macro_precision_scratch(y_true, y_pred, n_classes):
    precisions = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(p)
    return np.mean(precisions)

# ======= 从零实现 micro-average Precision =======
def micro_precision_scratch(y_true, y_pred, n_classes):
    tp_sum = fp_sum = 0
    for c in range(n_classes):
        tp_sum += np.sum((y_true == c) & (y_pred == c))
        fp_sum += np.sum((y_true != c) & (y_pred == c))
    return tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0

n_cls = len(np.unique(y_train))
macro_p = macro_precision_scratch(y_test, y_pred_multi, n_cls)
micro_p = micro_precision_scratch(y_test, y_pred_multi, n_cls)

print(f"手写 Macro Precision:  {macro_p:.4f}")
print(f"手写 Micro Precision:  {micro_p:.4f}")

print(f"\n=== sklearn Classification Report ===\n")
print(classification_report(y_test, y_pred_multi, target_names=iris.target_names))
```

#### 应用连接

| 策略 | 何时用 | 例子 |
|------|--------|------|
| Macro | 每个类别同等重要 | 情感分析（正面/中性/负面同等重要） |
| Micro | 关心全局表现 | 图像分类，总体准确率最重要 |
| Weighted | 尊重自然分布，但大小类都考虑 | 大多数多分类问题的默认选择 |

---

### 10. 选择正确的指标

没有放之四海皆准的指标。**正确的指标取决于错误的代价**。

| 场景 | 推荐的指标 | 原因 |
|------|-----------|------|
| 癌症筛查 | Recall, Sensitivity | 错过一个患者代价太高 |
| 垃圾邮件过滤 | Precision | 误杀一封重要邮件比漏过 10 封垃圾更糟 |
| 信用卡欺诈 | F1, PR-AUC | 需要平衡误报（打扰用户）和漏报（金钱损失） |
| 信息检索 / 推荐 | Precision@K, Recall@K | 用户只看前几条结果 |
| 类别平衡的竞赛 | Accuracy, F1 | 简单有效 |
| 模型对比（不关心阈值） | AUC | 衡量整体排序能力 |

#### 思维框架

面对一个分类问题，问自己两个问题：

1. **正例 vs 负例的代价是否对等？** 不等 → 精确率/召回率/F1；对等 → 准确率。
2. **正例是否极端稀少？** 是 → PR 曲线；否 → ROC 曲线。

---

### 11. 实战：完整分类评估

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成不平衡二分类数据（正例约 10%）
X, y = make_classification(
    n_samples=5000, n_features=20, weights=[0.9, 0.1],
    flip_y=0.05, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 训练多种分类器
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM (RBF)":           SVC(probability=True),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8),
}

results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    results.append({
        "模型": name,
        "准确率":  accuracy_score(y_test, y_pred),
        "精确率":  precision_score(y_test, y_pred),
        "召回率":  recall_score(y_test, y_pred),
        "F1":      f1_score(y_test, y_pred),
        "AUC":     roc_auc_score(y_test, y_prob),
        "AP":      average_precision_score(y_test, y_prob),
    })

# 打印对比表
print(f"{'模型':<22} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'AUC':>8} {'AP':>8}")
print("-" * 72)
for r in results:
    print(f"{r['模型']:<22} {r['准确率']:8.4f} {r['精确率']:8.4f} {r['召回率']:8.4f} "
          f"{r['F1']:8.4f} {r['AUC']:8.4f} {r['AP']:8.4f}")

# 重点：展示准确率的欺骗性
print(f"\n=== 如果只看准确率 ===")
best_acc = max(results, key=lambda r: r['准确率'])
best_f1  = max(results, key=lambda r: r['F1'])
print(f"最高准确率: {best_acc['模型']} (Acc={best_acc['准确率']:.4f}, F1={best_acc['F1']:.4f})")
print(f"最高 F1:     {best_f1['模型']} (Acc={best_f1['准确率']:.4f}, F1={best_f1['F1']:.4f})")
```

运行这段代码你会看到：某些模型准确率很高但 F1 很低——它们通过"牺牲少数类"换取了高准确率，这在真正关心少数类的场景中是致命的。

---

### 12. 思考题

*以下题目全部基于 Section 3 和 Section 6 的 10 个样本数据。强烈建议用纸笔手算后再看答案。*

---

**Q1**：如果模型把所有 10 个样本全部预测为正类（ŷ 全为 1），计算 Precision、Recall、Accuracy。

<details>
<summary><b>点击查看解答</b></summary>

全部预测为正：
- TP = 5（所有正例都被"命中"）
- FP = 5（所有负例都被误判）
- FN = 0（没有漏掉的正例）
- TN = 0（没有正确识别的负例）

$$Precision = \frac{5}{5+5} = 0.50$$
$$Recall = \frac{5}{5+0} = 1.00$$
$$Accuracy = \frac{5+0}{10} = 0.50$$

> 模型气势汹汹地说"全部是正！"——确实抓住了每一个正例（Recall=1），但精确率只有 50%，一半的警报是虚惊。这恰好印证了 P-R 的 tradeoff：**Recall 极高时 Precision 必然受损**（除非模型完美）。
</details>

---

**Q2**：如果模型把所有 10 个样本全部预测为负类（ŷ 全为 0），计算四个核心指标。这种模型有实用价值吗？

<details>
<summary><b>点击查看解答</b></summary>

全部预测为负：
- TP = 0, FP = 0, FN = 5, TN = 5

$$Accuracy = \frac{0+5}{10} = 0.50$$
$$Precision = \frac{0}{0+0} \rightarrow \text{未定义（一般记为 0 或 1，取决于约定）}$$
$$Recall = \frac{0}{0+5} = 0.00$$
$$F1 \rightarrow 0$$

> 召回率为 0——一个正例都没抓到。这种模型唯一的"价值"是说明：**当正负类各占一半时，全猜同一类只能拿到 50% 准确率**——比抛硬币好不了多少。
</details>

---

**Q3**：在 Section 3 的例子中（10 个样本，默认阈值 0.5），如果把样本 6 的预测从 1 改为 0（其他不变），Precision 和 Recall 如何变化？

<details>
<summary><b>点击查看解答</b></summary>

修改前：TP=3, FP=2, FN=2, TN=3

样本 6：y=0, ŷ=1 → 原来是 FP。改为 ŷ=0 后：

新的混淆矩阵：TP=3, FP=1, FN=2, TN=4

$$Precision = \frac{3}{3+1} = \frac{3}{4} = 0.75 \quad (\text{之前 0.60, 上升})$$
$$Recall = \frac{3}{3+2} = \frac{3}{5} = 0.60 \quad (\text{不变})$$

> 消除一个误报（FP → TN）：Precision 上升而 Recall 不变。这说明 **Precision 对 FP 敏感，Recall 不受 FP 影响**——因为 Recall 的分母里没有 FP。
</details>

---

**Q4**：继续用 Section 3 的例子。如果把样本 8 的预测从 0 改为 1（其他不变），Precision 和 Recall 如何变化？

<details>
<summary><b>点击查看解答</b></summary>

修改前：TP=3, FP=2, FN=2, TN=3

样本 8：y=1, ŷ=0 → 原来是 FN。改为 ŷ=1 后：

新的混淆矩阵：TP=4, FP=2, FN=1, TN=3

$$Precision = \frac{4}{4+2} = \frac{4}{6} \approx 0.667 \quad (\text{之前 0.60, 微升})$$
$$Recall = \frac{4}{4+1} = \frac{4}{5} = 0.80 \quad (\text{之前 0.60, 显著上升})$$

> 找回一个漏报（FN → TP）：Recall 显著上升，Precision 也略微上升（因为 TP 增加了，而 FP 没变）。这说明 **Recall 对 FN 敏感**——每减少一个漏报，Recall 都明显改善。
</details>

---

**Q5**：在 Section 6 的手算 ROC 中，阈值 = 0.50 时得到的 ROC 点是 (FPR=0.40, TPR=0.60)。请问 (0.40, 0.60) 这个点和随机猜测线 y=x 的相对位置说明了什么？

<details>
<summary><b>点击查看解答</b></summary>

点 (0.40, 0.60) 在 y=x 线的**上方**（因为 0.60 > 0.40）。

这说明在该阈值下，模型的 TPR（0.60）高于 FPR（0.40），即**模型识别正例的能力高于误判负例的能力**。

如果点在 y=x 上（如 (0.4, 0.4)），说明 TPR = FPR，模型的表现等价于抛一枚不均匀的硬币——对正负例没有区分能力。

如果点在 y=x 下方（如 (0.6, 0.4)），说明模型"搞反了"——它判为正的样本中负例更多。这种情况在实际中很少见（通常意味着模型训练有严重问题），但理论上可以将所有预测反过来获得一个 AUC > 0.5 的分类器。

> 核心结论：**ROC 曲线在 y=x 上方的面积代表模型优于随机的程度**。AUC 就是这个面积的度量。
</details>

---

**Q6**：手动计算 Section 6 中三个 ROC 点的"阶梯状" AUC 近似值，并与随机猜测（AUC=0.5）比较。

<details>
<summary><b>点击查看解答</b></summary>

三个点：(0, 0) → (0.00, 0.40) → (0.40, 0.60) → (0.60, 0.80) → (1, 1)

用梯形法近似 AUC：

梯形 1（x: 0 → 0.00）：宽度 0，面积 0

梯形 2（x: 0.00 → 0.40）：高度 (0.40+0.60)/2，宽度 0.40 → 面积 = 0.50 × 0.40 = **0.200**

梯形 3（x: 0.40 → 0.60）：高度 (0.60+0.80)/2，宽度 0.20 → 面积 = 0.70 × 0.20 = **0.140**

梯形 4（x: 0.60 → 1.00）：高度 (0.80+1.00)/2，宽度 0.40 → 面积 = 0.90 × 0.40 = **0.360**

总 AUC ≈ 0.200 + 0.140 + 0.360 = **0.700**

> 0.70 > 0.50，说明模型排序能力优于随机。仅仅 3 个阈值就能得到一个合理的 AUC 近似——这就是 ROC 思路的优雅之处。
</details>

---

**Q7**：为什么 F1 使用调和平均而不是算术平均？用一个极端例子说明。

<details>
<summary><b>点击查看解答</b></summary>

考虑一个极端模型：Precision = 0.99, Recall = 0.01。

- 算术平均：(0.99 + 0.01) / 2 = **0.50**
- 调和平均（F1）：2 × 0.99 × 0.01 / (0.99 + 0.01) = 0.0198 / 1.0 ≈ **0.02**

算术平均给出 0.50——看起来"及格了"。但实际模型只抓住了 1% 的正例，几乎毫无用处。

调和平均的数学性质：**只要有一个值接近 0，结果就接近 0**。这正是我们需要的——一个 Recall 接近 0 的模型，不管 Precision 多高都不该得分高。

> 结论：F1 = 高 当且仅当 P 和 R **同时**高。这就是调和平均"惩罚极端值"的直观含义。
</details>

---

**Q8**：假设你正在训练一个癌症筛查模型。Precision = 0.1, Recall = 0.99。这个模型值得部署吗？为什么？

<details>
<summary><b>点击查看解答</b></summary>

**值得部署。**

Precision = 0.1 意味着每 10 个被标记为"可能患癌"的人中，只有 1 个真正患癌——9 个是虚惊（FP）。

Recall = 0.99 意味着 99% 的真正患者被筛查出来——只漏掉 1%（FN）。

在癌症筛查的场景中：
- **FN（漏诊）的代价**：患者被错误告知"没病"，延误治疗，可能危及生命——代价**极高**。
- **FP（虚惊）的代价**：健康人被要求做进一步检查，多花时间和金钱——代价**相对较低**，且后续的精确检查（如活检）可以快速排除 FP。

> 部署策略：这个模型适合作为**初筛工具**。Recall 极高确保不漏掉患者；Precision 低由后续的精确检查来弥补。这就是为什么医疗初筛中的阳性率设置得很宽松。
</details>

---

**Q9**：解释为什么在处理极端不平衡数据（正例 < 1%）时，PR 曲线比 ROC 曲线更能反映模型质量。

<details>
<summary><b>点击查看解答</b></summary>

关键在于两个曲线的分母性质不同。

**ROC 的横轴是 FPR = FP / (FP + TN)**。当负例占 99% 以上时，TN 极其庞大。即使 FP 从 0 增加到 100，FPR 仍然接近 0（因为分母被 TN 主导）。ROC 曲线死死地趴在 y 轴左侧——看起来"很好"，但它掩盖了 FP 的绝对数量。

**PR 的纵轴是 Precision = TP / (TP + FP)**。分母不受 TN 影响，只关心真正的正例和误报。当正例稀少时，即使少量 FP 也能把 Precision 从接近 1 拉到很低。

举例：10000 个样本，100 个正例，9900 个负例。

模型 A 在阈值 0.5 时：TP=80, FP=50

- ROC: FPR = 50/(50+9850) = 0.005 → 几乎为 0，"看起来很完美"
- PR: Precision = 80/(80+50) = 0.615 → 不到 2/3，暴露了真实问题

> **ROC 对类别比例不敏感（这是它的优点也是缺点），PR 对不平衡数据更诚实。**
</details>

---

**Q10**：如果你拿到一个 AUC = 0.92 的模型，是否可以断定它比 AUC = 0.88 的模型更好？什么情况下不能这样比？

<details>
<summary><b>点击查看解答</b></summary>

**不能简单断定。** 以下情况需要谨慎：

1. **不同数据集**：AUC 是数据集相关的。在自己精心筛选的 1000 样本上拿到 0.92，不能和对方在 100 万嘈杂样本上拿到的 0.88 直接比较。

2. **AUC 不反映校准**：AUC 衡量的是排序能力，而不是概率校准质量。一个 AUC=0.92 但严重过拟合的模型，其预测概率可能不可靠（比如把所有正例预测概率集中在 0.51-0.55），而 AUC=0.88 的模型可能概率校准得很好。

3. **业务场景不匹配**：AUC 综合了整个 ROC 曲线。当你只关心 FPR < 0.1 的区域时（比如欺诈检测，容忍的误报率很低），应该比较**部分 AUC（partial AUC, pAUC）**而非全局 AUC。模型 A 在低 FPR 区域可能不如模型 B，但高 FPR 区域拉高了整体 AUC。

4. **统计显著性**：0.92 vs 0.88 的差异是否显著？需要用 DeLong 检验或 bootstrap 来确认——尤其是在小样本上。

> 结论：AUC 是很好的初筛指标，但做最终决策时需结合业务需求（关注哪个 FPR 区间？）和统计检验。

*补充——DeLong 检验的 Python 实现思路（不展开，仅供查阅）：*
```python
# 使用 scipy 计算 AUC 的方差和协方差，构造 z 统计量
# from scipy.stats import norm
# 参考 sklearn.metrics.roc_auc_score 配合手动实现 DeLong
```
</details>

---

### 总结

本章你学到了分类评估的完整武器库：从最基础的混淆矩阵出发，在 10 个样本上逐行手算 TP/FP/FN/TN，一路推导出精确率、召回率、F1，再到三个阈值下手算 ROC 坐标，最后用 Python 验证了所有结论。

核心心法只有一句：**没有万能指标，只有正确的代价判断**。在开始评估之前，先想清楚你更怕 FP（虚警）还是 FN（漏报）——你的指标选择应该直接反映这个答案。

下一步：[集成学习](./12-ensemble-methods.md)
