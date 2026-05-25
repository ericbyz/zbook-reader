# 模型评估与调参：让模型达到最佳状态

> **核心问题**：训练集上表现完美的模型，为什么一到测试集就"翻车"？交叉验证到底在验证什么？应该用网格搜索暴力穷举参数，还是让贝叶斯优化"聪明地猜"？

前面的章节你学了十几种算法，但每个算法都有一堆旋钮——KNN 的 K 取多少？SVM 的 C 和 gamma 怎么设？随机森林要多少棵树？本章回答的是一个本质问题：**如何系统性地评估一个模型并找到最佳参数配置**。这比选择算法本身至少重要一半——一个调好的 Logistic Regression 打败未调参的 XGBoost 并不稀奇。

读完本章你将能够：
- 熟练使用 K-Fold、分层 K-Fold、时序拆分等交叉验证策略
- 用 Grid Search、Random Search 和贝叶斯优化高效搜索超参数空间
- 绘制学习曲线和验证曲线诊断过拟合/欠拟合
- 处理不平衡数据，并通过统计检验科学地比较模型

---

## 1. 模型评估的重要性

### 直觉理解

你在考试前做了一套模拟题，拿了满分。这能保证你正式考试也满分吗？不能——因为你可能恰好做过这套题里的同类题目（记住了答案），也可能这套题出得太简单（覆盖不了真实考试的难度）。**训练集上的表现，不等同于真实场景的表现。**

机器学习模型面临同样的问题。一个复杂模型可以把训练数据"背下来"（过拟合），一个太简单的模型连训练数据都拟合不好（欠拟合）。我们需要的是**泛化能力**——模型对未见过的数据也能做出准确预测。

评估策略取决于两个因素：

| 数据量 | 推荐策略 | 原因 |
|--------|---------|------|
| 大数据（>10万） | 单次留出法（Train/Valid/Test 分割） | 数据足够多，一次分割就够可靠 |
| 中等数据（1千~10万） | K 折交叉验证 | 用多次平均抵消划分的偶然性 |
| 小数据（<1千） | 留一法（LOO）或重复 K 折 | 每一条数据都不能浪费 |

### 形式化定义

设训练集 $D_{\text{train}}$、验证集 $D_{\text{val}}$、测试集 $D_{\text{test}}$。训练的目标是找到参数 $\theta$ 使经验风险最小：

$$\hat{\theta} = \mathop{\arg\min}_{\theta} \frac{1}{|D_{\text{train}}|} \sum_{(x,y) \in D_{\text{train}}} L(f_\theta(x), y)$$

但我们真正关心的是**泛化误差**（期望风险）：

$$R(f) = \mathbb{E}_{(x,y) \sim P} [L(f(x), y)]$$

用测试集上的误差 $\hat{R}_{\text{test}}$ 来近似泛化误差时，有一个简单但常被违反的铁律：

> **测试集只能用一次——用在最终评估。绝对不能用测试集指导调参，否则测试集就变成了验证集，你的泛化误差估计不再可靠。**

### Python 代码验证：训练误差 vs 测试误差

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n = 200
X = np.linspace(-5, 5, n).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.15, n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

depths = range(1, 21)
train_errs, test_errs = [], []

for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    train_errs.append(mean_squared_error(y_train, tree.predict(X_train)))
    test_errs.append(mean_squared_error(y_test, tree.predict(X_test)))

best_d = depths[np.argmin(test_errs)]
print(f"训练误差最低的深度: max_depth=20, 训练MSE={train_errs[-1]:.4f}")
print(f"测试误差最低的深度: max_depth={best_d}, 测试MSE={min(test_errs):.4f}")
print(f"\\n深度 20 的测试误差是深度 {best_d} 的 {test_errs[-1]/min(test_errs):.1f} 倍")
```

**输出示例：**
```
训练误差最低的深度: max_depth=20, 训练MSE=0.0042
测试误差最低的深度: max_depth=5,  测试MSE=0.0248

深度 20 的测试误差是深度 5 的 3.2 倍
```

### 应用连接

只看训练误差，你会选 `max_depth=20`——它把训练数据拟合得近乎完美。但它的测试误差却是最优模型的 **3 倍还多**。这就是为什么面试官会反复问你："你的验证集和测试集是怎么划分的？"

---

## 2. 交叉验证深度解析

### 直觉理解

把数据切一刀分成训练和测试集，结果依赖这一刀切在哪——换一刀，评估结果可能大不相同。交叉验证的思路很简单：**多切几刀，取平均值**。这样既利用了所有数据评估，又让评估结果更稳定。

### K-Fold 交叉验证

把数据均匀分成 K 份。每次用 K-1 份训练、1 份验证，重复 K 次（每次轮换验证集）。最终报告 K 次得分的均值 ± 标准差。

```
第 1 折: [val][train][train][train][train]
第 2 折: [train][val][train][train][train]
第 3 折: [train][train][val][train][train]
第 4 折: [train][train][train][val][train]
第 5 折: [train][train][train][train][val]
```

### 分层 K 折（Stratified K-Fold）

分类任务中，如果某折的验证集恰好全是同一类，评估就毫无意义。分层 K 折保证**每折中各类别的比例 ≈ 原始数据中各类别的比例**。

### 留一法（Leave-One-Out, LOO）

K = N（样本数）的极端情况：每次只留一个样本验证。优点是几乎无偏（训练集最大），缺点是计算量巨大（训练 N 次模型）。

### 时序拆分（Time Series Split）

时间序列数据不能随机打乱——你不能用"未来"数据训练去预测"过去"。时序拆分保持时间顺序，永远用较早的数据训练、较晚的数据验证。

### 嵌套交叉验证

当需要同时调参和评估时，简单交叉验证会"泄露"验证集信息到参数选择中。正确做法是嵌套两层：**外层循环**评估泛化性能，**内层循环**选择最优超参数。

### Python 实现

```python
import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold, LeaveOneOut,
                                     TimeSeriesSplit, cross_val_score,
                                     GridSearchCV, cross_validate)
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=200, n_features=5, n_classes=2,
                           weights=[0.7, 0.3], random_state=42)

# 1. 普通 K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(SVC(kernel='rbf'), X, y, cv=kf, scoring='f1')
print(f"K-Fold         F1: {scores_kf.mean():.3f} ± {scores_kf.std():.3f}")

# 2. Stratified K-Fold（分类任务默认使用这个！）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(SVC(kernel='rbf'), X, y, cv=skf, scoring='f1')
print(f"Stratified K-Fold F1: {scores_skf.mean():.3f} ± {scores_skf.std():.3f}")

# 3. Leave-One-Out（数据很少时使用）
loo = LeaveOneOut()
scores_loo = cross_val_score(SVC(kernel='rbf'), X[:30], y[:30],
                              cv=loo, scoring='f1')
print(f"LOO (n=30)     F1: {scores_loo.mean():.3f} ± {scores_loo.std():.3f}")

# 4. Time Series Split
tscv = TimeSeriesSplit(n_splits=5, test_size=30)
X_seq = np.arange(300).reshape(-1, 1)
y_seq = np.sin(np.arange(300) * 0.1)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
    print(f"时序拆分 折{fold+1}: train=[{train_idx[0]}:{train_idx[-1]}], "
          f"val=[{val_idx[0]}:{val_idx[-1]}]")

# 5. 嵌套交叉验证
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=inner_cv, scoring='f1')
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='f1')
print(f"\\n嵌套CV 泛化 F1: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

**输出示例：**
```
K-Fold         F1: 0.724 ± 0.086
Stratified K-Fold F1: 0.778 ± 0.047
LOO (n=30)     F1: 0.800 ± 0.400
时序拆分 折1: train=[0:149], val=[150:179]
时序拆分 折2: train=[0:179], val=[180:209]
时序拆分 折3: train=[0:209], val=[210:239]
时序拆分 折4: train=[0:239], val=[240:269]
时序拆分 折5: train=[0:269], val=[270:299]

嵌套CV 泛化 F1: 0.734 ± 0.061
```

分层 K 折的 F1 标准差（0.047）远小于普通 K 折（0.086）——这就是分层采样带来的稳定性。

### 应用连接

| 场景 | 推荐 CV 策略 |
|------|------------|
| 普通分类/回归 | StratifiedKFold / KFold，K=5 或 10 |
| 类别极度不平衡 | Repeated StratifiedKFold（多次重复取平均）|
| 时间序列预测 | TimeSeriesSplit |
| 样本 < 100 | LeaveOneOut 或 10轮 repeated 5-fold |
| 调参 + 评估同时 | 嵌套交叉验证 |

---

## 3. 超参数调优

### 直觉理解

模型参数（如线性回归的系数 $w$）是**模型自己从数据中学出来的**。超参数（如 KNN 的 K、SVM 的 C）是**你在训练前手动设定的**。调参就是寻找一组最优超参数，使模型在验证集上表现最好。

搜索方式有三种思路：

| 方式 | 比喻 | 优点 | 缺点 |
|------|------|------|------|
| **Grid Search** | 把地图画成网格，每个交叉点都走一遍 | 保证找到网格内最优 | 维度稍高就爆炸 |
| **Random Search** | 在地图上随机撒点 | 高维空间更高效 | 可能错过最优区域 |
| **Bayesian Optimization** | 用已有观测推测哪里可能更好，优先探索 | 聪明且高效 | 串行，不能并行 |

### 形式化定义

超参数优化问题可以形式化为：

$$\lambda^* = \mathop{\arg\min}_{\lambda} \mathbb{E}_{(x,y) \sim P_{\text{val}}} [L(f_{\hat{\theta}(\lambda)}(x), y)]$$

贝叶斯优化的核心是构建一个**代理模型**（通常是高斯过程）来近似目标函数 $f(\lambda) = \text{score}(\lambda)$，然后通过**采集函数**（如 Expected Improvement）决定下一个评估点：

$$EI(\lambda) = \mathbb{E}[\max(f(\lambda) - f^*, 0)]$$

每次迭代都在"有利可图"和"不确定区域"之间权衡（Exploitation vs Exploration）。

### Python 实现

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)

# === Grid Search ===
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)
print(f"GridSearch 最佳参数: {grid.best_params_}")
print(f"GridSearch 最佳得分: {grid.best_score_:.4f}")
print(f"搜索的组合数: {4 * 4 * 2}")

# === Random Search ===
param_dist = {
    'C': loguniform(1e-3, 1e3),
    'gamma': loguniform(1e-4, 1e0),
    'kernel': ['rbf', 'poly']
}
random_search = RandomizedSearchCV(
    SVC(), param_dist, n_iter=30, cv=5,
    scoring='accuracy', random_state=42, n_jobs=-1
)
random_search.fit(X, y)
print(f"\\nRandomSearch 最佳参数: {random_search.best_params_}")
print(f"RandomSearch 最佳得分: {random_search.best_score_:.4f}")
print(f"搜索的组合数: 30（远少于 Grid 的 {4*4*2}）")
```

**输出示例：**
```
GridSearch 最佳参数: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
GridSearch 最佳得分: 0.9855
搜索的组合数: 32

RandomSearch 最佳参数: {'C': 12.34, 'gamma': 0.008, 'kernel': 'rbf'}
RandomSearch 最佳得分: 0.9860
搜索的组合数: 30（远少于 Grid 的 32）
```

Random Search 只搜了 30 组就胜过了 Grid 的 32 组全搜——因为它能在关键维度（C 和 gamma）上尝试更多非网格值。

### 应用连接

实战中默认使用 **RandomizedSearchCV 粗搜**（50-100 组）找到大致范围，再在最佳区域用 **GridSearchCV 精搜**。如果计算资源充裕且参数量大，推荐使用 **Optuna** 或 **Hyperopt** 进行贝叶斯优化。核心原则：**随机优于网格，聪明优于随机**。

---

## 4. 学习曲线与验证曲线

### 直觉理解

模型表现不好，到底是**数据不够**，还是**模型不当**？学习曲线能告诉你答案。它的 x 轴是训练集大小，y 轴是得分——随着数据增多，训练集和验证集的得分如何变化？

- **高偏差**：两条曲线快速趋于平坦且靠拢——加数据没用，得换更强模型
- **高方差**：两条曲线之间有宽大鸿沟——加数据或加正则化

验证曲线则固定数据量，改变某个超参数的值，看训练/验证得分如何变化——帮你在"因太简单而欠拟合"和"因太复杂而过拟合"之间找到最优平衡。

### 形式化定义

学习曲线 $L(n)$：用 $n$ 个训练样本拟合模型，记录训练得分 $S_{\text{train}}(n)$ 和验证得分 $S_{\text{val}}(n)$。

验证曲线 $V(\lambda)$：固定超参数 $\lambda$ 的值，记录训练和验证得分随 $\lambda$ 的变化。

$$V_{\text{train}}(\lambda) = \mathbb{E}[S_{\text{train}}(\lambda)], \quad V_{\text{val}}(\lambda) = \mathbb{E}[S_{\text{val}}(\lambda)]$$

### Python 实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# === 学习曲线（高方差模型: max_depth=15） ===
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(max_depth=15, random_state=42),
    X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

print("=== 学习曲线 (max_depth=15, 高方差) ===")
for i, s in enumerate(train_sizes):
    gap = train_scores.mean(axis=1)[i] - val_scores.mean(axis=1)[i]
    print(f"  训练量={s:3d}  训练得分={train_scores.mean(axis=1)[i]:.3f}  "
          f"验证得分={val_scores.mean(axis=1)[i]:.3f}  差距={gap:.3f}")

# === 学习曲线（高偏差模型: max_depth=1） ===
train_sizes2, train_scores2, val_scores2 = learning_curve(
    DecisionTreeClassifier(max_depth=1, random_state=42),
    X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

print("\\n=== 学习曲线 (max_depth=1, 高偏差) ===")
for i, s in enumerate(train_sizes2):
    gap = train_scores2.mean(axis=1)[i] - val_scores2.mean(axis=1)[i]
    print(f"  训练量={s:3d}  训练得分={train_scores2.mean(axis=1)[i]:.3f}  "
          f"验证得分={val_scores2.mean(axis=1)[i]:.3f}  差距={gap:.3f}")

# === 验证曲线 ===
param_range = range(1, 21)
train_scores_vc, val_scores_vc = validation_curve(
    DecisionTreeClassifier(random_state=42), X, y,
    param_name='max_depth', param_range=param_range, cv=5,
    scoring='accuracy'
)

best_depth = param_range[np.argmax(val_scores_vc.mean(axis=1))]
print(f"\\n=== 验证曲线 ===")
print(f"最优 max_depth: {best_depth}, 验证得分: {val_scores_vc.mean(axis=1).max():.3f}")
print(f"深度=1  验证得分: {val_scores_vc.mean(axis=1)[0]:.3f}  (高偏差)")
print(f"深度=20 验证得分: {val_scores_vc.mean(axis=1)[-1]:.3f}  (高方差)")
```

**输出示例：**
```
=== 学习曲线 (max_depth=15, 高方差) ===
  训练量= 50  训练得分=1.000  验证得分=0.748  差距=0.252
  训练量=100  训练得分=1.000  验证得分=0.826  差距=0.174
  训练量=200  训练得分=0.995  验证得分=0.872  差距=0.123
  训练量=300  训练得分=0.990  验证得分=0.905  差距=0.085
  训练量=400  训练得分=0.994  验证得分=0.918  差距=0.076
  训练量=450  训练得分=0.996  验证得分=0.927  差距=0.069

=== 学习曲线 (max_depth=1, 高偏差) ===
  训练量= 50  训练得分=0.740  验证得分=0.726  差距=0.014
  训练量=100  训练得分=0.755  验证得分=0.738  差距=0.017
  训练量=200  训练得分=0.753  验证得分=0.750  差距=0.003
  训练量=300  训练得分=0.753  验证得分=0.755  差距=-0.002
  训练量=450  训练得分=0.758  验证得分=0.760  差距=-0.002

=== 验证曲线 ===
最优 max_depth: 8, 验证得分: 0.946
深度=1  验证得分: 0.758  (高偏差)
深度=20 验证得分: 0.858  (高方差)
```

### 应用连接

诊断表格：

| 现象 | 诊断 | 对策 |
|------|------|------|
| 训练和验证得分都很低且接近 | **高偏差（欠拟合）** | 换更强模型、加特征、减正则化 |
| 训练得分高、验证得分低，差距大 | **高方差（过拟合）** | 加数据、加正则化、减模型复杂度 |
| 加数据后验证得分持续上升 | 数据不足 | 收集更多数据 |
| 加数据后验证得分趋于平稳 | 模型瓶颈 | 换算法 |

---

## 5. 模型选择策略

### 直觉理解

面临 SVM、随机森林、XGBoost 三个选择，该选哪个？直觉做法是每个都跑一遍交叉验证，选平均得分最高的。但这还不够——如果两个模型的得分差异可能只是"运气"呢？我们需要**统计检验**来回答：这个差异是真实的，还是偶然的。

### 形式化定义：5×2 交叉验证配对 t 检验

在 5 次 2 折交叉验证中记录两个模型在每个 fold 上的得分差 $d_i^{(j)}$，构造 t 统计量：

$$t = \frac{\bar{d}}{\sqrt{\frac{1}{5} \cdot \frac{1}{k} \sum_{i=1}^{5} s_i^2}}$$

其中 $s_i^2$ 是第 $i$ 次 2 折交叉验证中得分差的方差。若 $|t| > t_{0.05}(5)$，则差异显著。

### Python 实现：多模型系统比较

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM (RBF)': SVC(kernel='rbf'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
}

results = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=5,
                            scoring=['accuracy', 'f1_macro'])
    results.append({
        '模型': name,
        '准确率均值': scores['test_accuracy'].mean(),
        '准确率标准差': scores['test_accuracy'].std(),
        'F1均值': scores['test_f1_macro'].mean(),
        'F1标准差': scores['test_f1_macro'].std(),
    })

results.sort(key=lambda r: r['F1均值'], reverse=True)

print(f"{'模型':<22} {'准确率':>10} {'F1 (macro)':>14}")
print("-" * 48)
for r in results:
    print(f"{r['模型']:<22} {r['准确率均值']:.3f}±{r['准确率标准差']:.3f}"
          f"   {r['F1均值']:.3f}±{r['F1标准差']:.3f}")
```

**输出示例：**
```
模型                     准确率       F1 (macro)
------------------------------------------------
GradientBoosting        0.977±0.029   0.977±0.028
RandomForest            0.972±0.017   0.972±0.016
SVM (RBF)               0.977±0.016   0.977±0.018
LogisticRegression      0.971±0.022   0.971±0.022
```

### 应用连接

算法选择的金字塔原则：

1. **先试简单模型**（Logistic Regression、决策树）——它们训练快，且作为基线不可替代
2. **再看集成模型**（Random Forest、Gradient Boosting）——它们是最稳定的"默认选择"
3. **最后上神经网络**——当你有海量数据，且简单模型已榨干性能时

如果你发现一个复杂模型只比简单模型好 1-2%，问自己：这额外的复杂度值得吗？

---

## 6. 处理不平衡数据

### 直觉理解

回顾第 11 章：欺诈检测中 99.9% 是正常交易，0.1% 是欺诈。一个"全部预测正常"的模型准确率 99.9%，但 F1 为 0。处理不平衡有三种路线：**改数据**（采样）、**改算法**（权重）、**改评估**（指标）。

### 形式化定义

**SMOTE（Synthetic Minority Oversampling Technique）**：对每个少数类样本 $x_i$，随机选它的一个同类近邻 $\hat{x}_i$，在两者连线上插值生成新样本：

$$x_{\text{new}} = x_i + \lambda (\hat{x}_i - x_i), \quad \lambda \sim \text{Uniform}(0, 1)$$

**类别权重**：修改损失函数，使少数类的错误惩罚更大。对于二分类，通常设：

$$w_{\text{class }k} = \frac{N_{\text{total}}}{C \cdot N_k}$$

其中 $C$ 是类别数，$N_k$ 是第 $k$ 类的样本数。

### Python 实现

```python
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2000, n_features=10, weights=[0.95, 0.05],
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

print(f"原始训练集分布: {dict(sorted(Counter(y_train).items()))}")
print(f"少数类占比: {np.mean(y_train):.3f}\\n")

# === 策略 1: 类别权重 ===
clf_weighted = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42
)
clf_weighted.fit(X_train, y_train)
y_pred_w = clf_weighted.predict(X_test)
print("=== Class Weight 结果 ===")
print(classification_report(y_test, y_pred_w, digits=3))

# === 策略 2: SMOTE 过采样 ===
pipeline_smote = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline_smote.fit(X_train, y_train)
y_pred_s = pipeline_smote.predict(X_test)
print("\\n=== SMOTE 结果 ===")
print(classification_report(y_test, y_pred_s, digits=3))

# === 策略 3: SMOTE + 类别权重 ===
pipeline_both = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100,
                                   class_weight='balanced', random_state=42))
])
pipeline_both.fit(X_train, y_train)
y_pred_b = pipeline_both.predict(X_test)
print("\\n=== SMOTE + Class Weight 结果 ===")
print(classification_report(y_test, y_pred_b, digits=3))
```

**输出示例：**
```
原始训练集分布: {0: 1330, 1: 70}
少数类占比: 0.050

=== Class Weight 结果 ===
              precision    recall  f1-score   support
           0      0.984     0.981     0.982       570
           1      0.702     0.733     0.718        30
    accuracy                          0.968       600

=== SMOTE 结果 ===
              precision    recall  f1-score   support
           0      0.987     0.981     0.984       570
           1      0.711     0.767     0.738        30
    accuracy                          0.970       600

=== SMOTE + Class Weight 结果 ===
              precision    recall  f1-score   support
           0      0.984     0.988     0.986       570
           1      0.741     0.667     0.702        30
    accuracy                          0.972       600
```

### 应用连接

| 策略 | 适用场景 | 注意事项 |
|------|---------|---------|
| **类别权重** | 任何不平衡场景 | 最简单、不改变数据分布，优先尝试 |
| **SMOTE 过采样** | 少数类样本极少（<1%） | 生成样本在特征空间线性插值，高维效果打折扣 |
| **随机欠采样** | 数据量大、多数类有冗余 | 会丢弃有用信息，慎用 |
| **评估指标** | 所有不平衡场景 | 永远不要只看 accuracy，用 AUPRC 或 F1 |

---

## 7. 实战：完整调参流程

### 直觉理解

把前面学的一切串起来：选数据 → 定基线 → 交叉验证评估 → 搜索超参数 → 最终评估。这是一个机器学习项目中模型开发阶段的标准流程。

### Python 实现：端到端流水线

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_validate)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from scipy.stats import randint, uniform

data = load_breast_cancer()
X, y = data.data, data.target

# 第一步：划分测试集（只用一次！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
print(f"训练集类别分布: {np.bincount(y_train)}")

# 第二步：建立基线（默认参数）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\\n=== 基线模型 ===")
for name, model in [
    ('RandomForest', RandomForestClassifier(random_state=42)),
    ('GradientBoosting', GradientBoostingClassifier(random_state=42))
]:
    scores = cross_validate(model, X_train, y_train, cv=cv,
                           scoring=['accuracy', 'f1'])
    print(f"{name}: Acc={scores['test_accuracy'].mean():.3f}±"
          f"{scores['test_accuracy'].std():.3f}  "
          f"F1={scores['test_f1'].mean():.3f}±"
          f"{scores['test_f1'].std():.3f}")

# 第三步：选最佳基线做超参数搜索
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 6),
    'max_features': ['sqrt', 'log2', None],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_dist,
    n_iter=200, cv=5, scoring='f1', n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

print(f"\\n=== RandomSearch 最佳结果 ===")
print(f"最佳参数: {random_search.best_params_}")
print(f"交叉验证 F1: {random_search.best_score_:.4f}")

# 第四步：最终评估（只用一次测试集！）
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\\n=== 测试集最终评估 ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"测试集准确率: {best_model.score(X_test, y_test):.4f}")

# 第五步：报告结果
print("\\n=== 最终报告 ===")
print(f"模型: RandomForestClassifier")
print(f"超参数: {random_search.best_params_}")
print(f"搜索范围: {dict((k, str(v)) for k, v in param_dist.items())}")
print(f"CV 分数 (n=5): F1 = {random_search.best_score_:.4f} ± "
      f"{random_search.cv_results_['std_test_score'][random_search.best_index_]:.4f}")
print(f"测试集分数: Acc = {best_model.score(X_test, y_test):.4f}")
```

**输出示例：**
```
训练集: 455 样本, 测试集: 114 样本
训练集类别分布: [170 285]

=== 基线模型 ===
RandomForest: Acc=0.958±0.018  F1=0.968±0.014
GradientBoosting: Acc=0.958±0.013  F1=0.967±0.010

=== RandomSearch 最佳结果 ===
最佳参数: {'max_depth': 6, 'max_features': 'log2',
          'min_samples_leaf': 1, 'min_samples_split': 7,
          'n_estimators': 218}
交叉验证 F1: 0.9713

=== 测试集最终评估 ===
              precision    recall  f1-score   support
   malignant       0.98      0.95      0.96        40
      benign       0.97      0.99      0.98        74
    accuracy                           0.97       114

测试集准确率: 0.9737

=== 最终报告 ===
模型: RandomForestClassifier
超参数: {'max_depth': 6, ...}
CV 分数 (n=5): F1 = 0.9713 ± 0.0108
测试集分数: Acc = 0.9737
```

---

## 总结

本章你掌握了从评估到调参的完整武器库。核心心法只有三条：

1. **训练误差不等于测试误差**——永远用交叉验证评估泛化性能，测试集只能看一次
2. **没有免费的午餐**——Grid Search 保证找到网格内最优，但 Random Search 在高维空间更高效，贝叶斯优化则真正"聪明地搜索"
3. **调参是手段，理解是目的**——学习曲线告诉你该加数据还是改模型，验证曲线告诉你参数的最优区间。不要盲目调参，先用曲线诊断问题

记住这句话：**先让模型过拟合（证明它有足够的容量），再通过正则化、调参和更多数据把它拉回来。**

---

下一步：[特征工程](./18-feature-engineering.md)
