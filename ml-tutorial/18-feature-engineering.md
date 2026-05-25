# 特征工程：数据科学家最重要的技能

> "Better features beat better algorithms." — 好特征胜过好算法。

---

## 1. 什么是特征工程？为什么重要？

**直觉理解：** 想象你要教一个孩子区分猫和狗。直接给他看 10 万像素的原始图片，他学得很慢；但如果你告诉他"耳朵是尖的还是垂的？""嘴巴是长的还是短的？""体型大小如何？"这些经过提炼的特征，他很快就能学会。特征工程就是帮模型做类似的"提炼"工作——把原始数据转换成模型更容易理解的表达形式。

这个"提炼"过程的本质是什么？是从原始数据中挖掘出对预测任务有帮助的信息，并以模型能理解的方式表达出来。这既是科学（需要统计学和编程知识），也是艺术（需要领域知识和创造力）。

**形式化定义：** 给定原始数据 `D = {(x_i, y_i)}`，特征工程是构造映射 `φ: X → X'`，使得新特征空间 `X'` 中的模式对模型更容易捕捉。一个好的 `φ` 能降低模型的学习难度，让线性模型也能拟合非线性关系。

为什么特征工程往往比调参更重要？一个直观类比：调参相当于给厨师换一口更好的锅，特征工程则是给厨师换更新鲜的食材。锅好有微量提升，食材好才是根本。

| 投入 | 常见提升 |
|------|----------|
| 超参调优 | 1%-3% |
| 换更复杂的模型 | 2%-5% |
| 好的特征工程 | 10%-30% |

一个极端例子：Kaggle 的很多冠军方案，模型就是简单的 XGBoost/LightGBM，差异全在特征上。模型决定上限，特征决定你能逼进上界多近。

特征工程通常占一个数据科学家 60%-80% 的时间，因为每个数据集都有自己独特的"金矿"等待挖掘。在房价预测中，你可能需要构造"人均居住面积"；在用户推荐中，你需要的可能是"过去7天浏览同类商品的次数"。这些特征没有通用公式，需要你对业务的理解。

---

## 2. 数值特征的处理

数值特征是"最像礼物"的输入——模型拿到就能用。但直接用并不总是最优，因为模型通常只能捕捉线性关系。

### 2.1 分箱

**直觉理解：** 年龄对收入的影响不是线性的——20→30 岁收入和 50→60 岁收入的变化模式完全不同。把年龄切成"青年/中年/老年"几个箱子，让模型自由拟合每段的权重。分箱的本质是用"分段常数"来逼近任意复杂的函数关系，相当于手动给模型提供了非线性的表达能力。

但要注意：分箱会损失信息（箱子内不同值的差异被抹平），且边界的选择很关键。常用的分箱策略有等宽（uniform）、等频（quantile）和基于业务含义的手动定义。在实际项目中，如果有明确的业务规则（比如"18岁以下、18-45岁、45岁以上"），应该优先使用业务分箱。

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

np.random.seed(42)
X = pd.DataFrame({'age': np.random.randint(18, 80, 1000)})

binner = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
X_binned = binner.fit_transform(X)
print("分箱后的形状:", X_binned.shape)
print("分箱边界:", binner.bin_edges_[0])
```

### 2.2 多项式特征

**直觉理解：** 正方形的面积与边长是 `x²` 的关系，如果用 `y = ax + b` 拟合，无论如何都拟合不好。加上 `x²` 这个特征，线性模型就能完美拟合抛物线。这背后的原理是：任何光滑函数都可以用多项式的线性组合来逼近（泰勒展开的思想）。

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['age']])
print("原始: 1个特征 → 多项式: ", X_poly.shape[1], "个特征")
print("特征名:", poly.get_feature_names_out(['age']))
```

**注意：** 多项式特征数量随 degree 指数增长，n 个特征 degree=d 时产生 `C(n+d, d)` 个特征，容易过拟合。

### 2.3 对数/指数变换

**直觉理解：** 收入、关注量、销量这类数据通常"富者愈富"——极端值离平均值很远，分布严重偏斜。取对数后，$100→1000$ 和 $10000→100000$ 变为相同的间距。对数变换相当于把"乘法世界"转换成"加法世界"，让基于距离的模型（如线性回归）更容易处理。

常见的变换选择：右偏分布（长尾在右）用 `log` 或 `Box-Cox`，左偏分布用指数变换。`Yeo-Johnson` 是 `Box-Cox` 的升级版，可以处理零值和负值。

```python
from sklearn.preprocessing import PowerTransformer

skewed = np.random.lognormal(mean=2, sigma=1, size=1000)
print(f"偏度: {pd.Series(skewed).skew():.2f}")

pt = PowerTransformer(method='yeo-johnson')  # 可处理负值
transformed = pt.fit_transform(skewed.reshape(-1, 1))
print(f"变换后偏度: {pd.Series(transformed.ravel()).skew():.2f}")
```

### 2.4 特征交互

**直觉理解：** 房价评估中，卧室数量 × 地段 的组合信息远超两者单独相加。"三环内的三居室"不能用 `bedrooms + location` 表示，需要 `bedrooms × location_score`。特征交互的本质是让模型获得"条件效应"的信息：某个特征的效果在另一个特征取不同值时是不一样的。

常见的交互形式包括乘法 `x₁ × x₂`、除法 `x₁ / x₂`、以及更复杂的组合如 `(x₁ - x₂)²`。注意：交互特征会急剧增加特征数量，建议先用领域知识筛选有意义的组合，而非暴力生成所有两两交互。

```python
df = pd.DataFrame({
    'bedrooms': [1, 2, 3, 1, 2, 3],
    'area': [30, 60, 90, 40, 80, 120],
    'location_score': [0.5, 0.6, 0.8, 0.4, 0.7, 0.9]
})
df['bedroom_location'] = df['bedrooms'] * df['location_score']
df['area_per_bedroom'] = df['area'] / df['bedrooms']
df['bedroom_over_loc'] = df['bedrooms'] / (df['location_score'] + 0.01)
print(df.head())
```

---

## 3. 日期时间特征

**直觉理解：** 一个时间戳在模型眼里就是一串无意义的数字。但"周一"和"周六"的行为模式完全不同，"凌晨3点"和"下午3点"也是天壤之别。

### 3.1 基础提取

```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=100, freq='13H')
})
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=周一
df['month'] = df['timestamp'].dt.month
df['quarter'] = df['timestamp'].dt.quarter
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
```

### 3.2 循环编码 (Cyclical Encoding)

**直觉理解：** 直接把 `hour=23` 和 `hour=0` 喂给模型，模型会认为它们差 23 个单位——但实际上午夜只差 1 个小时！用 sin/cos 把环形特征映射到圆上，23 和 0 的距离自然就近了。

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
print(df[['hour', 'hour_sin', 'hour_cos']].head(3))
```

### 3.3 距事件时间

```python
df['days_since_2025'] = (df['timestamp'] - pd.Timestamp('2025-01-01')).dt.days
```

---

## 4. 文本特征

**直觉理解：** 模型不吃文字，只吃数字。文本特征工程就是"翻译"——把人类语言翻译成数字向量。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = [
    '机器学习让计算机从数据中学习',
    '深度学习是机器学习的一个子领域',
    '特征工程是数据预处理的重要步骤'
]

# Bag of Words: 就是数词出现次数
cv = CountVectorizer()
bow = cv.fit_transform(docs)
print("词表大小:", len(cv.get_feature_names_out()))
print("第一个文档的 BoW:", bow[0].toarray())

# TF-IDF: 出现次数 × 稀有度权重（越是"专有词"权重越高）
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
tfidf_matrix = tfidf.fit_transform(docs)
print("TF-IDF 形状:", tfidf_matrix.shape)

# N-gram: 把相邻词组合起来，捕捉短语信息
bigrams = CountVectorizer(ngram_range=(2, 2))
print("Bigrams:", bigrams.fit(docs).get_feature_names_out()[:5])
```

> **词嵌入 (Word Embeddings)**：BoW/TF-IDF 把每个词视为独立符号，"猫"和"喵"的关系完全丢失。Word2Vec、GloVe 等嵌入方法把词映射到稠密向量空间，语义相近的词向量也相近。这部分会在神经网络章节深入讨论。

---

## 5. 特征选择

**直觉理解：** 数据管道里灌了 200 个特征，但真正有用的可能只有 15 个。多余的 185 个特征不仅拖慢训练，还引入噪声，让模型更容易过拟合。特征选择就是"减法艺术"——在保持模型性能的前提下，去掉冗余和无关的特征。

这背后的数学直觉是"奥卡姆剃刀"：简单且有效的模型比复杂模型更好。每多一个无意义的特征，模型就需要额外的数据来"忘记"它带来的噪声。在高维低样本的典型场景（如基因数据：几千个基因，几十个病人）中，特征选择不是可选项，而是必须项。

```python
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

X, y = make_classification(n_samples=500, n_features=30, n_informative=10,
                           n_redundant=10, n_repeated=5, noise=0.05,
                           random_state=42)
```

### 5.1 过滤法：互信息

```python
mi = mutual_info_classif(X, y)
top_k = np.argsort(mi)[-5:]  # 互信息最大的 5 个特征
print("互信息 Top 5 特征:", top_k)
```

### 5.2 包装法：递归特征消除

```python
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(X, y)
print("RFE 选中的特征:", np.where(rfe.support_)[0])
```

### 5.3 嵌入法：Lasso 正则化

```python
lasso = LassoCV(cv=5, random_state=42).fit(X, y)
selected = np.where(lasso.coef_ != 0)[0]
print("Lasso 非零系数特征数:", len(selected))
print("系数绝对值 Top 5:", np.argsort(np.abs(lasso.coef_))[-5:])
```

三种方法各有优劣：过滤法快但忽略特征组合效应，包装法准但计算量大，嵌入法是折中，训练时顺便完成选择。

---

## 6. 特征缩放回顾

之前的[数据预处理](./03-preprocessing.md)章节已详细介绍过标准化和归一化，这里强调一个关键点：

**需要缩放 → 基于距离/梯度的模型：** SVM、KNN、神经网络、线性/逻辑回归

**不需要缩放 → 树模型：** 决策树、随机森林、XGBoost、LightGBM（决策树在分割时只关心值的相对顺序，不关心绝对大小）

---

## 7. 自动化特征工程

手动构造特征费时费力，而且严重依赖经验和想象力。在实际项目中，数据科学家经常面对几十上百张表，手动挖掘特征几乎不可能。好在 sklearn 和第三方库提供了丰富的自动化工具。

```python
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline
import numpy as np

log_transformer = FunctionTransformer(np.log1p, feature_names_out='one-to-one')

pipeline = Pipeline([
    ('log_transform', log_transformer),
    ('binning', KBinsDiscretizer(n_bins=10, encode='onehot-dense'))
])

X_example = np.random.lognormal(size=(100, 5))
result = pipeline.fit_transform(X_example)
print("自动特征工程后形状:", result.shape)
```

更高级的工具如 [Featuretools](https://featuretools.alteryx.com/) 可以自动从关系型数据中生成聚合特征（`SUM(orders.amount)`、`AVG(user.age) per city`），通过定义实体和关系，自动深度遍历并生成数百甚至数千个候选特征，再通过特征选择筛选出有用的部分。不过要注意：自动生成的特征需要经过验证，不是所有特征都有业务含义。

另外，sklearn 的 `ColumnTransformer` + `Pipeline` 组合是实现端到端特征工程流水线的利器——它能确保训练集和测试集用完全一致的方式变换，杜绝数据泄露。

---

## 8. 实战：特征工程全流程

用 Titanic 数据展示特征工程前后的模型性能对比。为什么选 Titanic？因为它是一个经典的"特征工程比算法更重要"的数据集——简单模型加好特征，轻松超越复杂模型加差特征。

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns

titanic = sns.load_dataset('titanic')
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'fare',
                    'embarked', 'sibsp', 'parch']]
titanic = titanic.dropna(subset=['embarked'])
y = titanic.pop('survived')

# --- 基准：不做特征工程 ---
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='S')),
                     ('encoder', OneHotEncoder(drop='first'))])
preprocessor_base = ColumnTransformer([
    ('num', num_pipe, ['age', 'fare']),
    ('cat', cat_pipe, ['sex', 'embarked'])
])
base_model = Pipeline([
    ('preprocess', preprocessor_base),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
base_score = cross_val_score(base_model, X, y, cv=5).mean()
print(f"基准模型 AUC: {base_score:.3f}")

# --- 特征工程增强 ---
X_fe = titanic.copy()
X_fe['family_size'] = X_fe['sibsp'] + X_fe['parch'] + 1
X_fe['is_alone'] = (X_fe['family_size'] == 1).astype(int)
X_fe['fare_per_person'] = X_fe['fare'] / X_fe['family_size']
X_fe['age_group'] = pd.cut(X_fe['age'], bins=[0, 12, 18, 35, 50, 80],
                           labels=['child', 'teen', 'young', 'mid', 'senior'])
X_fe['fare_log'] = np.log1p(X_fe['fare'])

fe_num_cols = ['age', 'fare', 'family_size', 'fare_per_person', 'fare_log']
fe_cat_cols = ['sex', 'embarked', 'age_group']

fe_preprocessor = ColumnTransformer([
    ('num', num_pipe, fe_num_cols),
    ('cat', cat_pipe, fe_cat_cols)
])
fe_model = Pipeline([
    ('preprocess', fe_preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
fe_score = cross_val_score(fe_model, X_fe, y, cv=5).mean()
print(f"特征工程后 AUC: {fe_score:.3f}")
print(f"提升: {(fe_score - base_score) * 100:.1f} 个百分点")
```

从这段代码中可以观察到一个重要模式：好的特征工程往往不需要复杂的算法。`family_size`（家庭规模）、`is_alone`（是否独自一人）、`fare_per_person`（人均票价）这几个特征都来自对 Titanic 背景的理解——家庭一起旅行的人可能有不同的幸存模式，而人均票价比总票价更能反映经济阶层。

**关键洞察：** 特征工程的核心不是技巧堆砌，而是理解数据——`family_size` 是否有用？`fare_per_person` 是否比原始 `fare` 更有解释力？这些判断来自对问题的深入理解，而非机械套用。最好的特征往往诞生于这样的时刻：你看着数据，突然问自己"如果我是当时的乘客，什么因素会影响我的幸存概率？"

---

下一步：[ML流水线](./19-ml-pipeline.md)
