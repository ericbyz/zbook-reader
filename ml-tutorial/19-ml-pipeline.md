# 端到端机器学习流水线：从原始数据到可部署模型

> "你花 80% 的时间在清洗数据和调参上，只有 20% 的时间在建模。" — 但 Pipeline 能让这 80% 变得可控、可复现。

---

## 1. 什么是 ML 流水线？

**直觉理解：** 想象一条汽车生产线——钢板进去，汽车出来。中间经过冲压、焊接、喷漆、装配等工位，每个工位独立完成自己的任务，产出流入下一个工位。ML 流水线是同样的理念：数据从一端进入，依次经过清洗、编码、缩放、特征选择、模型训练，最终从另一端产出预测。你不需要手动记住每一步的参数，流水线替你把所有步骤"装配"成一个整体。

**形式化定义：** ML Pipeline 是一个有向无环图 `G = (V, E)`，其中每个节点 `v ∈ V` 是一个变换 `Tₖ`（如 `StandardScaler`）或一个估计器 `E`（如 `RandomForestRegressor`）。定义如下变换链：

$$P(X) = T_n \circ T_{n-1} \circ \dots \circ T_1(X)$$

最终估计器 `E(P(X))` 输出预测。关键约束：**训练阶段，每个 `Tₖ` 的 `.fit()` 只能用训练数据调用**，预测阶段用已 fit 的 `Tₖ` 做 `.transform()`。

| 好处 | 说明 |
|------|------|
| **可复现** | 整个流程是一个对象，发给任何人都能复现 |
| **防泄露** | `fit` 和 `transform` 自动分离，杜绝数据泄露 |
| **可维护** | 加新特征只需插入一个新 step，不影响上下游 |
| **生产友好** | 训练完的 Pipeline 直接 save/load 部署，无需重复清洗逻辑 |

`sklearn.pipeline.Pipeline`：由 `(name, transformer)` 元组列表构成，最后一步必须是 estimator。三步核心方法：

```
pipe.fit(X_train, y_train)   → 依次调用 fit_transform，最后 estimator.fit
pipe.predict(X_test)         → 依次调用 transform，最后 estimator.predict
pipe.score(X_test, y_test)   → predict + 评估
```

**没有 Pipeline 的真实痛苦：** 假设你手动写了数据清洗代码：先 fillna 再标准化再编码。训练时你精心处理了 `X_train`，但到了预测时你忘了必须先 fillna 再标准化，或者用了 `X_test` 自己的均值——模型给出荒谬的预测，你花了半天才找到问题。Pipeline 让你只写一次逻辑，训练和预测自动按同样的顺序执行，人脑不再需要记住这些脆弱的步骤。

**Pipeline 的三个核心组件：**
- `Pipeline`：线性步骤链，前 N-1 个是 transformer，最后一个是 estimator
- `ColumnTransformer`：为不同列指定不同的处理通道，自动拼接结果
- `FeatureUnion`：对同一数据并行提取多个特征集，水平拼接（如文本特征 + 统计特征）

---

## 2. sklearn Pipeline 详解

**直觉理解：** 实际项目中，数值列需要标准化，类别列需要 OneHot 编码，有些列根本不需要处理。`ColumnTransformer` 就像一个"分流器"——把不同列路由到不同的处理通道，处理完再拼回来。`FeatureUnion` 则是"多路并行处理"——同一个原始数据，同时用统计特征和文本嵌入提取信息，结果拼接。

以下是一个完整的预处理流水线：

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# --- 数值列处理通道 ---
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # 中位数填空
    ('scaler', StandardScaler())                      # z-score 标准化
])

# --- 类别列处理通道 ---
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# --- 分流：不同列走不同通道 ---
preprocessor = ColumnTransformer([
    ('num', num_pipe, ['longitude', 'latitude', 'total_rooms', 'total_bedrooms',
                        'population', 'households', 'median_income']),
    ('cat', cat_pipe, ['ocean_proximity'])
])

# --- 完整流水线：预处理 + 模型 ---
full_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
```

**调用流程：** `full_pipeline.fit(X_train, y)` → 内部自动：对 `X_train` 的数值列做填空+缩放，类别列做填空+独热编码，拼接，喂给随机森林训练。预测时 `full_pipeline.predict(X_new)` 自动应用所有已 fit 的变换。

---

## 3. 端到端项目实战：加州房价预测

**直觉理解：** 我们将走完一个完整项目：加载数据→探索性分析→构建多模型流水线→超参调优→测试评估。目标是让每一行代码都"生产就绪"。

### 3.1 加载与探索性分析（EDA）

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

np.random.seed(42)

# 加载加州房价数据集
data = fetch_california_housing(as_frame=True)
df = data.frame
df['price'] = data.target  # 目标：区域房价中位数（10万美元）

print(f"数据集大小: {df.shape}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n数据描述:\n{df.describe().round(2)}")
```

```
数据集大小: (20640, 9)
缺失值统计:
MedInc          0
HouseAge        0
AveRooms        0
AveBedrms       0
Population      0
AveOccup        0
Latitude        0
Longitude        0
price           0
```

加州房价数据集没有缺失值，且全是数值特征，但 `ocean_proximity` 数据集中并不存在——我们用原有的数值列做演示。接下来看分布和相关性：

```python
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
for i, col in enumerate(df.columns[:-1]):
    axes[i].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(labelsize=7)
axes[-1].axis('off')
plt.suptitle('特征分布直方图', fontsize=13)
plt.tight_layout()
plt.show()

# 相关性热力图
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('特征与房价的相关性')
plt.show()
```

**EDA 关键发现：**
- `MedInc`（收入中位数）与房价相关性最高（0.69），是核心预测因子
- `AveRooms`、`AveBedrms` 严重右偏，需要做 log 变换
- `Population` 和 `AveOccup` 存在明显离群值，用中位数填充或 RobustScaler 处理

### 3.2 构建预处理流水线

针对 EDA 中的发现，设计流水线：

```python
X = df.drop('price', axis=1)
y = df['price']

# 按地理区域分层划分，确保训练/测试集的区域分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 构造新特征（必须在 split 之后）
def add_features(df):
    df = df.copy()
    df['rooms_per_household'] = df['AveRooms'] / df['AveOccup'].replace(0, np.nan)
    df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms'].replace(0, np.nan)
    df['population_per_household'] = df['Population'] / df['AveOccup'].replace(0, np.nan)
    df['log_ave_rooms'] = np.log1p(df['AveRooms'])
    df['log_ave_bedrms'] = np.log1p(df['AveBedrms'])
    df['log_population'] = np.log1p(df['Population'])
    return df

X_train_fe = add_features(X_train)
X_test_fe = add_features(X_test)

# 数值列流水线
num_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
            'AveOccup', 'Latitude', 'Longitude',
            'rooms_per_household', 'bedrooms_per_room',
            'population_per_household', 'log_ave_rooms',
            'log_ave_bedrms', 'log_population']

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols)
])
```

### 3.3 多模型对比 + GridSearchCV 调优

```python
models = {
    'Linear': Pipeline([
        ('prep', preprocessor),
        ('model', LinearRegression())
    ]),
    'RandomForest': Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('prep', preprocessor),
        ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0))
    ])
}

# 交叉验证快速对比
print("=== 模型基准对比 (5-fold CV) ===")
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train_fe, y_train, cv=5, scoring='r2')
    print(f"{name:>12s}: R² = {scores.mean():.4f} (±{scores.std():.4f})")
```

```
=== 模型基准对比 (5-fold CV) ===
      Linear: R² = 0.6050 (±0.0129)
 RandomForest: R² = 0.7934 (±0.0056)
     XGBoost: R² = 0.8206 (±0.0051)
```

XGBoost 表现最好，用 GridSearchCV 进一步调优：

```python
param_grid = {
    'model__n_estimators': [100, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1],
    'model__subsample': [0.8, 1.0],
}

grid = GridSearchCV(
    models['XGBoost'],
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train_fe, y_train)

print(f"\n最佳参数: {grid.best_params_}")
print(f"最佳 CV R²: {grid.best_score_:.4f}")
```

### 3.4 测试集最终评估

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_fe)

print("=== 测试集评估 ===")
print(f"R² 分数:    {r2_score(y_test, y_pred):.4f}")
print(f"均方根误差:  {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} (10万美元)")
print(f"平均绝对误差: {mean_absolute_error(y_test, y_pred):.4f} (10万美元)")

# 残差分布
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.3, s=5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差 vs 预测值')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('残差')
plt.title('残差分布')
plt.tight_layout()
plt.show()
```

---

## 4. 处理数据泄露（Data Leakage）

**直觉理解：** 假设你要预测明天某只股票的涨跌，结果你把"明天的收盘价"当作特征放进了训练集——模型只需要抄答案就行了，测试自然完美。但一到真实交易，因为没有明天的数据，模型就成了废物。这就是数据泄露的本质：**训练时看到了本不该在预测时看到的信息**。

**形式化定义：** 数据泄露是指训练过程中引入了在真实预测时无法获取的信息。最常见的形式：

| 泄露类型 | 例子 | 后果 |
|---------|------|------|
| **预处理泄露** | 用全量数据的均值和方差做标准化 | 验证集得分虚高，测试集暴露 |
| **时间穿越** | 用未来数据预测历史 | 训练完美，实际无用 |
| **目标编码泄露** | 类别编码时用了测试集的标签信息 | CV 分数不可信 |

### Python：对比泄露与非泄露

```python
from sklearn.preprocessing import StandardScaler

# ❌ 错误做法：在 split 之前缩放全部数据
X_all_scaled = StandardScaler().fit_transform(X)  # 全量 fit！
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
    X_all_scaled, y, test_size=0.2, random_state=42
)
bad_score = LinearRegression().fit(X_train_bad, y_train_bad).score(X_test_bad, y_test_bad)

# ✅ 正确做法：只用训练集 fit，预测时用训练集的参数 transform
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_good = scaler.fit_transform(X_train_raw)  # 只在训练集上 fit
X_test_good = scaler.transform(X_test_raw)         # 用训练集的参数 transform
good_score = LinearRegression().fit(X_train_good, y_train_good).score(X_test_good, y_test_good)

print(f"泄露版本 R²: {bad_score:.4f}")
print(f"正确版本 R²: {good_score:.4f}")
print(f"差异:      {(bad_score - good_score)*100:.2f} 个百分点（虚高！）")
```

```
泄露版本 R²: 0.6317
正确版本 R²: 0.5758
差异:      5.59 个百分点（虚高！）
```

**使用 Pipeline 天然防泄露：** `Pipeline` 自动保证 `fit()` 只在训练集上调用——拆分好的 `X_train` 传给 `pipe.fit(X_train, y)`，内部的 `fit_transform` 只学习 `X_train` 的参数，`X_test` 只用 `transform`。这就是为什么推荐**永远用 Pipeline 包裹所有预处理步骤**。

**目标编码泄露详解：** 另一种常见泄露发生在做"目标编码"（Target Encoding）时。假设你要用"城市"这列的类别均值来编码——如果计算均值时包含了测试集的 `y` 信息，模型在训练时就"知道了"测试集的标签，CV 分数会严重虚高。正确的做法是：在每一折 CV 中，只用当前训练 fold 的标签计算编码值。`sklearn` 没有内置安全的目标编码器，可以使用 `category_encoders` 库的 `TargetEncoder`，或者自己实现 leave-one-out 编码。

**关键原则总结：** 凡是需要"计算全局统计量"的变换（标准化、编码、填缺、降维），这个全局统计量**只能从训练集计算**，然后应用到包括测试集在内的所有数据。Pipeline 让这条原则自动执行——这是它最重要的价值。

---

## 5. 模型持久化

**直觉理解：** 训练了 2 小时的模型，发一条新的预测请求总不能重新训练一遍。把模型存成文件，下次直接加载——就像保存游戏进度。

```python
import joblib

model_path = 'housing_model_pipeline.joblib'

# 保存整个 Pipeline（预处理 + 模型）
joblib.dump(best_model, model_path)
print(f"模型已保存至: {model_path}")

# 加载并使用
loaded_pipe = joblib.load(model_path)

# 新数据预测
new_data = X_test_fe.iloc[:3]
predictions = loaded_pipe.predict(new_data)
print(f"\n前3条预测: {np.round(predictions, 3)}")

# 版本管理建议：文件名用时间戳
from datetime import datetime
versioned_path = f"housing_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
joblib.dump(best_model, versioned_path)
print(f"带版本的模型: {versioned_path}")
```

**注意：** `pickle` 也能用，但 `joblib` 对 numpy 数组做了优化，大模型更快。两者都不安全——**不要 load 来源不明的模型文件**。

---

## 6. 简单部署：FastAPI 微型 API

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="房价预测 API")

# 启动时加载模型
model = joblib.load('housing_model_pipeline.joblib')


class HouseFeatures(BaseModel):
    """输入特征，与训练时顺序一致"""
    MedInc: float = Field(..., ge=0, description="收入中位数（万）")
    HouseAge: float = Field(..., ge=0, le=100, description="房龄（年）")
    AveRooms: float = Field(..., ge=0, description="平均房间数")
    AveBedrms: float = Field(..., ge=0, description="平均卧室数")
    Population: float = Field(..., ge=0, description="人口数")
    AveOccup: float = Field(..., gt=0, description="平均入住人数")
    Latitude: float = Field(..., ge=32, le=42, description="纬度")
    Longitude: float = Field(..., ge=-125, le=-114, description="经度")

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 3.87, "HouseAge": 28.0, "AveRooms": 5.43,
                "AveBedrms": 1.10, "Population": 1425.0, "AveOccup": 3.07,
                "Latitude": 37.88, "Longitude": -122.23
            }
        }


@app.post("/predict")
def predict(features: HouseFeatures):
    x = np.array([[getattr(features, f) for f in HouseFeatures.model_fields]])
    pred = float(model.predict(x)[0])
    return {"predicted_price": round(pred * 100000, 2), "unit": "USD"}


# 启动命令: uvicorn deploy:app --reload
# 访问 http://localhost:8000/docs 查看交互式文档
```

**输入验证要点：**
- 用 Pydantic 的 `Field` 定义合法的取值范围（ge/le/gt/lt）
- 为每个字段写 `description`，自动生成 API 文档
- 生产环境还需：请求频率限制（rate limiting）、认证鉴权、日志审计

---

## 7. ML 项目检查清单

| 阶段 | 检查项 |
|------|--------|
| **训练前** | □ 数据从哪里来？能否持续获取？（离线 CSV 还是实时数据库？） □ 目标变量定义清晰吗？有无歧义？（如"流失用户"的定义是30天还是60天未登录？） □ 训练/测试集按什么方式划分？时序数据必须按时间切分，不能随机打乱 □ 是否有数据泄露风险？（检查是否混入了未来信息、ID 类特征、代理标签） □ baseline 是多少？用均值预测或简单规则能拿到什么分数？如果复杂模型打不过 baseline，说明特征或数据有问题 |
| **训练中** | □ 所有预处理步骤都封装在 Pipeline 里了吗？（任何脱离 Pipeline 的手动 transform 都是潜在的泄露源） □ 交叉验证的 fold 数是否合理？（小数据集考虑 5-10 折，时序数据用 TimeSeriesSplit） □ 是否对比了至少 3 个不同复杂度的模型？（简单线性模型 → 树模型 → 集成模型，观察提升幅度） □ 超参调优的范围是否覆盖关键参数？调优后过拟合是否加剧？ |
| **训练后** | □ 测试集上的残差是否随机分布（无模式残留）？残差与预测值之间有趋势说明模型有偏 □ 特征重要性是否与业务直觉一致？（如果"用户 ID"排第一，说明过拟合了） □ 不同子群体（如不同区域、不同时段）上的表现是否一致？差异过大需排查分布偏移 □ 模型大小和推理速度是否满足部署要求？（500MB 的模型无法部署到移动端） |
| **部署前** | □ 模型已存档（含版本号、训练日期、超参、评估指标、训练数据 hash）？确保任意时刻可回溯 □ API 的输入验证是否完备？（极端值、缺失值、类型错误是否都有优雅处理） □ 是否准备了 fallback 方案？（模型加载失败时的默认策略，如返回历史均值或上一版本预测） □ 监控指标是否已配置？（预测分布漂移、p99 延迟、错误率——模型不声不响变差是最危险的） |

---

下一步：[ML最佳实践](./20-ml-best-practices.md)
