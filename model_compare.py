"""
LightGBM vs XGBoost 对比测试
独立脚本，不改你原来的任何代码
"""
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

print("=" * 60)
print("LightGBM vs XGBoost 对比")
print("=" * 60)

# 加载数据（和你 XGBoost 用的一模一样）
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# 定义标签（预测1天）
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna(subset=['target'])

# 准备特征
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'return', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values
y = df['target'].values

print(f"数据量: {len(X)} 行, 特征数: {len(feature_cols)}")

# 切分（80%训练，20%测试）
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

# ===== 1. 训练 XGBoost（基准） =====
print("\n训练 XGBoost...")
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# ===== 2. 训练 LightGBM =====
print("训练 LightGBM...")
start = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start
lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))

# ===== 3. 输出对比结果 =====
print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)
print(f"XGBoost 测试集准确率: {xgb_acc:.2%} (训练时间: {xgb_time:.2f}秒)")
print(f"LightGBM 测试集准确率: {lgb_acc:.2%} (训练时间: {lgb_time:.2f}秒)")
print(f"准确率差异: {(lgb_acc - xgb_acc):.2%}")