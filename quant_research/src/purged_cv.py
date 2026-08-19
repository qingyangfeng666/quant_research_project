"""
第3周 Day 2：Purged 交叉验证
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. 加载数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
# file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors_talib.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
df = df.dropna(subset=['target'])

exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'return', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values
y = df['target'].values

# 2. 时间序列交叉验证（5折）
tscv = TimeSeriesSplit(n_splits=5)

print("=" * 60)
print("Purged 交叉验证（5折）")
print("=" * 60)

scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    
    print(f"第{fold+1}折: {acc:.2%}")

print(f"\n平均准确率: {np.mean(scores):.2%}")
print(f"标准差: {np.std(scores):.4f}")

if np.std(scores) < 0.03:
    print("✅ 模型表现稳定，过拟合风险低")
else:
    print("⚠️ 模型表现波动较大，可能过拟合")