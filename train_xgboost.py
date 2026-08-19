"""
第2周：XGBoost 训练
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os

# 1. 读取带因子的数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
# file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors_talib.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

print("=" * 50)
print("第2周：XGBoost 训练")
print("=" * 50)

print(f"数据量: {len(df)} 行")
print(f"因子数: {len(df.columns) - 7}")

# 2. 定义标签（预测目标）
# 用今天的因子，预测明天的涨跌
# 改成：预测3天后（减少噪声，信号更稳）
df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
df = df.dropna(subset=['target'])
print(f"标签分布: 涨={df['target'].sum()}, 跌={len(df)-df['target'].sum()}")

# 3. 选择特征列
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'return', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"特征数量: {len(feature_cols)}")

X = df[feature_cols].values
y = df['target'].values

# 4. 按时间顺序切分（80%训练，20%测试）
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

# 5. 训练 XGBoost 模型
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train)

# ===== 插入位置：在 model.fit(X_train, y_train) 后面 =====
# ===== 开始插入对比代码 =====
print("\n" + "=" * 60)
print("模型对比：XGBoost vs LightGBM")
print("=" * 60)

# 1. 复制一份数据用于对比（不影响原模型）
X_train_cmp, X_test_cmp = X_train, X_test
y_train_cmp, y_test_cmp = y_train, y_test

# 2. 训练 LightGBM（用同样的参数）
import lightgbm as lgb
import time

print("训练 LightGBM...")
start_lgb = time.time()
model_lgb = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
model_lgb.fit(X_train_cmp, y_train_cmp)
lgb_time = time.time() - start_lgb
lgb_acc = accuracy_score(y_test_cmp, model_lgb.predict(X_test_cmp))

# 3. 获取 XGBoost 的准确率（用你刚训练好的模型）
xgb_acc = accuracy_score(y_test, model.predict(X_test))

# 4. 输出对比结果
print("\n对比结果：")
print(f"  XGBoost 准确率: {xgb_acc:.2%} (训练时间: {time.time() - start_lgb:.2f}秒)")
print(f"  LightGBM 准确率: {lgb_acc:.2%} (训练时间: {lgb_time:.2f}秒)")
print(f"  差异: {(lgb_acc - xgb_acc):.2%}")

if lgb_acc > xgb_acc:
    print("  ✅ LightGBM 胜出，建议切换")
else:
    print("  ⚠️ XGBoost 胜出或持平，保持原样")
# ===== 插入结束 =====



# 6. 评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.2%}")

# 7. 特征重要性
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 重要特征:")
print(importance.head(10))

# 8. 保存模型
os.makedirs("C:/Users/95722/projects/quant_research/models", exist_ok=True)
model.save_model("C:/Users/95722/projects/quant_research/models/xgboost_model.json")
print("\n✅ 模型已保存: models/xgboost_model.json")