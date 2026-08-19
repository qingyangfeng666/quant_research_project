"""
检查回撤数据（修复版）
"""
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. 加载数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# 2. 定义标签
df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
df = df.dropna(subset=['target'])

# 3. 准备特征
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'return', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values
y = df['target'].values

# 4. 切分
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 5. 训练模型
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
model.fit(X_train, y_train)

# 6. 预测
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_signal = (y_pred_prob > 0.55).astype(int)

# 7. 回测
df_test = df.iloc[split_idx:].copy()
df_test['pred_signal'] = y_pred_signal
df_test['position'] = df_test['pred_signal'].shift(1)
df_test['strategy_return'] = df_test['position'] * df_test['return']

# 🔴 关键：删除空值
df_test = df_test.dropna(subset=['strategy_return', 'return'])

# 计算累计收益
df_test['cumulative_return'] = (1 + df_test['strategy_return']).cumprod()

# 8. 计算回撤
cumulative = df_test['cumulative_return'].values
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max

# 9. 打印统计
print("=" * 60)
print("回撤数据分析（修复版）")
print("=" * 60)
print(f"有效数据量: {len(df_test)} 条")
print(f"累计收益最大值: {cumulative.max():.2%}")
print(f"累计收益最小值: {cumulative.min():.2%}")
print(f"最大回撤: {drawdown.min():.2%}")
print(f"回撤标准差: {drawdown.std():.4f}")
print(f"回撤等于0的天数: {(drawdown == 0).sum()} / {len(drawdown)}")
print(f"回撤小于-1%的天数: {(drawdown < -0.01).sum()} / {len(drawdown)}")
print(f"回撤小于-5%的天数: {(drawdown < -0.05).sum()} / {len(drawdown)}")

# 10. 显示有回撤的日期
drawdown_dates = df_test['date'][drawdown < -0.01]
if len(drawdown_dates) > 0:
    print(f"\n有回撤（<-1%）的日期: {drawdown_dates.head(10).tolist()} ...")
else:
    print("\n没有回撤超过1%的日期")