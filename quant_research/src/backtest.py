"""
第2周 Day 2：回测引擎（含年化收益率 + 多阈值对比）
"""
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. 加载数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
# file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors_talib.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# 2. 定义标签（预测3天后涨跌）
df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
df = df.dropna(subset=['target'])

# 3. 准备特征
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'return', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values
y = df['target'].values

# 4. 按时间顺序切分（80%训练，20%测试）
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = df['date'].values[split_idx:]

# 5. 加载模型
model = xgb.XGBClassifier()
model.load_model("C:/Users/95722/projects/quant_research/models/xgboost_model.json")

# 6. 只在测试集上预测
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 7. 测试不同阈值
thresholds = [0.5, 0.52, 0.55, 0.58]
results = []

print("=" * 60)
print("不同阈值下的回测结果对比（单日收益±5%封顶）")
print("=" * 60)

for threshold in thresholds:
    y_pred_signal = (y_pred_prob > threshold).astype(int)
    
    df_test = df.iloc[split_idx:].copy()
    df_test['pred_signal'] = y_pred_signal
    df_test['position'] = df_test['pred_signal'].shift(1)
    
    # 手续费
    fee_rate = 0.0003
    df_test['trade_cost'] = abs(df_test['position'] - df_test['position'].shift(1)) * fee_rate
    df_test['strategy_return'] = df_test['position'] * df_test['return'] - df_test['trade_cost']
    
    # 🔴 新增：单日收益封顶 ±5%（股指期货正常情况下不会超过这个范围）
    df_test['strategy_return'] = df_test['strategy_return'].clip(-0.05, 0.05)
    # 删除空值（防止 cumprod 变成 nan）
    df_test = df_test.dropna(subset=['strategy_return', 'return'])
    df_bt = df_test.dropna(subset=['strategy_return', 'return'])
    
    if len(df_bt) == 0:
        continue
    
    # 计算指标
    total_return = (1 + df_bt['strategy_return']).prod() - 1
    
    n_days = len(df_bt)
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    sharpe = df_bt['strategy_return'].mean() / df_bt['strategy_return'].std() * (252 ** 0.5)
    max_drawdown = (df_bt['strategy_return'].cumsum() - df_bt['strategy_return'].cumsum().cummax()).min()
    win_rate = (df_bt['strategy_return'] > 0).mean()
    trade_count = df_bt['trade_cost'].gt(0).sum()
    
    results.append({
        '阈值': threshold,
        '总收益率': f"{total_return:.2%}",
        '年化收益率': f"{annual_return:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_drawdown:.2%}",
        '胜率': f"{win_rate:.2%}",
        '交易次数': trade_count
    })

# 显示结果
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# 推荐阈值
best_sharpe = df_results.loc[df_results['夏普比率'].astype(float).idxmax()]
print("\n" + "=" * 60)
print(f"✅ 推荐阈值: {best_sharpe['阈值']}")
print(f"   年化收益率: {best_sharpe['年化收益率']}")
print(f"   夏普比率: {best_sharpe['夏普比率']}")
print("=" * 60)