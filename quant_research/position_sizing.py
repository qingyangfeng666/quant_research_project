"""
第3周 Day 4：仓位管理（凯利公式 + 波动率调整）
预测目标：1天
"""
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. 加载数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# 2. 定义标签（预测1天）
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

# 6. 预测信号
y_pred_prob = model.predict_proba(X_test)[:, 1]

# ===== 7. 三种仓位策略对比 =====

# 7.1 固定仓位（基准）
df_test = df.iloc[split_idx:].copy()
df_test['pred_signal'] = (y_pred_prob > 0.55).astype(int)
df_test['position_fixed'] = df_test['pred_signal'] * 0.2  # 每次固定20%仓位

# 7.2 凯利公式仓位
train_pred = model.predict(X_train)
train_acc = (train_pred == y_train).mean()
win_rate = train_acc
loss_rate = 1 - win_rate

train_returns = df['return'].iloc[:split_idx].values
win_returns = train_returns[train_pred == 1]
loss_returns = train_returns[train_pred == 0]
avg_win = win_returns[win_returns > 0].mean() if len(win_returns[win_returns > 0]) > 0 else 0.01
avg_loss = abs(loss_returns[loss_returns < 0].mean()) if len(loss_returns[loss_returns < 0]) > 0 else 0.01
profit_ratio = avg_win / avg_loss if avg_loss > 0 else 1

kelly_fraction = (win_rate * profit_ratio - loss_rate) / profit_ratio
kelly_fraction = max(0, min(kelly_fraction, 0.3))

print("=" * 60)
print("凯利公式参数（预测1天）")
print("=" * 60)
print(f"胜率: {win_rate:.2%}")
print(f"盈亏比: {profit_ratio:.2f}")
print(f"凯利建议仓位: {kelly_fraction:.2%}")

df_test['position_kelly'] = df_test['pred_signal'] * kelly_fraction

# 7.3 波动率调整仓位
df_test['volatility'] = df_test['return'].rolling(20).std()
df_test['position_vol'] = df_test['pred_signal'] * 0.2 * (0.02 / df_test['volatility'])
df_test['position_vol'] = df_test['position_vol'].clip(0, 0.3)

# 8. 计算三种策略的收益
fee_rate = 0.0003

for pos_col in ['position_fixed', 'position_kelly', 'position_vol']:
    df_test[f'trade_cost_{pos_col}'] = abs(df_test[pos_col] - df_test[pos_col].shift(1)) * fee_rate
    df_test[f'strategy_return_{pos_col}'] = df_test[pos_col] * df_test['return'] - df_test[f'trade_cost_{pos_col}']
    df_test[f'strategy_return_{pos_col}'] = df_test[f'strategy_return_{pos_col}'].clip(-0.05, 0.05)

# 9. 计算累计收益
for pos_col in ['position_fixed', 'position_kelly', 'position_vol']:
    df_test[f'cumulative_{pos_col}'] = (1 + df_test[f'strategy_return_{pos_col}']).cumprod()

# 10. 对比结果
print("\n" + "=" * 60)
print("三种仓位策略对比（预测1天）")
print("=" * 60)

results = []
for pos_col, name in [
    ('position_fixed', '固定仓位(20%)'),
    ('position_kelly', '凯利公式'),
    ('position_vol', '波动率调整')
]:
    ret_col = f'strategy_return_{pos_col}'
    cum_col = f'cumulative_{pos_col}'
    
    df_temp = df_test.dropna(subset=[ret_col])
    total_ret = df_temp[cum_col].iloc[-1] - 1
    n_days = len(df_temp)
    n_years = n_days / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    sharpe = df_temp[ret_col].mean() / df_temp[ret_col].std() * (252 ** 0.5) if df_temp[ret_col].std() > 0 else 0
    
    cum = df_temp[cum_col].values
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    
    results.append({
        '策略': name,
        '总收益率': f"{total_ret:.2%}",
        '年化收益率': f"{annual_ret:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '平均仓位': f"{df_test[pos_col].mean():.2%}"
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

best = df_results.loc[df_results['夏普比率'].astype(float).idxmax()]
print("\n" + "=" * 60)
print(f"✅ 推荐: {best['策略']}")
print(f"   年化收益率: {best['年化收益率']}")
print(f"   夏普比率: {best['夏普比率']}")
print("=" * 60)