"""
第3周 Day 3：生成可视化回测报告（完整修复版）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("生成回测报告...")
print("=" * 60)

# 1. 加载数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
df = pd.read_csv(file_path, parse_dates=['date'])
print(f"加载数据: {len(df)} 行")

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
print("模型训练完成")

# 6. 预测
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_signal = (y_pred_prob > 0.55).astype(int)

# 7. 回测
df_test = df.iloc[split_idx:].copy()
df_test['pred_signal'] = y_pred_signal
df_test['position'] = df_test['pred_signal'].shift(1)
df_test['strategy_return'] = df_test['position'] * df_test['return']
df_test = df_test.dropna(subset=['strategy_return', 'return'])
df_test['cumulative_return'] = (1 + df_test['strategy_return']).cumprod()
df_test['buy_hold_return'] = (1 + df_test['return']).cumprod()

print(f"回测完成，测试集 {len(df_test)} 行")

# ===== 8. 创建图表 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ----- 图1：净值曲线 -----
ax1 = axes[0, 0]
ax1.plot(df_test['date'], df_test['cumulative_return'], label='策略', color='blue', linewidth=1.5)
ax1.plot(df_test['date'], df_test['buy_hold_return'], label='买入持有', color='gray', linewidth=1.5, linestyle='--')
ax1.set_title('策略净值 vs 买入持有', fontsize=12)
ax1.set_xlabel('日期')
ax1.set_ylabel('累计收益')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ----- 图2：回撤曲线 -----
ax2 = axes[0, 1]

# 计算回撤
cumulative = df_test['cumulative_return'].values
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max

# 画回撤填充区域
ax2.fill_between(df_test['date'], drawdown * 100, 0, 
                 where=drawdown < 0, color='red', alpha=0.3, interpolate=True)
# 画回撤曲线
ax2.plot(df_test['date'], drawdown * 100, color='red', linewidth=1.0)
# 零线
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

ax2.set_title('回撤曲线', fontsize=12)
ax2.set_xlabel('日期')
ax2.set_ylabel('回撤（%）')
ax2.grid(True, alpha=0.3)

# 如果完全没有回撤，显示提示
if np.all(drawdown >= 0):
    ax2.text(0.5, 0.5, '没有回撤（策略持续创新高）', 
             transform=ax2.transAxes, ha='center', va='center', fontsize=14, color='green')

# ----- 图3：每日收益分布 -----
ax3 = axes[1, 0]
returns = df_test['strategy_return'].dropna() * 100
ax3.hist(returns, bins=50, color='blue', alpha=0.5, edgecolor='black')
ax3.axvline(0, color='red', linestyle='--')
ax3.set_title('每日收益分布', fontsize=12)
ax3.set_xlabel('收益率（%）')
ax3.set_ylabel('频次')
ax3.grid(True, alpha=0.3)

# 显示统计信息
mean_ret = returns.mean()
std_ret = returns.std()
ax3.text(0.95, 0.95, f'均值: {mean_ret:.2f}%\n标准差: {std_ret:.2f}%', 
         transform=ax3.transAxes, ha='right', va='top', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ----- 图4：特征重要性 -----
ax4 = axes[1, 1]
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True).tail(15)
ax4.barh(importance['feature'], importance['importance'], color='green', alpha=0.7)
ax4.set_title('Top 15 重要特征', fontsize=12)
ax4.set_xlabel('重要性')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
os.makedirs("C:/Users/95722/projects/quant_research/reports", exist_ok=True)
output_path = "C:/Users/95722/projects/quant_research/reports/backtest_report.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 报告已保存: {output_path}")
plt.show()
input("按回车键关闭...")  # ← 加这行，让窗口保持打开

print("✅ 全部完成！")