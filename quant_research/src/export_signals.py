"""
导出带预测信号的完整表格（含0/1）
"""
import pandas as pd
import xgboost as xgb
import os

# 1. 加载原始数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

print(f"原始数据: {len(df)} 行")

# 2. 准备特征
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'return']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values

# 3. 加载模型
model = xgb.XGBClassifier()
model.load_model("C:/Users/95722/projects/quant_research/models/xgboost_model.json")

# 4. 生成预测信号
df['pred_prob'] = model.predict_proba(X)[:, 1]
df['pred_signal'] = (df['pred_prob'] > 0.55).astype(int)

# 5. 显示列名
print(f"\n列名: {df.columns.tolist()}")

# 6. 查看前20行（含0/1）
print("\n前20行数据（含0/1信号）:")
print(df[['date', 'close', 'pred_prob', 'pred_signal']].head(20))

# 7. 统计信号分布
print(f"\n信号分布:")
print(df['pred_signal'].value_counts())

# 8. 导出完整的带信号表格
output_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_signals.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 已导出带信号的完整表格: {output_path}")
print(f"   包含 {len(df)} 行, {len(df.columns)} 列")
print(f"   新增列: pred_prob (上涨概率), pred_signal (0/1)")