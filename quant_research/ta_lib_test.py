"""
对比手写因子 vs TA-Lib 因子
验证数值是否一致
"""
import pandas as pd
import numpy as np
import talib

# 1. 加载数据
file_path = "C:/Users/95722/projects/quant_research/data/processed/ICL9_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# 提取 OHLCV 数组（TA-Lib 要求 numpy 数组格式）
high = df['high'].values
low = df['low'].values
close = df['close'].values
volume = df['volume'].values

print("=" * 60)
print("TA-Lib 测试：手写 vs TA-Lib")
print("=" * 60)

# ===== 1. RSI（14日） =====
# 你手写的 RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi_manual = 100 - (100 / (1 + rs))

# TA-Lib 的 RSI
rsi_talib = talib.RSI(close, timeperiod=14)

print(f"RSI 对比:")
print(f"  手写 RSI (最后5个): {rsi_manual.tail(5).values}")
print(f"  TA-Lib RSI (最后5个): {rsi_talib[-5:]}")
print(f"  差值: {np.abs(rsi_manual.tail(5).values - rsi_talib[-5:]).mean():.6f}")

# ===== 2. 布林带（20日，2倍标准差） =====
bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

print(f"\n布林带对比:")
print(f"  TA-Lib 上轨 (最后5个): {bb_upper[-5:]}")
print(f"  TA-Lib 中轨 (最后5个): {bb_middle[-5:]}")
print(f"  TA-Lib 下轨 (最后5个): {bb_lower[-5:]}")

# ===== 3. ATR（14日） =====
atr_talib = talib.ATR(high, low, close, timeperiod=14)
print(f"\nATR (最后5个): {atr_talib[-5:]}")

# ===== 4. MACD =====
macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
print(f"\nMACD (最后5个):")
print(f"  MACD: {macd[-5:]}")
print(f"  Signal: {macd_signal[-5:]}")
print(f"  Histogram: {macd_hist[-5:]}")

print("\n✅ TA-Lib 测试完成！")