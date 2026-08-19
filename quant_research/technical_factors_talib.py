"""
技术因子计算 - TA-Lib 版本
直接替换手写因子，输出带因子的 CSV 文件
"""
import pandas as pd
import numpy as np
import talib
import os

def add_technical_factors_talib(df):
    """使用 TA-Lib 计算技术因子"""
    df = df.copy()
    
    # 提取数组
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    
    # ========== 1. 收益率（手写，TA-Lib 没有） ==========
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['ret_20d'] = df['close'].pct_change(20)
    
    # ========== 2. 均线（TA-Lib） ==========
    for period in [5, 10, 20, 60]:
        df[f'ma_{period}'] = talib.SMA(close, timeperiod=period)
        df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}'] - 1
    
    # ========== 3. 波动率（手写） ==========
    df['volatility_10'] = df['ret_1d'].rolling(10).std()
    df['volatility_20'] = df['ret_1d'].rolling(20).std()
    df['volatility_60'] = df['ret_1d'].rolling(60).std()
    
    # ========== 4. RSI（TA-Lib） ==========
    df['rsi'] = talib.RSI(close, timeperiod=14)
    
    # ========== 5. 布林带（TA-Lib） ==========
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ========== 6. ATR（TA-Lib） ==========
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['atr_ratio'] = df['atr'] / df['close']
    
    # ========== 7. 动量（手写） ==========
    for period in [10, 20, 60]:
        df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
    
    # ========== 8. 成交量相关（手写） ==========
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    df['volume_change'] = df['volume'].pct_change()
    
    # ========== 9. 新增：CCI（TA-Lib） ==========
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)
    
    # ========== 10. 新增：MACD（TA-Lib） ==========
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
    
    # ========== 11. 新增：威廉指标（TA-Lib） ==========
    df['williams'] = talib.WILLR(high, low, close, timeperiod=14)
    
    # ========== 12. 新增：OBV（TA-Lib） ==========
    df['obv'] = talib.OBV(close, volume)
    
    # ========== 13. 乖离率（手写，用 TA-Lib 的 SMA） ==========
    for period in [6, 12, 24]:
        ma = talib.SMA(close, timeperiod=period)
        df[f'bias_{period}'] = (df['close'] - ma) / ma * 100
    
    # ========== 14. KDJ（手写，TA-Lib 没有） ==========
    low_9 = df['low'].rolling(9).min()
    high_9 = df['high'].rolling(9).max()
    rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # ========== 15. 价格位置（手写） ==========
    for period in [10, 20, 60]:
        high_period = df['high'].rolling(period).max()
        low_period = df['low'].rolling(period).min()
        df[f'price_position_{period}'] = (df['close'] - low_period) / (high_period - low_period)
    
    # ========== 16. 跳空缺口（手写） ==========
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_ratio'] = df['gap'] / df['close'].shift(1)
    
    # ========== 17. 振幅（手写） ==========
    df['amplitude'] = (df['high'] - df['low']) / df['close']
    df['amplitude_ma'] = df['amplitude'].rolling(20).mean()
    df['amplitude_ratio'] = df['amplitude'] / df['amplitude_ma']
    
    # ========== 18. 滚动夏普（手写） ==========
    df['rolling_sharpe_20'] = df['ret_1d'].rolling(20).mean() / df['ret_1d'].rolling(20).std() * (252 ** 0.5)
    df['rolling_sharpe_60'] = df['ret_1d'].rolling(60).mean() / df['ret_1d'].rolling(60).std() * (252 ** 0.5)
    
    # ========== 19. 偏度/峰度（手写） ==========
    df['skew_20'] = df['ret_1d'].rolling(20).skew()
    df['kurt_20'] = df['ret_1d'].rolling(20).kurt()
    
    return df

if __name__ == "__main__":
    input_file = "C:/Users/95722/projects/quant_research/data/processed/ICL9_cleaned.csv"
    output_file = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors_talib.csv"
    
    print("=" * 60)
    print("计算因子 - TA-Lib 版本")
    print("=" * 60)
    
    df = pd.read_csv(input_file, parse_dates=['date'])
    print(f"原始数据: {len(df)} 行")
    
    df = add_technical_factors_talib(df)
    df = df.dropna()
    
    print(f"删除空值后: {len(df)} 行")
    print(f"因子数量: {len(df.columns) - 7}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存: {output_file}")