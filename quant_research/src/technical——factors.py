"""
第一周 Day 4：计算技术因子
"""
import pandas as pd
import numpy as np
import os

def add_technical_factors(df):
    """在原始 OHLCV 数据上添加技术因子"""
    df = df.copy()
    
    # 1. 收益率
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['ret_20d'] = df['close'].pct_change(20)
    
    # 2. 均线
    for period in [5, 10, 20, 60]:
        df[f'ma_{period}'] = df['close'].rolling(period).mean()
        df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}'] - 1
    
    # 3. 波动率
    df['volatility_10'] = df['ret_1d'].rolling(10).std()
    df['volatility_20'] = df['ret_1d'].rolling(20).std()
    df['volatility_60'] = df['ret_1d'].rolling(60).std()
    
    # 4. RSI（14日）
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 5. 布林带（20日，2倍标准差）
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 6. ATR（14日）
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # 7. 动量
    for period in [10, 20, 60]:
        df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
    
    # 8. 成交量相关
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    df['volume_change'] = df['volume'].pct_change()

    
    
    return df

if __name__ == "__main__":
    # 读取清洗后的数据
    input_file = "C:/Users/95722/projects/quant_research/data/processed/ICL9_cleaned.csv"
    output_file = "C:/Users/95722/projects/quant_research/data/processed/ICL9_with_factors.csv"
    
    print("=" * 50)
    print("第一周：计算技术因子")
    print("=" * 50)
    
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['date'])
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    print("计算技术因子...")
    df = add_technical_factors(df)
    
    # 删除空值（因子计算过程中产生的）
    df = df.dropna()
    print(f"删除空值后: {len(df)} 行")
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存: {output_file}")
    
    # 显示因子列表
    print(f"\n因子列表 ({len(df.columns)-7} 个因子):")
    for col in df.columns:
        if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'return']:
            print(f"  - {col}")