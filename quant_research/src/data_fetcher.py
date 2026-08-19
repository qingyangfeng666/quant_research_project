import akshare as ak
import pandas as pd
import os
import numpy as np
from datetime import datetime

# ========== 配置 ==========
LOCAL_DATA_PATH = "C:/Users/95722/projects/quant_research/data/"

# ========== 列名映射（支持不同格式） ==========
COLUMN_MAPPING = {
    'date': ['date', 'Date', 'DATE', '日期', 'trade_date', 'tradeDate'],
    'open': ['open', 'Open', '开盘', '开盘价', 'OPEN'],
    'high': ['high', 'High', '最高', '最高价', 'HIGH'],
    'low': ['low', 'Low', '最低', '最低价', 'LOW'],
    'close': ['close', 'Close', '收盘', '收盘价', 'CLOSE'],
    'volume': ['volume', 'Volume', '成交量', 'VOLUME', 'vol']
}

def find_column(df, standard_name):
    """
    在 DataFrame 中查找对应的列名
    """
    possible_names = COLUMN_MAPPING.get(standard_name, [standard_name])
    for col in df.columns:
        if col in possible_names:
            return col
    return None

def normalize_columns(df):
    """
    标准化列名：将各种别名转换为标准列名
    """
    df = df.copy()
    new_columns = {}
    
    for standard_name in COLUMN_MAPPING.keys():
        found = find_column(df, standard_name)
        if found:
            new_columns[found] = standard_name
    
    if new_columns:
        df = df.rename(columns=new_columns)
    
    return df

def clean_raw_data(df, symbol="IF888"):
    """
    数据清洗函数：读取本地数据后立即执行
    """
    df = df.copy()
    
    # 1. 标准化列名
    df = normalize_columns(df)
    
    # 2. 检查必需列
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        print(f"   当前列: {df.columns.tolist()}")
        print(f"   请确保数据包含: date, open, high, low, close, volume")
        return None
    
    # 3. 转换日期格式
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"❌ 日期格式转换失败: {e}")
        print(f"   请确保日期格式为 YYYY-MM-DD 或 YYYYMMDD")
        return None
    
    # 4. 删除空值
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # 5. 检查价格是否为正
    for col in ['open', 'high', 'low', 'close']:
        if (df[col] <= 0).any():
            print(f"⚠️ {col} 列存在零值或负值，自动修正...")
            df[col] = df[col].clip(lower=0.01)
    
    # 6. 确保 high >= low
    invalid_hl = df[df['high'] < df['low']]
    if len(invalid_hl) > 0:
        print(f"⚠️ 发现 {len(invalid_hl)} 条 high < low 的数据，自动修正...")
        df['high'] = df[['high', 'low']].max(axis=1)
        df['low'] = df[['high', 'low']].min(axis=1)
    
    # 7. 确保 open 和 close 在 high 和 low 之间
    df['open'] = df['open'].clip(df['low'], df['high'])
    df['close'] = df['close'].clip(df['low'], df['high'])
    
    # 8. 按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    # 9. 计算收益率（用于后续分析）
    df['return'] = df['close'].pct_change()
    
    # 10. 去极值：3倍标准差
    if df['return'].std() > 0:
        mean = df['return'].mean()
        std = df['return'].std()
        df['return'] = df['return'].clip(mean - 3*std, mean + 3*std)
    
    print(f"✅ 数据清洗完成！共 {len(df)} 条数据")
    print(f"   日期范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"   价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    return df

# ========== 读取本地数据 ==========
def load_local_data(symbol="IF888", start_date="2020-01-01", end_date=None):
    """
    从本地 CSV 文件读取数据
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 支持多种文件名格式
    possible_files = [
        f"{LOCAL_DATA_PATH}{symbol}.csv",
        f"{LOCAL_DATA_PATH}{symbol}.CSV",
        f"{LOCAL_DATA_PATH}{symbol}.txt",
        f"{LOCAL_DATA_PATH}{symbol}_data.csv",
    ]
    
    filename = None
    for f in possible_files:
        if os.path.exists(f):
            filename = f
            break
    
    if filename is None:
        print(f"❌ 本地数据不存在: {LOCAL_DATA_PATH}{symbol}.csv")
        print(f"   请确保数据文件放在: {LOCAL_DATA_PATH}")
        return None
    
    try:
        # 尝试多种编码格式读取
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        df = None
        
        for enc in encodings:
            try:
                df = pd.read_csv(filename, encoding=enc)
                print(f"✅ 读取本地数据成功: {filename} (编码: {enc})")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"❌ 无法读取文件，请检查编码格式")
            return None
        
        # 立即清洗数据
        df = clean_raw_data(df, symbol)
        if df is None:
            return None
        
        # 按日期筛选
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✅ 从本地读取 {symbol} 数据成功！共 {len(df)} 条数据")
        return df
        
    except Exception as e:
        print(f"❌ 本地数据读取失败: {e}")
        return None

# ========== 保存数据到本地 ==========
def save_data_to_local(df, symbol="IF888"):
    """
    将数据保存到本地 CSV
    """
    if not os.path.exists(LOCAL_DATA_PATH):
        os.makedirs(LOCAL_DATA_PATH)
    
    filename = f"{LOCAL_DATA_PATH}{symbol}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ 数据已保存到: {filename}")

# ========== 从网络获取数据 ==========
def fetch_from_network(symbol="IF888", start_date="2020-01-01", end_date=None):
    """
    从网络获取数据（备选方案）
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"正在从网络获取 {symbol} 数据...")
    
    # 尝试新浪数据源
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is not None and len(df) > 0:
            df = clean_raw_data(df, symbol)
            if df is not None:
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                print(f"✅ 从新浪获取 {symbol} 数据成功！")
                save_data_to_local(df, symbol)
                return df
    except Exception as e:
        print(f"新浪数据源失败: {e}")
    
    # 所有数据源失败
    print("所有数据源均失败，生成模拟数据...")
    df = generate_mock_data(start_date, end_date)
    save_data_to_local(df, symbol)
    return df

# ========== 生成模拟数据 ==========
def generate_mock_data(start_date, end_date):
    """
    生成模拟的期货日线数据
    """
    print("生成模拟数据用于学习测试...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]
    dates = dates[np.random.choice([True, False], len(dates), p=[0.7, 0.3])]
    dates = sorted(dates)
    
    n = len(dates)
    if n == 0:
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:100]
        n = len(dates)
    
    np.random.seed(42)
    price = 4000 + np.cumsum(np.random.randn(n) * 20)
    price = np.maximum(price, 3000)
    
    df = pd.DataFrame({
        'date': dates,
        'open': price + np.random.randn(n) * 10,
        'high': price + np.abs(np.random.randn(n) * 20) + 5,
        'low': price - np.abs(np.random.randn(n) * 20) - 5,
        'close': price,
        'volume': np.random.randint(10000, 100000, n)
    })
    
    df['high'] = df[['high', 'low']].max(axis=1)
    df['low'] = df[['high', 'low']].min(axis=1)
    df['open'] = df['open'].clip(df['low'], df['high'])
    df['close'] = df['close'].clip(df['low'], df['high'])
    
    df = clean_raw_data(df)
    print(f"✅ 生成模拟数据成功！共 {len(df)} 条数据")
    return df

# ========== 主入口 ==========
def get_future_data(symbol="IF888", start_date="2020-01-01", end_date=None, use_local=True):
    """
    获取期货日线数据（优先使用本地数据）
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n{'='*50}")
    print(f"正在获取 {symbol} 数据...")
    print(f"日期范围: {start_date} 到 {end_date}")
    print(f"{'='*50}")
    
    # 优先本地数据
    if use_local:
        df = load_local_data(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
        print("本地数据不可用，尝试联网获取...")
    
    # 网络获取
    return fetch_from_network(symbol, start_date, end_date)

# ========== 测试 ==========
if __name__ == "__main__":
    # 测试1：读取本地数据
    print("\n【测试1】尝试读取本地数据...")
    df = get_future_data("IF888", "2022-01-01", use_local=True)
    
    if df is not None:
        print("\n✅ 最终数据预览：")
        print(df[['date', 'open', 'high', 'low', 'close', 'volume']].head(10))
        print(f"\n数据量: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
    else:
        print("\n❌ 没有获取到有效数据")