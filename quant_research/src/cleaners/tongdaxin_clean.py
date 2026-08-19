"""
通达信数据专用清洗脚本（修复版）
"""
import pandas as pd
import os

RAW_FILE = "C:/Users/95722/projects/quant_research/data/raw/ICL9.csv"
OUTPUT_FILE = "C:/Users/95722/projects/quant_research/data/processed/ICL9_cleaned.csv"

print("=" * 50)
print("通达信数据清洗工具")
print("=" * 50)

# 1. 读取文件
with open(RAW_FILE, 'r', encoding='gbk') as f:
    content = f.read()

# 2. 按行分割
lines = content.strip().split('\n')

print(f"总行数: {len(lines)}")

# 3. 找到真正的列名行和数据行
header_line = None
data_start_index = 0

for i, line in enumerate(lines):
    line_clean = line.strip().strip('"')
    # 如果这一行包含"时间"和"开盘"，说明是真正的列名行
    if '时间' in line_clean and '开盘' in line_clean:
        header_line = line_clean
        data_start_index = i + 1
        print(f"找到列名行，位于第 {i} 行")
        break

if header_line is None:
    print("❌ 未找到列名行！")
    print("前5行预览：")
    for i, line in enumerate(lines[:5]):
        print(f"  {i}: {line[:100]}...")
    exit()

# 4. 解析列名
header_parts = header_line.split('\t')
print(f"列名: {header_parts}")

# 5. 找到需要的列的索引
col_map = {}
for i, col in enumerate(header_parts):
    col_clean = col.strip()
    if '时间' in col_clean or '日期' in col_clean:
        col_map['date'] = i
    elif '开盘' in col_clean:
        col_map['open'] = i
    elif '最高' in col_clean:
        col_map['high'] = i
    elif '最低' in col_clean:
        col_map['low'] = i
    elif '收盘' in col_clean:
        col_map['close'] = i
    elif '成交量' in col_clean:
        col_map['volume'] = i

print(f"找到列索引: {col_map}")

if len(col_map) < 6:
    print("❌ 未找到所有必需的列！")
    exit()

# 6. 提取数据
df_data = []
for i in range(data_start_index, len(lines)):
    line = lines[i].strip()
    if not line:
        continue
    # 去掉首尾的双引号
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    parts = line.split('\t')
    # 确保行有足够的列
    if len(parts) <= max(col_map.values()):
        continue
    try:
        df_data.append([
            parts[col_map['date']].strip(),
            float(parts[col_map['open']].strip()),
            float(parts[col_map['high']].strip()),
            float(parts[col_map['low']].strip()),
            float(parts[col_map['close']].strip()),
            float(parts[col_map['volume']].strip())
        ])
    except (ValueError, IndexError) as e:
        # 跳过无法解析的行
        continue

# 7. 创建DataFrame
df_clean = pd.DataFrame(df_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

print(f"解析后数据量: {len(df_clean)} 行")

if len(df_clean) == 0:
    print("❌ 没有解析到任何数据！")
    exit()

# 8. 转换日期
df_clean['date'] = pd.to_datetime(df_clean['date'])

# 9. 按日期排序
df_clean = df_clean.sort_values('date').reset_index(drop=True)

# 10. 计算收益率
df_clean['return'] = df_clean['close'].pct_change()

print(f"清洗后: {len(df_clean)} 行")
print(f"日期范围: {df_clean['date'].min()} 到 {df_clean['date'].max()}")
print(f"价格范围: {df_clean['close'].min():.2f} - {df_clean['close'].max():.2f}")

# 在 tongdaxin_clean.py 的 main() 函数中，保存之前加：

# ===== 去极值：去掉收益率超过 ±10% 的极端值 =====
print(f"去极值前: {len(df_clean)} 行")

# 删除收益率超过 ±10% 的极端值（期货很少一天涨跌超过10%）
df_clean = df_clean[df_clean['return'].abs() < 0.10]

print(f"去极值后: {len(df_clean)} 行")



# 11. 保存
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"✅ 已保存: {OUTPUT_FILE}")

# 12. 预览
print("\n数据预览:")
print(df_clean[['date', 'open', 'high', 'low', 'close', 'volume']].head(10))