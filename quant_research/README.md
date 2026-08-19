# 量化研究项目：中证500股指期货（IC）CTA策略

## 项目简介
本项目基于中证500股指期货（IC）的日线数据，构建了一个**XGBoost驱动的CTA策略**。完整覆盖了数据获取、因子计算、模型训练、回测验证和绩效归因的全流程。

## 项目结构
quant_research/
├── data/
│ ├── raw/ # 原始数据（通达信导出）
│ └── processed/ # 清洗后数据 + 因子表
├── src/
│ ├── cleaners/ # 数据清洗脚本
│ ├── data_fetcher.py # 数据获取
│ ├── technical_factors.py # 28个技术因子计算
│ ├── train_xgboost.py # XGBoost模型训练
│ ├── backtest_fixed.py # 回测引擎（含多阈值对比）
│ ├── purged_cv.py # Purged交叉验证
│ ├── export_signals.py # 导出预测信号
│ └── generate_report.py # 可视化报告生成
├── reports/ # 回测报告图片
├── models/ # 保存的模型文件
└── README.md

## 因子列表（28个）
- 收益率类：ret_1d, ret_5d, ret_10d, ret_20d
- 均线类：ma_5/10/20/60, ma_ratio_5/10/20/60
- 波动率类：volatility_10/20/60
- 技术指标：rsi, bb_position, atr, atr_ratio
- 动量类：momentum_10/20/60
- 成交量类：volume_ma_5, volume_ratio, volume_change

## 回测结果（测试集，阈值0.55）
| 指标 | 数值 |
| :--- | :--- |
| 总收益率 | XX% |
| 年化收益率 | XX% |
| 夏普比率 | X.XX |
| 最大回撤 | -X.XX% |
| 胜率 | XX% |
| 交易次数 | XX次 |

## 交叉验证结果
- 5折Purged交叉验证平均准确率：XX%
- 标准差：X.XXXX

## 技术栈
- Python 3.14
- pandas, numpy
- XGBoost
- scikit-learn
- matplotlib
- akshare

## 运行方式
```bash
# 1. 数据清洗
python src/cleaners/tongdaxin_clean.py

# 2. 计算因子
python src/technical_factors.py

# 3. 训练模型
python src/train_xgboost.py

# 4. 回测
python src/backtest_fixed.py

# 5. 生成报告
python src/generate_report.py
作者
令狐冲

