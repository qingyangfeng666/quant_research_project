# 量化研究项目：中证500股指期货CTA策略

## 项目简介
基于中证500股指期货（IC）日线数据，使用XGBoost构建CTA策略，完整覆盖数据清洗、因子计算、模型训练、回测验证和绩效归因。

## 项目结构
```
quant_research/
├── data/processed/          # 清洗后的数据 + 因子
├── src/                     # 源代码
│   ├── technical_factors.py # 28个因子计算
│   ├── train_xgboost.py     # XGBoost训练 + LightGBM对比
│   ├── model_compare.py     # 独立模型对比脚本
│   ├── backtest_fixed.py    # 回测引擎
│   ├── purged_cv.py         # Purged交叉验证
│   └── cleaners/            # 数据清洗
├── reports/                 # 回测报告图片
└── models/                  # 保存的模型文件
```

## 因子列表（28个）
- 收益率类：ret_1d, ret_5d, ret_10d, ret_20d
- 均线类：ma_5/10/20/60, ma_ratio_5/10/20/60
- 波动率类：volatility_10/20/60
- 技术指标：rsi, bb_position, atr, atr_ratio
- 动量类：momentum_10/20/60
- 成交量类：volume_ma_5, volume_ratio, volume_change

## 模型对比
在相同数据上对比了XGBoost和LightGBM：

| 模型 | 测试集准确率 | 训练时间 |
| :--- | :--- | :--- |
| XGBoost | 51.23% | 0.09秒 |
| LightGBM | 50.62% | 1.37秒 |

XGBoost 准确率略高且训练更快，最终选用 XGBoost 作为主模型。

## 回测结果（测试集，阈值0.50）
| 指标 | 数值 |
| :--- | :--- |
| 年化收益率 | 26.20% |
| 夏普比率 | 1.89 |
| 最大回撤 | -7.29% |
| 胜率 | 13.04% |
| 交易次数 | 68次 |

> ⚠️ **注意**：Purged交叉验证显示，纯技术因子在中证500上的平均准确率约为50%，表明预测能力有限。回测收益可能来源于特定市场环境，需进一步引入基本面/宏观因子验证。

## 技术栈
- Python 3.14
- pandas, numpy
- XGBoost / LightGBM
- scikit-learn
- matplotlib

## 运行方式
```bash
python src/technical_factors.py    # 计算因子
python src/train_xgboost.py        # 训练模型（含对比）
python src/backtest_fixed.py       # 回测
python src/generate_report.py      # 生成报告
```

## 作者
qingyangfeng666
