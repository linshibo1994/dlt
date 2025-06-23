# 大乐透马尔可夫链预测系统

基于真实历史数据的大乐透号码预测系统，使用马尔可夫链算法进行预测分析。

## 快速开始

### 1. 安装依赖
```bash
pip3 install -r requirements.txt
```

### 2. 获取数据
```bash
# 获取最近100期数据
python3 dlt_500_crawler.py -c 100 -o data/dlt_data.csv

# 获取所有历史数据（推荐）
python3 dlt_500_crawler.py -a -o data/dlt_data.csv
```

### 3. 马尔可夫链预测

#### 预测1注最稳定的号码
```bash
python3 markov_predictor.py -d data/dlt_data.csv -n 1 --explain
```

#### 预测5注号码
```bash
python3 markov_predictor.py -d data/dlt_data.csv -n 5
```

#### 预测10注号码
```bash
python3 markov_predictor.py -d data/dlt_data.csv -n 10
```

## 主要功能

### 数据爬取 (dlt_500_crawler.py)
- 从500彩票网获取真实大乐透历史数据
- 支持获取指定期数或全量历史数据
- 支持数据更新和去重

**使用示例：**
```bash
# 获取最近50期数据
python3 dlt_500_crawler.py -c 50

# 获取所有历史数据
python3 dlt_500_crawler.py -a

# 更新现有数据
python3 dlt_500_crawler.py -u data/dlt_data.csv -n 10
```

### 马尔可夫链预测 (markov_predictor.py)
- 基于历史号码转移概率进行预测
- 支持预测1注到多注号码
- 提供详细的预测过程说明

**使用示例：**
```bash
# 预测1注，显示详细过程
python3 markov_predictor.py -n 1 --explain

# 预测5注号码
python3 markov_predictor.py -n 5

# 使用指定数据文件预测
python3 markov_predictor.py -d data/dlt_data.csv -n 3
```

### 高级马尔可夫链分析 (advanced_markov_analyzer.py) ⭐推荐
- 逐期分析概率转移，记录历史概率变化
- 计算每个号码的稳定性得分
- 提供最稳定和最高概率的预测
- 支持渐进式分析和预测对比

**使用示例：**
```bash
# 运行完整分析（首次使用）
python3 advanced_markov_analyzer.py -d data/dlt_data.csv --analyze -n 5

# 仅进行预测（使用已有分析）
python3 advanced_markov_analyzer.py -d data/dlt_data.csv --predict-only -n 5

# 预测1注最稳定的号码
python3 advanced_markov_analyzer.py -d data/dlt_data.csv --predict-only -n 1
```

### 数据分析 (main.py)
- 基础统计分析
- 高级分析（马尔可夫链、贝叶斯、集成预测）
- 完整的命令行界面

**使用示例：**
```bash
# 基础分析
python3 main.py analyze -d data/dlt_data.csv

# 马尔可夫链预测
python3 main.py markov -d data/dlt_data.csv -c 3 --explain

# 集成预测
python3 main.py ensemble -d data/dlt_data.csv -c 5 --explain
```

### 数据管理工具

#### 数据去重 (dedup.py)
```bash
python3 dedup.py data/dlt_data.csv -c
```

#### 重复检查 (check_duplicates.py)
```bash
python3 check_duplicates.py data/dlt_data.csv -q
```

## 预测算法说明

### 马尔可夫链算法
- **原理**：基于历史号码的状态转移概率
- **特点**：考虑号码之间的关联性和转移规律
- **优势**：能够捕捉号码间的依赖关系
- **适用**：中短期预测，寻找号码间的关联模式

### 预测策略
1. **单注预测**：选择转移概率最高的号码组合
2. **多注预测**：在保证多样性的同时选择高概率号码
3. **稳定性**：基于大量历史数据确保预测的稳定性

## 文件结构

```
├── dlt_500_crawler.py           # 500彩票网数据爬虫
├── markov_predictor.py          # 马尔可夫链预测器
├── advanced_markov_analyzer.py  # 高级马尔可夫链分析器（⭐推荐）
├── main.py                      # 主程序（完整功能）
├── advanced_analyzer.py         # 高级分析器
├── basic_analyzer.py            # 基础分析器
├── utils.py                     # 工具函数
├── dedup.py                     # 数据去重工具
├── check_duplicates.py          # 重复检查工具
├── requirements.txt             # 依赖包列表
├── data/                        # 数据目录
│   └── dlt_data.csv            # 大乐透历史数据
├── analysis/                    # 分析结果目录
│   ├── historical_analysis.json # 历史分析数据
│   └── analysis_report.json    # 分析报告
└── output/                      # 输出目录
    ├── basic/                  # 基础分析结果
    └── advanced/               # 高级分析结果
```

## 使用建议

### 1. 数据准备
```bash
# 首次使用，建议获取所有历史数据
python3 dlt_500_crawler.py -a -o data/dlt_data.csv

# 定期更新（每周1-2次）
python3 dlt_500_crawler.py -u data/dlt_data.csv -n 10
```

### 2. 预测使用
```bash
# 🎯 最推荐：使用高级马尔可夫链分析器
python3 advanced_markov_analyzer.py -d data/dlt_data.csv --analyze -n 5

# 快速预测（使用已有分析）
python3 advanced_markov_analyzer.py -d data/dlt_data.csv --predict-only -n 5

# 或者使用基础马尔可夫链预测器
python3 markov_predictor.py -n 5 --explain

# 或者使用主程序的马尔可夫链功能
python3 main.py markov -d data/dlt_data.csv -c 5 --explain
```

### 3. 数据验证
```bash
# 检查数据质量
python3 check_duplicates.py data/dlt_data.csv -q

# 如有重复，进行去重
python3 dedup.py data/dlt_data.csv -c
```

## 预测结果示例

```
使用马尔可夫链预测 5 注号码...

第 1 注: 前区 06 08 10 22 29 | 后区 01 03
第 2 注: 前区 03 12 18 25 31 | 后区 05 09
第 3 注: 前区 07 14 20 26 33 | 后区 02 11
第 4 注: 前区 01 09 15 23 35 | 后区 04 08
第 5 注: 前区 05 11 17 24 30 | 后区 06 12
```

## 注意事项

1. **数据质量**：确保使用真实、完整的历史数据
2. **预测准确性**：彩票具有随机性，预测结果仅供参考
3. **理性购彩**：请理性对待预测结果，适度购彩
4. **数据更新**：建议定期更新数据以保持预测的时效性

## 免责声明

本系统仅用于技术研究和学习目的，预测结果不构成购彩建议。彩票具有随机性，请理性购彩，适度娱乐。
