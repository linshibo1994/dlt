# CLI 交互模式指南

## 命令行界面标准

### 主要命令结构
- `python3 dlt_main.py <command> [options]`
- 支持的主命令：data, analyze, predict, learn, smart, backtest, optimize, system

### 数据管理命令
```bash
# 数据状态查看
python3 dlt_main.py data status

# 获取最新开奖
python3 dlt_main.py data latest [--compare]

# 数据更新
python3 dlt_main.py data update [--source zhcw|500] [--periods N]
```

### 分析命令
```bash
# 基础分析
python3 dlt_main.py analyze -t basic -p 500

# 高级分析
python3 dlt_main.py analyze -t advanced -p 300

# 综合分析
python3 dlt_main.py analyze -t comprehensive -p 1000 [--report] [--visualize]
```

### 预测命令
```bash
# 传统预测
python3 dlt_main.py predict -m frequency|hot_cold|missing -c 5

# 高级预测
python3 dlt_main.py predict -m markov|bayesian|ensemble -c 3

# 复式预测
python3 dlt_main.py predict -m compound --front-count 8 --back-count 4

# 9种数学模型预测
python3 dlt_main.py predict -m nine_models -c 3
```

### 输出格式标准
- 使用中文提示信息和状态显示
- 号码显示格式：前区用空格分隔，后区用"+"连接
- 时间戳使用中文格式显示
- 进度条显示中文状态信息

### 错误处理模式
- 友好的中文错误提示
- 提供解决建议
- 记录详细错误日志
- 优雅降级处理