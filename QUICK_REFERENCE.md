# 🚀 大乐透预测系统快速参考

## 📋 常用命令速查

### 🔧 系统管理
```bash
python3 dlt_main.py version                    # 查看版本信息
python3 dlt_main.py data status                # 查看数据状态
python3 dlt_main.py data latest                # 查看最新开奖
python3 dlt_main.py data latest --compare      # 中奖比较
python3 dlt_main.py system cache info          # 查看缓存信息
python3 dlt_main.py system cache clear --type all  # 清理缓存
```

### 📊 数据更新
```bash
python3 dlt_main.py data update --source zhcw          # 更新所有数据
python3 dlt_main.py data update --source zhcw --periods 100  # 更新100期
python3 dlt_main.py data update --source 500           # 从500彩票网更新
```

### 🔍 数据分析
```bash
python3 dlt_main.py analyze -t basic -p 500                    # 基础分析
python3 dlt_main.py analyze -t advanced -p 300                 # 高级分析
python3 dlt_main.py analyze -t comprehensive -p 1000 --report  # 综合分析+报告
python3 dlt_main.py analyze -t comprehensive -p 1000 --visualize  # 生成图表
```

### 🎯 号码预测
```bash
# 传统方法
python3 dlt_main.py predict -m frequency -c 5      # 频率预测
python3 dlt_main.py predict -m hot_cold -c 3       # 冷热号预测
python3 dlt_main.py predict -m missing -c 5        # 遗漏预测

# 高级方法
python3 dlt_main.py predict -m markov -c 3         # 马尔可夫链
python3 dlt_main.py predict -m bayesian -c 5       # 贝叶斯预测
python3 dlt_main.py predict -m ensemble -c 5       # 集成预测

# 机器学习
python3 dlt_main.py predict -m super -c 3          # 超级预测器
python3 dlt_main.py predict -m adaptive -c 5       # 自适应预测

# 复式投注
python3 dlt_main.py predict -m compound --front-count 8 --back-count 4  # 8+4复式
python3 dlt_main.py predict -m duplex              # 胆拖投注

# 混合策略
python3 dlt_main.py predict -m mixed_strategy --strategy conservative -c 3  # 保守策略
python3 dlt_main.py predict -m mixed_strategy --strategy aggressive -c 3    # 激进策略
python3 dlt_main.py predict -m mixed_strategy --strategy balanced -c 3      # 平衡策略

# 马尔可夫自定义
python3 dlt_main.py predict -m markov_custom -c 3 --analysis-periods 300 --predict-periods 2
```

### 🧠 自适应学习
```bash
# 学习过程
python3 dlt_main.py learn -s 100 -t 1000 --algorithm ucb1 --save ucb1.json
python3 dlt_main.py learn -s 100 -t 1000 --algorithm epsilon_greedy --save eg.json
python3 dlt_main.py learn -s 100 -t 1000 --algorithm thompson_sampling --save ts.json

# 智能预测
python3 dlt_main.py smart -c 5 --load output/learning/learning_ucb1_*.json
python3 dlt_main.py smart -c 3  # 使用默认配置
```

### ⚙️ 参数优化
```bash
python3 dlt_main.py optimize -t 100 -r 10                      # 基础优化
python3 dlt_main.py optimize -t 500 -r 20 --save opt.json     # 高级优化
```

### 📈 历史回测
```bash
python3 dlt_main.py backtest -s 100 -t 500 -m frequency    # 频率回测
python3 dlt_main.py backtest -s 100 -t 1000 -m ensemble    # 集成回测
python3 dlt_main.py backtest -s 200 -t 800 -m markov       # 马尔可夫回测
```

## 📁 输出目录结构

```
output/
├── predictions/     # 预测结果 (.json)
├── learning/        # 学习结果 (.json)
├── optimization/    # 优化结果 (.json)
├── reports/         # 分析报告 (.txt)
├── visualization/   # 可视化图表 (.png)
└── backtest/        # 回测结果 (.log)
```

## 🎯 预测方法对比

| 方法 | 类型 | 适用场景 | 准确率 | 计算速度 |
|------|------|----------|--------|----------|
| frequency | 传统 | 长期稳定 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| hot_cold | 传统 | 短期趋势 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| missing | 传统 | 遗漏补偿 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| markov | 高级 | 状态转移 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| bayesian | 高级 | 条件概率 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| ensemble | 高级 | 综合预测 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| super | ML | 深度学习 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| adaptive | ML | 自适应 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| mixed_strategy | 混合 | 策略组合 | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🔧 常见问题解决

### 数据问题
```bash
# 数据加载失败
python3 dlt_main.py system cache clear --type data
python3 dlt_main.py data update --source zhcw

# 数据过期
python3 dlt_main.py data update --source zhcw --periods 50
```

### 性能问题
```bash
# 内存不足
python3 dlt_main.py analyze -t basic -p 200  # 减少期数
python3 dlt_main.py system cache clear --type all  # 清理缓存

# 速度慢
python3 dlt_main.py predict -m frequency -c 3  # 使用快速方法
```

### 预测问题
```bash
# 预测失败
python3 dlt_main.py data status  # 检查数据
python3 dlt_main.py analyze -t basic -p 100  # 重新分析

# 结果异常
python3 dlt_main.py predict -m ensemble -c 1  # 使用稳定方法
```

## 📊 参数说明

### 通用参数
- `-c, --count`: 生成注数 (1-20)
- `-p, --periods`: 分析期数 (100-3000)
- `--save`: 保存文件名
- `--source`: 数据源 (zhcw/500)

### 预测参数
- `--front-count`: 复式前区号码数 (6-15)
- `--back-count`: 复式后区号码数 (3-12)
- `--strategy`: 混合策略 (conservative/aggressive/balanced)
- `--analysis-periods`: 马尔可夫分析期数
- `--predict-periods`: 马尔可夫预测期数

### 学习参数
- `-s, --start`: 起始期数
- `-t, --test`: 测试期数
- `--algorithm`: 学习算法 (ucb1/epsilon_greedy/thompson_sampling)

### 回测参数
- `-s, --start`: 起始期数
- `-t, --test`: 测试期数
- `-m, --method`: 预测方法

### 优化参数
- `-t, --test-periods`: 测试期数
- `-r, --rounds`: 优化轮数

## 🎨 可视化图表

### 生成图表
```bash
python3 dlt_main.py analyze -t basic -p 500 --visualize
```

### 图表类型
- **frequency_distribution.png**: 号码频率分布图
- **trend_charts.png**: 和值走势图
- **correlation_charts.png**: 相关性热力图

## 📝 日志文件

```bash
# 查看系统日志
tail -f logs/dlt_predictor.log

# 查看错误日志
tail -f logs/errors.log
```

## 🔗 相关文件

- **README.md**: 项目说明文档
- **USAGE_GUIDE.md**: 详细使用指南
- **PROJECT_STRUCTURE.md**: 项目结构说明
- **requirements.txt**: 依赖包列表

---

**🎯 快速上手，智能预测！**
