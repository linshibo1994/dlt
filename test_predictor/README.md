# 大乐透预测方法测试系统

## 🎯 项目简介

这是一个专为大乐透预测算法设计的全面测试评估系统。它可以自动化测试您项目中的各种预测方法，通过与真实开奖结果对比，找出最有效的中奖预测算法。

### 核心功能

✅ **全方位测试** - 支持27种项目内置预测算法（频率分析、马尔可夫链、贝叶斯、深度学习等）  
✅ **智能中奖判断** - 基于2019年新规则的9个奖级准确判断  
✅ **自动停止机制** - 中得一等奖或二等奖时自动停止测试  
✅ **参数优化** - 智能调整periods（分析期数）和count（生成注数）参数  
✅ **并行处理** - 多线程并行测试，提高效率  
✅ **详细报告** - JSON、CSV、HTML多格式结果输出  

## 🚀 快速开始

### 1. 一键启动

```bash
cd test_predictor
./start_test.sh
```

然后按照菜单选择测试模式即可。

### 2. 命令行使用

```bash
# 系统检查
python3 test_predictor.py check

# 查看可用预测方法、并发配置等
python3 test_predictor.py config

# 快速测试（5-10分钟）
python3 test_predictor.py quick

# 全面测试（2-8小时）
python3 test_predictor.py comprehensive

# 自定义测试（示例：马尔可夫方法，从10期到2000期）
python3 test_predictor.py custom markov 10 2000 50 1
# 所有自定义测试都会验证方法名是否为项目真实实现
```

## 📊 测试策略

### 支持的预测方法

项目当前自动同步27种预测方法，按照配置分类如下：

| 分类 | 方法 |
| --- | --- |
| 基础分析 | `frequency`, `hot_cold`, `missing` |
| 马尔可夫系列 | `markov`, `markov_2nd`, `markov_3rd`, `adaptive_markov`, `markov_custom`, `markov_compound` |
| 概率模型 | `bayesian` |
| 集成算法 | `ensemble`, `stacking`, `adaptive_ensemble`, `ultimate_ensemble` |
| 智能/增强 | `super`, `adaptive`, `mixed_strategy`, `highly_integrated`, `advanced_integration`, `enhanced` |
| 复式/胆拖 | `compound`, `duplex`, `nine_models`, `nine_models_compound`, `markov_compound` |
| 深度学习 | `lstm`, `transformer`, `gan` |

> 任何自定义或策略测试都会基于上述真实方法执行，并在启动前自动校验非法方法名。

### 快速测试模式
- **用途**: 快速验证系统功能
- **方法**: frequency, markov, markov_2nd, bayesian, ensemble, lstm
- **时间**: 5-10分钟

### 全面测试模式  
- **用途**: 评估所有预测方法
- **方法**: 全部27种算法
- **时间**: 2-8小时

### 优化测试模式
- **用途**: 随机参数寻找最优配置
- **方法**: 高级算法
- **时间**: 可自定义

### 自定义测试模式
- **用途**: 针对特定方法深度测试
- **示例**: 测试马尔可夫链在不同期数下的表现
- **参数说明**:
  - `method`: 上表列出的任意真实方法名
  - `periods_start/periods_end/periods_step`: 分析期数范围与步长
  - `count`: 每个测试用例生成的注数

## 🏆 中奖等级

基于2019年大乐透新规则（9个奖级）：

| 奖级 | 中奖条件 | 说明 |
|------|----------|------|
| 一等奖 | 5+2 | 前区5个号码全中 + 后区2个号码全中 |
| 二等奖 | 5+1 | 前区5个号码全中 + 后区1个号码中 |
| 三等奖 | 5+0 | 前区5个号码全中 |
| 四等奖 | 4+2 | 前区4个号码中 + 后区2个号码全中 |
| 五等奖 | 4+1 | 前区4个号码中 + 后区1个号码中 |
| 六等奖 | 3+2 | 前区3个号码中 + 后区2个号码全中 |
| 七等奖 | 4+0 | 前区4个号码中 |
| 八等奖 | 3+1 或 2+2 | 前区3个+后区1个 或 前区2个+后区2个 |
| 九等奖 | 3+0 或 2+1 或 1+2 或 0+2 | 多种组合 |

## 📁 项目结构

```
test_predictor/
├── test_predictor.py          # 主程序入口
├── start_test.sh              # 一键启动脚本
├── modules/                   # 核心模块
│   ├── config_manager.py      # 配置管理
│   ├── predictor_caller.py    # 预测调用
│   ├── lottery_judge.py       # 中奖判断
│   ├── lottery_data.py        # 数据管理
│   ├── test_controller.py     # 测试控制
│   └── result_recorder.py     # 结果记录
├── config/                    # 配置文件
│   └── config.json           # 主配置
├── logs/                      # 运行日志
└── results/                   # 测试结果
    ├── *_results_*.json      # 详细结果
    ├── *_summary_*.json      # 汇总统计
    ├── *_results_*.csv       # CSV数据
    ├── *_report_*.html       # 可视化报告
    └── *_winners_*.txt       # 中奖记录
```

## ⚙️ 配置说明

主要配置项（`config/config.json`）：

```json
{
  "test_settings": {
    "timeout_seconds": 60,
    "max_retries": 2,
    "parallel_workers": 4,
    "stop_on_major_prize": true,
    "major_prize_levels": [1, 2]
  },
  "prediction_methods": {
    "basic": ["frequency", "hot_cold", "missing"],
    "markov": ["markov", "markov_2nd", "markov_3rd", "adaptive_markov", "markov_custom", "markov_compound"],
    "probabilistic": ["bayesian"],
    "ensemble": ["ensemble", "stacking", "adaptive_ensemble", "ultimate_ensemble"],
    "intelligent": ["super", "adaptive", "mixed_strategy", "highly_integrated", "advanced_integration", "enhanced"],
    "compound": ["compound", "duplex", "nine_models", "nine_models_compound", "markov_compound"],
    "deep_learning": ["lstm", "transformer", "gan"]
  },
  "test_strategies": {
    "quick": {
      "methods": ["frequency", "markov", "markov_2nd", "bayesian", "ensemble", "lstm"],
      "periods_list": [100, 500],
      "count_list": [1, 2]
    },
    "comprehensive": {
      "methods": "all",
      "periods_range": [50, 1000],
      "count_range": [1, 5]
    },
    "optimization": {
      "methods": ["markov", "markov_2nd", "markov_3rd", "adaptive_markov", "markov_custom", "bayesian", "ensemble", "stacking", "adaptive_ensemble", "ultimate_ensemble", "lstm", "transformer", "gan", "super", "adaptive"],
      "periods_range": [10, 2000],
      "count_range": [1, 10]
    }
  }
}
```

> 配置管理器会自动校验并补全上述列表中的真实方法，避免遗漏或误填无效方法。

## 📈 结果分析

### 关键指标

- **测试中奖率**: 中奖测试次数 / 总测试次数
- **预测中奖率**: 中奖预测数量 / 总预测数量  
- **最佳奖级**: 各方法达到的最高奖级
- **方法排名**: 按中奖表现排序的算法排行

### 输出报告

1. **HTML报告** - 可视化图表和统计分析
2. **CSV数据** - 适合进一步数据分析
3. **JSON结果** - 完整的结构化数据
4. **中奖记录** - 专门的中奖详情报告

## 💡 使用建议

### 寻找最佳方法

1. 先运行**全面测试**，了解各方法整体表现
2. 选择表现优秀的方法进行**自定义深度测试**
3. 关注**测试中奖率**和**最佳奖级**两个指标

### 验证特定算法

```bash
# 测试不同阶数的马尔可夫方法
python3 test_predictor.py custom markov 100 1000 100 1
python3 test_predictor.py custom markov_2nd 100 1000 100 1
python3 test_predictor.py custom markov_3rd 100 1000 100 1

# 深度学习或集成方法
python3 test_predictor.py custom transformer 200 1200 100 2
python3 test_predictor.py custom ultimate_ensemble 100 800 100 3
```

### 寻找一等奖

```bash
# 使用优化模式进行大量随机测试
./start_test.sh optimization
```

## ⚠️ 注意事项

1. **测试时间**: 全面测试可能需要数小时，建议在空闲时运行
2. **资源占用**: 并行测试会占用较多CPU资源
3. **存储空间**: 注意结果文件占用的磁盘空间
4. **安全中断**: 使用Ctrl+C可以安全停止，已完成结果会保存
5. **研究用途**: 结果仅供学术研究，不建议用于实际投注

## 🛠️ 故障排除

### 常见问题

**Q: "无法连接预测系统"**  
A: 检查 `python3 dlt_main.py --help` 是否正常运行

**Q: "数据文件错误"**  
A: 确认 `data/dlt_data_all.csv` 文件存在且格式正确

**Q: "无法解析预测结果"**  
A: 预测输出格式可能发生变化，需要调整解析逻辑

**Q: 测试运行缓慢**  
A: 减少并行线程数或跳过深度学习方法

### 系统检查

```bash
./start_test.sh check
```

这个命令会检查所有系统组件的状态。

## 📚 详细文档

- [完整使用指南](test_predictor_guide.md) - 详细的使用说明和配置指南
- [需求文档](test_predictor_requirements.md) - 项目需求和功能规格
- [任务文档](test_predictor_tasks.md) - 开发任务和实现计划

## 🎉 示例结果

```
=== 测试完成摘要 ===
测试策略: comprehensive
执行时间: 3600.5 秒
总测试数: 156
中奖测试: 23
重大奖项: 2
测试中奖率: 14.74%

🏆 中奖等级统计:
  1等奖 (一等奖): 1 次
  2等奖 (二等奖): 1 次  
  3等奖 (三等奖): 8 次
  
🔍 方法表现排行:
  1. markov_3rd: 1等奖 (中奖率 18.2%)
  2. bayesian: 2等奖 (中奖率 15.8%)
  3. ensemble: 3等奖 (中奖率 12.5%)
```

---

**🎯 祝您测试顺利，找到最佳的预测方法！**