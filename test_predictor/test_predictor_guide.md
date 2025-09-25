# 大乐透预测方法测试工具使用指南

## 概述

这是一个全面的大乐透预测方法评估系统，可以自动测试各种预测算法的中奖效果，帮助找出最有效的预测方法。

## 功能特性

- ✅ 支持所有现有预测方法（频率、马尔可夫、贝叶斯、深度学习等）
- ✅ 自动中奖等级判断（基于2019年新规则的9个奖级）
- ✅ 智能参数优化和策略选择
- ✅ 获得一等奖/二等奖时自动停止
- ✅ 并行测试支持，提高效率
- ✅ 详细的结果记录和统计分析
- ✅ 多格式报告输出（JSON、CSV、HTML）

## 安装要求

确保您的系统满足以下要求：

- Python 3.8+
- 大乐透预测系统已正确安装（dlt_main.py可正常运行）
- 历史开奖数据文件存在（data/dlt_data_all.csv）

## 快速开始

### 1. 系统检查

首先检查系统是否正常：

```bash
cd test_predictor
python3 test_predictor.py check
```

### 2. 快速测试

运行快速测试，体验基本功能：

```bash
python3 test_predictor.py quick
```

这将测试几种基础预测方法，用时约5-10分钟。

### 3. 全面测试

运行完整的测试流程：

```bash
python3 test_predictor.py comprehensive
```

这将测试所有可用的预测方法，可能需要数小时完成。

## 使用方法

### 命令行界面

```bash
python3 test_predictor.py <command> [options]
```

#### 可用命令

- `check` - 检查系统状态
- `quick` - 快速测试模式
- `comprehensive` - 全面测试模式
- `optimization` - 优化测试模式
- `custom` - 自定义测试
- `config` - 显示当前配置
- `create-config` - 创建默认配置文件

### 测试模式详解

#### 1. 快速测试模式 (quick)

- **目的**: 快速验证系统功能
- **测试方法**: frequency, markov, bayesian
- **参数范围**: periods=[100, 500], count=[1, 2]
- **预计时间**: 5-10分钟

```bash
python3 test_predictor.py quick
```

#### 2. 全面测试模式 (comprehensive)

- **目的**: 测试所有预测方法
- **测试方法**: 所有可用方法
- **参数范围**: periods=50-1000（累进式），count=1-5
- **预计时间**: 2-8小时

```bash
python3 test_predictor.py comprehensive
```

#### 3. 优化测试模式 (optimization)

- **目的**: 随机参数优化
- **测试方法**: 高级算法
- **参数范围**: periods=10-2000（随机），count=1-10（随机）
- **最大测试数**: 1000次
- **预计时间**: 根据设置而定

```bash
python3 test_predictor.py optimization
```

#### 4. 自定义测试模式 (custom)

针对特定方法进行深度测试：

```bash
python3 test_predictor.py custom <方法> <起始期数> <结束期数> [步长] [注数]
```

示例：

```bash
# 测试马尔可夫方法，从10期到2000期，步长50，每次生成1注
python3 test_predictor.py custom markov 10 2000 50 1

# 测试频率方法，从100期到1000期，步长100，每次生成3注
python3 test_predictor.py custom frequency 100 1000 100 3
```

## 配置说明

### 配置文件位置

- 主配置：`test_predictor/config/config.json`

### 主要配置项

```json
{
  "test_settings": {
    "timeout_seconds": 60,        // 单次预测超时时间
    "max_retries": 2,             // 失败重试次数
    "parallel_workers": 4,        // 并行线程数
    "stop_on_major_prize": true,  // 中重大奖项时停止
    "major_prize_levels": [1, 2]  // 重大奖项级别
  }
}
```

### 修改配置

1. 直接编辑配置文件
2. 或使用命令查看当前配置：

```bash
python3 test_predictor.py config
```

## 结果分析

### 输出文件

测试完成后，结果保存在 `test_predictor/results/` 目录：

- `*_results_*.json` - 详细测试结果
- `*_summary_*.json` - 汇总统计
- `*_results_*.csv` - CSV格式数据
- `*_report_*.html` - 可视化HTML报告
- `*_winners_*.txt` - 中奖者专用报告

### 关键指标

- **测试中奖率**: 中奖测试数 / 总测试数
- **预测中奖率**: 中奖预测数 / 总预测数
- **最佳奖级**: 各方法达到的最高奖级
- **方法排名**: 按中奖表现排序

### 中奖等级

基于2019年大乐透新规则：

1. **一等奖**: 5+2 (前区全中+后区全中)
2. **二等奖**: 5+1 (前区全中+后区中1个)
3. **三等奖**: 5+0 (前区全中)
4. **四等奖**: 4+2
5. **五等奖**: 4+1
6. **六等奖**: 3+2
7. **七等奖**: 4+0
8. **八等奖**: 3+1 或 2+2
9. **九等奖**: 3+0 或 2+1 或 1+2 或 0+2

## 实际使用示例

### 示例1：寻找最佳方法

目标：找出最容易中奖的预测方法

```bash
# 运行全面测试
python3 test_predictor.py comprehensive

# 查看HTML报告，重点关注"方法表现排行"
# 选择排名靠前的方法进行深度测试
python3 test_predictor.py custom markov 10 2000 50 1
```

### 示例2：验证特定方法

目标：验证马尔可夫方法在不同参数下的表现

```bash
# 测试不同阶数的马尔可夫方法
python3 test_predictor.py custom markov 100 1000 100 1
python3 test_predictor.py custom markov_2nd 100 1000 100 1
python3 test_predictor.py custom markov_3rd 100 1000 100 1
```

### 示例3：寻找一等奖

目标：持续测试直到中一等奖

```bash
# 使用优化模式，随机参数大量测试
python3 test_predictor.py optimization

# 或者针对性测试高级方法
python3 test_predictor.py custom bayesian 500 1500 100 5
```

## 性能优化

### 1. 调整并行数

根据您的CPU核心数调整：

```json
{
  "test_settings": {
    "parallel_workers": 8  // 设置为CPU核心数
  }
}
```

### 2. 限制深度学习方法

深度学习方法耗时较长，可以限制测试：

```json
{
  "prediction_methods": {
    "deep_learning": []  // 清空列表跳过深度学习方法
  }
}
```

### 3. 调整超时时间

如果经常出现超时，可以增加时间：

```json
{
  "test_settings": {
    "timeout_seconds": 120  // 增加到2分钟
  }
}
```

## 故障排除

### 常见问题

#### 1. "无法连接预测系统"

**原因**: dlt_main.py无法正常运行
**解决**: 
- 检查Python环境和依赖
- 确认dlt_main.py路径正确
- 手动运行 `python3 dlt_main.py --help` 验证

#### 2. "数据文件错误"

**原因**: 开奖数据文件不存在或格式错误
**解决**:
- 确认 `data/dlt_data_all.csv` 文件存在
- 检查文件格式是否正确
- 运行 `python3 dlt_main.py data check` 验证数据

#### 3. "无法解析预测结果"

**原因**: CLI输出格式变化
**解决**:
- 手动运行一次预测命令查看输出格式
- 如有需要，修改 `predictor_caller.py` 中的解析逻辑

#### 4. 测试运行缓慢

**原因**: 参数设置过大或方法复杂
**解决**:
- 减少并行线程数
- 跳过深度学习方法
- 使用更小的参数范围

### 调试模式

如需调试，可以修改日志级别：

```json
{
  "output_settings": {
    "log_level": "DEBUG"
  }
}
```

## 注意事项

1. **测试时间**: 全面测试可能需要数小时，建议在空闲时运行
2. **资源占用**: 并行测试会占用较多CPU和内存资源
3. **存储空间**: 大量测试会产生较多结果文件，注意磁盘空间
4. **中断恢复**: 使用Ctrl+C可以安全中断测试，已完成的结果会被保存
5. **仅供研究**: 测试结果仅用于学术研究，不建议用于实际投注

## 技术支持

如果遇到问题，可以：

1. 检查配置文件是否正确
2. 运行系统检查命令
3. 查看错误日志
4. 检查相关文档

---

**祝您测试顺利，找到最佳的预测方法！**