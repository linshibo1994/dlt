# 🎯 大乐透智能预测系统

[![python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 🚀 **项目简介**

大乐透智能预测系统是一个**企业级AI预测平台**，基于2748期真实历史数据，集成25+种完整算法，支持深度学习、传统统计、概率模型和自适应学习等多种预测方法。

### ✨ **核心特性**
- **🧠 25+种完整算法**：所有算法均为完整数学模型实现，无简化版本
- **📊 真实数据驱动**：基于2748期真实大乐透历史数据
- **🔧 灵活参数配置**：支持自定义分析期数（50-2748期）和生成注数（1-100注）
- **🎲 多种投注方式**：单式、复式、胆拖等7种投注方式
- **🚀 智能化系统**：自适应学习、GPU加速、异常检测、自动重训练

## 🧠 **算法分类**

### 🔢 **传统统计算法**
- **频率分析**：概率分布建模、置信区间计算、趋势分析
- **冷热号分析**：温度量化计算、动态阈值调整、稳定性分析
- **遗漏值分析**：回补概率模型、期望回补时间、紧迫度评分
- **贝叶斯分析**：完整贝叶斯推理、多维似然函数、信息增益计算

### 🔗 **马尔可夫链算法**
- **1-3阶马尔可夫链**：完整状态转移矩阵、真实序列生成
- **自适应马尔可夫**：动态阶数选择、智能权重分配

### 🧠 **深度学习算法**
- **LSTM时序预测**：完整TensorFlow实现、双向LSTM、注意力机制
- **Transformer注意力**：真实多头注意力、位置编码、稀疏注意力
- **GAN生成对抗**：真实生成器判别器、对抗训练、条件生成
- **集成深度学习**：智能模型融合、权重自适应、性能监控

### 🎯 **智能预测算法**
- **自适应预测**：基于多臂老虎机算法的智能预测器选择
- **超级预测**：多种算法智能融合的超级预测系统
- **9种数学模型**：统计学、概率论、决策树等综合分析

### 🎲 **复式投注算法**
- **标准复式**：指定前区和后区号码数量的复式投注
- **胆拖投注**：胆码+拖码的智能胆拖投注
- **高级复式**：多算法融合的高级复式投注

## 🛠️ **技术栈**

| 组件 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **核心语言** | Python | 3.8+ | 主要开发语言 |
| **深度学习** | TensorFlow | 2.8+ | 神经网络框架 |
| **数据处理** | Pandas, NumPy | 1.3+, 1.19+ | 数据科学栈 |
| **机器学习** | Scikit-learn | 1.0+ | 传统ML算法 |
| **硬件加速** | CUDA, Metal | 可选 | GPU加速 |

## 🛠️ **安装指南**

### 📋 **系统要求**
- **Python**: 3.8+ (推荐3.10+)
- **操作系统**: Windows 10+、macOS 10.15+、Linux (Ubuntu 18.04+)
- **可选**: TensorFlow 2.8+ (深度学习功能)

### ⚡ **快速安装**

```bash
# 1. 克隆项目
git clone https://github.com/linshibo1994/dlt.git
cd dlt

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装深度学习支持（可选）
pip install tensorflow

# 4. 验证安装
python3 dlt_main.py data status
python3 dlt_main.py predict -m frequency -c 1
```

## 📖 **使用指南**

### 🎯 **基本语法**

```bash
python3 dlt_main.py predict -m <方法名> -p <期数> -c <注数> [其他参数]
```

**核心参数**：
- `-m, --method`: 预测方法（必需）
- `-p, --periods`: 分析期数（50-2748，默认500）
- `-c, --count`: 生成注数（1-100，默认1）
- `--save`: 保存结果到文件
- `--format`: 输出格式（txt/json/csv）

### 💡 **快速开始**

```bash
# 1. 最简单的预测
python3 dlt_main.py predict -m frequency

# 2. 指定期数和注数
python3 dlt_main.py predict -m lstm -p 1000 -c 3

# 3. 复式投注
python3 dlt_main.py predict -m compound --front-count 8 --back-count 4

# 4. 保存结果
python3 dlt_main.py predict -m ensemble -c 5 --save --format json
```

### 📊 **预测方法详解**

#### 🔢 **传统统计方法**

| 方法 | 命令 | 说明 | 推荐参数 |
|------|------|------|----------|
| **频率分析** | `frequency` | 基于历史频率的概率预测 | `-p 1000 -c 5` |
| **冷热号分析** | `hot_cold` | 识别冷热号，预测回补趋势 | `-p 800 -c 3` |
| **遗漏值分析** | `missing` | 分析遗漏周期，预测回补 | `-p 1200 -c 2` |
| **贝叶斯分析** | `bayesian` | 贝叶斯推理概率预测 | `-p 1000 -c 3` |

```bash
# 示例：频率分析预测
python3 dlt_main.py predict -m frequency -p 1000 -c 5
```

#### 🔗 **马尔可夫链方法**

| 方法 | 命令 | 说明 | 推荐参数 |
|------|------|------|----------|
| **1阶马尔可夫** | `markov` | 基于前一期状态转移 | `-p 1500 -c 5` |
| **2阶马尔可夫** | `markov_2nd` | 考虑前两期状态 | `-p 1500 -c 3` |
| **3阶马尔可夫** | `markov_3rd` | 考虑前三期状态 | `-p 2000 -c 3` |
| **自适应马尔可夫** | `adaptive_markov` | 动态选择最优阶数 | `-p 1800 -c 4` |

```bash
# 示例：自适应马尔可夫预测
python3 dlt_main.py predict -m adaptive_markov -p 1800 -c 4
```

#### 🧠 **深度学习方法**

| 方法 | 命令 | 说明 | 推荐参数 |
|------|------|------|----------|
| **LSTM预测** | `lstm` | 长短期记忆网络时序预测 | `-p 1000 -c 3` |
| **Transformer** | `transformer` | 多头注意力机制预测 | `-p 1500 -c 2` |
| **GAN预测** | `gan` | 生成对抗网络预测 | `-p 800 -c 5` |
| **集成学习** | `ensemble` | 多模型智能融合 | `-p 2000 -c 3` |

```bash
# 示例：LSTM深度学习预测
python3 dlt_main.py predict -m lstm -p 1000 -c 3
```

#### 🎯 **智能预测方法**

| 方法 | 命令 | 说明 | 推荐参数 |
|------|------|------|----------|
| **超级预测** | `super` | 多算法智能融合系统 | `-p 1000 -c 3` |
| **自适应预测** | `adaptive` | 智能预测器选择 | `-p 1000 -c 5` |
| **9种数学模型** | `nine_models` | 综合数学分析 | `-p 800 -c 2` |
| **高级集成** | `advanced_integration` | 多维度权重计算 | `-p 1500 -c 4` |

```bash
# 示例：超级预测
python3 dlt_main.py predict -m super -p 1000 -c 3
```

#### 🎲 **复式投注方法**

| 方法 | 命令 | 说明 | 推荐参数 |
|------|------|------|----------|
| **标准复式** | `compound` | 指定前后区号码数量 | `--front-count 8 --back-count 3` |
| **胆拖投注** | `duplex` | 胆码+拖码投注 | `--front-dan 2 --back-dan 1` |
| **马尔可夫复式** | `markov_compound` | 基于马尔可夫链 | `--front-count 9 --back-count 4` |

```bash
# 示例：标准复式投注
python3 dlt_main.py predict -m compound -p 1000 --front-count 8 --back-count 3

# 示例：胆拖投注
python3 dlt_main.py predict -m duplex -p 800 --front-dan 2 --back-dan 1
```

## 📊 **详细预测方法说明**

### 🔢 **传统统计预测详解**

#### 频率分析预测
基于历史出现频率的概率统计预测，适合寻找高频号码。
```bash
python3 dlt_main.py predict -m frequency -p 1000 -c 5    # 分析1000期，预测5注
python3 dlt_main.py predict -m frequency -p 500 -c 3     # 分析500期，预测3注
python3 dlt_main.py predict -m frequency --save --format json  # 保存为JSON格式
```

#### 冷热号分析预测
识别热号和冷号，预测冷号回补趋势，适合平衡投注。
```bash
python3 dlt_main.py predict -m hot_cold -p 800 -c 3      # 分析800期，预测3注
python3 dlt_main.py predict -m hot_cold -p 1200 -c 2     # 长期分析，预测2注
```

#### 遗漏值分析预测
分析号码遗漏周期，预测回补概率，适合长期跟踪。
```bash
python3 dlt_main.py predict -m missing -p 1200 -c 2      # 分析1200期，预测2注
python3 dlt_main.py predict -m missing -p 800 -c 5       # 分析800期，预测5注
```

#### 贝叶斯分析预测
基于贝叶斯定理的概率推理预测，适合概率分析。
```bash
python3 dlt_main.py predict -m bayesian -p 1000 -c 3     # 分析1000期，预测3注
python3 dlt_main.py predict -m bayesian -p 1500 -c 2     # 长期分析，预测2注
```

### 🔗 **马尔可夫链预测详解**

#### 1阶马尔可夫链预测
基于前一期状态的转移概率预测，适合短期趋势分析。
```bash
python3 dlt_main.py predict -m markov -p 1500 -c 5       # 分析1500期，预测5注
python3 dlt_main.py predict -m markov -p 800 -c 3        # 短期分析，预测3注
```

#### 2阶马尔可夫链预测
考虑前两期状态的复合转移预测，适合中期模式识别。
```bash
python3 dlt_main.py predict -m markov_2nd -p 1500 -c 5   # 分析1500期，预测5注
python3 dlt_main.py predict -m markov_2nd -p 1000 -c 3   # 中期分析，预测3注
```

#### 3阶马尔可夫链预测
基于前三期状态的高阶依赖预测，适合长期模式分析。
```bash
python3 dlt_main.py predict -m markov_3rd -p 2000 -c 3   # 分析2000期，预测3注
python3 dlt_main.py predict -m markov_3rd -p 1500 -c 2   # 长期分析，预测2注
```

#### 自适应马尔可夫预测
动态选择最优阶数的智能融合预测，适合综合分析。
```bash
python3 dlt_main.py predict -m adaptive_markov -p 1800 -c 4  # 分析1800期，预测4注
python3 dlt_main.py predict -m adaptive_markov -p 1200 -c 3  # 自适应分析，预测3注
```

#### 马尔可夫自定义预测
支持自定义分析期数和预测期数的马尔可夫链预测。
```bash
python3 dlt_main.py predict -m markov_custom -p 1000 -c 2    # 自定义马尔可夫预测
```

### 🧠 **深度学习预测详解**

#### LSTM时序预测
长短期记忆网络，专门处理时序数据，擅长捕捉长期依赖关系。
```bash
python3 dlt_main.py predict -m lstm -p 1000 -c 3         # 分析1000期，LSTM预测3注
python3 dlt_main.py predict -m lstm -p 1500 -c 2         # 长期训练，预测2注
python3 dlt_main.py predict -m lstm -p 800 -c 5          # 短期训练，预测5注
```

#### Transformer注意力预测
多头注意力机制，捕捉长距离依赖，适合复杂模式识别。
```bash
python3 dlt_main.py predict -m transformer -p 1500 -c 2  # 分析1500期，Transformer预测2注
python3 dlt_main.py predict -m transformer -p 1000 -c 3  # 注意力分析，预测3注
```

#### GAN生成对抗预测
生成对抗网络，创新号码组合生成，适合探索新的号码模式。
```bash
python3 dlt_main.py predict -m gan -p 800 -c 5           # 分析800期，GAN预测5注
python3 dlt_main.py predict -m gan -p 1200 -c 3          # 生成对抗，预测3注
```

#### 集成深度学习预测
融合LSTM+Transformer+GAN的智能预测，综合多种深度学习优势。
```bash
python3 dlt_main.py predict -m ensemble -p 2000 -c 3     # 分析2000期，集成预测3注
python3 dlt_main.py predict -m ensemble -p 1500 -c 5     # 深度集成，预测5注
```

### 🧠 **智能预测算法详解**

#### 超级预测
多种算法智能融合的超级预测系统，综合传统和高级算法优势。
```bash
python3 dlt_main.py predict -m super -p 1000 -c 3        # 分析1000期，超级预测3注
python3 dlt_main.py predict -m super -p 1500 -c 5        # 超级融合，预测5注
```

#### 自适应预测
基于多臂老虎机算法的智能预测器选择，动态优化预测策略。
```bash
python3 dlt_main.py predict -m adaptive -p 1000 -c 5     # 分析1000期，自适应预测5注
python3 dlt_main.py predict -m adaptive -p 1200 -c 3     # 智能选择，预测3注
```

#### 9种数学模型预测
统计学、概率论、决策树等9种数学模型的综合分析预测。
```bash
python3 dlt_main.py predict -m nine_models -p 800 -c 2   # 分析800期，9种模型预测2注
python3 dlt_main.py predict -m nine_models -p 1000 -c 4  # 数学建模，预测4注
```

#### 高级集成分析预测
多维度权重计算和智能评分系统的高级集成预测。
```bash
python3 dlt_main.py predict -m advanced_integration -p 1500 -c 4  # 分析1500期，高级集成预测4注
python3 dlt_main.py predict -m advanced_integration -p 1000 -c 3 --integration-type comprehensive  # 综合集成
```

#### 混合策略预测
支持保守、激进、平衡三种策略的混合预测方法。
```bash
python3 dlt_main.py predict -m mixed_strategy -p 1500 -c 3 --strategy balanced  # 平衡策略预测3注
python3 dlt_main.py predict -m mixed_strategy -p 1000 -c 2 --strategy conservative  # 保守策略
python3 dlt_main.py predict -m mixed_strategy -p 800 -c 5 --strategy aggressive   # 激进策略
```

### 🎲 **复式投注预测详解**

#### 标准复式投注
指定前区和后区号码数量的复式投注，适合大额投注。
```bash
python3 dlt_main.py predict -m compound -p 1000 --front-count 8 --back-count 3   # 8+3复式
python3 dlt_main.py predict -m compound -p 1500 --front-count 12 --back-count 5  # 12+5大复式
python3 dlt_main.py predict -m compound -p 800 --front-count 6 --back-count 3    # 6+3小复式
```

#### 胆拖投注
胆码+拖码的智能胆拖投注，降低投注成本。
```bash
python3 dlt_main.py predict -m duplex -p 800 --front-dan 2 --back-dan 1          # 胆拖投注
python3 dlt_main.py predict -m duplex -p 1000 --front-dan 3 --back-dan 1 --front-tuo 8 --back-tuo 5  # 详细胆拖
```

#### 9种模型复式投注
基于9种数学模型的复式投注预测。
```bash
python3 dlt_main.py predict -m nine_models_compound -p 1000 --front-count 10 --back-count 4  # 9种模型复式
```

#### 马尔可夫链复式
基于马尔可夫链的复式投注，结合概率模型优势。
```bash
python3 dlt_main.py predict -m markov_compound -p 1000 --front-count 9 --back-count 4  # 马尔可夫复式
python3 dlt_main.py predict -m markov_compound -p 1500 --front-count 12 --back-count 5 # 大复式
```

#### 高度集成复式
多算法融合的高级复式投注，综合多种预测优势。
```bash
python3 dlt_main.py predict -m highly_integrated -p 1000 --front-count 12 --back-count 5 --integration-level ultimate
python3 dlt_main.py predict -m highly_integrated -p 1500 --front-count 10 --back-count 4 --integration-level high
```

### 🤖 **智能学习预测详解**

#### 智能复式预测
基于学习结果的智能复式预测，动态优化投注策略。
```bash
python3 dlt_main.py smart -p 1200 --compound --front-count 10 --back-count 4  # 智能复式预测
python3 dlt_main.py smart -p 1000 --compound --front-count 8 --back-count 3   # 智能复式
```

#### 智能胆拖预测
基于学习结果的智能胆拖预测，优化胆码选择。
```bash
python3 dlt_main.py smart -p 800 --duplex --front-dan 3 --back-dan 1 --front-tuo 8 --back-tuo 5  # 智能胆拖预测
python3 dlt_main.py smart -p 1000 --duplex --front-dan 2 --back-dan 1 --front-tuo 6 --back-tuo 4 # 智能胆拖
```

### 🚀 **高级集成预测详解**

#### Stacking集成预测
使用Stacking算法的高级集成预测。
```bash
python3 dlt_main.py predict -m stacking -p 1500 -c 3     # Stacking集成预测3注
```

#### 自适应集成预测
动态调整权重的自适应集成预测。
```bash
python3 dlt_main.py predict -m adaptive_ensemble -p 1000 -c 5  # 自适应集成预测5注
```

#### 终极集成预测
融合所有算法的终极集成预测系统。
```bash
python3 dlt_main.py predict -m ultimate_ensemble -p 2000 -c 3  # 终极集成预测3注
```

#### 增强预测
使用增强系统的自动预测功能。
```bash
python3 dlt_main.py predict -m enhanced -p 1000 -c 3     # 增强预测3注
```

## 📊 **数据管理功能详解**

#### 数据状态查看
查看当前数据状态，包括数据量、最新期数、数据完整性等。
```bash
python3 dlt_main.py data status                          # 查看数据状态
```

#### 数据更新
更新最新开奖数据，支持增量更新和全量更新。
```bash
python3 dlt_main.py data update                          # 全量更新数据
python3 dlt_main.py data update --incremental            # 增量更新数据
```

#### 最新开奖查询
获取最新开奖结果并与预测结果进行比较。
```bash
python3 dlt_main.py data latest                          # 获取最新开奖
python3 dlt_main.py data latest --compare                # 获取最新开奖并比较预测
```

#### 数据质量检查
检查数据质量，包括异常检测、数据完整性验证等。
```bash
python3 dlt_main.py data check                           # 基础数据检查
python3 dlt_main.py data check --anomaly                 # 异常检测分析
```

## 🔍 **高级分析功能详解**

#### 基础统计分析
提供频率、遗漏、冷热号等基础统计分析。
```bash
python3 dlt_main.py analyze -t basic -p 500              # 基础分析（500期）
python3 dlt_main.py analyze -t basic -p 1000             # 基础分析（1000期）
```

#### 高级模式分析
深度模式分析，包括马尔可夫链、贝叶斯等高级分析。
```bash
python3 dlt_main.py analyze -t advanced -p 1000          # 高级分析（1000期）
python3 dlt_main.py analyze -t advanced -p 1500          # 高级分析（1500期）
```

#### 综合分析
9种数学模型的综合分析，提供全面的数据洞察。
```bash
python3 dlt_main.py analyze -t comprehensive -p 800      # 综合分析（800期）
python3 dlt_main.py analyze -t comprehensive -p 1200     # 综合分析（1200期）
```

#### 异常检测分析
检测数据中的异常模式和趋势变化。
```bash
python3 dlt_main.py analyze -t anomaly                   # 异常检测分析
python3 dlt_main.py analyze -t anomaly -p 1000           # 指定期数异常检测
```

## 🎓 **自适应学习功能详解**

#### UCB1算法学习
使用UCB1（Upper Confidence Bound）算法进行预测器选择学习。
```bash
python3 dlt_main.py learn --algorithm ucb1 -t 1000       # UCB1学习1000轮
python3 dlt_main.py learn --algorithm ucb1 -t 500        # UCB1学习500轮
```

#### Thompson采样学习
使用Thompson采样算法进行贝叶斯优化学习。
```bash
python3 dlt_main.py learn --algorithm thompson_sampling -t 1000  # Thompson采样学习1000轮
python3 dlt_main.py learn --algorithm thompson_sampling -t 800   # Thompson采样学习800轮
```

#### Epsilon贪婪学习
使用Epsilon贪婪算法平衡探索和利用。
```bash
python3 dlt_main.py learn --algorithm epsilon_greedy -t 1000     # Epsilon贪婪学习1000轮
python3 dlt_main.py learn --algorithm epsilon_greedy -t 600      # Epsilon贪婪学习600轮
```

## 🚀 **性能优化功能详解**

#### GPU加速优化
启用GPU加速，提升深度学习模型训练和预测速度。
```bash
python3 dlt_main.py optimize -t gpu                      # GPU加速优化
python3 dlt_main.py optimize -t gpu --device cuda        # 指定CUDA设备
```

#### 内存优化
优化内存使用，适合大数据量处理。
```bash
python3 dlt_main.py optimize -t memory                   # 内存优化
python3 dlt_main.py optimize -t memory --cache-size 1000 # 指定缓存大小
```

#### 批处理优化
启用批处理模式，提升大量预测任务的处理效率。
```bash
python3 dlt_main.py optimize -t batch                    # 批处理优化
python3 dlt_main.py optimize -t batch --batch-size 50    # 指定批处理大小
```

## 📈 **回测和验证功能详解**

#### 性能回测
对预测算法进行历史回测，评估预测准确性和稳定性。
```bash
python3 dlt_main.py backtest -m ensemble -t 100          # 集成算法回测100期
python3 dlt_main.py backtest -m lstm -t 50               # LSTM算法回测50期
python3 dlt_main.py backtest -m frequency -t 200         # 频率分析回测200期
python3 dlt_main.py backtest -m markov -t 150            # 马尔可夫链回测150期
```

#### 算法比较
比较不同算法的预测性能，选择最优算法。
```bash
python3 dlt_main.py compare -m frequency,markov,lstm -t 100  # 比较三种算法性能
python3 dlt_main.py compare -m ensemble,super,adaptive -t 80 # 比较高级算法性能
```

## 🚀 **增强功能模块详解**

#### 增强系统信息
查看增强功能模块的系统信息和兼容性。
```bash
python3 dlt_main.py enhanced info                        # 查看增强系统信息
```

#### 系统兼容性测试
运行系统兼容性测试，确保所有功能正常运行。
```bash
python3 dlt_main.py enhanced test                        # 运行兼容性测试
```

#### 增强预测功能
使用增强功能模块进行高级预测。
```bash
python3 dlt_main.py enhanced predict -d "sample_data" -m auto  # 增强预测功能
python3 dlt_main.py enhanced predict -d "predict_5_numbers" -m lstm  # 指定LSTM模型
```

#### 增强可视化功能
生成交互式可视化图表和分析报告。
```bash
python3 dlt_main.py enhanced visualize -d "sample_data" -t interactive  # 交互式可视化
python3 dlt_main.py enhanced visualize -d "analysis_data" -t static     # 静态图表
```

## 💡 **完整使用示例**

### 🎯 **快速开始示例**
```bash
# 1. 最简单的预测（使用默认参数）
python3 dlt_main.py predict -m ensemble

# 2. 指定期数和注数的预测
python3 dlt_main.py predict -m lstm -p 1000 -c 3

# 3. 复式投注预测
python3 dlt_main.py predict -m compound -p 800 --front-count 8 --back-count 4

# 4. 智能学习预测
python3 dlt_main.py smart -p 1000 --compound --front-count 10 --back-count 5

# 5. 数据管理
python3 dlt_main.py data status
python3 dlt_main.py data update --incremental

# 6. 高级分析
python3 dlt_main.py analyze -t comprehensive -p 1000

# 7. 性能回测
python3 dlt_main.py backtest -m ensemble -t 100
```

### 🔥 **高级使用示例**
```bash
# 深度学习组合预测
python3 dlt_main.py predict -m lstm -p 1500 -c 2 --save --format json
python3 dlt_main.py predict -m transformer -p 1000 -c 3 --save
python3 dlt_main.py predict -m gan -p 800 -c 5

# 马尔可夫链系列预测
python3 dlt_main.py predict -m markov -p 1200 -c 3
python3 dlt_main.py predict -m markov_2nd -p 1500 -c 2
python3 dlt_main.py predict -m adaptive_markov -p 1800 -c 4

# 复式投注组合
python3 dlt_main.py predict -m compound --front-count 12 --back-count 5
python3 dlt_main.py predict -m duplex --front-dan 3 --back-dan 1 --front-tuo 8
python3 dlt_main.py predict -m highly_integrated --front-count 10 --back-count 4

# 学习和优化
python3 dlt_main.py learn --algorithm ucb1 -t 1000
python3 dlt_main.py optimize -t gpu
python3 dlt_main.py backtest -m super -t 200
```

### 💡 **常用示例**

```bash
# 基础预测
python3 dlt_main.py predict -m frequency -c 3           # 频率分析预测3注
python3 dlt_main.py predict -m lstm -p 1000 -c 2        # LSTM预测2注

# 复式投注
python3 dlt_main.py predict -m compound --front-count 8 --back-count 3

# 数据管理
python3 dlt_main.py data status                         # 查看数据状态
python3 dlt_main.py data update                         # 更新数据

# 分析功能
python3 dlt_main.py analyze -t basic -p 500             # 基础分析
python3 dlt_main.py backtest -m ensemble -t 100         # 性能回测
```

## 📋 **参数说明**

### 🎯 **核心参数**
| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `-m, --method` | string | 见方法表格 | ensemble | 预测方法 |
| `-p, --periods` | int | 50-2748 | 500 | 分析期数 |
| `-c, --count` | int | 1-100 | 1 | 生成注数 |
| `--save` | flag | - | False | 保存结果 |
| `--format` | string | txt/json/csv | txt | 输出格式 |

### 🎲 **复式投注参数**
| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--front-count` | int | 6-15 | 8 | 前区号码数量 |
| `--back-count` | int | 3-12 | 4 | 后区号码数量 |
| `--front-dan` | int | 1-5 | 2 | 前区胆码数量 |
| `--back-dan` | int | 1-2 | 1 | 后区胆码数量 |

## 🌟 **项目特色**

### ✅ **算法完整性**
- **🔬 完整数学模型**：所有算法均为完整实现，无简化版本
- **🧮 真实神经网络**：使用真实TensorFlow深度学习框架
- **📊 完整贝叶斯推理**：包含完整的贝叶斯统计过程
- **🔗 真实马尔可夫链**：实现真正的状态序列生成

### 🚀 **系统优势**
- **🎯 25+种算法**：涵盖传统统计到深度学习的完整算法库
- **📊 真实数据**：基于2748期真实历史数据
- **🔧 灵活配置**：支持自定义参数和多种投注方式
- **🧠 智能学习**：自适应算法选择和参数优化
- **⚡ 高性能**：支持GPU加速和跨平台运行

## � **联系方式**

- **GitHub**: [https://github.com/linshibo1994/dlt](https://github.com/linshibo1994/dlt)
- **问题反馈**: 通过GitHub Issues提交

## 📄 **许可证**

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## ⚠️ **免责声明**

本系统仅供学习和研究使用，不构成任何投资建议。彩票投注有风险，请理性参与。

---

**🏆 项目状态**: ✅ 生产就绪 | 🧠 AI驱动 | 📊 数据驱动 | 🚀 高性能优化