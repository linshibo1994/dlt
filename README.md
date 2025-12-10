# 🎯 大乐透智能预测系统

[![python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 📖 项目简介

大乐透智能预测系统是一个基于Python的AI预测平台，集成26+种算法，使用2756期真实历史数据进行大乐透号码预测。系统支持命令行和图形界面两种使用方式，提供从基础统计到深度学习的完整预测方案。

### ✨ 核心特色
- **🧠 26+种算法** - 涵盖传统统计、马尔可夫链、深度学习、集成学习等
- **📊 真实数据** - 基于2756期历史开奖数据，支持增量更新
- **🔧 灵活配置** - 支持自定义分析期数（50-2756期）和生成注数（1-100注）
- **🎲 多种投注** - 单式、复式、胆拖等投注模式，智能成本控制
- **🚀 智能系统** - 自适应学习、GPU/CPU自动加速、智能缓存
- **⚡ 硬件加速** - 支持GPU加速和多线程并行处理
- **🎯 批量对比** - 批量预测验证，统计分析，Excel报告导出

## 🧠 预测算法体系

### 📊 传统统计算法
- **frequency** - 频率分析：基于历史出现频率的概率分布建模
- **hot_cold** - 冷热号分析：动态温度量化计算和阈值调整
- **missing** - 遗漏值分析：基于回补概率的期望时间模型
- **bayesian** - 贝叶斯分析：完整贝叶斯推理和似然函数计算

### 🔗 马尔可夫链算法
- **markov** - 1阶马尔可夫链：状态转移矩阵和序列生成
- **markov_2nd** - 2阶马尔可夫链：联合状态概率计算
- **markov_3rd** - 3阶马尔可夫链：长期依赖性建模
- **adaptive_markov** - 自适应马尔可夫：动态阶数选择

### 🧠 深度学习算法
- **lstm** - LSTM时序预测：双向LSTM网络和注意力机制
- **transformer** - Transformer预测：多头注意力和位置编码
- **gan** - GAN生成对抗：生成器判别器对抗训练
- **ensemble** - 集成深度学习：多模型智能融合

### 📊 聚类算法
- **clustering** - 聚类预测：基于K-means聚类的智能预测

### ⚡ 增强算法
- **enhanced** - 增强预测：集成多种增强算法的高级预测系统

### 🎯 智能预测算法
- **super** - 超级预测：多算法智能融合系统
- **adaptive** - 自适应预测：基于多臂老虎机的预测器选择
- **nine_models** - 九种数学模型：统计学、概率论综合分析
- **advanced_integration** - 高级集成分析：多维度权重计算
- **mixed_strategy** - 混合策略：保守/激进/平衡策略选择
- **highly_integrated** - 高度集成：全算法融合系统

### 🎲 复式投注算法
- **compound** - 标准复式：指定前区和后区号码数量
- **duplex** - 胆拖投注：胆码+拖码智能投注
- **markov_compound** - 马尔可夫复式：基于马尔可夫链的复式
- **nine_models_compound** - 九模型复式：多算法融合复式

### 🔄 集成学习算法
- **stacking** - Stacking集成：基于元学习器的高级集成
- **adaptive_ensemble** - 自适应集成：动态权重调整
- **ultimate_ensemble** - 终极集成：最高级别集成预测

## 🛠️ 系统架构

### 🏗️ 前后端分离架构 (v2.0)

本系统采用现代前后端分离架构：

```
┌─────────────────────────────────────────────────────────────┐
│                         前端 (Vue 3)                         │
│   ┌─────────┬─────────┬─────────┬─────────┬─────────┐      │
│   │Dashboard│Prediction│ Analysis│ Compare │Settings │      │
│   └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘      │
│        └─────────┴─────────┴─────────┴─────────┘            │
│                         ↓ Axios                              │
│   ┌─────────────────────────────────────────────────┐       │
│   │               Nginx (端口 80)                    │       │
│   │     静态文件服务 + API 反向代理                  │       │
│   └────────────────────────┬────────────────────────┘       │
└────────────────────────────┼────────────────────────────────┘
                             │ /api/*
┌────────────────────────────┴────────────────────────────────┐
│                     后端 (FastAPI)                           │
│   ┌──────────────────────────────────────────────────┐      │
│   │           REST API (端口 8000)                    │      │
│   │  /api/data/* | /api/predict | /api/analysis/*    │      │
│   └────────────────────────┬─────────────────────────┘      │
│                            ↓                                 │
│   ┌──────────────────────────────────────────────────┐      │
│   │               核心预测引擎                        │      │
│   │  26+ 算法 | 数据管理 | 缓存系统 | 分析模块       │      │
│   └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**技术栈：**
| 层级 | 技术 | 说明 |
|------|------|------|
| 前端框架 | Vue 3 + TypeScript | 响应式 UI 框架 |
| 状态管理 | Pinia | Vue 官方状态管理 |
| UI 组件 | Naive UI | 企业级 Vue 组件库 |
| 图表 | ECharts | 数据可视化 |
| 样式 | Tailwind CSS | 原子化 CSS |
| 构建工具 | Vite | 快速构建 |
| 后端框架 | FastAPI | 高性能 Python API |
| 数据验证 | Pydantic | 类型安全 |
| Web 服务器 | Nginx | 反向代理 |

### 📊 核心模块
- **backend/api/server.py** - FastAPI REST API 服务器
- **backend/app/core/** - 核心模块（数据/缓存/日志/任务管理）
- **backend/app/predictors/** - 预测算法模块（26+种算法实现）
- **backend/app/analyzers/** - 数据分析模块（统计分析和异常检测）
- **frontend/src/views/** - Vue 页面组件
- **frontend/src/stores/** - Pinia 状态管理
- **frontend/src/api/** - API 接口层

### 💾 智能缓存系统
- **版本控制** - 基于数据内容的版本签名，自动失效机制
- **期数隔离** - 不同期数的分析结果独立缓存
- **双层缓存** - 内存缓存(LRU) + 文件缓存
- **自动管理** - 智能过期检查和LRU淘汰机制

### ⚡ 硬件加速系统
- **智能检测** - 自动检测CPU核心、内存、GPU、CUDA版本
- **多级策略** - GPU优先 → CPU多线程 → CPU单线程
- **优雅降级** - GPU不可用时自动降级，确保稳定运行
- **性能基准** - 实时评估硬件性能，推荐最优配置

## 🎯 大乐透中奖规则 (2019年新规则)

系统采用大乐透2019年新规则，共9个奖级：

| 奖级 | 中奖条件 | 说明 |
|-----|---------|------|
| **一等奖** | 5+2 | 前区5个号码全中 + 后区2个号码全中 |
| **二等奖** | 5+1 | 前区5个号码全中 + 后区2个号码中1个 |
| **三等奖** | 5+0 | 前区5个号码全中 + 后区2个号码全不中 |
| **四等奖** | 4+2 | 前区5个号码中4个 + 后区2个号码全中 |
| **五等奖** | 4+1 | 前区5个号码中4个 + 后区2个号码中1个 |
| **六等奖** | 4+0 或 3+2 | 前区中4个后区不中 或 前区中3个后区全中 |
| **七等奖** | 3+1 | 前区5个号码中3个 + 后区2个号码中1个 |
| **八等奖** | 3+0 或 2+2 | 前区中3个后区不中 或 前区中2个后区全中 |
| **九等奖** | 2+1 或 1+2 或 0+2 | 多种组合的最低中奖等级 |

### 🎲 玩法说明
- **前区**：从01-35号码中选择5个不重复号码
- **后区**：从01-12号码中选择2个不重复号码  
- **投注方式**：支持单式、复式、胆拖等多种投注方式
- **中奖判断**：系统自动对比预测号码与开奖号码，精确判断中奖等级

## 🛠️ 安装与配置

### 📋 系统要求
- **Python** 3.8+ (推荐3.10+)
- **操作系统** Windows 10+、macOS 10.15+、Linux (Ubuntu 18.04+)
- **可选** TensorFlow 2.8+ (深度学习功能)

### ⚡ 快速安装
```bash
# 1. 克隆项目
git clone https://github.com/linshibo1994/dlt.git
cd dlt

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装GUI依赖（推荐）
pip install -r requirements_gui.txt

# 4. 验证安装
python3 dlt_main.py data status
```

### 🔧 技术栈
| 组件 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **核心语言** | Python | 3.8+ | 主要开发语言 |
| **深度学习** | TensorFlow | 2.8+ | 神经网络框架(可选) |
| **数据处理** | Pandas, NumPy | 1.5+, 1.24+ | 数据科学栈 |
| **机器学习** | Scikit-learn | 1.3+ | 传统ML算法 |
| **图形界面** | Streamlit | 1.28+ | Web界面框架 |
| **数据可视化** | Plotly | 5.15+ | 交互式图表 |

## 📖 使用指南

### 🐳 Docker 部署（推荐）

最简单的部署方式，一键启动前后端服务：

```bash
# 1. 克隆项目
git clone https://github.com/linshibo1994/dlt.git
cd dlt

# 2. 启动服务
./deploy.sh start

# 3. 访问应用
# 前端界面: http://localhost
# 后端API: http://localhost:8000/api
# API文档: http://localhost:8000/docs
```

**部署脚本命令：**
```bash
./deploy.sh start    # 启动所有服务
./deploy.sh stop     # 停止所有服务
./deploy.sh restart  # 重启所有服务
./deploy.sh logs     # 查看服务日志
./deploy.sh build    # 重新构建镜像
./deploy.sh status   # 查看服务状态
```

### 🖥️ 本地开发

**前后端分离开发：**
```bash
# 终端 1：启动后端 API
cd dlt
pip install -r backend/requirements.txt
python -m uvicorn backend.api.server:app --reload --port 8000

# 终端 2：启动前端开发服务器
cd dlt/frontend
npm install
npm run dev
# 访问 http://localhost:3000
```

**传统 Streamlit GUI（可选）：**
```bash
# 启动GUI界面
./start_gui.sh      # Linux/macOS
start_gui.bat       # Windows
python3 run_gui.py  # 通用方式
```

浏览器打开 http://localhost:8501 使用完整的图形界面功能。

### 💻 命令行使用

#### 基础语法
```bash
python3 dlt_main.py predict -m <方法名> -p <期数> -c <注数>
```

#### 快速开始示例
```bash
# 基础预测
python3 dlt_main.py predict -m frequency -c 1
python3 dlt_main.py predict -m markov -c 3
python3 dlt_main.py predict -m bayesian -p 800 -c 2

# 复式投注
python3 dlt_main.py predict -m compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m duplex --front-dan 2 --back-dan 1

# 硬件加速
python3 dlt_main.py predict -m lstm --acceleration gpu
python3 dlt_main.py predict -m markov --acceleration cpu_multi

# 数据管理
python3 dlt_main.py data status
python3 dlt_main.py data update --incremental
python3 dlt_main.py data check --fix

# 分析功能
python3 dlt_main.py analyze --type comprehensive -p 1000
python3 dlt_main.py backtest -m ensemble -t 100
```

### 📊 预测方法列表

#### 基础统计方法 (支持CPU多线程加速 + 复式预测)
- **frequency** - 频率分析：基于历史出现频率的概率分布建模
```bash
# 基础预测
python3 dlt_main.py predict -m frequency -c 1
python3 dlt_main.py predict -m frequency -p 800 -c 3

# CPU多线程加速
python3 dlt_main.py predict -m frequency --acceleration cpu_multi -c 2
python3 dlt_main.py predict -m frequency --acceleration cpu_multi --cpu-threads 8 -c 3

# 复式预测
python3 dlt_main.py predict -m frequency --compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m frequency --compound --front-count 10 --back-count 5 -p 1000
```

- **hot_cold** - 冷热号分析：动态温度量化计算和阈值调整
```bash
# 基础预测
python3 dlt_main.py predict -m hot_cold -c 1
python3 dlt_main.py predict -m hot_cold -p 600 -c 2

# CPU多线程加速
python3 dlt_main.py predict -m hot_cold --acceleration cpu_multi -c 3
python3 dlt_main.py predict -m hot_cold --acceleration cpu_multi --cpu-threads 6 -c 2

# 复式预测
python3 dlt_main.py predict -m hot_cold --compound --front-count 9 --back-count 4
python3 dlt_main.py predict -m hot_cold --compound --front-count 12 --back-count 6 -p 800
```

- **missing** - 遗漏值分析：基于回补概率的期望时间模型
```bash
# 基础预测
python3 dlt_main.py predict -m missing -c 1
python3 dlt_main.py predict -m missing -p 1000 -c 2

# 自动加速
python3 dlt_main.py predict -m missing --acceleration auto -c 3
python3 dlt_main.py predict -m missing --acceleration cpu_multi -c 2

# 复式预测
python3 dlt_main.py predict -m missing --compound --front-count 8 --back-count 3
python3 dlt_main.py predict -m missing --compound --front-count 11 --back-count 5 -p 1200
```

- **bayesian** - 贝叶斯分析：完整贝叶斯推理和似然函数计算
```bash
# 基础预测
python3 dlt_main.py predict -m bayesian -c 1
python3 dlt_main.py predict -m bayesian -p 500 -c 3

# CPU多线程加速
python3 dlt_main.py predict -m bayesian --acceleration cpu_multi -c 2
python3 dlt_main.py predict -m bayesian --acceleration cpu_multi --cpu-threads 4 -c 3

# 复式预测
python3 dlt_main.py predict -m bayesian --compound --front-count 10 --back-count 4
python3 dlt_main.py predict -m bayesian --compound --front-count 13 --back-count 6 -p 600
```

#### 马尔可夫链方法 (支持CPU多线程加速 + 复式预测)
- **markov** - 1阶马尔可夫链：状态转移矩阵和序列生成
```bash
# 基础预测
python3 dlt_main.py predict -m markov -c 1
python3 dlt_main.py predict -m markov -p 800 -c 3

# CPU多线程加速
python3 dlt_main.py predict -m markov --acceleration cpu_multi -c 2
python3 dlt_main.py predict -m markov --acceleration cpu_multi --cpu-threads 8 -c 4

# 复式预测
python3 dlt_main.py predict -m markov --compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m markov --compound --front-count 12 --back-count 5 -p 1000
```

- **markov_2nd** - 2阶马尔可夫链：联合状态概率计算
```bash
# 基础预测
python3 dlt_main.py predict -m markov_2nd -c 1
python3 dlt_main.py predict -m markov_2nd -p 600 -c 2

# CPU多线程加速
python3 dlt_main.py predict -m markov_2nd --acceleration cpu_multi -c 3
python3 dlt_main.py predict -m markov_2nd --acceleration cpu_multi --cpu-threads 6 -c 2

# 复式预测
python3 dlt_main.py predict -m markov_2nd --compound --front-count 9 --back-count 4
python3 dlt_main.py predict -m markov_2nd --compound --front-count 11 --back-count 6 -p 800
```

- **markov_3rd** - 3阶马尔可夫链：长期依赖性建模
```bash
# 基础预测
python3 dlt_main.py predict -m markov_3rd -c 1
python3 dlt_main.py predict -m markov_3rd -p 1000 -c 3

# 自动加速
python3 dlt_main.py predict -m markov_3rd --acceleration auto -c 2
python3 dlt_main.py predict -m markov_3rd --acceleration cpu_multi -c 3

# 复式预测
python3 dlt_main.py predict -m markov_3rd --compound --front-count 10 --back-count 4
python3 dlt_main.py predict -m markov_3rd --compound --front-count 14 --back-count 5 -p 1200
```

- **adaptive_markov** - 自适应马尔可夫：动态阶数选择
```bash
# 基础预测
python3 dlt_main.py predict -m adaptive_markov -c 1
python3 dlt_main.py predict -m adaptive_markov -p 500 -c 2

# CPU多线程加速
python3 dlt_main.py predict -m adaptive_markov --acceleration cpu_multi -c 3
python3 dlt_main.py predict -m adaptive_markov --acceleration cpu_multi --cpu-threads 4 -c 2

# 复式预测
python3 dlt_main.py predict -m adaptive_markov --compound --front-count 8 --back-count 3
python3 dlt_main.py predict -m adaptive_markov --compound --front-count 13 --back-count 6 -p 700
```

#### 深度学习方法 (支持GPU加速 + 复式预测)
- **lstm** - LSTM时序预测：双向LSTM网络和注意力机制
```bash
# 基础预测
python3 dlt_main.py predict -m lstm -c 1
python3 dlt_main.py predict -m lstm -p 1000 -c 2

# GPU加速
python3 dlt_main.py predict -m lstm --acceleration gpu -c 3
python3 dlt_main.py predict -m lstm --acceleration auto -c 2

# GPU CUDA加速
python3 dlt_main.py predict -m lstm --acceleration gpu_cuda --mixed-precision -c 2
python3 dlt_main.py predict -m lstm --acceleration gpu --gpu-memory-limit 4 -c 3

# 复式预测
python3 dlt_main.py predict -m lstm --compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m lstm --compound --front-count 12 --back-count 5 --acceleration gpu -p 1500
```

- **transformer** - Transformer预测：多头注意力和位置编码
```bash
# 基础预测
python3 dlt_main.py predict -m transformer -c 1
python3 dlt_main.py predict -m transformer -p 800 -c 2

# GPU CUDA加速
python3 dlt_main.py predict -m transformer --acceleration gpu_cuda -c 3
python3 dlt_main.py predict -m transformer --acceleration gpu --mixed-precision -c 2

# GPU优化
python3 dlt_main.py predict -m transformer --acceleration gpu --gpu-device 0 -c 2
python3 dlt_main.py predict -m transformer --acceleration gpu --gpu-memory-limit 6 -c 3

# 复式预测
python3 dlt_main.py predict -m transformer --compound --front-count 9 --back-count 4
python3 dlt_main.py predict -m transformer --compound --front-count 11 --back-count 6 --acceleration gpu -p 1000
```

- **gan** - GAN生成对抗：生成器判别器对抗训练
```bash
# 基础预测
python3 dlt_main.py predict -m gan -c 1
python3 dlt_main.py predict -m gan -p 600 -c 2

# GPU加速
python3 dlt_main.py predict -m gan --acceleration gpu -c 3
python3 dlt_main.py predict -m gan --acceleration auto -c 2

# GPU混合精度
python3 dlt_main.py predict -m gan --acceleration gpu_cuda --mixed-precision -c 2
python3 dlt_main.py predict -m gan --acceleration gpu --gpu-memory-limit 8 -c 3

# 复式预测
python3 dlt_main.py predict -m gan --compound --front-count 10 --back-count 4
python3 dlt_main.py predict -m gan --compound --front-count 13 --back-count 5 --acceleration gpu -p 800
```

- **ensemble** - 集成深度学习：多模型智能融合
```bash
# 基础预测
python3 dlt_main.py predict -m ensemble -c 1
python3 dlt_main.py predict -m ensemble -p 1200 -c 3

# 自动加速
python3 dlt_main.py predict -m ensemble --acceleration auto -c 2
python3 dlt_main.py predict -m ensemble --acceleration gpu -c 3

# 复式预测
python3 dlt_main.py predict -m ensemble --compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m ensemble --compound --front-count 15 --back-count 6 --acceleration auto -p 1000
```

#### 聚类算法方法 (支持自动加速 + 复式预测)
- **clustering** - 聚类预测：基于K-means聚类的智能预测
```bash
# 基础预测
python3 dlt_main.py predict -m clustering -c 1
python3 dlt_main.py predict -m clustering -p 800 -c 2

# 自动加速
python3 dlt_main.py predict -m clustering --acceleration auto -c 3
python3 dlt_main.py predict -m clustering --acceleration cpu_multi -c 2

# 复式预测
python3 dlt_main.py predict -m clustering --compound --front-count 9 --back-count 4
python3 dlt_main.py predict -m clustering --compound --front-count 12 --back-count 5 --acceleration auto -p 1000
```

#### 增强预测方法 (支持GPU加速 + 复式预测)  
- **enhanced** - 增强预测：集成多种增强算法的高级预测系统
```bash
# 基础预测
python3 dlt_main.py predict -m enhanced -c 1
python3 dlt_main.py predict -m enhanced -p 1000 -c 2

# GPU加速
python3 dlt_main.py predict -m enhanced --acceleration gpu -c 3
python3 dlt_main.py predict -m enhanced --acceleration auto -c 2

# 复式预测
python3 dlt_main.py predict -m enhanced --compound --front-count 10 --back-count 4
python3 dlt_main.py predict -m enhanced --compound --front-count 13 --back-count 6 --acceleration gpu -p 1200
```

#### 智能预测方法 (支持自动加速 + 复式预测)
- **super** - 超级预测：多算法智能融合系统
```bash
# 基础预测
python3 dlt_main.py predict -m super -c 1
python3 dlt_main.py predict -m super -p 1000 -c 3

# 自动加速
python3 dlt_main.py predict -m super --acceleration auto -c 2
python3 dlt_main.py predict -m super --acceleration cpu_multi -c 3

# 复式预测
python3 dlt_main.py predict -m super --compound --front-count 10 --back-count 4
python3 dlt_main.py predict -m super --compound --front-count 14 --back-count 6 --acceleration auto -p 1500
```

- **adaptive** - 自适应预测：基于多臂老虎机的预测器选择
```bash
# 基础预测
python3 dlt_main.py predict -m adaptive -c 1
python3 dlt_main.py predict -m adaptive -p 800 -c 2

# 自动加速
python3 dlt_main.py predict -m adaptive --acceleration auto -c 3
python3 dlt_main.py predict -m adaptive --acceleration cpu_multi -c 2

# 复式预测
python3 dlt_main.py predict -m adaptive --compound --front-count 9 --back-count 4
python3 dlt_main.py predict -m adaptive --compound --front-count 12 --back-count 5 --acceleration auto -p 1200
```

- **nine_models** - 九种数学模型：统计学、概率论综合分析
```bash
# 基础预测
python3 dlt_main.py predict -m nine_models -c 1
python3 dlt_main.py predict -m nine_models -p 600 -c 3

# CPU多线程加速
python3 dlt_main.py predict -m nine_models --acceleration cpu_multi -c 2
python3 dlt_main.py predict -m nine_models --acceleration cpu_multi --cpu-threads 6 -c 3

# 复式预测
python3 dlt_main.py predict -m nine_models --compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m nine_models --compound --front-count 11 --back-count 6 --acceleration cpu_multi -p 800
```

- **advanced_integration** - 高级集成分析：多维度权重计算
```bash
# 基础预测
python3 dlt_main.py predict -m advanced_integration -c 1
python3 dlt_main.py predict -m advanced_integration -p 1000 -c 2

# 自动加速
python3 dlt_main.py predict -m advanced_integration --acceleration auto -c 3
python3 dlt_main.py predict -m advanced_integration --acceleration cpu_multi -c 2

# 复式预测
python3 dlt_main.py predict -m advanced_integration --compound --front-count 10 --back-count 4
python3 dlt_main.py predict -m advanced_integration --compound --front-count 13 --back-count 5 --acceleration auto -p 1200
```

- **mixed_strategy** - 混合策略：保守/激进/平衡策略选择
```bash
# 基础预测
python3 dlt_main.py predict -m mixed_strategy -c 1
python3 dlt_main.py predict -m mixed_strategy -p 500 -c 3

# 自动加速
python3 dlt_main.py predict -m mixed_strategy --acceleration auto -c 2
python3 dlt_main.py predict -m mixed_strategy --acceleration cpu_multi -c 3

# 复式预测
python3 dlt_main.py predict -m mixed_strategy --compound --front-count 9 --back-count 4
python3 dlt_main.py predict -m mixed_strategy --compound --front-count 14 --back-count 6 --acceleration auto -p 700
```

- **highly_integrated** - 高度集成：全算法融合系统
```bash
# 基础预测
python3 dlt_main.py predict -m highly_integrated -c 1
python3 dlt_main.py predict -m highly_integrated -p 1500 -c 2

# 自动加速
python3 dlt_main.py predict -m highly_integrated --acceleration auto -c 3
python3 dlt_main.py predict -m highly_integrated --acceleration cpu_multi -c 2

# 复式预测
python3 dlt_main.py predict -m highly_integrated --compound --front-count 12 --back-count 5
python3 dlt_main.py predict -m highly_integrated --compound --front-count 15 --back-count 6 --acceleration auto -p 1800
```

#### 复式投注方法 (原生复式支持 + 自动加速)
- **compound** - 标准复式：指定前区和后区号码数量
```bash
# 标准复式预测
python3 dlt_main.py predict -m compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m compound --front-count 10 --back-count 5 -p 1000

# 大复式预测
python3 dlt_main.py predict -m compound --front-count 12 --back-count 6
python3 dlt_main.py predict -m compound --front-count 15 --back-count 8 -p 1500

# 自动加速复式
python3 dlt_main.py predict -m compound --front-count 9 --back-count 4 --acceleration auto
python3 dlt_main.py predict -m compound --front-count 13 --back-count 7 --acceleration cpu_multi
```

- **duplex** - 胆拖投注：胆码+拖码智能投注
```bash
# 胆拖投注
python3 dlt_main.py predict -m duplex --front-dan 2 --back-dan 1
python3 dlt_main.py predict -m duplex --front-dan 3 --back-dan 1 --front-tuo 8 -p 800

# 高级胆拖
python3 dlt_main.py predict -m duplex --front-dan 2 --back-dan 2 --front-tuo 10
python3 dlt_main.py predict -m duplex --front-dan 4 --back-dan 1 --front-tuo 12 -p 1200

# 自动加速胆拖
python3 dlt_main.py predict -m duplex --front-dan 2 --back-dan 1 --acceleration auto
python3 dlt_main.py predict -m duplex --front-dan 3 --back-dan 2 --acceleration cpu_multi
```

- **markov_compound** - 马尔可夫复式：基于马尔可夫链的复式
```bash
# 马尔可夫复式
python3 dlt_main.py predict -m markov_compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m markov_compound --front-count 10 --back-count 5 -p 1000

# CPU多线程加速
python3 dlt_main.py predict -m markov_compound --front-count 9 --back-count 4 --acceleration cpu_multi
python3 dlt_main.py predict -m markov_compound --front-count 12 --back-count 6 --acceleration cpu_multi --cpu-threads 8
```

- **nine_models_compound** - 九模型复式：多算法融合复式
```bash
# 九模型复式
python3 dlt_main.py predict -m nine_models_compound --front-count 8 --back-count 4
python3 dlt_main.py predict -m nine_models_compound --front-count 11 --back-count 5 -p 800

# CPU多线程加速
python3 dlt_main.py predict -m nine_models_compound --front-count 9 --back-count 4 --acceleration cpu_multi
python3 dlt_main.py predict -m nine_models_compound --front-count 13 --back-count 6 --acceleration cpu_multi --cpu-threads 6
```

#### 集成学习方法 (支持自动加速 + 复式预测)
- **stacking** - Stacking集成：基于元学习器的高级集成
```bash
# 基础预测
python3 dlt_main.py predict -m stacking -c 1
python3 dlt_main.py predict -m stacking -p 1000 -c 2

# 自动加速
python3 dlt_main.py predict -m stacking --acceleration auto -c 3
python3 dlt_main.py predict -m stacking --acceleration cpu_multi -c 2

# 复式预测
python3 dlt_main.py predict -m stacking --compound --front-count 10 --back-count 4
python3 dlt_main.py predict -m stacking --compound --front-count 14 --back-count 6 --acceleration auto -p 1200
```

- **adaptive_ensemble** - 自适应集成：动态权重调整
```bash
# 基础预测
python3 dlt_main.py predict -m adaptive_ensemble -c 1
python3 dlt_main.py predict -m adaptive_ensemble -p 800 -c 3

# 自动加速
python3 dlt_main.py predict -m adaptive_ensemble --acceleration auto -c 2
python3 dlt_main.py predict -m adaptive_ensemble --acceleration cpu_multi -c 3

# 复式预测
python3 dlt_main.py predict -m adaptive_ensemble --compound --front-count 9 --back-count 4
python3 dlt_main.py predict -m adaptive_ensemble --compound --front-count 12 --back-count 5 --acceleration auto -p 1000
```

- **ultimate_ensemble** - 终极集成：最高级别集成预测
```bash
# 基础预测
python3 dlt_main.py predict -m ultimate_ensemble -c 1
python3 dlt_main.py predict -m ultimate_ensemble -p 1500 -c 2

# 自动加速
python3 dlt_main.py predict -m ultimate_ensemble --acceleration auto -c 3
python3 dlt_main.py predict -m ultimate_ensemble --acceleration cpu_multi -c 2

# 复式预测
python3 dlt_main.py predict -m ultimate_ensemble --compound --front-count 11 --back-count 5
python3 dlt_main.py predict -m ultimate_ensemble --compound --front-count 15 --back-count 7 --acceleration auto -p 1800
```

### 🔧 常用参数说明

#### 基础参数
- `-m, --method` - 预测方法（必需）
- `-p, --periods` - 分析期数 (50-2756，默认500)
- `-c, --count` - 预测注数 (1-100，默认1)
- `--save` - 保存结果到文件

#### 硬件加速参数
- `--acceleration` - 加速模式 (auto/gpu/cpu_multi/gpu_cuda)
- `--cpu-threads` - CPU线程数 (-1表示所有核心)
- `--gpu-device` - GPU设备ID (0-7)
- `--gpu-memory-limit` - GPU内存限制 (GB)
- `--mixed-precision` - 启用混合精度训练
- `--batch-size-multiplier` - 批次大小倍数 (默认1.0)
- `--benchmark-hardware` - 运行硬件基准测试
- `--fallback-enabled` - 启用优雅降级 (默认true)

#### 训练优化参数
- `--auto-epochs` - 启用智能训练轮数
- `--min-epochs` - 最小训练轮数 (默认10)
- `--max-epochs` - 最大训练轮数 (默认1000)
- `--performance-mode` - 性能模式 (low/medium/high)
- `--training-intensity` - 训练强度倍数 (默认1.0)

#### 复式预测参数
- `--compound` - 启用复式预测
- `--front-count` - 前区号码数量 (6-15，默认8)
- `--back-count` - 后区号码数量 (3-12，默认4)
- `--front-dan` - 前区胆码数量 (1-5，默认2)
- `--back-dan` - 后区胆码数量 (1-2，默认1)  
- `--front-tuo` - 前区拖码数量 (默认6)
- `--back-tuo` - 后区拖码数量 (默认4)
- `--max-cost` - 最大投注成本 (默认10000元)
- `--min-confidence` - 最小置信度阈值 (默认0.5)

## 🎯 批量预测对比功能

批量预测对比功能支持对同一期号进行多次预测并与实际开奖结果比对，统计分析算法的稳定性和中奖概率。

### ✨ 核心功能
- **批量预测** - 对指定期号进行多次重复预测（5-10000次）
- **中奖判定** - 自动识别9个等级的中奖情况
- **统计分析** - 计算中奖率、各等级概率、执行时间等指标  
- **Excel导出** - 生成包含详细统计的专业报告
- **GUI集成** - 图形界面中提供完整的批量对比功能

### 🚀 使用示例
```bash
# 基础语法
python3 dlt_main.py compare --issue <期号> -m <方法> -p <期数> -t <次数>

# 使用示例
python3 dlt_main.py compare --issue 25104 -m markov -p 100 -t 50
python3 dlt_main.py compare --issue 25103 -m frequency -p 200 -t 30
python3 dlt_main.py compare --issue 25103 -m frequency -p 150 -t 60 --export
```

## 🖥️ 图形用户界面

### 🚀 GUI启动
```bash
./start_gui.sh      # Linux/macOS
start_gui.bat       # Windows
python3 run_gui.py  # 通用方式
```

### 📱 界面功能
- **系统首页** - 硬件监控、最新开奖结果、快速预测
- **数据管理** - 数据状态查看、更新、历史数据分析
- **预测功能** - 26+种预测方法，支持复式预测和成本计算
- **数据分析** - 统计图表、多维度分析报告
- **批量对比** - 批量预测对比功能，Excel报告导出
- **学习功能** - 自适应学习和算法优化
- **性能优化** - 硬件加速配置和性能监控
- **系统设置** - 界面设置和参数配置

## 📊 数据管理

### 基础命令
```bash
# 查看数据状态
python3 dlt_main.py data status

# 更新数据
python3 dlt_main.py data update                    # 全量更新
python3 dlt_main.py data update --incremental      # 增量更新

# 获取最新开奖结果
python3 dlt_main.py data latest

# 数据完整性检查
python3 dlt_main.py data check                     # 基础检查
python3 dlt_main.py data check --fix               # 自动修复
```

### 高级分析
```bash
# 统计分析
python3 dlt_main.py analyze --type basic -p 1000
python3 dlt_main.py analyze --type comprehensive -p 800

# 性能回测
python3 dlt_main.py backtest -m ensemble -t 100
```

## 🎓 自适应学习

```bash
# 学习算法
python3 dlt_main.py learn --algorithm ucb1 -t 1000
python3 dlt_main.py learn --algorithm thompson_sampling -t 800

# 智能预测（基于学习结果）
python3 dlt_main.py smart -p 1000
python3 dlt_main.py smart --compound --front-count 10 --back-count 4
```

## ⚡ 性能优化

### GPU加速
```bash
# 自动选择最优加速模式
python3 dlt_main.py predict -m lstm --acceleration auto

# 使用GPU加速
python3 dlt_main.py predict -m transformer --acceleration gpu

# 使用CPU多线程
python3 dlt_main.py predict -m markov --acceleration cpu_multi
```

### 缓存管理
```bash
# 查看缓存状态
python3 dlt_main.py system cache status

# 缓存清理
python3 dlt_main.py system cache clear
python3 dlt_main.py system cache refresh
```

## 🔧 常见问题与解决方案

### ⚡ GPU相关问题
```bash
# 问题：GPU检测不到
# 解决：运行诊断
python system_check.py

# 问题：GPU内存不足
# 解决：使用CPU多线程模式
python3 dlt_main.py predict -m lstm --acceleration cpu_multi
```

### 🐛 常见错误
```bash
# 模型加载失败
# 解决：删除旧模型文件，重新训练
rm -rf models/
python3 dlt_main.py predict -m lstm

# 依赖包问题
# 解决：重新安装依赖
pip install -r requirements.txt

# 数据完整性问题
# 解决：检查和修复数据
python3 dlt_main.py data check --fix
```

### 📊 性能优化建议
1. **使用自动加速模式**：`--acceleration auto`
2. **适当调整期数**：期数越多训练时间越长
3. **启用缓存**：重复预测会自动使用缓存
4. **GPU内存管理**：大模型建议使用混合精度训练

## 📞 联系方式

- **GitHub**: [https://github.com/linshibo1994/dlt](https://github.com/linshibo1994/dlt)
- **问题反馈**: 通过GitHub Issues提交

## 📄 许可证

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## ⚠️ 免责声明

本系统仅供学习和研究使用，不构成任何投资建议。彩票投注有风险，请理性参与。

---

**🎯 开始您的智能预测之旅！**

🏆 **项目状态**: ✅ 生产就绪 | 🧠 AI驱动 | 📊 数据驱动 | 🚀 高性能优化 | 🎯 批量验证