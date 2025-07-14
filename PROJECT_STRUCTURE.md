# 📁 项目结构说明

## 🏗️ 核心文件

### 主程序
- **`dlt_main.py`** - 主程序入口，统一命令行接口

### 核心模块
- **`core_modules.py`** - 核心功能模块
  - 缓存管理器 (CacheManager)
  - 进度管理器 (ProgressBar, TaskManager)
  - 日志管理器 (LoggerManager)
  - 数据管理器 (DataManager)

- **`analyzer_modules.py`** - 分析器模块
  - 基础分析器 (BasicAnalyzer)
  - 高级分析器 (AdvancedAnalyzer)
  - 综合分析器 (ComprehensiveAnalyzer)
  - 可视化分析器 (VisualizationAnalyzer)

- **`predictor_modules.py`** - 预测器模块
  - 传统预测器 (TraditionalPredictor)
  - 高级预测器 (AdvancedPredictor)
  - 超级预测器 (SuperPredictor)
  - 复式投注预测器 (CompoundPredictor)

- **`adaptive_learning_modules.py`** - 自适应学习模块
  - 多臂老虎机 (MultiArmedBandit)
  - 准确率跟踪器 (AccuracyTracker)
  - 增强学习预测器 (EnhancedAdaptiveLearningPredictor)

### 配置文件
- **`requirements.txt`** - Python依赖包列表
- **`README.md`** - 项目说明文档

## 📂 数据目录

### 数据文件
- **`data/`** - 数据目录
  - `dlt_data_all.csv` - 历史开奖数据（2745期）

### 缓存目录
- **`cache/`** - 缓存目录
  - `analysis/` - 分析结果缓存
  - `data/` - 数据缓存
  - `models/` - 模型缓存

### 日志目录
- **`logs/`** - 日志目录
  - `dlt_predictor.log` - 系统日志
  - `errors.log` - 错误日志

### 输出目录
- **`output/`** - 输出目录
  - `visualization/` - 可视化图表
  - `analysis_report.txt` - 分析报告
  - `comprehensive_analysis.json` - 综合分析结果

## 🎯 学习结果文件

- **`learning_*.json`** - 自适应学习结果文件
  - 包含多臂老虎机学习历史
  - 预测器性能统计
  - 算法权重调整记录

## 🚀 快速开始

```bash
# 查看帮助
python3 dlt_main.py --help

# 查看数据状态
python3 dlt_main.py data status

# 进行分析
python3 dlt_main.py analyze -t comprehensive -p 500 --report

# 号码预测
python3 dlt_main.py predict -m ensemble -c 5

# 自适应学习
python3 dlt_main.py learn -s 100 -t 1000 --algorithm ucb1

# 智能预测
python3 dlt_main.py smart -c 5 --load learning_*.json
```

## 📊 项目特色

- ✅ **模块化设计**：清晰的功能分离
- ✅ **延迟加载**：按需加载，快速启动
- ✅ **智能缓存**：提升性能
- ✅ **完整日志**：详细记录
- ✅ **自适应学习**：持续优化
- ✅ **多种算法**：传统+机器学习+深度学习

---

**🎯 大乐透预测系统 v2.0 - 让预测更智能！**
