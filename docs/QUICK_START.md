# 🚀 快速开始指南

## 📋 系统要求
- Python 3.8+
- 操作系统: Windows 10+、macOS 10.15+、Linux (Ubuntu 18.04+)

## ⚡ 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/linshibo1994/dlt.git
cd dlt

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装GUI依赖（推荐）
pip install -r requirements_gui.txt
```

## 🖥️ 启动GUI界面（推荐）

### 方式一：使用启动脚本
```bash
# Linux/macOS
./start_gui.sh

# Windows
start_gui.bat
```

### 方式二：使用Python脚本
```bash
python3 run_gui.py
```

### 方式三：直接启动
```bash
streamlit run gui_app.py
```

## 💻 命令行使用

### 基本预测
```bash
# 频率分析预测
python3 dlt_main.py predict -m frequency -c 1

# 马尔可夫链预测
python3 dlt_main.py predict -m markov -c 1

# 贝叶斯推理预测
python3 dlt_main.py predict -m bayesian -c 1
```

### 复式预测
```bash
# 8+4复式预测
python3 dlt_main.py predict -m frequency --compound --front-count 8 --back-count 4

# 10+5复式预测
python3 dlt_main.py predict -m markov --compound --front-count 10 --back-count 5
```

### 硬件加速
```bash
# 自动选择加速方式
python3 dlt_main.py predict -m bayesian --acceleration auto

# CPU多线程加速
python3 dlt_main.py predict -m markov --acceleration cpu_multi --cpu-threads 6
```

## 📊 数据管理

```bash
# 查看数据状态
python3 dlt_main.py data status

# 增量更新数据
python3 dlt_main.py data update --incremental

# 完整更新数据
python3 dlt_main.py data update
```

## 🎯 GUI功能说明

### 🏠 系统首页
- 实时硬件监控
- 最新开奖结果
- 快速预测入口

### 🔮 预测功能
- 传统方法：频率、冷热、遗漏分析
- 高级方法：马尔可夫、贝叶斯、聚类
- 复式预测：支持所有方法
- 成本计算：自动计算投注成本

### 📈 数据分析
- 统计图表可视化
- 多维度分析报告
- 异常检测分析

### ⚡ 性能优化
- 硬件加速配置
- 缓存管理
- 性能监控

## 🔧 常见问题

### Q: GUI无法启动？
A: 确保已安装GUI依赖：`pip install -r requirements_gui.txt`

### Q: 预测结果为空？
A: 检查数据是否正常加载：`python3 dlt_main.py data status`

### Q: 如何更新数据？
A: 运行：`python3 dlt_main.py data update --incremental`

### Q: 支持哪些预测方法？
A: 支持25+种算法，包括传统统计、机器学习、深度学习等

## 📞 获取帮助

- 📖 完整文档：查看 README.md
- 🐛 问题反馈：GitHub Issues
- 💬 讨论交流：GitHub Discussions

## 🎉 开始使用

推荐使用GUI界面，功能更直观：
```bash
python3 run_gui.py
```

然后在浏览器中打开：http://localhost:8501
