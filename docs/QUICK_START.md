# 快速开始指南

## 系统要求
- Python 3.8+ (推荐 3.10+)
- Node.js 18+ (前端开发)
- 操作系统: Windows 10+、macOS 10.15+、Linux (Ubuntu 18.04+)

## Docker 部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/linshibo1994/dlt.git
cd dlt

# 2. 启动服务
./deploy.sh start

# 3. 访问应用
# 前端界面: http://localhost
# 后端API: http://localhost:6000/api
# API文档: http://localhost:6000/docs
```

## 本地开发

### 后端 API
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m uvicorn backend.api.server:app --reload --port 6000
```

### 前端界面
```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:3000
```

## 命令行使用

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

## 数据管理

```bash
# 查看数据状态
python3 dlt_main.py data status

# 增量更新数据
python3 dlt_main.py data update --incremental

# 完整更新数据
python3 dlt_main.py data update
```

## 界面功能

| 页面 | 功能 |
|------|------|
| Dashboard | 系统概览、最新开奖、快速预测 |
| Prediction | 26+ 预测算法、复式投注、成本计算 |
| Analysis | 频率分析、冷热号、遗漏值分析 |
| Compare | 批量预测对比、中奖统计 |
| Settings | 主题切换、系统配置 |

## 常见问题

### Q: 后端无法启动？
A: 确保已安装依赖：`pip install -r requirements.txt`

### Q: 前端无法启动？
A: 确保已安装 Node.js 并运行 `npm install`

### Q: 预测结果为空？
A: 检查数据是否正常加载：`python3 dlt_main.py data status`

### Q: 如何更新数据？
A: 运行：`python3 dlt_main.py data update --incremental`

## 获取帮助

- 完整文档：查看 README.md
- 问题反馈：GitHub Issues
- API 文档：http://localhost:6000/docs
