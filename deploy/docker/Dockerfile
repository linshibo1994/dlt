# 大乐透预测系统Docker镜像
# DLT Prediction System Docker Image
# 注意: 此为旧版 Streamlit Dockerfile，推荐使用根目录的 docker-compose.yml

# 使用官方Python运行时作为基础镜像
FROM python:3.10-slim

# 设置维护者信息
LABEL maintainer="DLT Prediction System"
LABEL description="大乐透智能预测系统"
LABEL version="2.0.0"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt ./

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p cache logs models output data config

# 暴露端口
EXPOSE 6000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:6000/api/system/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "backend.api.server:app", "--host", "0.0.0.0", "--port", "6000"]
