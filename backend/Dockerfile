# DLT Backend - FastAPI
# 大乐透预测系统后端服务

FROM python:3.10-slim

# 设置维护者信息
LABEL maintainer="DLT Prediction System"
LABEL description="大乐透智能预测系统 - FastAPI 后端"
LABEL version="2.0.0"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/backend/app/core:/app/backend/app

# 安装系统依赖（使用阿里云镜像加速）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true && \
    apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件（从根目录）
COPY requirements.txt ./requirements.txt

# 安装Python依赖（使用清华镜像加速）
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制后端代码
COPY backend/ ./backend/
COPY config/ ./config/
COPY data/ ./data/
COPY scripts/ ./scripts/

# 创建必要的目录
RUN mkdir -p cache logs models output

# 暴露端口
EXPOSE 6000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:6000/api/system/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "backend.api.server:app", "--host", "0.0.0.0", "--port", "6000"]
