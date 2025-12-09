# 🚀 大乐透预测系统部署指南

本文档提供了在任意服务器上部署大乐透预测系统的完整指南。

## 📋 **部署方式概览**

### 🎯 **支持的部署方式**

1. **一键自动部署** (最简单)
2. **Docker容器化部署** (推荐)
3. **传统服务器部署**
4. **开发模式部署**
5. **云平台部署**
6. **Kubernetes集群部署**

### 🚀 **GitHub自动拉取特性**

所有部署脚本都支持直接从GitHub仓库拉取最新代码，无需手动下载：
- ✅ 自动克隆/更新代码
- ✅ 实时日志查看
- ✅ 版本信息显示
- ✅ 一键更新部署

### 🔧 **系统要求**

#### 最低配置
- **CPU**: 2核心
- **内存**: 4GB RAM
- **存储**: 10GB可用空间
- **操作系统**: Linux (Ubuntu 18.04+, CentOS 7+), macOS 10.15+

#### 推荐配置
- **CPU**: 4核心以上
- **内存**: 8GB RAM以上
- **存储**: 20GB可用空间
- **GPU**: NVIDIA GPU (可选，用于深度学习加速)

## 🚀 **一键自动部署 (最简单)**

### 超级简单部署

只需要一条命令，自动从GitHub拉取代码并部署：

```bash
# 下载并执行一键部署脚本
curl -fsSL https://raw.githubusercontent.com/linshibo1994/dlt/main/one_click_deploy.sh | bash

# 或者手动下载执行
wget https://raw.githubusercontent.com/linshibo1994/dlt/main/one_click_deploy.sh
chmod +x one_click_deploy.sh
./one_click_deploy.sh
```

### 部署选项

```bash
# 交互式部署 (推荐新手)
./one_click_deploy.sh

# Docker部署
./one_click_deploy.sh --docker

# 传统服务器部署
./one_click_deploy.sh --server

# 开发模式部署
./one_click_deploy.sh --dev

# 自动选择最佳部署方式
./one_click_deploy.sh --auto
```

### 部署后管理

```bash
# 查看服务状态
./deploy.sh --status          # 传统部署
./quick_deploy.sh --status    # Docker部署

# 查看实时日志
./deploy.sh --logs            # 传统部署
./quick_deploy.sh --logs      # Docker部署
./logs_viewer.sh              # 通用日志查看器

# 更新代码
./deploy.sh --update          # 传统部署
./quick_deploy.sh --update    # Docker部署
```

## 🐳 **Docker部署 (推荐)**

### 快速部署

```bash
# 1. 克隆项目
git clone https://github.com/your-username/dlt-prediction-system.git
cd dlt-prediction-system

# 2. 执行快速部署脚本
chmod +x quick_deploy.sh
./quick_deploy.sh
```

### 手动Docker部署

```bash
# 1. 构建镜像
docker build -t dlt-prediction:latest .

# 2. 运行容器
docker run -d \
  --name dlt-prediction \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  dlt-prediction:latest

# 3. 检查状态
docker ps
docker logs dlt-prediction
```

### Docker Compose部署

```bash
# 1. 启动所有服务
docker-compose up -d

# 2. 查看服务状态
docker-compose ps

# 3. 查看日志
docker-compose logs -f dlt-prediction

# 4. 停止服务
docker-compose down
```

## 🖥️ **传统服务器部署**

### 自动部署脚本

```bash
# 1. 下载部署脚本
wget https://raw.githubusercontent.com/your-username/dlt-prediction-system/main/deploy.sh

# 2. 执行部署
chmod +x deploy.sh
sudo ./deploy.sh

# 3. 自定义部署选项
sudo ./deploy.sh --install-dir /opt/dlt --port 8501 --user dlt
```

### 手动部署步骤

#### 1. 安装系统依赖

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev \
                    build-essential curl wget git nginx supervisor
```

**CentOS/RHEL:**
```bash
sudo yum update -y
sudo yum install -y python39 python39-devel \
                    gcc gcc-c++ make curl wget git nginx supervisor
```

#### 2. 创建服务用户
```bash
sudo useradd -r -s /bin/false -d /opt/dlt dlt
```

#### 3. 安装应用
```bash
# 创建安装目录
sudo mkdir -p /opt/dlt
cd /opt/dlt

# 克隆项目
sudo git clone https://github.com/your-username/dlt-prediction-system.git .

# 创建虚拟环境
sudo -u dlt python3.9 -m venv venv
sudo -u dlt bash -c "source venv/bin/activate && pip install -r requirements.txt && pip install -r requirements_gui.txt"

# 设置权限
sudo chown -R dlt:dlt /opt/dlt
```

#### 4. 配置systemd服务
```bash
sudo tee /etc/systemd/system/dlt-prediction.service > /dev/null <<EOF
[Unit]
Description=DLT Prediction System
After=network.target

[Service]
Type=simple
User=dlt
Group=dlt
WorkingDirectory=/opt/dlt
Environment=PATH=/opt/dlt/venv/bin
ExecStart=/opt/dlt/venv/bin/python -m streamlit run gui_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable dlt-prediction
sudo systemctl start dlt-prediction
```

#### 5. 配置Nginx反向代理
```bash
sudo tee /etc/nginx/sites-available/dlt-prediction > /dev/null <<EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# 启用站点
sudo ln -sf /etc/nginx/sites-available/dlt-prediction /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## ☁️ **云平台部署**

### AWS部署

#### 使用EC2实例
```bash
# 1. 创建EC2实例 (推荐t3.medium或更高配置)
# 2. 连接到实例
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. 执行部署脚本
wget https://raw.githubusercontent.com/your-username/dlt-prediction-system/main/deploy.sh
chmod +x deploy.sh
sudo ./deploy.sh
```

#### 使用ECS (Docker)
```bash
# 1. 构建并推送镜像到ECR
aws ecr create-repository --repository-name dlt-prediction
docker build -t dlt-prediction .
docker tag dlt-prediction:latest your-account.dkr.ecr.region.amazonaws.com/dlt-prediction:latest
docker push your-account.dkr.ecr.region.amazonaws.com/dlt-prediction:latest

# 2. 创建ECS任务定义和服务
# (使用AWS控制台或CLI)
```

### Azure部署

#### 使用Azure Container Instances
```bash
# 1. 创建资源组
az group create --name dlt-rg --location eastus

# 2. 部署容器
az container create \
  --resource-group dlt-rg \
  --name dlt-prediction \
  --image your-registry/dlt-prediction:latest \
  --ports 8501 \
  --dns-name-label dlt-prediction-unique
```

### Google Cloud Platform部署

#### 使用Cloud Run
```bash
# 1. 构建并推送镜像
gcloud builds submit --tag gcr.io/your-project/dlt-prediction

# 2. 部署到Cloud Run
gcloud run deploy dlt-prediction \
  --image gcr.io/your-project/dlt-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔧 **配置选项**

### 环境变量

```bash
# Streamlit配置
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# 应用配置
PYTHONPATH=/app
DLT_CONFIG_PATH=/app/config
DLT_DATA_PATH=/app/data
DLT_CACHE_PATH=/app/cache
DLT_LOG_LEVEL=INFO

# GPU配置 (如果有GPU)
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 端口配置

- **8501**: Streamlit应用端口
- **80**: HTTP端口 (Nginx)
- **443**: HTTPS端口 (Nginx)
- **6379**: Redis缓存端口
- **9090**: Prometheus监控端口
- **3000**: Grafana仪表板端口

## 📊 **监控和维护**

### 服务状态检查

```bash
# 检查Docker服务
docker-compose ps
docker-compose logs -f dlt-prediction

# 检查systemd服务
sudo systemctl status dlt-prediction
sudo journalctl -u dlt-prediction -f

# 检查Nginx
sudo systemctl status nginx
sudo nginx -t
```

### 日志管理

```bash
# 查看应用日志
tail -f logs/dlt_predictor.log

# 查看错误日志
tail -f logs/errors.log

# 查看深度学习日志
tail -f logs/deep_learning.log
```

### 备份和恢复

```bash
# 备份数据
tar -czf dlt-backup-$(date +%Y%m%d).tar.gz data/ cache/ models/ config/

# 恢复数据
tar -xzf dlt-backup-20231201.tar.gz
```

## 🔒 **安全配置**

### 防火墙设置

```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### SSL证书配置

```bash
# 使用Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 🚨 **故障排除**

### 常见问题

1. **端口被占用**
   ```bash
   sudo netstat -tulpn | grep :8501
   sudo kill -9 <PID>
   ```

2. **权限问题**
   ```bash
   sudo chown -R dlt:dlt /opt/dlt
   sudo chmod +x /opt/dlt/*.sh
   ```

3. **依赖安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

4. **GPU不可用**
   ```bash
   nvidia-smi
   pip install tensorflow-gpu
   ```

### 性能优化

1. **启用缓存**
   - 确保Redis服务运行正常
   - 配置适当的缓存大小

2. **GPU加速**
   - 安装NVIDIA驱动和CUDA
   - 配置TensorFlow GPU支持

3. **负载均衡**
   - 使用多个应用实例
   - 配置Nginx负载均衡

## 📞 **技术支持**

如果在部署过程中遇到问题，请：

1. 查看日志文件
2. 检查系统资源使用情况
3. 确认网络连接正常
4. 提交GitHub Issue并附上详细的错误信息

---

**🎯 祝您部署成功！**
