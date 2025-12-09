#!/bin/bash

# 大乐透预测系统部署脚本
# DLT Prediction System Deployment Script
# 
# 支持在任意Linux/macOS服务器上部署
# Supports deployment on any Linux/macOS server

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置变量
PROJECT_NAME="dlt-prediction-system"
INSTALL_DIR="/opt/${PROJECT_NAME}"
SERVICE_USER="dlt"
SERVICE_PORT="8501"
PYTHON_VERSION="3.8"
VIRTUAL_ENV_NAME="dlt_env"

# GitHub配置
GITHUB_REPO="https://github.com/linshibo1994/dlt.git"
GITHUB_BRANCH="main"
GITHUB_TOKEN=""  # 如果是私有仓库，需要设置token

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --port)
            SERVICE_PORT="$2"
            shift 2
            ;;
        --user)
            SERVICE_USER="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --github-repo)
            GITHUB_REPO="$2"
            shift 2
            ;;
        --github-branch)
            GITHUB_BRANCH="$2"
            shift 2
            ;;
        --github-token)
            GITHUB_TOKEN="$2"
            shift 2
            ;;
        --update)
            UPDATE_MODE=true
            shift
            ;;
        --logs)
            show_logs
            exit 0
            ;;
        --status)
            show_status
            exit 0
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --install-dir DIR     安装目录 (默认: /opt/dlt-prediction-system)"
            echo "  --port PORT          服务端口 (默认: 8501)"
            echo "  --user USER          服务用户 (默认: dlt)"
            echo "  --python-version VER Python版本 (默认: 3.8)"
            echo "  --github-repo URL    GitHub仓库地址"
            echo "  --github-branch NAME GitHub分支 (默认: main)"
            echo "  --github-token TOKEN GitHub访问令牌 (私有仓库)"
            echo "  --update             更新模式 (仅更新代码)"
            echo "  --logs               查看服务日志"
            echo "  --status             查看服务状态"
            echo "  --help               显示帮助信息"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
            PKG_MANAGER="apt"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
            PKG_MANAGER="yum"
        else
            OS="linux"
            PKG_MANAGER="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    log_info "检测到操作系统: $OS"
}

# 显示服务日志
show_logs() {
    echo "========================================"
    echo "📋 大乐透预测系统服务日志"
    echo "========================================"

    if systemctl is-active --quiet dlt-prediction; then
        log_info "显示实时日志 (按Ctrl+C退出)"
        echo ""
        journalctl -u dlt-prediction -f --no-pager
    else
        log_warning "服务未运行，显示最近日志"
        journalctl -u dlt-prediction -n 50 --no-pager
    fi
}

# 显示服务状态
show_status() {
    echo "========================================"
    echo "📊 大乐透预测系统状态"
    echo "========================================"

    # 检查服务状态
    if systemctl is-active --quiet dlt-prediction; then
        log_success "DLT预测服务: 运行中"
    else
        log_error "DLT预测服务: 已停止"
    fi

    if systemctl is-active --quiet nginx; then
        log_success "Nginx服务: 运行中"
    else
        log_warning "Nginx服务: 已停止"
    fi

    # 检查端口
    if netstat -tuln 2>/dev/null | grep -q ":$SERVICE_PORT "; then
        log_success "端口 $SERVICE_PORT: 正在监听"
    else
        log_error "端口 $SERVICE_PORT: 未监听"
    fi

    # 显示访问地址
    SERVER_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    echo ""
    echo "🌐 访问地址:"
    echo "  本地访问: http://localhost:$SERVICE_PORT"
    echo "  网络访问: http://$SERVER_IP:$SERVICE_PORT"
    if systemctl is-active --quiet nginx; then
        echo "  Nginx代理: http://$SERVER_IP"
    fi

    # 显示最近日志
    echo ""
    echo "📋 最近日志 (最后10行):"
    journalctl -u dlt-prediction -n 10 --no-pager | tail -10
}

# 检查权限
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        log_info "请使用: sudo $0"
        exit 1
    fi
}

# 安装系统依赖
install_system_dependencies() {
    log_info "安装系统依赖..."
    
    case $PKG_MANAGER in
        apt)
            apt update
            apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
                          build-essential curl wget git nginx supervisor \
                          libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev
            ;;
        yum)
            yum update -y
            yum install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-devel \
                          gcc gcc-c++ make curl wget git nginx supervisor \
                          openssl-devel libffi-devel bzip2-devel readline-devel sqlite-devel
            ;;
        brew)
            brew update
            brew install python@${PYTHON_VERSION} nginx supervisor
            ;;
        *)
            log_warning "未知的包管理器，请手动安装依赖"
            ;;
    esac
    
    log_success "系统依赖安装完成"
}

# 创建服务用户
create_service_user() {
    log_info "创建服务用户: $SERVICE_USER"
    
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
        log_success "用户 $SERVICE_USER 创建成功"
    else
        log_info "用户 $SERVICE_USER 已存在"
    fi
}

# 创建安装目录
create_install_directory() {
    log_info "创建安装目录: $INSTALL_DIR"
    
    mkdir -p "$INSTALL_DIR"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    
    log_success "安装目录创建完成"
}

# 从GitHub拉取项目代码
clone_or_update_project() {
    log_info "从GitHub拉取项目代码: $GITHUB_REPO"

    # 设置Git配置
    if [ -n "$GITHUB_TOKEN" ]; then
        # 使用token访问私有仓库
        REPO_URL=$(echo "$GITHUB_REPO" | sed "s|https://|https://$GITHUB_TOKEN@|")
    else
        REPO_URL="$GITHUB_REPO"
    fi

    if [ -d "$INSTALL_DIR/.git" ]; then
        # 目录已存在，更新代码
        log_info "更新现有代码库"
        cd "$INSTALL_DIR"

        # 备份本地修改
        if [ -n "$(git status --porcelain)" ]; then
            log_warning "检测到本地修改，创建备份"
            git stash push -m "Auto backup before update $(date)"
        fi

        # 拉取最新代码
        git fetch origin
        git reset --hard origin/$GITHUB_BRANCH
        git clean -fd

        log_success "代码更新完成"
    else
        # 首次部署，克隆仓库
        log_info "克隆代码库到 $INSTALL_DIR"

        # 确保目录为空
        rm -rf "$INSTALL_DIR"
        mkdir -p "$(dirname "$INSTALL_DIR")"

        # 克隆代码
        git clone -b "$GITHUB_BRANCH" "$REPO_URL" "$INSTALL_DIR"

        if [ $? -eq 0 ]; then
            log_success "代码克隆完成"
        else
            log_error "代码克隆失败"
            exit 1
        fi
    fi

    # 设置权限
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR"/*.sh 2>/dev/null || true
    chmod +x "$INSTALL_DIR"/*.py 2>/dev/null || true

    # 显示当前版本信息
    cd "$INSTALL_DIR"
    CURRENT_COMMIT=$(git rev-parse --short HEAD)
    COMMIT_MESSAGE=$(git log -1 --pretty=format:"%s")
    COMMIT_DATE=$(git log -1 --pretty=format:"%ci")

    log_info "当前版本信息:"
    echo "  提交ID: $CURRENT_COMMIT"
    echo "  提交信息: $COMMIT_MESSAGE"
    echo "  提交时间: $COMMIT_DATE"
}

# 创建Python虚拟环境
create_virtual_environment() {
    log_info "创建Python虚拟环境: $VIRTUAL_ENV_NAME"
    
    cd "$INSTALL_DIR"
    
    # 创建虚拟环境
    sudo -u "$SERVICE_USER" python${PYTHON_VERSION} -m venv "$VIRTUAL_ENV_NAME"
    
    # 激活虚拟环境并安装依赖
    sudo -u "$SERVICE_USER" bash -c "
        source $VIRTUAL_ENV_NAME/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements_gui.txt
    "
    
    log_success "Python虚拟环境创建完成"
}

# 配置Nginx
configure_nginx() {
    log_info "配置Nginx反向代理"
    
    cat > /etc/nginx/sites-available/dlt-prediction << EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:$SERVICE_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF
    
    # 启用站点
    if [ -d "/etc/nginx/sites-enabled" ]; then
        ln -sf /etc/nginx/sites-available/dlt-prediction /etc/nginx/sites-enabled/
        rm -f /etc/nginx/sites-enabled/default
    fi
    
    # 测试Nginx配置
    nginx -t
    
    log_success "Nginx配置完成"
}

# 创建systemd服务
create_systemd_service() {
    log_info "创建systemd服务"
    
    cat > /etc/systemd/system/dlt-prediction.service << EOF
[Unit]
Description=DLT Prediction System
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/$VIRTUAL_ENV_NAME/bin
ExecStart=$INSTALL_DIR/$VIRTUAL_ENV_NAME/bin/python -m streamlit run frontend/streamlit/app.py --server.port $SERVICE_PORT --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # 重新加载systemd
    systemctl daemon-reload
    systemctl enable dlt-prediction
    
    log_success "systemd服务创建完成"
}

# 启动服务
start_services() {
    log_info "启动服务"
    
    # 启动DLT预测服务
    systemctl start dlt-prediction
    
    # 启动Nginx
    systemctl restart nginx
    systemctl enable nginx
    
    log_success "服务启动完成"
}

# 检查服务状态
check_service_status() {
    log_info "检查服务状态"
    
    # 检查DLT服务
    if systemctl is-active --quiet dlt-prediction; then
        log_success "DLT预测服务运行正常"
    else
        log_error "DLT预测服务启动失败"
        systemctl status dlt-prediction
    fi
    
    # 检查Nginx
    if systemctl is-active --quiet nginx; then
        log_success "Nginx服务运行正常"
    else
        log_error "Nginx服务启动失败"
        systemctl status nginx
    fi
    
    # 显示访问信息
    SERVER_IP=$(hostname -I | awk '{print $1}')
    echo ""
    log_success "部署完成！"
    echo "访问地址:"
    echo "  本地访问: http://localhost"
    echo "  网络访问: http://$SERVER_IP"
    echo "  直接访问: http://$SERVER_IP:$SERVICE_PORT"
    echo ""
    echo "服务管理命令:"
    echo "  启动服务: systemctl start dlt-prediction"
    echo "  停止服务: systemctl stop dlt-prediction"
    echo "  重启服务: systemctl restart dlt-prediction"
    echo "  查看状态: systemctl status dlt-prediction"
    echo "  查看日志: journalctl -u dlt-prediction -f"
}

# 更新模式
update_deployment() {
    log_info "执行更新部署"

    # 停止服务
    log_info "停止服务"
    systemctl stop dlt-prediction 2>/dev/null || true

    # 更新代码
    clone_or_update_project

    # 更新依赖
    log_info "更新Python依赖"
    cd "$INSTALL_DIR"
    sudo -u "$SERVICE_USER" bash -c "
        source $VIRTUAL_ENV_NAME/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt --upgrade
        pip install -r requirements_gui.txt --upgrade
    "

    # 重启服务
    log_info "重启服务"
    systemctl start dlt-prediction

    # 检查状态
    sleep 5
    if systemctl is-active --quiet dlt-prediction; then
        log_success "服务更新完成并正常运行"
    else
        log_error "服务启动失败"
        systemctl status dlt-prediction
        exit 1
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "🎯 大乐透预测系统自动化部署脚本"
    echo "========================================"
    echo "GitHub仓库: $GITHUB_REPO"
    echo "分支: $GITHUB_BRANCH"
    echo "安装目录: $INSTALL_DIR"
    echo "========================================"

    detect_os
    check_permissions

    # 检查是否为更新模式
    if [ "$UPDATE_MODE" = true ]; then
        if [ -d "$INSTALL_DIR" ] && [ -f "/etc/systemd/system/dlt-prediction.service" ]; then
            update_deployment
            check_service_status
            return
        else
            log_warning "未检测到现有安装，执行完整部署"
        fi
    fi

    # 完整部署流程
    install_system_dependencies
    create_service_user
    create_install_directory
    clone_or_update_project
    create_virtual_environment

    if command -v nginx &> /dev/null; then
        configure_nginx
    else
        log_warning "Nginx未安装，跳过反向代理配置"
    fi

    create_systemd_service
    start_services
    check_service_status

    # 显示部署后的管理命令
    echo ""
    log_info "部署完成！常用管理命令:"
    echo "  查看状态: $0 --status"
    echo "  查看日志: $0 --logs"
    echo "  更新代码: $0 --update"
    echo "  重启服务: systemctl restart dlt-prediction"
    echo "  停止服务: systemctl stop dlt-prediction"
}

# 执行主函数
main "$@"
