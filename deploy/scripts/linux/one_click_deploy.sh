#!/bin/bash

# 大乐透预测系统一键部署脚本
# DLT Prediction System One-Click Deployment Script

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
GITHUB_REPO="https://github.com/linshibo1994/dlt.git"
GITHUB_BRANCH="main"
PROJECT_DIR="dlt-prediction-system"
DEPLOY_METHOD=""

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

# 检测系统环境
detect_environment() {
    log_info "检测系统环境..."
    
    # 检测操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [ -f /etc/debian_version ]; then
            DISTRO="debian"
        elif [ -f /etc/redhat-release ]; then
            DISTRO="redhat"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macos"
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    # 检测Docker
    DOCKER_AVAILABLE=false
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            DOCKER_AVAILABLE=true
        fi
    fi
    
    # 检测权限
    ROOT_ACCESS=false
    if [[ $EUID -eq 0 ]] || sudo -n true 2>/dev/null; then
        ROOT_ACCESS=true
    fi
    
    log_info "系统信息:"
    echo "  操作系统: $OS ($DISTRO)"
    echo "  Docker可用: $DOCKER_AVAILABLE"
    echo "  Root权限: $ROOT_ACCESS"
}

# 选择部署方法
choose_deployment_method() {
    echo ""
    echo "========================================"
    echo "🚀 选择部署方法"
    echo "========================================"
    echo "1. Docker部署 (推荐)"
    echo "2. 传统服务器部署"
    echo "3. 开发模式部署"
    echo "4. 自动选择"
    echo "========================================"
    
    if [ -n "$1" ]; then
        # 非交互模式
        case "$1" in
            docker|1) DEPLOY_METHOD="docker" ;;
            server|2) DEPLOY_METHOD="server" ;;
            dev|3) DEPLOY_METHOD="dev" ;;
            auto|4) DEPLOY_METHOD="auto" ;;
            *) 
                log_error "无效的部署方法: $1"
                exit 1
                ;;
        esac
    else
        # 交互模式
        read -p "请选择部署方法 (1-4): " choice
        case $choice in
            1) DEPLOY_METHOD="docker" ;;
            2) DEPLOY_METHOD="server" ;;
            3) DEPLOY_METHOD="dev" ;;
            4) DEPLOY_METHOD="auto" ;;
            *)
                log_error "无效选择"
                exit 1
                ;;
        esac
    fi
    
    # 自动选择逻辑
    if [ "$DEPLOY_METHOD" = "auto" ]; then
        if [ "$DOCKER_AVAILABLE" = true ]; then
            DEPLOY_METHOD="docker"
            log_info "自动选择: Docker部署"
        elif [ "$ROOT_ACCESS" = true ]; then
            DEPLOY_METHOD="server"
            log_info "自动选择: 传统服务器部署"
        else
            DEPLOY_METHOD="dev"
            log_info "自动选择: 开发模式部署"
        fi
    fi
    
    log_success "选择的部署方法: $DEPLOY_METHOD"
}

# 克隆项目代码
clone_project() {
    log_info "从GitHub克隆项目代码..."
    
    if [ -d "$PROJECT_DIR" ]; then
        log_warning "项目目录已存在，正在更新..."
        cd "$PROJECT_DIR"
        git pull origin "$GITHUB_BRANCH"
    else
        git clone -b "$GITHUB_BRANCH" "$GITHUB_REPO" "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi
    
    # 显示版本信息
    CURRENT_COMMIT=$(git rev-parse --short HEAD)
    COMMIT_MESSAGE=$(git log -1 --pretty=format:"%s")
    
    log_success "代码获取完成"
    echo "  提交ID: $CURRENT_COMMIT"
    echo "  提交信息: $COMMIT_MESSAGE"
}

# Docker部署
deploy_with_docker() {
    log_info "执行Docker部署..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        echo "安装命令:"
        echo "  Ubuntu/Debian: curl -fsSL https://get.docker.com | sh"
        echo "  CentOS/RHEL: curl -fsSL https://get.docker.com | sh"
        echo "  macOS: brew install docker"
        exit 1
    fi
    
    # 使用项目的快速部署脚本
    chmod +x quick_deploy.sh
    ./quick_deploy.sh
}

# 传统服务器部署
deploy_with_server() {
    log_info "执行传统服务器部署..."
    
    if [ "$ROOT_ACCESS" != true ]; then
        log_error "传统服务器部署需要root权限"
        exit 1
    fi
    
    # 使用项目的部署脚本
    chmod +x deploy.sh
    sudo ./deploy.sh
}

# 开发模式部署
deploy_with_dev() {
    log_info "执行开发模式部署..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装"
        exit 1
    fi
    
    # 创建虚拟环境
    if [ ! -d "venv" ]; then
        log_info "创建Python虚拟环境..."
        python3 -m venv venv
    fi

    # 激活虚拟环境并安装依赖
    log_info "安装依赖包..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    # 启动应用
    log_info "启动应用..."
    python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
}

# 显示部署结果
show_deployment_result() {
    local server_ip=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    
    echo ""
    echo "========================================"
    log_success "🎉 部署完成！"
    echo "========================================"
    echo ""
    echo "🌐 访问地址:"
    
    case $DEPLOY_METHOD in
        docker)
            echo "  主应用: http://localhost:8501"
            echo "  主应用: http://$server_ip:8501"
            echo ""
            echo "🔧 管理命令:"
            echo "  查看状态: ./quick_deploy.sh --status"
            echo "  查看日志: ./quick_deploy.sh --logs"
            echo "  更新代码: ./quick_deploy.sh --update"
            ;;
        server)
            echo "  主应用: http://localhost:8501"
            echo "  主应用: http://$server_ip:8501"
            if systemctl is-active --quiet nginx; then
                echo "  Nginx代理: http://$server_ip"
            fi
            echo ""
            echo "🔧 管理命令:"
            echo "  查看状态: sudo ./deploy.sh --status"
            echo "  查看日志: sudo ./deploy.sh --logs"
            echo "  更新代码: sudo ./deploy.sh --update"
            ;;
        dev)
            echo "  主应用: http://localhost:8501"
            echo ""
            echo "🔧 管理命令:"
            echo "  重启应用: source venv/bin/activate && python3 run_gui.py"
            echo "  查看日志: ./logs_viewer.sh"
            ;;
    esac
    
    echo ""
    echo "📚 文档:"
    echo "  部署文档: cat DEPLOYMENT.md"
    echo "  使用说明: cat README.md"
    echo "  快速开始: cat QUICK_START.md"
}

# 主函数
main() {
    echo "========================================"
    echo "🎯 大乐透预测系统一键部署"
    echo "========================================"
    echo "GitHub仓库: $GITHUB_REPO"
    echo "分支: $GITHUB_BRANCH"
    echo "========================================"
    
    # 解析命令行参数
    local deploy_method=""
    case "${1:-}" in
        --docker) deploy_method="docker" ;;
        --server) deploy_method="server" ;;
        --dev) deploy_method="dev" ;;
        --auto) deploy_method="auto" ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --docker     Docker部署"
            echo "  --server     传统服务器部署"
            echo "  --dev        开发模式部署"
            echo "  --auto       自动选择部署方法"
            echo "  --help       显示帮助信息"
            echo ""
            echo "无参数时启动交互式部署"
            exit 0
            ;;
        "")
            # 交互模式
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
    
    detect_environment
    choose_deployment_method "$deploy_method"
    clone_project
    
    case $DEPLOY_METHOD in
        docker)
            deploy_with_docker
            ;;
        server)
            deploy_with_server
            ;;
        dev)
            deploy_with_dev
            ;;
    esac
    
    show_deployment_result
}

# 错误处理
trap 'log_error "部署失败，请检查错误信息"; exit 1' ERR

# 执行主函数
main "$@"
