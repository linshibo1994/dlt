#!/bin/bash

# 大乐透预测系统快速部署脚本 (GitHub自动拉取版)
# DLT Prediction System Quick Deployment Script with GitHub Auto-Pull

set -e

# GitHub配置
GITHUB_REPO="https://github.com/linshibo1994/dlt.git"
GITHUB_BRANCH="main"
GITHUB_TOKEN=""
PROJECT_DIR="dlt-prediction-system"
UPDATE_MODE=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
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
            --project-dir)
                PROJECT_DIR="$2"
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
                echo "  --github-repo URL    GitHub仓库地址"
                echo "  --github-branch NAME GitHub分支 (默认: main)"
                echo "  --github-token TOKEN GitHub访问令牌"
                echo "  --project-dir DIR    项目目录名 (默认: dlt-prediction-system)"
                echo "  --update             更新模式"
                echo "  --logs               查看服务日志"
                echo "  --status             查看服务状态"
                echo "  --help               显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                echo "使用 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
}

# 显示服务日志
show_logs() {
    if docker-compose ps | grep -q "dlt-prediction.*Up"; then
        log_info "显示DLT预测服务实时日志 (按Ctrl+C退出)"
        docker-compose logs -f dlt-prediction
    else
        log_warning "服务未运行，显示最近日志"
        docker-compose logs --tail=50 dlt-prediction
    fi
}

# 显示服务状态
show_status() {
    echo "========================================"
    echo "📊 Docker服务状态"
    echo "========================================"

    docker-compose ps

    echo ""
    echo "🌐 访问地址:"
    local server_ip=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    echo "  主应用: http://localhost:8501"
    echo "  主应用: http://$server_ip:8501"

    if docker-compose ps nginx | grep -q "Up"; then
        echo "  Nginx代理: http://localhost"
        echo "  Nginx代理: http://$server_ip"
    fi

    if docker-compose ps prometheus | grep -q "Up"; then
        echo "  Prometheus: http://localhost:9090"
    fi

    if docker-compose ps grafana | grep -q "Up"; then
        echo "  Grafana: http://localhost:3000"
    fi
}

# 从GitHub拉取或更新代码
clone_or_update_project() {
    log_info "从GitHub获取项目代码: $GITHUB_REPO"

    # 设置Git配置
    if [ -n "$GITHUB_TOKEN" ]; then
        REPO_URL=$(echo "$GITHUB_REPO" | sed "s|https://|https://$GITHUB_TOKEN@|")
    else
        REPO_URL="$GITHUB_REPO"
    fi

    if [ -d "$PROJECT_DIR/.git" ]; then
        log_info "更新现有代码库"
        cd "$PROJECT_DIR"

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
        log_info "克隆代码库"

        # 删除现有目录
        rm -rf "$PROJECT_DIR"

        # 克隆代码
        git clone -b "$GITHUB_BRANCH" "$REPO_URL" "$PROJECT_DIR"

        if [ $? -eq 0 ]; then
            log_success "代码克隆完成"
        else
            log_error "代码克隆失败"
            exit 1
        fi

        cd "$PROJECT_DIR"
    fi

    # 显示当前版本信息
    CURRENT_COMMIT=$(git rev-parse --short HEAD)
    COMMIT_MESSAGE=$(git log -1 --pretty=format:"%s")
    COMMIT_DATE=$(git log -1 --pretty=format:"%ci")

    log_info "当前版本信息:"
    echo "  提交ID: $CURRENT_COMMIT"
    echo "  提交信息: $COMMIT_MESSAGE"
    echo "  提交时间: $COMMIT_DATE"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        echo "安装Docker:"
        echo "  Ubuntu/Debian: curl -fsSL https://get.docker.com | sh"
        echo "  CentOS/RHEL: curl -fsSL https://get.docker.com | sh"
        echo "  macOS: brew install docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_warning "docker-compose未安装，尝试安装..."
        
        # 尝试安装docker-compose
        if command -v pip3 &> /dev/null; then
            pip3 install docker-compose
        elif command -v apt &> /dev/null; then
            apt update && apt install -y docker-compose
        elif command -v yum &> /dev/null; then
            yum install -y docker-compose
        else
            log_error "无法自动安装docker-compose，请手动安装"
            exit 1
        fi
    fi
    
    log_success "Docker环境检查通过"
}

# 检查端口是否被占用
check_ports() {
    local ports=(8501 80 443 6379 9090 3000)
    local occupied_ports=()
    
    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            occupied_ports+=($port)
        fi
    done
    
    if [ ${#occupied_ports[@]} -gt 0 ]; then
        log_warning "以下端口被占用: ${occupied_ports[*]}"
        log_info "您可以选择:"
        log_info "1. 停止占用端口的服务"
        log_info "2. 修改docker-compose.yml中的端口映射"
        
        read -p "是否继续部署? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p {data,cache,logs,models,output,config,nginx/ssl,monitoring/{prometheus,grafana/{dashboards,datasources}}}
    
    # 设置权限
    chmod 755 {data,cache,logs,models,output,config}
    
    log_success "目录创建完成"
}

# 生成配置文件
generate_configs() {
    log_info "生成配置文件..."
    
    # 生成Prometheus配置
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'dlt-prediction'
    static_configs:
      - targets: ['dlt-prediction:8501']
    metrics_path: '/_stcore/metrics'
    scrape_interval: 30s
EOF

    # 生成Grafana数据源配置
    cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    log_success "配置文件生成完成"
}

# 构建和启动服务
deploy_services() {
    log_info "构建Docker镜像..."
    docker-compose build --no-cache
    
    log_info "启动服务..."
    docker-compose up -d
    
    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
            log_success "DLT预测服务已就绪"
            break
        fi
        
        log_info "等待服务启动... ($attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "服务启动超时"
        docker-compose logs dlt-prediction
        exit 1
    fi
}

# 显示部署信息
show_deployment_info() {
    local server_ip=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    
    echo ""
    echo "========================================"
    log_success "🎉 部署完成！"
    echo "========================================"
    echo ""
    echo "📊 服务访问地址:"
    echo "  主应用: http://localhost:8501"
    echo "  主应用: http://$server_ip:8501"
    if docker-compose ps nginx | grep -q "Up"; then
        echo "  Nginx代理: http://localhost"
        echo "  Nginx代理: http://$server_ip"
    fi
    if docker-compose ps prometheus | grep -q "Up"; then
        echo "  Prometheus: http://localhost:9090"
    fi
    if docker-compose ps grafana | grep -q "Up"; then
        echo "  Grafana: http://localhost:3000 (admin/admin123)"
    fi
    echo ""
    echo "🔧 管理命令:"
    echo "  查看状态: docker-compose ps"
    echo "  查看日志: docker-compose logs -f"
    echo "  停止服务: docker-compose down"
    echo "  重启服务: docker-compose restart"
    echo "  更新服务: docker-compose pull && docker-compose up -d"
    echo ""
    echo "📁 数据目录:"
    echo "  数据文件: ./data/"
    echo "  缓存文件: ./cache/"
    echo "  日志文件: ./logs/"
    echo "  模型文件: ./models/"
    echo "  输出文件: ./output/"
    echo ""
}

# 更新模式
update_deployment() {
    log_info "执行更新部署"

    # 进入项目目录
    cd "$PROJECT_DIR"

    # 停止服务
    log_info "停止服务"
    docker-compose down

    # 更新代码
    clone_or_update_project

    # 重新构建和启动
    log_info "重新构建镜像"
    docker-compose build --no-cache dlt-prediction

    log_info "启动服务"
    docker-compose up -d

    # 等待服务就绪
    wait_for_services

    log_success "服务更新完成"
}

# 主函数
main() {
    echo "========================================"
    echo "🚀 大乐透预测系统自动化部署"
    echo "========================================"
    echo "GitHub仓库: $GITHUB_REPO"
    echo "分支: $GITHUB_BRANCH"
    echo "项目目录: $PROJECT_DIR"
    echo "========================================"

    # 解析命令行参数
    parse_arguments "$@"

    check_docker

    # 检查是否为更新模式
    if [ "$UPDATE_MODE" = true ]; then
        if [ -d "$PROJECT_DIR" ] && [ -f "$PROJECT_DIR/docker-compose.yml" ]; then
            update_deployment
            show_deployment_info
            return
        else
            log_warning "未检测到现有部署，执行完整部署"
        fi
    fi

    # 完整部署流程
    clone_or_update_project
    check_ports
    create_directories
    generate_configs
    deploy_services
    wait_for_services
    show_deployment_info

    # 显示管理命令
    echo ""
    log_info "常用管理命令:"
    echo "  查看状态: $0 --status"
    echo "  查看日志: $0 --logs"
    echo "  更新代码: $0 --update"
    echo "  重启服务: cd $PROJECT_DIR && docker-compose restart"
    echo "  停止服务: cd $PROJECT_DIR && docker-compose down"
}

# 错误处理
trap 'log_error "部署失败，正在清理..."; docker-compose down 2>/dev/null || true; exit 1' ERR

# 执行主函数
main "$@"
