#!/bin/bash

# 大乐透预测系统日志查看器
# DLT Prediction System Logs Viewer

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

# 检测部署类型
detect_deployment_type() {
    if [ -f "docker-compose.yml" ] && command -v docker-compose &> /dev/null; then
        echo "docker"
    elif systemctl list-units --type=service | grep -q "dlt-prediction"; then
        echo "systemd"
    else
        echo "unknown"
    fi
}

# Docker日志查看
show_docker_logs() {
    local service="$1"
    local lines="$2"
    local follow="$3"
    
    if [ "$follow" = "true" ]; then
        log_info "显示 $service 实时日志 (按Ctrl+C退出)"
        docker-compose logs -f "$service"
    else
        log_info "显示 $service 最近 $lines 行日志"
        docker-compose logs --tail="$lines" "$service"
    fi
}

# Systemd日志查看
show_systemd_logs() {
    local service="$1"
    local lines="$2"
    local follow="$3"
    
    if [ "$follow" = "true" ]; then
        log_info "显示 $service 实时日志 (按Ctrl+C退出)"
        journalctl -u "$service" -f --no-pager
    else
        log_info "显示 $service 最近 $lines 行日志"
        journalctl -u "$service" -n "$lines" --no-pager
    fi
}

# 应用日志查看
show_app_logs() {
    local lines="$1"
    local follow="$2"
    
    if [ -f "logs/dlt_predictor.log" ]; then
        if [ "$follow" = "true" ]; then
            log_info "显示应用实时日志 (按Ctrl+C退出)"
            tail -f logs/dlt_predictor.log
        else
            log_info "显示应用最近 $lines 行日志"
            tail -n "$lines" logs/dlt_predictor.log
        fi
    else
        log_warning "应用日志文件不存在: logs/dlt_predictor.log"
    fi
}

# 错误日志查看
show_error_logs() {
    local lines="$1"
    
    if [ -f "logs/errors.log" ]; then
        log_info "显示错误日志最近 $lines 行"
        tail -n "$lines" logs/errors.log
    else
        log_warning "错误日志文件不存在: logs/errors.log"
    fi
}

# 深度学习日志查看
show_dl_logs() {
    local lines="$1"
    
    if [ -f "logs/deep_learning.log" ]; then
        log_info "显示深度学习日志最近 $lines 行"
        tail -n "$lines" logs/deep_learning.log
    else
        log_warning "深度学习日志文件不存在: logs/deep_learning.log"
    fi
}

# 显示所有日志概览
show_logs_overview() {
    echo "========================================"
    echo "📋 大乐透预测系统日志概览"
    echo "========================================"
    
    local deployment_type=$(detect_deployment_type)
    
    case $deployment_type in
        docker)
            log_info "检测到Docker部署"
            echo ""
            echo "🐳 Docker服务状态:"
            docker-compose ps
            echo ""
            echo "📋 最近日志 (最后10行):"
            docker-compose logs --tail=10 dlt-prediction
            ;;
        systemd)
            log_info "检测到Systemd部署"
            echo ""
            echo "🔧 服务状态:"
            systemctl status dlt-prediction --no-pager -l
            echo ""
            echo "📋 最近日志 (最后10行):"
            journalctl -u dlt-prediction -n 10 --no-pager
            ;;
        *)
            log_warning "未检测到已部署的服务"
            ;;
    esac
    
    # 显示应用日志文件
    echo ""
    echo "📁 应用日志文件:"
    if [ -d "logs" ]; then
        ls -la logs/ | grep -E "\.(log|txt)$" || echo "  无日志文件"
    else
        echo "  logs目录不存在"
    fi
}

# 交互式日志查看器
interactive_logs_viewer() {
    while true; do
        echo ""
        echo "========================================"
        echo "📋 大乐透预测系统日志查看器"
        echo "========================================"
        echo "1. 服务日志 (实时)"
        echo "2. 服务日志 (最近50行)"
        echo "3. 应用日志 (实时)"
        echo "4. 应用日志 (最近50行)"
        echo "5. 错误日志"
        echo "6. 深度学习日志"
        echo "7. 日志概览"
        echo "8. 退出"
        echo "========================================"
        
        read -p "请选择 (1-8): " choice
        
        case $choice in
            1)
                local deployment_type=$(detect_deployment_type)
                case $deployment_type in
                    docker) show_docker_logs "dlt-prediction" 50 true ;;
                    systemd) show_systemd_logs "dlt-prediction" 50 true ;;
                    *) log_error "未检测到已部署的服务" ;;
                esac
                ;;
            2)
                local deployment_type=$(detect_deployment_type)
                case $deployment_type in
                    docker) show_docker_logs "dlt-prediction" 50 false ;;
                    systemd) show_systemd_logs "dlt-prediction" 50 false ;;
                    *) log_error "未检测到已部署的服务" ;;
                esac
                ;;
            3)
                show_app_logs 50 true
                ;;
            4)
                show_app_logs 50 false
                ;;
            5)
                show_error_logs 50
                ;;
            6)
                show_dl_logs 50
                ;;
            7)
                show_logs_overview
                ;;
            8)
                log_info "退出日志查看器"
                exit 0
                ;;
            *)
                log_error "无效选择，请输入1-8"
                ;;
        esac
        
        echo ""
        read -p "按Enter键继续..."
    done
}

# 主函数
main() {
    # 解析命令行参数
    case "${1:-}" in
        --service)
            local deployment_type=$(detect_deployment_type)
            case $deployment_type in
                docker) show_docker_logs "dlt-prediction" 50 true ;;
                systemd) show_systemd_logs "dlt-prediction" 50 true ;;
                *) log_error "未检测到已部署的服务" ;;
            esac
            ;;
        --app)
            show_app_logs 50 true
            ;;
        --error)
            show_error_logs 100
            ;;
        --dl)
            show_dl_logs 100
            ;;
        --overview)
            show_logs_overview
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --service    显示服务日志 (实时)"
            echo "  --app        显示应用日志 (实时)"
            echo "  --error      显示错误日志"
            echo "  --dl         显示深度学习日志"
            echo "  --overview   显示日志概览"
            echo "  --help       显示帮助信息"
            echo ""
            echo "无参数时启动交互式日志查看器"
            ;;
        "")
            interactive_logs_viewer
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
