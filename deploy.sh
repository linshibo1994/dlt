#!/bin/bash
# DLT Prediction System - 一键部署脚本
# 使用方法: ./deploy.sh [start|stop|restart|logs|build]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 帮助信息
show_help() {
    echo "大乐透智能预测系统 - 部署脚本"
    echo ""
    echo "使用方法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start     启动所有服务"
    echo "  stop      停止所有服务"
    echo "  restart   重启所有服务"
    echo "  logs      查看服务日志"
    echo "  build     重新构建镜像"
    echo "  status    查看服务状态"
    echo "  clean     清理未使用的镜像和容器"
    echo "  help      显示此帮助信息"
}

# 启动服务
start_services() {
    echo -e "${GREEN}启动服务...${NC}"
    docker compose up -d
    echo -e "${GREEN}服务启动完成!${NC}"
    echo ""
    echo "访问地址:"
    echo "  前端: http://localhost"
    echo "  后端API: http://localhost:6000/api"
    echo "  健康检查: http://localhost:6000/api/system/health"
}

# 停止服务
stop_services() {
    echo -e "${YELLOW}停止服务...${NC}"
    docker compose down
    echo -e "${GREEN}服务已停止${NC}"
}

# 重启服务
restart_services() {
    echo -e "${YELLOW}重启服务...${NC}"
    docker compose restart
    echo -e "${GREEN}服务重启完成!${NC}"
}

# 查看日志
view_logs() {
    docker compose logs -f
}

# 构建镜像（构建后自动清理废弃镜像和构建缓存）
build_images() {
    echo -e "${GREEN}构建镜像...${NC}"
    docker compose build --no-cache
    echo -e "${GREEN}镜像构建完成!${NC}"

    # 自动清理废弃镜像（dangling images）
    DANGLING=$(docker images -f "dangling=true" -q 2>/dev/null)
    if [ -n "$DANGLING" ]; then
        DANGLING_SIZE=$(docker images -f "dangling=true" --format "{{.Size}}" | head -5)
        echo -e "${YELLOW}清理废弃镜像...${NC}"
        docker image prune -f
        echo -e "${GREEN}废弃镜像已清理${NC}"
    fi

    # 自动清理构建缓存
    CACHE_SIZE=$(docker builder du 2>/dev/null | tail -1 | awk '{print $NF}' || echo "0")
    if [ "$CACHE_SIZE" != "0" ] && [ "$CACHE_SIZE" != "0B" ]; then
        echo -e "${YELLOW}清理构建缓存...${NC}"
        docker builder prune -f
        echo -e "${GREEN}构建缓存已清理${NC}"
    fi
}

# 查看状态
check_status() {
    echo -e "${GREEN}服务状态:${NC}"
    docker compose ps
}

# 清理
clean_up() {
    echo -e "${YELLOW}清理未使用的Docker资源...${NC}"
    echo ""

    # 显示可清理资源概览
    DANGLING_COUNT=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l | tr -d ' ')
    CACHE_INFO=$(docker builder du --verbose 2>/dev/null | tail -1 || echo "无法获取")
    echo "  废弃镜像数量: ${DANGLING_COUNT}"
    echo "  构建缓存: ${CACHE_INFO}"
    echo ""

    echo -e "${RED}警告: 此操作将删除废弃镜像、构建缓存和未使用的网络${NC}"
    read -p "确定要继续吗? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}[1/3] 清理废弃镜像...${NC}"
        docker image prune -f
        echo -e "${YELLOW}[2/3] 清理构建缓存...${NC}"
        docker builder prune -f
        echo -e "${YELLOW}[3/3] 清理未使用的网络...${NC}"
        docker network prune -f
        echo ""
        echo -e "${GREEN}清理完成! 当前磁盘使用:${NC}"
        docker system df
    else
        echo "已取消"
    fi
}

# 主函数
main() {
    case "${1:-help}" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            view_logs
            ;;
        build)
            build_images
            ;;
        status)
            check_status
            ;;
        clean)
            clean_up
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
