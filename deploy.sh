#!/bin/bash

# 大乐透智能预测系统 - Docker 部署脚本
# 使用方法: ./deploy.sh [command]

set -e

# 脚本路径信息
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="deploy.sh"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_FILE}"

# 切换到脚本所在目录执行，避免在其他目录调用时路径不一致
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 打印带颜色的消息
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装"
        exit 1
    fi
}

# 使用正确的 docker-compose 命令
get_compose_cmd() {
    if docker compose version &> /dev/null; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

COMPOSE_CMD=$(get_compose_cmd)

# 同步远程 deploy.sh，若脚本更新则重新执行 update，确保后续逻辑使用最新脚本
sync_remote_deploy_script() {
    if [[ "${DEPLOY_SCRIPT_RELOADED:-0}" == "1" ]]; then
        return
    fi

    local branch
    branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
    if [[ -z "$branch" || "$branch" == "HEAD" ]]; then
        branch="main"
    fi

    log_step "0/4 同步部署脚本..."
    if ! git fetch origin "$branch"; then
        log_error "无法从远程拉取分支 $branch，请检查网络或权限"
        exit 1
    fi

    local before_hash=""
    local after_hash=""
    if [[ -f "$SCRIPT_PATH" ]]; then
        before_hash=$(shasum -a 256 "$SCRIPT_PATH" | awk '{print $1}')
    fi

    if ! git checkout "origin/$branch" -- "$SCRIPT_FILE"; then
        log_error "同步远程 deploy.sh 失败"
        exit 1
    fi
    chmod +x "$SCRIPT_PATH" || true

    if [[ -f "$SCRIPT_PATH" ]]; then
        after_hash=$(shasum -a 256 "$SCRIPT_PATH" | awk '{print $1}')
    fi

    if [[ -n "$before_hash" && -n "$after_hash" && "$before_hash" != "$after_hash" ]]; then
        log_warn "检测到远程 deploy.sh 更新，已覆盖本地脚本并重新执行 update..."
        DEPLOY_SCRIPT_RELOADED=1 exec bash "$SCRIPT_PATH" update
    fi

    log_info "部署脚本已与远程保持一致"
}

# 拉取最新代码 + 重新构建 + 部署
update() {
    log_info "开始执行完整更新流程..."
    sync_remote_deploy_script
    local stash_ref=""
    local stash_msg=""
    local has_local_changes=false

    # 检测并暂存本地改动，避免 git pull 因覆盖失败
    if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        has_local_changes=true
        stash_msg="deploy-auto-stash-$(date +%Y%m%d_%H%M%S)"
        log_warn "检测到本地未提交改动，自动暂存后再更新..."
        git stash push -u -m "$stash_msg" > /dev/null
        stash_ref=$(git stash list | grep -m1 "$stash_msg" | cut -d: -f1 || true)

        if [[ -z "$stash_ref" ]]; then
            log_error "自动暂存失败，请手动执行 git stash 后重试"
            exit 1
        fi
        log_info "已自动暂存本地改动: $stash_ref"
    fi

    # 1. 拉取最新代码
    log_step "1/4 拉取最新代码..."
    if ! git pull --ff-only; then
        log_warn "fast-forward 合并失败，尝试 rebase..."
        if ! git pull --rebase; then
            if [[ "$has_local_changes" == true ]]; then
                log_warn "更新失败，本地改动仍保存在: $stash_ref"
                log_warn "可用以下命令恢复: git stash apply --index $stash_ref"
            fi
            log_error "代码拉取失败，请手动解决冲突后重试"
            exit 1
        fi
    fi
    log_info "代码已更新至: $(git log --oneline -1)"

    # 更新成功后恢复暂存改动
    if [[ "$has_local_changes" == true ]]; then
        log_step "恢复本地改动..."
        if git stash apply --index "$stash_ref"; then
            git stash drop "$stash_ref" > /dev/null || true
            log_info "本地改动已恢复"
        else
            log_error "本地改动恢复时发生冲突，请手动处理"
            log_warn "你的改动仍保存在: $stash_ref"
            log_warn "冲突处理后可删除暂存: git stash drop $stash_ref"
            exit 1
        fi
    fi

    # 2. 停止当前服务
    log_step "2/4 停止当前服务..."
    $COMPOSE_CMD down || true

    # 3. 重新构建镜像
    log_step "3/4 重新构建镜像..."
    $COMPOSE_CMD build --no-cache

    # 4. 启动服务
    log_step "4/4 启动服务..."
    $COMPOSE_CMD up -d

    # 自动清理废弃镜像和构建缓存
    log_info "清理废弃镜像和构建缓存..."
    docker image prune -f 2>/dev/null || true
    docker builder prune -f 2>/dev/null || true

    log_info "更新部署完成!"
    log_info "前端: http://localhost"
    log_info "后端API: http://localhost:6000/api"
    log_info "健康检查: http://localhost:6000/api/system/health"
    echo ""
    status
}

# 构建镜像
build() {
    local no_cache=""
    if [[ "$1" == "--no-cache" ]]; then
        no_cache="--no-cache"
        log_info "开始构建 Docker 镜像 (无缓存)..."
    else
        log_info "开始构建 Docker 镜像..."
    fi
    $COMPOSE_CMD build $no_cache
    log_info "构建完成!"

    # 自动清理废弃镜像（dangling images）
    local dangling
    dangling=$(docker images -f "dangling=true" -q 2>/dev/null)
    if [[ -n "$dangling" ]]; then
        log_info "清理废弃镜像..."
        docker image prune -f
        log_info "废弃镜像已清理"
    fi

    # 自动清理构建缓存
    local cache_size
    cache_size=$(docker builder du 2>/dev/null | tail -1 | awk '{print $NF}' || echo "0")
    if [[ "$cache_size" != "0" && "$cache_size" != "0B" ]]; then
        log_info "清理构建缓存..."
        docker builder prune -f
        log_info "构建缓存已清理"
    fi
}

# 启动服务
start() {
    log_info "启动服务..."
    $COMPOSE_CMD up -d
    log_info "服务已启动!"
    log_info "前端: http://localhost"
    log_info "后端API: http://localhost:6000/api"
    log_info "健康检查: http://localhost:6000/api/system/health"
}

# 停止服务
stop() {
    log_info "停止服务..."
    $COMPOSE_CMD down
    log_info "服务已停止"
}

# 重启服务
restart() {
    log_info "重启服务..."
    $COMPOSE_CMD restart
    log_info "服务已重启"
}

# 查看日志（支持指定服务名）
logs() {
    $COMPOSE_CMD logs -f "$@"
}

# 交互式清理 Docker 资源
clean() {
    echo ""
    echo "Docker 资源清理选项:"
    echo "  1) 清理悬空镜像和已停止的容器 (安全，推荐)"
    echo "  2) 清理所有未使用的镜像 (包括未被容器引用的镜像)"
    echo "  3) 清理构建缓存"
    echo "  4) 全部清理 (悬空镜像 + 未使用镜像 + 构建缓存 + 未使用网络)"
    echo "  0) 取消"
    echo ""
    read -p "请选择 [0-4]: " choice

    case "$choice" in
        1)
            log_info "清理悬空镜像..."
            local dangling_images
            dangling_images=$(docker images -f "dangling=true" -q 2>/dev/null)
            if [[ -n "$dangling_images" ]]; then
                echo "$dangling_images" | xargs docker rmi 2>/dev/null || true
                log_info "悬空镜像已清理"
            else
                log_info "没有悬空镜像"
            fi

            log_info "清理已停止的容器..."
            local stopped_containers
            stopped_containers=$(docker ps -a -f "status=exited" -f "status=dead" -q 2>/dev/null)
            if [[ -n "$stopped_containers" ]]; then
                echo "$stopped_containers" | xargs docker rm 2>/dev/null || true
                log_info "已停止的容器已清理"
            else
                log_info "没有已停止的容器"
            fi
            ;;
        2)
            log_warn "将清理所有未被容器使用的镜像..."
            read -p "确定要继续吗? (y/N): " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                docker image prune -a -f
                log_info "未使用的镜像已清理"
            else
                log_info "已取消"
            fi
            ;;
        3)
            log_info "清理 Docker 构建缓存..."
            docker builder prune -f
            log_info "构建缓存已清理"
            ;;
        4)
            log_warn "将执行全面清理 (悬空镜像 + 未使用镜像 + 构建缓存 + 未使用网络)..."
            log_warn "注意: 不会删除正在运行的容器和已挂载的卷"
            read -p "确定要继续吗? (y/N): " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                log_step "清理已停止的容器..."
                docker container prune -f

                log_step "清理未使用的镜像..."
                docker image prune -a -f

                log_step "清理构建缓存..."
                docker builder prune -f

                log_step "清理未使用的网络..."
                docker network prune -f

                log_info "全面清理完成!"
                echo ""
                log_info "当前磁盘使用情况:"
                docker system df
            else
                log_info "已取消"
            fi
            ;;
        0|"")
            log_info "已取消"
            ;;
        *)
            log_error "无效选项: $choice"
            ;;
    esac
}

# 状态检查
status() {
    log_info "服务状态:"
    $COMPOSE_CMD ps
}

# 帮助信息
show_help() {
    echo "大乐透智能预测系统 - Docker 部署脚本"
    echo ""
    echo "使用方法: $0 <command>"
    echo ""
    echo "命令:"
    echo "  update             拉取最新代码 + 重新构建 + 部署 (一键更新)"
    echo "  build [--no-cache] 构建 Docker 镜像 (默认使用缓存)"
    echo "  start              启动所有服务"
    echo "  stop               停止所有服务"
    echo "  restart            重启所有服务"
    echo "  logs [service]     查看日志 (可加服务名: logs backend)"
    echo "  status             查看服务状态"
    echo "  clean              交互式清理 Docker 资源"
    echo ""
    echo "别名:"
    echo "  up = start, down = stop"
    echo ""
}

# 主函数
main() {
    check_docker

    case "${1:-help}" in
        update)
            update
            ;;
        build)
            shift
            build "$@"
            ;;
        start|up)
            start
            ;;
        stop|down)
            stop
            ;;
        restart)
            restart
            ;;
        logs)
            shift
            logs "$@"
            ;;
        status)
            status
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
