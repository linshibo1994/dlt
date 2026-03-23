#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAINTENANCE_ENV_FILE="${MAINTENANCE_ENV_FILE:-${SCRIPT_DIR}/maintenance.env}"

if [[ -f "${MAINTENANCE_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${MAINTENANCE_ENV_FILE}"
fi

DOCKER_CLEAN_STOPPED_CONTAINERS="${DOCKER_CLEAN_STOPPED_CONTAINERS:-true}"
DOCKER_CLEAN_UNUSED_IMAGES="${DOCKER_CLEAN_UNUSED_IMAGES:-true}"
DOCKER_IMAGE_MODE="${DOCKER_IMAGE_MODE:-all}"
DOCKER_CLEAN_BUILD_CACHE="${DOCKER_CLEAN_BUILD_CACHE:-true}"
DOCKER_BUILDER_MODE="${DOCKER_BUILDER_MODE:-all}"
DOCKER_CLEAN_UNUSED_NETWORKS="${DOCKER_CLEAN_UNUSED_NETWORKS:-true}"
DOCKER_CLEAN_UNUSED_VOLUMES="${DOCKER_CLEAN_UNUSED_VOLUMES:-false}"
DOCKER_PRUNE_FILTER="${DOCKER_PRUNE_FILTER:-}"
DOCKER_CLEANUP_LOG_FILE="${DOCKER_CLEANUP_LOG_FILE:-${PROJECT_ROOT}/logs/maintenance/docker_cleanup.log}"

log() {
  printf '[%s] [Docker清理] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${DOCKER_CLEANUP_LOG_FILE}"
}

is_true() {
  local value="${1:-false}"
  [[ "${value,,}" = "true" || "${value}" = "1" || "${value,,}" = "yes" || "${value,,}" = "y" ]]
}

as_abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

DOCKER_CLEANUP_LOG_FILE="$(as_abs_path "${DOCKER_CLEANUP_LOG_FILE}")"
mkdir -p "$(dirname "${DOCKER_CLEANUP_LOG_FILE}")"

run_docker_cmd() {
  log "执行: $*"
  "$@"
}

main() {
  local -a cmd

  if ! command -v docker >/dev/null 2>&1; then
    log "未检测到 docker 命令，跳过 Docker 清理"
    exit 0
  fi

  if ! docker info >/dev/null 2>&1; then
    log "Docker daemon 不可用，跳过 Docker 清理"
    exit 0
  fi

  log "开始清理 Docker 资源..."

  if is_true "${DOCKER_CLEAN_STOPPED_CONTAINERS}"; then
    cmd=(docker container prune -f)
    [[ -n "${DOCKER_PRUNE_FILTER}" ]] && cmd+=(--filter "${DOCKER_PRUNE_FILTER}")
    run_docker_cmd "${cmd[@]}"
  fi

  if is_true "${DOCKER_CLEAN_UNUSED_IMAGES}"; then
    cmd=(docker image prune -f)
    if [[ "${DOCKER_IMAGE_MODE}" = "all" ]]; then
      cmd=(docker image prune -a -f)
    fi
    [[ -n "${DOCKER_PRUNE_FILTER}" ]] && cmd+=(--filter "${DOCKER_PRUNE_FILTER}")
    run_docker_cmd "${cmd[@]}"
  fi

  if is_true "${DOCKER_CLEAN_BUILD_CACHE}"; then
    cmd=(docker builder prune -f)
    if [[ "${DOCKER_BUILDER_MODE}" = "all" ]]; then
      cmd=(docker builder prune -a -f)
    fi
    [[ -n "${DOCKER_PRUNE_FILTER}" ]] && cmd+=(--filter "${DOCKER_PRUNE_FILTER}")
    run_docker_cmd "${cmd[@]}"
  fi

  if is_true "${DOCKER_CLEAN_UNUSED_NETWORKS}"; then
    cmd=(docker network prune -f)
    [[ -n "${DOCKER_PRUNE_FILTER}" ]] && cmd+=(--filter "${DOCKER_PRUNE_FILTER}")
    run_docker_cmd "${cmd[@]}"
  fi

  if is_true "${DOCKER_CLEAN_UNUSED_VOLUMES}"; then
    cmd=(docker volume prune -f)
    [[ -n "${DOCKER_PRUNE_FILTER}" ]] && cmd+=(--filter "${DOCKER_PRUNE_FILTER}")
    run_docker_cmd "${cmd[@]}"
  else
    log "已跳过 volume 清理 (DOCKER_CLEAN_UNUSED_VOLUMES=false)"
  fi

  log "清理后磁盘占用:"
  docker system df | tee -a "${DOCKER_CLEANUP_LOG_FILE}"
  log "Docker 资源清理完成"
}

main "$@"
