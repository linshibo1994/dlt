#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAINTENANCE_ENV_FILE="${MAINTENANCE_ENV_FILE:-${SCRIPT_DIR}/maintenance.env}"

if [[ -f "${MAINTENANCE_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${MAINTENANCE_ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
NIGHTLY_MAINTENANCE_LOG_FILE="${NIGHTLY_MAINTENANCE_LOG_FILE:-${PROJECT_ROOT}/logs/maintenance/nightly_maintenance.log}"
as_abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

NIGHTLY_MAINTENANCE_LOG_FILE="$(as_abs_path "${NIGHTLY_MAINTENANCE_LOG_FILE}")"
mkdir -p "$(dirname "${NIGHTLY_MAINTENANCE_LOG_FILE}")"

log() {
  printf '[%s] [夜间维护] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${NIGHTLY_MAINTENANCE_LOG_FILE}"
}

main() {
  log "开始执行夜间维护任务..."
  log "开始更新最新开奖结果..."
  if "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/auto_update_latest_data.py" --once >> "${NIGHTLY_MAINTENANCE_LOG_FILE}" 2>&1; then
    log "最新开奖结果更新完成"
  else
    log "最新开奖结果更新失败，继续执行后续维护任务"
  fi
  MAINTENANCE_ENV_FILE="${MAINTENANCE_ENV_FILE}" "${SCRIPT_DIR}/cleanup_local_artifacts.sh"
  MAINTENANCE_ENV_FILE="${MAINTENANCE_ENV_FILE}" "${SCRIPT_DIR}/cleanup_docker_resources.sh"
  log "夜间维护任务执行完成"
}

main "$@"
