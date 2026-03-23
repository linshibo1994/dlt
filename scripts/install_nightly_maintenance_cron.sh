#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAINTENANCE_ENV_FILE="${MAINTENANCE_ENV_FILE:-${SCRIPT_DIR}/maintenance.env}"

if [[ -f "${MAINTENANCE_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${MAINTENANCE_ENV_FILE}"
fi

CRON_SCHEDULE="${CRON_SCHEDULE:-30 0 * * *}"
CRON_TASK_TAG="${CRON_TASK_TAG:-DLT_NIGHTLY_MAINTENANCE}"
CRON_LOG_FILE="${CRON_LOG_FILE:-${PROJECT_ROOT}/logs/maintenance/cron.log}"
RUN_SCRIPT="${SCRIPT_DIR}/run_nightly_maintenance.sh"

as_abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

CRON_LOG_FILE="$(as_abs_path "${CRON_LOG_FILE}")"
mkdir -p "$(dirname "${CRON_LOG_FILE}")"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  chmod +x "${RUN_SCRIPT}"
fi

if ! command -v crontab >/dev/null 2>&1; then
  echo "未检测到 crontab 命令，请先安装 cron/crontab"
  exit 1
fi

cron_command="/bin/bash '${RUN_SCRIPT}' >> '${CRON_LOG_FILE}' 2>&1"
cron_line="${CRON_SCHEDULE} ${cron_command} # ${CRON_TASK_TAG}"

current_crontab="$(crontab -l 2>/dev/null || true)"
filtered_crontab="$(printf '%s\n' "${current_crontab}" | grep -v "${CRON_TASK_TAG}" || true)"

if [[ -n "${filtered_crontab}" ]]; then
  new_crontab="${filtered_crontab}"$'\n'"${cron_line}"
else
  new_crontab="${cron_line}"
fi

if ! printf '%s\n' "${new_crontab}" | crontab -; then
  echo "写入 crontab 失败，请手动执行以下命令安装:"
  echo "(crontab -l 2>/dev/null | grep -v '${CRON_TASK_TAG}'; echo \"${cron_line}\") | crontab -"
  exit 1
fi

echo "已安装定时任务:"
echo "  ${cron_line}"
echo "查看当前任务: crontab -l"
