#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAINTENANCE_ENV_FILE="${MAINTENANCE_ENV_FILE:-${SCRIPT_DIR}/maintenance.env}"

if [[ -f "${MAINTENANCE_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${MAINTENANCE_ENV_FILE}"
fi

LOCAL_CLEAN_DIRS="${LOCAL_CLEAN_DIRS:-logs:cache:artifacts/cache:test_results/logs:test_results/reports:test_results/tmp:tmp}"
LOCAL_PATTERN_ROOTS="${LOCAL_PATTERN_ROOTS:-logs:test_results:artifacts}"
LOCAL_FILE_PATTERNS="${LOCAL_FILE_PATTERNS:-*.log:*.out:*.err:*.tmp}"
LOCAL_CLEAN_EXCLUDE_DIRS="${LOCAL_CLEAN_EXCLUDE_DIRS:-logs/maintenance}"
LOCAL_CLEANUP_LOG_FILE="${LOCAL_CLEANUP_LOG_FILE:-${PROJECT_ROOT}/logs/maintenance/local_cleanup.log}"

log() {
  printf '[%s] [本地清理] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOCAL_CLEANUP_LOG_FILE}"
}

as_abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

normalize_path() {
  local path="$1"
  while [[ "${path}" != "/" && "${path}" == */ ]]; do
    path="${path%/}"
  done
  printf '%s\n' "${path}"
}

build_exclude_paths() {
  local exclude
  EXCLUDE_PATHS=()
  IFS=':' read -r -a exclude_list <<< "${LOCAL_CLEAN_EXCLUDE_DIRS}"
  for exclude in "${exclude_list[@]}"; do
    [[ -n "${exclude}" ]] || continue
    EXCLUDE_PATHS+=("$(normalize_path "$(as_abs_path "${exclude}")")")
  done
  EXCLUDE_PATHS+=("$(normalize_path "$(dirname "${LOCAL_CLEANUP_LOG_FILE}")")")
}

is_excluded_path() {
  local path normalized excluded
  path="$1"
  normalized="$(normalize_path "${path}")"

  for excluded in "${EXCLUDE_PATHS[@]}"; do
    if [[ "${normalized}" == "${excluded}" || "${normalized}" == "${excluded}"/* ]]; then
      return 0
    fi
  done
  return 1
}

LOCAL_CLEANUP_LOG_FILE="$(as_abs_path "${LOCAL_CLEANUP_LOG_FILE}")"
mkdir -p "$(dirname "${LOCAL_CLEANUP_LOG_FILE}")"

is_dangerous_path() {
  local abs_path normalized project_root
  abs_path="$1"
  normalized="$(normalize_path "${abs_path}")"
  project_root="$(normalize_path "${PROJECT_ROOT}")"
  [[ -z "${normalized}" || "${normalized}" = "/" || "${normalized}" = "${HOME}" || "${normalized}" = "${project_root}" ]]
}

cleanup_directories() {
  local dir abs_dir entry
  local -a dir_entries
  IFS=':' read -r -a dir_list <<< "${LOCAL_CLEAN_DIRS}"

  for dir in "${dir_list[@]}"; do
    [[ -n "${dir}" ]] || continue
    abs_dir="$(as_abs_path "${dir}")"

    if is_dangerous_path "${abs_dir}"; then
      log "跳过危险路径: ${abs_dir}"
      continue
    fi

    if [[ -d "${abs_dir}" ]]; then
      log "清理目录内容: ${abs_dir}"
      shopt -s dotglob nullglob
      dir_entries=("${abs_dir}"/*)
      shopt -u dotglob nullglob

      for entry in "${dir_entries[@]}"; do
        if is_excluded_path "${entry}"; then
          log "保留排除路径: ${entry}"
          continue
        fi
        rm -rf "${entry}"
      done
    else
      log "目录不存在，跳过: ${abs_dir}"
    fi
  done
}

cleanup_pattern_files() {
  local root pattern abs_root deleted_count file
  IFS=':' read -r -a root_list <<< "${LOCAL_PATTERN_ROOTS}"
  IFS=':' read -r -a pattern_list <<< "${LOCAL_FILE_PATTERNS}"

  for root in "${root_list[@]}"; do
    [[ -n "${root}" ]] || continue
    abs_root="$(as_abs_path "${root}")"

    if [[ ! -d "${abs_root}" ]]; then
      continue
    fi

    for pattern in "${pattern_list[@]}"; do
      [[ -n "${pattern}" ]] || continue
      deleted_count=0
      while IFS= read -r -d '' file; do
        if is_excluded_path "${file}"; then
          continue
        fi
        rm -f "${file}"
        deleted_count=$((deleted_count + 1))
      done < <(find "${abs_root}" -type f -name "${pattern}" -print0 2>/dev/null)
      if [[ "${deleted_count}" != "0" ]]; then
        log "已删除 ${deleted_count} 个匹配文件 (${pattern})，目录: ${abs_root}"
      fi
    done
  done
}

main() {
  log "开始清理日志、缓存和测试产物..."
  build_exclude_paths
  cleanup_directories
  cleanup_pattern_files
  log "本地文件清理完成"
}

main "$@"
