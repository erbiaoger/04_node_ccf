#!/usr/bin/env bash
# 节点 SAC 互相关一键入口：参数说明见下方默认值。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_shell_common.sh"
CSV="${1:-}" # 排序 CSV；若配置含 station_csv 可省略
DATA_DIR="${2:-}" # 可由配置中的 node_data_dir 提供；空值时使用 CLI 默认节点目录
OUT_DIR="${3:-}" # 可由配置中的 node_output_dir 提供；空值时使用 CLI 默认输出目录
CONFIG="${CONFIG:-${SCRIPT_DIR}/config/cc_config.jsonc}" # 节点配置；内部继承 DAS cc_config.jsonc

# 外部 DAS 仓库运行时会切换工作目录；先把显式相对参数固定为启动脚本所在
# 工作目录下的绝对路径，避免 CSV、数据目录和自定义 CONFIG 被错误解析。
_node_abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

CONFIG="$(_node_abs_path "${CONFIG}")"
[[ -n "${CSV}" ]] && CSV="$(_node_abs_path "${CSV}")"
[[ -n "${DATA_DIR}" ]] && DATA_DIR="$(_node_abs_path "${DATA_DIR}")"
[[ -n "${OUT_DIR}" ]] && OUT_DIR="$(_node_abs_path "${OUT_DIR}")"

# 只把显式传入的位置参数作为覆盖项传给 CLI；留空时由 cc_config.jsonc 提供默认值。
# 这样 station_csv/node_output_dir 中的相对路径会按 04_node_ccf 仓库解析，
# 不会因为运行环境切换到外部 DASQT_REPO_DIR 而失效。
ARGS=(--config "${CONFIG}")
[[ -n "${CSV}" ]] && ARGS+=(--csv "${CSV}")
[[ -n "${DATA_DIR}" ]] && ARGS+=(--data-dir "${DATA_DIR}")
[[ -n "${OUT_DIR}" ]] && ARGS+=(--output-dir "${OUT_DIR}")
[[ -n "${READ_MODE:-}" ]] && ARGS+=(--read-mode "${READ_MODE}") # preload 或 window
[[ "${INCLUDE_AUTOCORR:-0}" == "1" ]] && ARGS+=(--include-autocorr) # 是否计算自相关
[[ -n "${PAIR_MODE:-}" ]] && ARGS+=(--pair-mode "${PAIR_MODE}") # all_pairs 或 sliding
[[ -n "${OFFSET_M:-}" ]] && ARGS+=(--offset-m "${OFFSET_M}") # sliding 接收范围，米
[[ -n "${DSHOT_M:-}" ]] && ARGS+=(--dshot-m "${DSHOT_M}") # sliding 震源步长，米
[[ -n "${CC_LEN:-}" ]] && ARGS+=(--cc-len "${CC_LEN}") # 短窗长度，秒
[[ -n "${STEP_S:-}" ]] && ARGS+=(--step-s "${STEP_S}") # 短窗步长，秒
[[ -n "${CC_BACKEND:-}" ]] && ARGS+=(--cc-backend "${CC_BACKEND}") # auto/cpu/torch/cupy/cpp
[[ -n "${CC_DEVICE:-}" ]] && ARGS+=(--cc-device "${CC_DEVICE}") # auto/cpu/cuda/mps
[[ -n "${CC_BATCH_CHUNKS:-}" ]] && ARGS+=(--cc-batch-chunks "${CC_BATCH_CHUNKS}") # GPU 一批短窗数
[[ -n "${MINUTE_STACK_S:-}" ]] && ARGS+=(--minute-stack-s "${MINUTE_STACK_S}") # 第一层分钟段长度，秒
[[ -n "${MAXLAG:-}" ]] && ARGS+=(--maxlag "${MAXLAG}") # 最大延迟，秒
[[ -n "${SAVE_EVERY:-}" ]] && ARGS+=(--save-every "${SAVE_EVERY}") # 每多少分钟保存一个 MAT
[[ -n "${START_UTC:-}" ]] && ARGS+=(--start-utc "${START_UTC}") # 可选 UTC 起始时间
[[ -n "${END_UTC:-}" ]] && ARGS+=(--end-utc "${END_UTC}") # 可选 UTC 结束时间
run_uv_python "${SCRIPT_DIR}/scripts/cli.py" "${ARGS[@]}"
