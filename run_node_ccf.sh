#!/usr/bin/env bash
# 节点全配对入口：项目 .venv + uv，C++/CUDA 默认；--help显示CLI全部参数。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_shell_common.sh"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  run_uv_python "${SCRIPT_DIR}/scripts/cli.py" --help
  exit 0
fi
CSV="${1:-}" # 台站排序CSV；空值使用配置
DATA_DIR="${2:-}" # SAC数据目录；空值使用配置
OUT_DIR="${3:-}" # 输出目录；空值使用配置，新口径默认独立spectral_v1目录
CONFIG="${CONFIG:-${SCRIPT_DIR}/config/cc_config.jsonc}" # 独立节点JSONC配置
_node_abs_path() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }
ARGS=(--config "$(_node_abs_path "$CONFIG")")
[[ -n "$CSV" ]] && ARGS+=(--csv "$(_node_abs_path "$CSV")")
[[ -n "$DATA_DIR" ]] && ARGS+=(--data-dir "$(_node_abs_path "$DATA_DIR")")
[[ -n "$OUT_DIR" ]] && ARGS+=(--output-dir "$(_node_abs_path "$OUT_DIR")")
[[ -n "${READ_MODE:-}" ]] && ARGS+=(--read-mode "$READ_MODE") # auto/preload/block
[[ -n "${CC_BACKEND:-}" ]] && ARGS+=(--cc-backend "$CC_BACKEND") # cpp_cuda/cpu/torch
[[ -n "${CC_DEVICE:-}" ]] && ARGS+=(--cc-device "$CC_DEVICE") # CUDA原生必须cuda
[[ -n "${CC_LEN:-}" ]] && ARGS+=(--cc-len "$CC_LEN") # 短窗秒数，默认10
[[ -n "${STEP_S:-}" ]] && ARGS+=(--step-s "$STEP_S") # 步长秒数，默认5
[[ -n "${MINUTE_STACK_S:-}" ]] && ARGS+=(--minute-stack-s "$MINUTE_STACK_S") # 第一层块秒数
[[ -n "${SAVE_EVERY:-}" ]] && ARGS+=(--save-every "$SAVE_EVERY") # 每多少第一层块保存
[[ -n "${MAXLAG:-}" ]] && ARGS+=(--maxlag "$MAXLAG") # 最大延迟秒数
[[ -n "${FREQMIN:-}" ]] && ARGS+=(--freqmin "$FREQMIN") # 频带下界Hz
[[ -n "${FREQMAX:-}" ]] && ARGS+=(--freqmax "$FREQMAX") # 频带上界Hz
[[ -n "${TIME_DOWNSAMPLE:-}" ]] && ARGS+=(--time-downsample "$TIME_DOWNSAMPLE") # GPU降采样整数因子
[[ -n "${CC_BATCH_CHUNKS:-}" ]] && ARGS+=(--cc-batch-chunks "$CC_BATCH_CHUNKS") # 最大窗口批次数
[[ -n "${CC_PAIR_BATCH:-}" ]] && ARGS+=(--cc-pair-batch "$CC_PAIR_BATCH") # 最大配对批次数，0自动
[[ -n "${CC_MEMORY_BUDGET_BYTES:-}" ]] && ARGS+=(--cc-memory-budget-bytes "$CC_MEMORY_BUDGET_BYTES") # 工作区字节
[[ -n "${CC_CUDA_DEVICE:-}" ]] && ARGS+=(--cc-cuda-device "$CC_CUDA_DEVICE") # GPU编号
[[ -n "${STACKING_METHOD:-}" ]] && ARGS+=(--stacking-method "$STACKING_METHOD") # pws/linear
[[ -n "${START_UTC:-}" ]] && ARGS+=(--start-utc "$START_UTC") # 可选UTC起点，带Z时区
[[ -n "${END_UTC:-}" ]] && ARGS+=(--end-utc "$END_UTC") # 可选UTC终点，半开区间
[[ "${RESUME:-1}" == "0" ]] && ARGS+=(--no-resume) # 0禁止复用现有任务目录
[[ "${PREFLIGHT:-0}" == "1" ]] && ARGS+=(--preflight) # 1只做输入预检
[[ "${PRINT_OUTPUT_DIR:-0}" == "1" ]] && ARGS+=(--print-output-dir) # 完整入口内部使用
run_uv_python "${SCRIPT_DIR}/scripts/cli.py" "${ARGS[@]}"
