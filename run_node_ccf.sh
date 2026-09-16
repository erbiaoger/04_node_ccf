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
# 独立仓库运行时，当前目录是 DASQT_REPO_DIR；因此将本仓库的默认输入/输出
# 绝对路径显式传给 CLI，避免把相对路径误解析到外部 DAS 仓库。
[[ -z "${CSV}" ]] && CSV="${SCRIPT_DIR}/config/line_380_stations.csv"
[[ -z "${OUT_DIR}" ]] && OUT_DIR="${SCRIPT_DIR}/outputs/node_ccf"
ARGS=(--config "${CONFIG}" --csv "${CSV}" --output-dir "${OUT_DIR}")
[[ -n "${CSV}" ]] && ARGS+=(--csv "${CSV}")
[[ -n "${DATA_DIR}" ]] && ARGS+=(--data-dir "${DATA_DIR}")
[[ -n "${OUT_DIR}" ]] && ARGS+=(--output-dir "${OUT_DIR}")
[[ -n "${READ_MODE:-}" ]] && ARGS+=(--read-mode "${READ_MODE}") # preload 或 window
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
