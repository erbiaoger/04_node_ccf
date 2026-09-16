#!/usr/bin/env bash
# 节点 CCF 后处理：复用 DAS 的二次叠加，并按节点 profileX 坐标绘图。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_shell_common.sh"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --input-dir PATH        第一层 cc_stack_*.mat 目录；默认：${SCRIPT_DIR}/outputs/node_ccf
  --output-dir PATH       restack MAT 和图片输出目录；默认：${SCRIPT_DIR}/outputs/node_ccf_restack
  --glob PATTERN          MAT 文件模式；默认：cc_stack_*.mat
  --stack-size VALUE      二次叠加文件数；默认：None（全部文件）
  --interval VALUE        滑动窗口步长；默认：None（只输出一组）
  --start-index N         起始 MAT 索引；默认：0
  --stack-method NAME     pws | robust | linear | mean；默认：pws
  --cc-backend NAME       auto | cpu | torch；默认：torch
  --cc-device NAME        auto | cpu | mps | cuda；默认：cuda
  --cc-dtype NAME         float32 | float64；默认：float32
  --plot-name NAME        汇总图名；默认：node_cc_preview.png
  --no-save-shot-figures  不保存逐源节点图片，只保存汇总图
  --help                  显示帮助信息
EOF
}

INPUT_DIR="${SCRIPT_DIR}/outputs/node_ccf"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/node_ccf_restack"
GLOB_PATTERN="cc_stack_*.mat"
STACK_SIZE="None"
INTERVAL="None"
START_INDEX="0"
STACK_METHOD="pws"
CC_BACKEND="torch"
CC_DEVICE="cuda"
CC_DTYPE="float32"
PLOT_NAME="node_cc_preview.png"
SAVE_SHOT_FIGURES="1"

_node_abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --glob) GLOB_PATTERN="$2"; shift 2 ;;
    --stack-size) STACK_SIZE="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --stack-method) STACK_METHOD="$2"; shift 2 ;;
    --cc-backend) CC_BACKEND="$2"; shift 2 ;;
    --cc-device) CC_DEVICE="$2"; shift 2 ;;
    --cc-dtype) CC_DTYPE="$2"; shift 2 ;;
    --plot-name) PLOT_NAME="$2"; shift 2 ;;
    --no-save-shot-figures) SAVE_SHOT_FIGURES="0"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "错误：未知参数：$1" >&2; usage; exit 1 ;;
  esac
done

INPUT_DIR="$(_node_abs_path "${INPUT_DIR}")"
OUTPUT_DIR="$(_node_abs_path "${OUTPUT_DIR}")"
ensure_dasqt_env
STACK_SCRIPT="${DASQT_REPO_DIR}/examples/03_ccf/scripts/stack_saved_cc_mat.py"
if [[ ! -f "${STACK_SCRIPT}" ]]; then
  echo "错误：未找到 DAS 二次叠加脚本：${STACK_SCRIPT}" >&2
  exit 1
fi

# 先复用 DAS 脚本生成 restack_*.mat；关闭其默认绘图，避免用均匀 dx
# 覆盖节点真实坐标图。节点专用 plot_node_ccf.py 随后读取 profileX 出图。
STACK_ARGS=(
  "${STACK_SCRIPT}"
  --input-dir "${INPUT_DIR}"
  --glob "${GLOB_PATTERN}"
  --output-dir "${OUTPUT_DIR}"
  --stack-size "${STACK_SIZE}"
  --interval "${INTERVAL}"
  --start-index "${START_INDEX}"
  --stack-method "${STACK_METHOD}"
  --cc-backend "${CC_BACKEND}"
  --cc-device "${CC_DEVICE}"
  --cc-dtype "${CC_DTYPE}"
  --no-save-summary-plot
  --no-save-shot-figures
)
run_uv_python "${STACK_ARGS[@]}"

PLOT_ARGS=(
  "${SCRIPT_DIR}/scripts/plot_node_ccf.py"
  --input-dir "${OUTPUT_DIR}"
  --profile-source-dir "${INPUT_DIR}"
  --plot-name "${PLOT_NAME}"
)
[[ "${SAVE_SHOT_FIGURES}" == "0" ]] && PLOT_ARGS+=(--no-save-shot-figures)
run_uv_python "${PLOT_ARGS[@]}"
