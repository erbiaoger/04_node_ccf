#!/usr/bin/env bash
# 节点 CCF 完整一键流程：计算 -> 二次叠加 -> 节点坐标绘图。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_OUTPUT_DIR="${3:-${SCRIPT_DIR}/outputs/node_ccf}"
RESTACK_OUTPUT_DIR="${COMPUTE_OUTPUT_DIR%/}_restack"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage:
  ./run_node_ccf_all.sh [station_csv] [node_data_dir] [node_output_dir]

先运行 run_node_ccf.sh，再运行 run_node_ccf_stack.sh，完成计算、二次叠加和绘图。
后处理输出目录默认为 node_output_dir 后追加 _restack。
EOF
  exit 0
fi

# 先执行第一层节点互相关；可通过位置参数覆盖 CSV、数据目录和输出目录。
"${SCRIPT_DIR}/run_node_ccf.sh" "$@"

# 计算脚本默认输出到 outputs/node_ccf，后处理脚本默认从该目录读取。
# 如需修改后处理选项，直接单独运行 run_node_ccf_stack.sh。
"${SCRIPT_DIR}/run_node_ccf_stack.sh" \
  --input-dir "${COMPUTE_OUTPUT_DIR}" \
  --output-dir "${RESTACK_OUTPUT_DIR}"
