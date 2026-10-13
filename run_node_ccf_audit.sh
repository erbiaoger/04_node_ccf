#!/usr/bin/env bash
# 结果审计：每次只读一份MAT，检查完整性、源接收反转、有效计数和总叠加来源。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_shell_common.sh"
INPUT_DIR="${1:-$SCRIPT_DIR/outputs/node_ccf_spectral_v1}" # 完成的30分钟MAT目录
RESTACK_DIR="${2:-${INPUT_DIR%/}_restack}" # 全时段总叠加MAT目录
AUDIT_JSON="${3:-$SCRIPT_DIR/outputs/node_ccf_audit.json}" # 质量报告JSON输出
run_uv_python "$SCRIPT_DIR/scripts/audit_node_ccf.py" --input-dir "$INPUT_DIR" \
  --restack-dir "$RESTACK_DIR" --output-json "$AUDIT_JSON"
