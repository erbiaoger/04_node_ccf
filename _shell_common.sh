#!/usr/bin/env bash
# 节点 CCF 与 DAS CCF 共用的运行环境辅助函数。

set -euo pipefail

# 可通过 DASQT_REPO_DIR 指向实际的 dasQt/DAS_Procee_Show 源码仓库。
# 不设置时依次尝试：当前源码树、03_ccf 使用的服务器路径和本机路径。
if [[ -z "${DASQT_REPO_DIR:-}" ]]; then
  _NODE_REPO_DIR="${SCRIPT_DIR}"
  _DASQT_CANDIDATES=(
    "${_NODE_REPO_DIR}/../.."
    "/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show"
    "/Users/zhangzhiyu/MyProjects/dasQt"
  )
  for _candidate in "${_DASQT_CANDIDATES[@]}"; do
    if [[ -d "${_candidate}/dasqt" && -f "${_candidate}/.venv/bin/activate" ]]; then
      DASQT_REPO_DIR="${_candidate}"
      break
    fi
  done
fi

DASQT_VENV_ACTIVATE="${DASQT_VENV_ACTIVATE:-${DASQT_REPO_DIR:-}/.venv/bin/activate}"

ensure_dasqt_env() {
  if [[ -z "${DASQT_REPO_DIR:-}" || ! -d "${DASQT_REPO_DIR}" ]]; then
    echo "错误：未找到 DAS/dasQt 源码仓库。" >&2
    echo "请设置 DASQT_REPO_DIR，例如：" >&2
    echo "  DASQT_REPO_DIR=/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show ./run_node_ccf.sh" >&2
    exit 1
  fi
  if [[ ! -f "${DASQT_VENV_ACTIVATE}" ]]; then
    echo "错误：未找到项目 Python 环境：${DASQT_VENV_ACTIVATE}" >&2
    echo "请在 DASQT_REPO_DIR 下执行：uv sync" >&2
    exit 1
  fi
}

run_uv_python() {
  ensure_dasqt_env
  (
    cd "${DASQT_REPO_DIR}"
    # shellcheck disable=SC1090
    source "${DASQT_VENV_ACTIVATE}"
    uv run --no-sync python "$@"
  )
}
