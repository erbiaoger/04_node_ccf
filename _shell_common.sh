#!/usr/bin/env bash
# 节点 CCF 与 DAS CCF 共用的运行环境辅助函数。

set -euo pipefail

# 可通过 dasqt_repo_dir 指向实际的 dasQt/DAS_Procee_Show 源码仓库。
# 不设置时依次尝试：当前源码树、03_ccf 使用的服务器路径和本机路径。
if [[ -z "${dasqt_repo_dir:-}" ]]; then
  # 内部自动发现的项目目录，无需配置。
  _node_repo_dir="${script_dir}"
  # 自动搜索路径数组；可选添加本机已有dasQt仓库路径。
  _dasqt_candidates=(
    "${_node_repo_dir}/../.."
    "/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show"
    "/Users/zhangzhiyu/MyProjects/dasQt"
  )
  # 内部循环变量，逐一检查候选路径。
  for _candidate in "${_dasqt_candidates[@]}"; do
    if [[ -d "${_candidate}/dasqt" && -f "${_candidate}/.venv/bin/activate" ]]; then
      dasqt_repo_dir="${_candidate}"
      break
    fi
  done
fi

# Python环境activate文件路径；可选已有activate文件，同名小写环境变量覆盖，默认仓库/.venv/bin/activate。
dasqt_venv_activate="${dasqt_venv_activate:-${dasqt_repo_dir:-}/.venv/bin/activate}"

ensure_dasqt_env() {
  if [[ -z "${dasqt_repo_dir:-}" || ! -d "${dasqt_repo_dir}" ]]; then
    echo "错误：未找到 DAS/dasQt 源码仓库。" >&2
    echo "请设置 dasqt_repo_dir，例如：" >&2
    echo "  dasqt_repo_dir=/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show ./run_mohe_node_ccf.sh" >&2
    exit 1
  fi
  if [[ ! -f "${dasqt_venv_activate}" ]]; then
    echo "错误：未找到项目 Python 环境：${dasqt_venv_activate}" >&2
    echo "请在 dasqt_repo_dir 下执行：uv sync" >&2
    exit 1
  fi
}

run_uv_python() {
  ensure_dasqt_env
  (
    cd "${dasqt_repo_dir}"
    # shellcheck disable=SC1090
    source "${dasqt_venv_activate}"
    uv run --no-sync python "$@"
  )
}
