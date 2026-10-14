#!/usr/bin/env bash
# 节点互相关/TSI一键入口；计算日期、时间、坐标模式和batch等参数见config/cc_config.jsonc。
# 运行：./ccf.sh；仅预检：./ccf.sh --preflight；单日：./ccf.sh --date 2026-07-13。
set -euo pipefail
# 脚本目录，自动解析，无需修改；BASH_SOURCE为Bash内置变量，保持原名。
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# dasQt仓库路径；可选已有仓库绝对路径，同名小写环境变量可覆盖。
dasqt_repo_dir="${dasqt_repo_dir:-/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show}"
# 配置文件路径；可选兼容节点JSONC，同名小写环境变量可覆盖；相对路径按调用目录解析。
config="${config:-${script_dir}/config/cc_config.jsonc}"
[[ "${config}" == /* ]] || config="${PWD}/${config}"
# shellcheck disable=SC1091
source "${script_dir}/_shell_common.sh"
run_uv_python "${script_dir}/scripts/run_node_ccf_config.py" --config "${config}" "$@"
