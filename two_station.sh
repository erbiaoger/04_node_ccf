#!/usr/bin/env bash
# 单独启动与DAS主界面同一个节点双站窗口；正常使用也可点击主界面的“节点双站频散”。
set -euo pipefail
# 自动定位项目，无需修改；BASH_SOURCE为Bash内置变量。
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# DAS仓库路径，可选已有仓库绝对路径，同名小写环境变量覆盖。
dasqt_repo_dir="${dasqt_repo_dir:-/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show}"
# 参数配置路径，可选兼容JSONC，同名小写环境变量覆盖。
config="${config:-${script_dir}/config/two_station_config.jsonc}"
[[ "${config}" == /* ]] || config="${PWD}/${config}"
source "${script_dir}/_shell_common.sh"
run_uv_python -m dasqt.features.node_two_station.gui --config "${config}" "$@"
