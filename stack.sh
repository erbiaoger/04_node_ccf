#!/usr/bin/env bash
# 多日叠加：递归读取日期子目录；默认叠加全部，日期/时间/数量见config/stack_config.jsonc。
# 用法：./run_mohe_day_stack.sh；仅查看清单：./run_mohe_day_stack.sh --list-only。
set -euo pipefail
# 脚本目录，自动解析；BASH_SOURCE为Bash内置变量，保留原名。
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# dasQt仓库路径；可选任意已有仓库绝对路径，同名小写环境变量可覆盖。
dasqt_repo_dir="${dasqt_repo_dir:-/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show}"
# 参数配置路径；可选兼容叠加JSONC，同名小写环境变量可覆盖；相对路径按调用目录解析。
config="${config:-${script_dir}/config/stack_config.jsonc}"
[[ "${config}" == /* ]] || config="${PWD}/${config}"
# 可选输入/输出目录覆盖；空值使用JSONC配置，input_dir递归读取各日期子目录。
input_dir="${input_dir:-}"
output_dir="${output_dir:-}"
# 内部参数数组，自动构建，不需要修改。
args=("${script_dir}/scripts/run_node_stack_config.py" --config "${config}")
[[ -z "${input_dir}" ]] || args+=(--input-dir "${input_dir}")
[[ -z "${output_dir}" ]] || args+=(--output-dir "${output_dir}")
# shellcheck disable=SC1091
source "${script_dir}/_shell_common.sh"
run_uv_python "${args[@]}" "$@"
