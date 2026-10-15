#!/usr/bin/env bash
# 一键提取指定虚拟源的互相关，并绘制单幅剖面及Park/F-J频散谱。
# 使用：./source_gather.sh；也可用同名小写环境变量覆盖以下参数。
set -euo pipefail
# 自动定位项目；BASH_SOURCE为Bash内置变量，无需修改。
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DAS源码及Python环境位置，可选已有DAS仓库路径。
dasqt_repo_dir="${dasqt_repo_dir:-/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show}"
# 输入：最终全台站叠加H5，必须包含台站编号、经纬度、距离、延迟轴、data和有效掩码。
input_file="${input_file:-${script_dir}/outputs/stack/stack_final.h5}"
# 虚拟源编号；可选输入H5中存在的一个原始台站编号。
source_id="${source_id:-39}"
# 接收台站：all=除源外所有台站；或空格/中英文逗号分隔的编号，例如 "47 16 18 62 20 33 41 48"。
# 无需把源放进列表；若包含源会自动去除；重复编号去重，不存在的编号报错。
receiver_ids="${receiver_ids:-all}"
# 输出根目录；按源、接收数量及参数摘要自动建子目录。相同参数重跑覆盖该任务结果。
output_root="${output_root:-${script_dir}/outputs/source_gathers}"
# 最低/最高频率，Hz；0 < fmin < fmax，最高频率不应超出原互相关有效频带。
fmin="${fmin:-1}"
fmax="${fmax:-10}"
# 频率网格步长，Hz，正数；插值不增加原始FFT频率分辨率。
df="${df:-0.1}"
# 最低/最高相速度，m/s；0 < vmin < vmax。
vmin="${vmin:-500}"
vmax="${vmax:-4000}"
# 相速度网格步长，m/s，正数。
dv="${dv:-10}"
# 波形图归一化后的最大横向摆幅，米，正数；仅影响显示，近距离道可能重叠。
wiggle_amplitude_m="${wiggle_amplitude_m:-20}"
# 固定处理：原始data，不做零延迟压制；频散采用CPU，计算Park三个分支和F-J。
# 每个任务保存互相关H5/MAT、频散H5、PNG/PDF/SVG、receivers.csv、task.json及run.log。
source "${script_dir}/_shell_common.sh"
# 相对路径统一按项目目录解析，支持从其他工作目录调用。
[[ "${input_file}" == /* ]] || input_file="${script_dir}/${input_file}"
[[ "${output_root}" == /* ]] || output_root="${script_dir}/${output_root}"
run_uv_python "${script_dir}/scripts/run_source_gather.py" \
  --input "${input_file}" --source "${source_id}" --receivers "${receiver_ids}" \
  --output-root "${output_root}" --fmin "${fmin}" --fmax "${fmax}" --df "${df}" \
  --vmin "${vmin}" --vmax "${vmax}" --dv "${dv}" --wiggle-amplitude-m "${wiggle_amplitude_m}"
