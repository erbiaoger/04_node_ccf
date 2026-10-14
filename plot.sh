#!/usr/bin/env bash
# 一键频散与绘图：只输入最终叠加H5，自动计算对应频散。
# 使用：bash run_mohe_day_plot.sh；可从任意目录调用本脚本。
set -euo pipefail
# 脚本目录，自动解析；无需修改。BASH_SOURCE 是 Bash 内置变量，保留大写。
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 所有可调参数集中在这里 ──
# dasQt 源码仓库路径；可选任意已有仓库绝对路径，也可通过同名小写环境变量覆盖。
dasqt_repo_dir="${dasqt_repo_dir:-/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show}"
# 最终叠加 H5 路径；可选任意兼容节点 H5；同名小写环境变量可覆盖。
input_file="${input_file:-${script_dir}/outputs/stack/stack_final.h5}"
# 输出目录；可选可写目录；绘图默认输入文件旁的 figures，同名小写环境变量可覆盖。
output_dir="${output_dir:-$(dirname "${input_file}")/figures}"
# 频散方法；可选 fj（F-J）| park（Park相移法），默认fj；小写环境变量可覆盖。
dispersion_method="${dispersion_method:-fj}"
# Park的互相关分支；可选 positive（正延迟）| negative（负延迟反转）| symmetric（两分支线性平均）。
park_branch="${park_branch:-symmetric}"
# 频散输出目录，自动设为 output_dir/dispersion；不是额外输入。
dispersion_dir="${output_dir}/dispersion/${dispersion_method}"
# 频散频率扫描步长，Hz；可选正数，默认0.1；插值步长不代表实际频率分辨率。
df="0.05"
# 相速度扫描步长，m/s；可选正数，默认10。
dc="10"
# 频散后端；可选 torch_cuda（GPU）| cpu（CPU参考），默认 torch_cuda。
backend="torch_cuda"
# GPU编号；可选机器上的非负设备编号，默认0；cpu后端忽略。
cuda_device="0"
# 一次构建核的频率数量；可选正整数，默认8；增大会增加显存需求。
frequency_batch="8"
# 可选1（计算频散并绘制六图）| 0（仅四幅互相关/几何图）；默认1。
draw_dispersion="1"
# 频散最低频率，Hz；可选正数且小于 fmax，默认2。
fmin="1"
# 频散最高频率，Hz；可选大于 fmin 且不超过数据奈奎斯特频率，当前10。
fmax="10"
# 最低相速度，m/s；可选正数且小于 vmax，当前1500；图中换算为km/s。
vmin="1500"
# 最高相速度，m/s；可选大于 vmin，默认4000。
vmax="4000"
# 距离分箱宽度，m；可选正数，默认50；互相关与分箱频散使用同一值。
distance_bin_m="50"
# 互相关对称色标裁剪百分位；可选0＜值≤100，默认99.5；100表示不裁剪。
clip_percentile="99.5"
# 互相关色图；可选 Matplotlib 已注册名称，如 RdBu_r | seismic | coolwarm，默认 RdBu_r。
ccf_cmap="RdBu_r"
# 频散色图；可选 Matplotlib 已注册名称，如 magma | inferno | viridis，默认 magma。
dispersion_cmap="magma"
# 总览宽度，英寸；可选正数，默认7.2。
figure_width_in="7.2"
# 总览高度，英寸；可选正数，六图默认9.75，四图建议6.5。
figure_height_in="9.75"
# 字体；可选系统已安装字体，如 DejaVu Sans | Arial；默认 DejaVu Sans。
font_family="DejaVu Sans"
# 基础字号，pt；可选≥7，默认8。
font_size="8"
# PNG及矢量图内热图分辨率，dpi；可选整数≥72，默认300。
dpi="300"
# 导出格式数组；可选 png | pdf | svg 的非空组合，默认三种全部导出。
export_formats=("png" "pdf" "svg")
# 可选1（所有源台站PNG）| 0（仅中心源）；默认1，中心源另导出指定格式。
save_source_figures="1"

# 六图固定顺序：a地图，b台站对距离分布，c距离分箱互相关，d全配对互相关，
# e对应c的分箱频散，f对应d的全配对频散；布局三行两列。
# 延迟轴读取H5；频散显示范围由上面的FMIN/fmax/vmin/VMAX控制。
# shellcheck disable=SC1091
source "${script_dir}/_shell_common.sh"
ensure_dasqt_env
if [[ ! -f "${input_file}" ]]; then
  echo "错误：找不到最终叠加互相关H5。请检查脚本开头的路径。" >&2
  exit 1
fi
if [[ "${draw_dispersion}" != "0" && "${draw_dispersion}" != "1" ]] ||
   [[ "${save_source_figures}" != "0" && "${save_source_figures}" != "1" ]]; then
  echo "错误：DRAW_DISPERSION和SAVE_SOURCE_FIGURES必须为0或1。" >&2
  exit 1
fi

# 内部Python命令参数数组，自动构建；不要手动修改。
args=("${script_dir}/scripts/plot_geographic_ccf.py"
  --input "${input_file}" --output-dir "${output_dir}"
  --distance-bin-m "${distance_bin_m}" --clip-percentile "${clip_percentile}"
  --fmin "${fmin}" --fmax "${fmax}" --vmin "${vmin}" --vmax "${vmax}"
  --ccf-cmap "${ccf_cmap}" --dispersion-cmap "${dispersion_cmap}"
  --figure-width-in "${figure_width_in}" --figure-height-in "${figure_height_in}"
  --font-family "${font_family}" --font-size "${font_size}"
  --dpi "${dpi}" --export-formats "${export_formats[@]}")
if [[ "${draw_dispersion}" == "1" ]]; then
  mkdir -p "${dispersion_dir}"
  run_uv_python "${script_dir}/scripts/compute_geographic_dispersion.py" \
    --input "${input_file}" --output-dir "${dispersion_dir}" --method "${dispersion_method}" --park-branch "${park_branch}" \
    --fmin "${fmin}" --fmax "${fmax}" --df "${df}" \
    --cmin "${vmin}" --cmax "${vmax}" --dc "${dc}" \
    --distance-bin-m "${distance_bin_m}" --backend "${backend}" \
    --cuda-device "${cuda_device}" --frequency-batch "${frequency_batch}" \
    2>&1 | tee "${dispersion_dir}/run.log"
  args+=(--dispersion-dir "${dispersion_dir}")
fi
if [[ "${save_source_figures}" == "1" ]]; then
  args+=(--save-source-figures)
else
  args+=(--no-save-source-figures)
fi
mkdir -p "${output_dir}"
run_uv_python "${args[@]}" 2>&1 | tee "${output_dir}/run.log"
echo "绘图完成：${output_dir}/ccf_overview；导出格式：${export_formats[*]}"
