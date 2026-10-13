#!/usr/bin/env bash
# 节点CUDA原生/Torch对照：每次重新计算，包含上传和下载，不包含SAC读取。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_shell_common.sh"
STATIONS="${STATIONS:-34}" # 节点数；第380线34个
SECONDS_COUNT="${SECONDS_COUNT:-120}" # 合成数据时长秒数
SAMPLING_RATE="${SAMPLING_RATE:-250}" # 原始采样率Hz
REPEATS="${REPEATS:-3}" # 预热后重复次数，报告中位数
BENCHMARK_OUT="${BENCHMARK_OUT:-$SCRIPT_DIR/outputs/node_benchmark.json}" # JSON输出路径
run_uv_python "$SCRIPT_DIR/scripts/benchmark_node_pairs.py" --stations "$STATIONS" \
  --seconds "$SECONDS_COUNT" --sampling-rate "$SAMPLING_RATE" --repeats "$REPEATS" --output "$BENCHMARK_OUT"
