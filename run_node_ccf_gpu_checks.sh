#!/usr/bin/env bash
# 用途：RTX4090上构建独立wheel、验证新旧CUDA接口，不替换环境中已有扩展。
# 运行：bash examples/04_node_ccf/run_node_ccf_gpu_checks.sh
# 输出：outputs/node_cuda_validation/{wheel,extension,tests.xml}和终端构建/测试日志。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_shell_common.sh"
ensure_dasqt_env
cd "$DASQT_REPO_DIR"
source "$DASQT_VENV_ACTIVATE"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.6}" # CUDA Toolkit目录，必须支持sm_89
CHECK_OUT="${CHECK_OUT:-$DASQT_REPO_DIR/outputs/node_cuda_validation}" # wheel和测试报告目录
BUILD_NATIVE="${BUILD_NATIVE:-1}" # 1重新构建；0复用CHECK_OUT/extension用于重复测试
mkdir -p "$CHECK_OUT/wheel" "$CHECK_OUT/extension"
if [[ "$BUILD_NATIVE" == "1" ]]; then
  CUDACXX="$CUDA_ROOT/bin/nvcc" uv build --wheel --out-dir "$CHECK_OUT/wheel" \
    --config-setting="cmake.define.CMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc" \
    --config-setting="cmake.define.CUDAToolkit_ROOT=$CUDA_ROOT" native/cc_cuda_engine
  for wheel in "$CHECK_OUT/wheel"/dasqt_cc_cuda-*.whl; do
    uv run --no-sync python -m zipfile -e "$wheel" "$CHECK_OUT/extension"
  done
fi
export PYTHONPATH="$CHECK_OUT/extension:$DASQT_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export REQUIRE_NODE_CUDA=1 # 缺少新原生扩展或CUDA必须失败，禁止以skip代替验收
uv run --no-sync pytest -q tests/node_ccf tests/core/ccf/test_cpp_cuda_backend.py \
  --junitxml="$CHECK_OUT/tests.xml"
