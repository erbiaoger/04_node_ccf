#!/usr/bin/env bash
# 用途：更新节点示例所使用的主项目.venv中的C++/CUDA扩展。
# 运行：./build_node_ccf_cuda.sh；成功后可直接运行./run_node_ccf_all.sh。
# 输出：终端打印主项目、Python、nvcc、扩展路径和节点接口版本。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_shell_common.sh"
ensure_dasqt_env
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.6}" # CUDA Toolkit目录，RTX4090需支持sm_89
# DASQT_REPO_DIR可显式指定依赖的主项目；默认与计算入口采用相同查找规则。
cd "$DASQT_REPO_DIR"
source "$DASQT_VENV_ACTIVATE"
if [[ ! -f native/cc_cuda_engine/include/node_pairs.hpp ]]; then
  echo "错误：主项目源码尚未包含节点CUDA接口，请先更新主项目：" >&2
  echo "  git -C \"$DASQT_REPO_DIR\" pull --ff-only origin main" >&2
  exit 1
fi
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "错误：未找到 $CUDA_ROOT/bin/nvcc；请通过CUDA_ROOT指定Toolkit目录。" >&2
  exit 1
fi
printf '主项目：%s\nPython：%s\n' "$PWD" "$PWD/.venv/bin/python"
"$CUDA_ROOT/bin/nvcc" --version
CUDACXX="$CUDA_ROOT/bin/nvcc" uv pip install --python .venv/bin/python \
  --reinstall --no-deps \
  --config-settings="cmake.define.CMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc" \
  --config-settings="cmake.define.CUDAToolkit_ROOT=$CUDA_ROOT" \
  ./native/cc_cuda_engine
# 清除旧的独立wheel覆盖路径，确认默认项目环境实际安装成功。
env -u PYTHONPATH uv run --no-sync python -c 'import _dasqt_cc_cuda as m; print("扩展：", m.__file__); print("节点接口：", getattr(m, "node_pairwise_version", None)); assert getattr(m, "node_pairwise_version", None) == "node_spectral_pair_pws_v1"; assert m.runtime_available(); print("节点CUDA默认环境验证通过")'
