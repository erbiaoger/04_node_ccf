# 第380线节点全配对互相关：C++/CUDA

默认34节点、561个无序节点对；CSV顺序保持不变。A–B只计算一次，B–A由延迟
反转得到，对角线不计算。节点流程已经移除sliding、offset和dshot计算模式。

## 计算定义

10秒窗/5秒步长，2–30Hz，±4秒延迟，时域RMA+频域RMA。
去趋势/加窗/带通/时间归一化后，补零到至少2N−1并FFT，在该频谱上白化，
**直接 `conj(F_A)*F_B` 后逆FFT**。白化后不还原短窗、不再次FFT。

每个节点对独立沿延迟轴做PWS（指数2），短窗→1分钟→每30分钟保存→全时段
总叠加。每层有效输入等权，峰值归一化并去均值；有效计数只作质量记录。

预处理与带通复用现有CUDA核；短窗和配对分块、共享频谱、分钟与组统计量
保留在GPU，只有30分钟成品下载。坏窗只使涉及该节点的配对无效。
详细公式、边界和旧结果差异见 [方法说明](METHODS.md)。

## 文件与目录

- `config/`：独立JSONC配置和第380线CSV，不继承DAS几何/FK/TSI。
- `scripts/node_io/`：SAC头、参考时间+b、采样网格、memmap读取和缺测掩码。
- `scripts/node_compute/`：参数、整数窗口调度、严格CUDA边界、CPU/Torch参考。
- `scripts/node_output/`：原子MAT/JSON、配对稠密化、输入指纹和续算校验。
- `scripts/cli.py`、`node_pipeline.py`：命令行和两级时间调度。
- `scripts/stack_node_cc_mat.py`：逐文件流式总叠加，保留几何和有效计数。
- `scripts/plot_node_ccf.py`：真实profileX坐标绘图，无效配对留空，Times New Roman。
- `scripts/benchmark_node_pairs.py`：包含传输的原生CUDA/Torch数值及计时对照。
- `outputs/`：运行生成的MAT、任务清单、图和性能记录，不提交版本管理。

## 环境与编译

脚本复用dasQt项目 `.venv`，激活后通过 `uv run --no-sync` 运行。若示例独立
克隆，指定 `DASQT_REPO_DIR=/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show`。
默认后端 `cpp_cuda/cuda/float32`，要求RTX4090/sm_89、CUDA Toolkit12.6和
新编译的 `node_pairwise_version=node_spectral_pair_pws_v1` 扩展，不自动回退。

可在主项目根目录安装更新后的扩展：

```bash
source .venv/bin/activate
CUDACXX=/usr/local/cuda-12.6/bin/nvcc uv pip install --python .venv/bin/python \
  --reinstall --no-deps \
  --config-settings=cmake.define.CMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  --config-settings=cmake.define.CUDAToolkit_ROOT=/usr/local/cuda-12.6 \
  ./native/cc_cuda_engine
```

也可先构建独立wheel验证，不替换已有扩展：

```bash
bash examples/04_node_ccf/run_node_ccf_gpu_checks.sh
```

该入口生成 `outputs/node_cuda_validation/extension`；使用它运行时设置：
`PYTHONPATH="$PWD/outputs/node_cuda_validation/extension:$PWD"`。

## 运行

先确认NAS挂载，再做头信息预检：

```bash
PREFLIGHT=1 bash examples/04_node_ccf/run_node_ccf.sh
```

一小时试算，保存结果、总叠加和全部节点图：

```bash
START_UTC=2026-07-09T11:00:00Z END_UTC=2026-07-09T12:00:00Z \
bash examples/04_node_ccf/run_node_ccf_all.sh \
  examples/04_node_ccf/config/line_380_stations.csv \
  /mnt/nas-changbai/2026SaErTuoHai_passive \
  examples/04_node_ccf/outputs/validation_1h
```

正式共同时间段运行，默认auto读取和校验续算：

```bash
bash examples/04_node_ccf/run_node_ccf_all.sh
```

默认输出为 `outputs/node_ccf_spectral_v1`，总叠加和图位于同名 `_restack`
目录。新输出目录避免混入旧周期相关/展平PWS文件。中断后原命令重跑，逐个
检查SHA256并跳过完成组；损坏组重算，未完成分钟不落盘。输入文件属性、
SAC头、顺序、时间段或科学参数变更必须换输出目录。读取方式和GPU批次可调。

`READ_MODE=auto|preload|block`：预加载峰值估计不超过当前可用内存50%时auto
选择preload，否则每30分钟分块。两种模式使用相同全局窗口，跨分钟、组边界
保留重叠，末尾只计算完整短窗。SAC采样率来自头，不使用配置中的旧sps。
用户UTC边界就近对齐采样点，实际边界记入MAT；台站间采样网格仍严格检查。

示例环境变量：`CC_BATCH_CHUNKS=8 CC_PAIR_BATCH=64 READ_MODE=block`。
显式CPU参考：`CC_BACKEND=cpu CC_DEVICE=cpu`；Torch参考：
`CC_BACKEND=torch CC_DEVICE=cuda`。`RESUME=0`要求空的新任务目录。
所有命令行参数见 `bash run_node_ccf.sh --help`；shell脚本中每项有中文注释。

只做计算运行 `run_node_ccf.sh`；只做总叠加/绘图运行 `run_node_ccf_stack.sh`。
生产目录不要同时运行两个写入任务。长任务建议tmux并重定向日志。

## 输出

`cc_stack_0001_n30.mat` 的 `data` 为 `(source, lag, receiver)` float32，含
`dt`、`profileX`、`station_id`、显式 `lags_s` 和从0开始的上三角pair索引。
`valid_mask`、`valid_short_window_count`、`valid_minute_count` 为节点×节点。
无效配对填零并标记无效，对角线也为无效。计数为uint64。

`dx`和`dshot`写相邻坐标差的中位数，仅兼容旧读取器，不参与计算。
SAC缺失节点保留位置；内部间隙、非有限值、全零窗及质量失败逐窗标记。
重叠段、组件歧义和采样率不一致报错。共同时间段按有文件台站的外部覆盖
范围交集确定，内部缺口由掩码处理。尾组可无有效窗，仍保存时间轴和掩码。

总叠加包含 `valid_file_count`、累计有效短窗/分钟数量、文件名和SHA256。
二次叠加拒绝计算版本、台站顺序、频带/叠加参数或时间轴不一致的文件。

## 验证

本地：激活主项目 `.venv` 后执行 `uv run --no-sync pytest -q tests/node_ccf`，
无CUDA时显式跳过硬件测试；RTX4090上以 `REQUIRE_NODE_CUDA=1` 禁止跳过。
`run_node_ccf_gpu_checks.sh` 同时运行节点测试和既有CUDA后端回归。

性能对照：`bash examples/04_node_ccf/run_node_ccf_benchmark.sh`，默认34节点、
250Hz、120秒合成数据，预热后3次重新计算，记录包含传输的中位数和实际后端。
