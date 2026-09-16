# 节点 SAC 互相关

本目录把排序 CSV 和 `/Volumes/CSIM/2026SaErTuoHai_passive` 这类节点 SAC 数据接入现有 DAS CCF 计算。默认 `read_mode=preload`：先将 CSV 中选定台站的共同时间段一次性读入运行内存，再按一分钟切片并把多个短窗组成 batch，交给现有 DAS 的 `compute_cc_shot` 和 GPU/CPU 后端。`read_mode=window` 可作为低内存回退模式。

本仓库依赖已有 DAS/dasQt 源码仓库和其中的 `.venv`。服务器上默认寻找
`/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show`，本机默认寻找
`/Users/zhangzhiyu/MyProjects/dasQt`；也可以显式指定：

```bash
DASQT_REPO_DIR=/path/to/DAS_Procee_Show ./run_node_ccf.sh
```

CSV 最少包含两列：

```csv
station_id,distance_m
450400001,0.0
450400002,2.5
```

运行示例：

```bash
cd /Users/zhangzhiyu/MyProjects/dasQt
bash examples/04_node_ccf/run_node_ccf.sh /path/to/stations.csv
```

从独立克隆目录运行时直接执行 `./run_node_ccf.sh` 即可；脚本会自动调用外部
DAS 仓库的 `_shell_common.sh` 同等环境逻辑，并使用本仓库的配置、CSV 和输出目录。

服务器使用 CUDA 时可显式指定后端和设备：

```bash
READ_MODE=preload CC_BACKEND=torch CC_DEVICE=cuda CC_BATCH_CHUNKS=8 \
./run_node_ccf.sh ./config/line_380_stations.csv /path/to/node_sac ./outputs/node_ccf
```

`CC_BATCH_CHUNKS` 是一次送入后端的短窗数量；一分钟内的短窗会先按该数量分批处理，再完成一分钟叠加。预加载模式会按实际数据长度申请内存，请先确认服务器可用内存足够。

预加载内存可按 `样本数 × 台站数 × dtype 字节数` 估算；例如日志中的 `515540500 × 34` 个 `float32` 样本约为 65.3 GiB。程序会在分配前打印估算值，并由 `preload_max_gib`（默认 800 GiB）拦截异常大的配置。相同输出目录同时只允许一个计算进程，避免重复启动多个预加载进程叠加占用内存；SAC 文件读取产生的 Linux 文件页缓存属于系统缓存，不等于该 Python 进程的 RSS。

当 `CC_BACKEND=torch` 且 `CC_DEVICE=cuda` 时，短窗预处理、互相关 FFT、分钟叠加和 `save_every` 汇总都会在同一块 CUDA 设备上执行；日志会打印 `Torch CUDA device=...`、`GPU pairwise correlation active` 和 `GPU pairwise stacking active`。CUDA 模式遇到设备不可用或预处理无效数据时会直接报错，不会静默改用 CPU。`nvidia-smi` 的利用率是采样瞬时值，单分钟只有十几个短窗时可能显示较低，但上述日志可以确认实际执行路径。

当前配置已经填入第380线的 CSV，因此也可以直接运行整条第380线：

```bash
bash examples/04_node_ccf/run_node_ccf.sh
```

先做时间段试算时可设置：

```bash
START_UTC=2026-07-09T11:00:00Z END_UTC=2026-07-09T12:00:00Z \
bash examples/04_node_ccf/run_node_ccf.sh
```

脚本默认读取本目录的 `config/cc_config.jsonc`，并自动继承 `examples/03_ccf/config/cc_config.jsonc` 的 DAS 参数。频带、短窗、归一化、后端、数据类型和 `time_downsample` 等设置都会传给节点流程。配置中的相对路径以 `04_node_ccf` 仓库根目录为基准，因此下面的配置可直接使用：

```jsonc
"station_csv": "./config/line_380_stations.csv",
"node_data_dir": "/mnt/nas-changbai/2026SaErTuoHai_passive",
"node_output_dir": "./outputs/node_ccf"
```

绝对路径按原样使用；一键脚本中显式传入的 CSV、数据目录和输出目录会覆盖配置值。需要换配置时设置 `CONFIG=/path/to/cc_config.jsonc`；只有设置 `PAIR_MODE`、`CC_LEN` 等环境变量时才覆盖对应配置值。脚本会在切换到外部 DAS 仓库前把这些显式参数转换为绝对路径。

也可以只传 CSV，让脚本自动使用默认 DAS 配置和节点目录；或显式指定配置：

```bash
CONFIG=examples/04_node_ccf/config/cc_config.jsonc \
bash examples/04_node_ccf/run_node_ccf.sh /path/to/stations.csv
```

当前第380线默认只计算不同台站的无序对：34 个台站对应 `34 × 33 / 2 = 561` 个互相关。B–A 不重复计算，而由 A–B 沿延迟轴反转得到；输出矩阵对角线默认置零。如需自相关，设置 `INCLUDE_AUTOCORR=1`，此时增加 34 个对角线结果。计算先将 `minute_stack_s=60` 秒内的短窗叠加为一个一分钟段，再将 `save_every=30` 个一分钟段叠加，输出一个与 DAS CCF 兼容的 MAT 文件：`data` 形状为 `(源节点, 延迟采样点, 接收节点)`。末尾不足一个时间段的短窗也会单独保存。

计算结束后，可用下面的命令执行与 DAS 相同的二次叠加流程，并生成基于节点真实 `profileX` 坐标的图片：

```bash
./run_node_ccf_stack.sh
```

它会读取 `outputs/node_ccf/cc_stack_*.mat`，在 `outputs/node_ccf_restack/` 中写入 `restack_*.mat`、`node_cc_preview.png` 和 `shots_fig/`。如果希望计算、二次叠加和绘图一次完成，运行：

```bash
./run_node_ccf_all.sh
```

后处理默认使用 DAS 的 `stack_cc_chunks`，配置为 `pws + torch + cuda`；绘图脚本会使用 MAT 中的 `profileX`，不会把不规则节点距离强行当成等间距 `dx`。

输出文件按 `cc_stack_0001_n30.mat`、`cc_stack_0002_n30.mat` 等命名；文件名中的 `n` 是该文件包含的一分钟段数量，最后不足 30 段时也会单独保存。每个 MAT 文件包含 `data`、`dt`、`dx`、`profileX`、台站号和台站对索引等字段，可直接交给现有 DAS 的 MAT 后处理读取。`actual_short_window_count` 记录该文件实际叠加了多少个短窗，便于检查边界时间段。
