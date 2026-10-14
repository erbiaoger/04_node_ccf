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
- `scripts/node_output/`：原子H5/JSON、配对稠密化、输入指纹和续算校验。
- `scripts/cli.py`、`node_pipeline.py`：命令行和两级时间调度。
- `scripts/stack_node_cc_mat.py`：逐文件流式总叠加，保留几何和有效计数。
- `scripts/plot_node_ccf.py`：真实profileX坐标绘图，无效配对留空，Times New Roman。
- `scripts/benchmark_node_pairs.py`：包含传输的原生CUDA/Torch数值及计时对照。
- `outputs/`：运行生成的H5、任务清单、图和性能记录，不提交版本管理。

## 环境与编译

脚本复用dasQt项目 `.venv`，激活后通过 `uv run --no-sync` 运行。若示例独立
克隆，指定 `dasqt_repo_dir=/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show`。
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

正式共同时间段运行，默认并行分块读取和校验续算：

```bash
bash examples/04_node_ccf/run_node_ccf_all.sh
```

默认输出为 `outputs/node_ccf_spectral_v1`，总叠加和图位于同名 `_restack`
目录。新输出目录避免混入旧周期相关/展平PWS文件。中断后原命令重跑，逐个
检查SHA256并跳过完成组；损坏组重算，未完成分钟不落盘。输入文件属性、
SAC头、顺序、时间段或科学参数变更必须换输出目录。读取方式和GPU批次可调。

`READ_MODE=auto|preload|block`：默认block，主机按 `READ_HOURS` 缓存，默认24小时。
24/48小时缓存读完后，按30分钟切片送GPU；`SAVE_EVERY=30`控制H5输出片段长度。
两项独立配置，读取时长须为输出片段时长的整数倍。`READ_HOURS=0`恢复按输出片段读取；
`READ_MODE=preload`显式加载整个任务；auto在read_hours>0时使用主机分块缓存。
read_hours=0时auto仅对估计峰值不超过512 MiB及可用内存50%的小任务全量加载。两种模式使用相同全局窗口，跨分钟、组边界
保留重叠，末尾只计算完整短窗。SAC采样率来自头，不使用配置中的旧sps。
用户UTC边界就近对齐采样点，实际边界记入H5；台站间采样网格仍严格检查。

默认 `IO_WORKERS=8 PREFETCH_BLOCKS=1 ASYNC_SAVE=1`：台站SAC并行读取，最多预取下一份24/48小时主机缓存，
与当前缓存中的30分钟CUDA批次计算重叠；主机只保留当前时间缓存和下一时间缓存，不全量加载整个长任务。
CUDA会释放GIL，预取线程可同时运行，科学窗口和叠加顺序保持不变。
`IO_WORKERS=1 PREFETCH_BLOCKS=0 ASYNC_SAVE=0` 可关闭并行作串行对照。

示例环境变量：`READ_HOURS=24 IO_WORKERS=8 PREFETCH_BLOCKS=1 CC_BATCH_CHUNKS=12 READ_MODE=block`。
日志逐组显示读取时间、等待预取时间和计算时间；任务清单保存总耗时。
读取总时间包含与计算重叠的部分，因此不能与计算时间简单相加。
显式CPU参考：`cc_backend=cpu cc_device=cpu`；Torch参考：
`cc_backend=torch cc_device=cuda`。`RESUME=0`要求空的新任务目录。
所有命令行参数见 `bash run_node_ccf.sh --help`；shell脚本中每项有中文注释。

只做计算运行 `run_node_ccf.sh`；只做总叠加/绘图运行 `run_node_ccf_stack.sh`。
生产目录不要同时运行两个写入任务。长任务建议tmux并重定向日志。

## 输出

`cc_stack_0001_n30.h5` 的 `data` 为 `(source, lag, receiver)` float32，含
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

独立示例中的回归测试可在本目录用主项目Python执行：
`/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show/.venv/bin/python -m pytest -q tests`。
包含并行读取掩码一致性、预取与计算重叠、预取上限、续算和CUDA数值测试。

真实NAS流水线基准（输出目录必须不存在）：

```bash
/path/to/dasQt/.venv/bin/python scripts/benchmark_node_stream.py \
  --start-utc 2026-07-09T11:00:00Z --end-utc 2026-07-09T12:00:00Z \
  --output-dir outputs/stream_benchmark_new --repeats 2
```

先按块预热文件缓存，再交替运行串行与并行，包含读取、CUDA计算和H5写入，
校验结果与有效计数，生成 `results.json`。NAS/GPU共享负载仍会影响耗时。

完成后的逐文件审计：`bash examples/04_node_ccf/run_node_ccf_audit.sh`，校验全部
任务组SHA256、精确计数、有效掩码、延迟互易性及总叠加文件来源，流式处理，
生成质量报告JSON。自定义目录可传三个位置参数：输入目录、总叠加目录、JSON。

H5保存规则：每个30分钟时间片段一个 `cc_stack_XXXX_n30.h5`，末尾不足30分钟
单独保存。`/data` 为 `(source, lag, receiver)`，包含全部台站；`/station_id`、
`/profileX`、`/lags_s`、`/valid_mask`、有效计数和实际起止UTC时间都保存在同一文件。
这是原生HDF5，可直接用h5py读取。默认无压缩，按source分块，便于取单个虚拟炮。

`ASYNC_SAVE=1`（默认）最多排队一个输出块，H5写盘和下一组CUDA计算重叠；
所有写入与校验完成后才将任务标记为完成。可设置 `ASYNC_SAVE=0` 作同步对照。
现有完成的MAT任务可用 `scripts/migrate_node_h5.py --input-dir PATH` 无损转换，
逐项校验并更新清单，将原MAT保留在 `legacy_mat/`；叠加和绘图可读取显式指定的旧MAT文件。

NAS读取期间每台站复用一个SAC文件句柄，避免每个时间片段重新建立CIFS打开/关闭请求；
最多保留台站数个句柄，结束或异常退出时关闭。

指定主机一次读入时间：

```bash
READ_HOURS=24 bash run_node_ccf.sh
READ_HOURS=48 bash run_node_ccf.sh
# 不预取下一大块，只保留一份主机缓存：
READ_HOURS=48 PREFETCH_BLOCKS=0 bash run_node_ccf.sh
```

34台站、约250Hz、float32波形加bool掩码，24小时缓存约3.42 GiB，48小时约6.84 GiB。
预取下一块时最多保留两份缓存，另有工作区开销；开始读取前检查内存容量。
跨缓存边界补齐短窗/FIR所需重叠，30分钟输出的窗口归属与两层PWS定义保持一致。
改变read_hours、prefetch_blocks或cc_upload_block不会改变续算任务身份。

`CC_UPLOAD_BLOCK=1` 默认将一个30分钟片段一次上传显存，再在显存内按分钟处理；
小显存预算会自动回到小批次上传。`CC_UPLOAD_BLOCK=0` 可作旧传输路径对照。
短窗预处理与互相关batch已解除分钟边界限制，默认12（当前真实数据测速最快），可通过 `CC_BATCH_CHUNKS=12|128|256|512` 指定上限。
批次可以跨分钟；分钟PWS按minute_id分别累计，再进入30分钟PWS，科学定义保持一致。
每个30分钟输出片段通常约360窗，512的请求也受可用窗口数限制。显存预算不足时自动缩小
窗口或配对批次。日志记录实际window_batch_max、window_batch_last、pair_batch_last、
compute_batches、minute_span_max和input_transfers。

原生整块上传需要主项目应用本示例的 `native/node_block_upload.patch` 后重新编译：
在主项目根目录执行 `git apply /path/to/04_node_ccf/native/node_block_upload.patch`，
然后按上面的CUDA安装步骤安装扩展。当前服务器已应用并安装这一版本。

真实数据的短窗batch对照（只读一次SAC块，随后在RAM中反复计算）：

```bash
/path/to/dasQt/.venv/bin/python scripts/benchmark_node_batch.py \
  --batches 12 32 64 128 256 512 --repeats 5 --output outputs/batch_tuning.json
```

对照包括传输、预处理、相关、两层PWS及下载，排除NAS读取和H5写盘；
每个候选预热后计时，验证计数精确一致及CCF数值容差，报告实际采用的批次大小。
较大的请求不保证更快，应按测得耗时及显存选择。

默认 `CC_REUSE_WORKSPACE=1` 复用单个CUDA工作区，包括短窗预处理、互相关及分钟统计缓冲，
减少大batch反复分配显存的开销；Session结束时释放。设置为0可作分配路径对照。
`workspace_cached_bytes`记录工作区缓存大小。批次参数可调整且不影响任务续算身份。

### 漠河60台站：单日白天、每小时H5

支持TSI的配置式互相关一键入口：

```bash
bash run_mohe_node_ccf.sh --preflight
bash run_mohe_node_ccf.sh
# 换一天，输出日期目录自动更新：
bash run_mohe_node_ccf.sh --date 2026-07-09
```

`processing_time_mode`控制时间选择：

| 模式 | 日期来源 | 每天计算时段 |
| --- | --- | --- |
| `date_range` | 指定起止日期，含首尾 | `start_time`至`end_time` |
| `all_date` | 自动发现SAC覆盖的所有日期 | `start_time`至`end_time` |
| `all_date_time` | 自动发现SAC覆盖的所有日期 | 全天00:00–24:00 |

自动模式忽略配置中的起止日期；`all_date_time`还忽略每日起止时间。
所有日期以`local_timezone`为准，UTC切日的SAC可能覆盖更多北京时间日期。
全天模式首尾按实际数据时间裁剪，保留跨日的全部样本；内部缺测仍使用有效性掩码。
`--date`在所有模式下都可限制为指定一天，全天模式仍按全天计算。

日期范围使用 `processing_start_date`、`processing_end_date`（首尾日期都包含），
每日时段使用字符串 `start_time="08:00"`、`end_time="18:00"`（结束时刻不包含）。
例如2026-07-08至2026-07-11会依次计算4天，每天10个小时H5，分别写入日期子目录。
一次只缓存当天数据；断点续算逐日校验，已完成的文件会跳过。
`--preflight`检查范围内各天，`--date YYYY-MM-DD`覆盖日期范围只跑指定一天。
发现缺失波形/GPS或计算错误会停止并报出当天错误，不静默跳过；修复后重新运行可续算。
仍兼容旧版`processing_date/day_start_hour/day_end_hour`配置。

所有参数集中在 `config/mohe_cc_config.jsonc`，采用与 `CC_ChangBai_select` 相同的扁平JSONC风格。
可用 `config=/path/to/config.jsonc bash run_mohe_node_ccf.sh` 指定另一份配置。
默认2026-07-08至2026-07-11，每天北京时间08–18点，输出根目录下自动创建
`20260708_day_0800_1800_tsi/`，每小时一个全台站H5，原始CC与TSI结果保存在同一文件。

`minute_stack_s=60`时，保存/TSI时长这样设置：

| 保存间隔 | TSI间隔 | save_every | tsi_files_per_group |
| --- | --- | --- | --- |
| 1小时 | 1小时 | 60 | 60 |
| 1小时 | 30分钟 | 60 | 30 |
| 30分钟 | 30分钟 | 30 | 30 |

节点SAC通常是一整天，不能直接按SAC文件数定义TSI时长，所以这里两个计数均以
`minute_stack_s`时间块为单位；0表示TSI跟随保存间隔。保存间隔必须是TSI间隔的整数倍。
当TSI间隔小于保存间隔，先分别形成每个TSI片段的CC、做增强，再把这些增强片段等权叠加到
一个保存片段；原始CC仍独立按整个保存片段计算，保存为 `data_raw`。
该模式会额外计算子片段CC，但复用同一主机缓存，不重新读取NAS。

`cc_enhance_method`可选`raw`/`tsi`；`file_stacking_method`控制短窗→分钟，
`stacking_method`控制分钟→片段及多个TSI片段→输出，均支持独立选择`linear`或`pws`。
互相关使用C++/CUDA；节点TSI使用Torch CUDA，可显式切换到CPU参考路径。
`cc_memory_budget_gib`约束互相关工作区规划；TSI工作区由`tsi_pair_batch_size`控制。

GPS节点TSI使用 `tsi_mode=geographic_all_pairs`：按实际三站距离的路径和/差选择组合，
默认路径误差不超过10米、至少1个第三台站；不按数字编号判断空间先后。
内部第三台站使用定向分支卷积，外部第三台站使用相关，FFT补零保证线性运算；
正负延迟分支分别处理，输出满足源/接收互换的延迟反转关系。
速度软窗使用真实台站间距，默认500–4000 m/s；4次迭代，相对L1变化≤0.005可提前结束。
相关方法参见 [Song et al. (2022)](https://doi.org/10.3389/feart.2022.952410)；
二维节点几何筛选和定向分支处理是本例的适配，增强效果仍需与原始CC对照。

TSI开启后，`data`保存增强结果，`data_raw`保存原始互相关；
`tsi_applied_mask`、`tsi_applied_group_count`、`tsi_common_station_count_min`及
`tsi_diagnostics_json`记录实际增强范围、三站支持和各组迭代历史。
没有足够支持的台站对保留原始CC，缺测配对不会被填成新观测。
原有短窗/分钟有效计数描述原始CC输入，TSI支持单独记录。
参数改变会拒绝复用不兼容的任务目录；建议同时修改输出根目录进行新实验。

已有10个小时结果的一键PWS叠加：`bash run_mohe_day_stack.sh`。
输入、输出、叠加数量、方法、后端和精度均集中写在脚本开头；使用CUDA生成一个
`outputs/mohe_20260708_day_0800_1800_restack/restack_0001_n10_idx0000-0009.h5`，
随后自动验收，保存 `validation.json` 和 `run.log`。

叠加后的地理布局绘图：`bash run_mohe_day_plot.sh`。参数集中在脚本开头，输出到
`outputs/mohe_20260708_day_0800_1800_restack/figures/`：总览 `ccf_overview.png`、
全部1770对的真实距离剖面 `ccf_all_pairs.png`、完整编号GPS地图 `station_map.png`，
以及 `sources/` 下60个源台站的互相关剖面。总览、全配对图、地图及中心源另有PDF/SVG。
波形逐条按绝对峰值归一化，仅改变显示；50米分箱图是这些波形的线性平均，不强制正负延迟对称。
绘图只需一个 `input_file`，指向最终叠加H5；频散直接从该文件计算。
`output_dir`默认位于输入文件旁的`figures/`，频散H5自动写入其`dispersion/`子目录，均为输出。
可用 `input_file=/path/to/final.h5 bash run_mohe_day_plot.sh` 切换最终结果。

参数集中在`run_mohe_day_plot.sh`开头，包含`fmin/fmax`（Hz）、
`vmin/vmax`（m/s）、`df/dc`、CUDA设备、频率batch、分箱宽度和全部绘图参数。
频率/速度范围同时用于计算和显示；保留当前设置2–10 Hz、1500–4000 m/s，
需要原定全范围时改为2–30 Hz、500–4000 m/s。
`draw_dispersion=1`自动计算两份频散并画六图；设为0仅画四图。

六图顺序为a台站地图、b台站对距离分布、c距离分箱互相关、d全部台站对互相关、
e对应c的分箱频散、f对应d的全配对频散。b显示每50米箱内有效台站对数量。
不再读取各小时文件；Python入口仍允许显式`--hourly-dir`进行额外小时一致性比较。
`bash run_mohe_day_dispersion.sh`调用同一个入口，避免两套参数不一致。
实际绘图设置与来源校验值记录在`plot_diagnostics.json`。

F-J使用零阶贝塞尔核J0：把CCF零延迟移到FFT起点，取频谱实部，按真实配对距离进行
`Re(C(f,r))*J0(2πfr/c)*r`的梯形积分。距离分箱先对每个台站对按绝对峰值归一化，
再线性平均，以非空箱中心为代表距离；不对平均后的箱再次逐条归一化。
频散能量为积分幅度平方、逐频率归一化0–1，原积分、归一化因子及有符号幅度均保留。
理论参考：[Wang et al. (2019), Frequency-Bessel transform](https://doi.org/10.1029/2018JB016595)。

F-J径向积分使用Torch CUDA float64批量J0核（与互相关的C++/CUDA后端分开），
FFT及插值使用NumPy；可显式改用`backend=cpu`。两份H5保存到`restack/dispersion/`，
`dispersion_validation.json`记录输入SHA256、实际FFT频点间隔、耗时及GPU/CPU抽查差异。
0.1 Hz是输出插值步长，原8.004秒CCF对应FFT间隔约0.125 Hz。
50米分箱可削弱短波长信号，e图白色虚线仅为每波长两个距离采样点的参考线；
径向平均也会混合台站对方位。谱峰与交叉条纹不自动作为已确认的基阶频散曲线。

在已安装原生CUDA扩展的主项目虚拟环境中运行：

```bash
/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show/.venv/bin/python scripts/run_mohe_day.py --preflight
/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show/.venv/bin/python scripts/run_mohe_day.py
# 完成后重新检查文件、时间、计数和SHA256，无需重新计算：
/csim2/zhangzhiyu/MyProjects/DAS_Procee_Show/.venv/bin/python scripts/run_mohe_day.py --verify-only
```

默认读取 `/mnt/nas-downstairs1/mohe_cut/z_cut` 的所有数字编号台站子目录，
计算2026-07-08北京时间08:00–18:00（UTC00:00–10:00）。`--date`、
`--start-hour`、`--end-hour`、`--data-dir` 和 `--output-dir` 可指定其他单日任务。
输出为 `outputs/mohe_20260708_day_0800_1800/cc_stack_0001_n60.h5` 至
`cc_stack_0010_n60.h5`；每个文件包含全部1770对，float32数据轴为源、延迟、接收，
形状 `(60, 2001, 60)`。本入口只产生小时结果，不额外做全日叠加。

一次读取所选10小时波形及有效性掩码（约5.03 GiB），每小时整块上传CUDA，
短窗batch为12，`time_downsample=2`使用抗混叠FIR后降至250 Hz。
参数为10秒窗、5秒步长、2–30 Hz、±4秒延迟，短窗→分钟→小时逐对PWS。
跨小时短窗按起点归属并补齐重叠，计算终点外的数据不参与。同步保存并逐小时验收，
第一个小时验收通过后继续；每次重跑都校验续算清单，跳过完整文件并修复损坏文件。

台站坐标来自各自 `DigiSolo.LOG` 在所选UTC日期的有效GPS定位，取经纬度中位数，
保存数量、离散程度、海拔和来源到 `stations_gps.csv` 及H5元数据。
H5另存WGS84坐标、上三角1770对的 `pair_distance_m` 和对称 `station_distance_m`。
二维地理布局不生成 `profileX`/`dx`/`dshot`；现有一维测线绘图器会明确拒绝此布局。
CSV现在接受 `station_id,latitude_deg,longitude_deg`，仍兼容已有 `station_id,distance_m`。

`sac_sampling_rate_hz=500`确认SAC头间隔与 `float32(1/500)` 完全一致，
使用准确标称采样网格并保留原头间隔；不会接受真实采样率不匹配的数据。
生成的 `run_config.json` 包含完整路径、参数及UTC范围，可以交给 `scripts/cli.py --config` 使用。
`preflight.json`、`cuda_reference_validation.json`、`first_hour_validation.json`、
`validation.json` 和 `run.log` 记录输入预检、真实小样本CPU/CUDA比较、首小时验收、
最终文件验收和分阶段性能；`task_manifest.json` 记录输入身份及文件SHA256。

### 跨日期筛选与最终叠加

```bash
bash run_node_stack.sh --list-only  # 仅生成selection.json清单
bash run_node_stack.sh             # 筛选并叠加到stack_final.h5
```

所有参数见`config/node_stack_config.jsonc`。默认递归读取`outputs/node_ccf/`所有
`cc_stack_*.h5`，按文件内UTC时间排序，以北京时间筛选，全部文件叠加为一个H5。
文件路径包含日期子目录，来源路径、SHA256、时间区间及各文件GPS均保存到结果。
输出固定为`outputs/node_ccf_restack/stack_final.h5`；改筛选条件进行新实验时可更换`output_dir`。

- 全部数据：`start_date/end_date/start_time/end_time/first_n_days/stack_count`均为`null`。
- 每天08–18点：`start_time="08:00"`、`end_time="18:00"`，日期不限。
- 前3个有匹配数据的日期：`first_n_days=3`。
- 指定日期：`start_date="2026-07-08"`、`end_date="2026-07-11"`，首尾均包含。
- 只叠加前20个文件：`stack_count=20`；`null`为全部。数量不足明确报错。

条件可组合，顺序是日期/每日时间筛选→前N个有效日期→前N个文件。
H5已经是一个时段的叠加结果，不能恢复子时段，只有完整落入每日时段的文件才选中；
跨边界文件记入`selection.json`并排除，不进行伪裁剪。重叠文件会报错，避免混入不同参数任务。
每对有效文件等权叠加，缺测配对不参与该对叠加，支持PWS或线性平均。
跨天允许任务ID不同，但台站顺序、延迟轴、采样率、处理和TSI参数必须兼容。
每天GPS相对首文件的位移限值由`geometry_tolerance_m`控制，默认10米；
输出几何使用首文件，所有源几何保留在`source_geometry_json`中，超限会停止。

绘图直接使用该最终文件：

```bash
input_file="$PWD/outputs/node_ccf_restack/stack_final.h5" bash run_mohe_day_plot.sh
```

Shell参数统一使用小写，例如：

```bash
config=/path/to/config.jsonc ./run_mohe_node_ccf.sh --preflight
input_file=/path/to/final.h5 output_dir=/path/to/figures ./run_mohe_day_plot.sh
```

当前shell文件的参数说明与可选项直接写在对应赋值前。
`dasqt_repo_dir`、`dasqt_venv_activate`也使用小写；Bash内置`BASH_SOURCE`、`PWD`保持原名。

### 固定台站几何

当前`config/cc_config.jsonc`使用`station_csv="./config/stations_fixed.csv"`。
固定表来自2026-07-07完整60台站GPS表，来源与SHA256记录于
`config/stations_fixed_source.json`，原始定位数量和离散程度保留在CSV中。
全任务按此表计算距离和台站顺序，逐天读取波形，不再要求每日GPS。
输出到`outputs/node_ccf_fixed/`；旧`outputs/node_ccf/`保留，不与固定几何结果混用。
这要求台站在部署期内没有移动。需要更新几何时，应创建另一份CSV并更换输出根目录。
每日`preflight.json`保存坐标模式、固定表路径和SHA256，H5保留固定坐标及GPS来源。

```bash
./run_mohe_node_ccf.sh
# 叠加固定几何任务：把config/stack_config.jsonc的input_dir改为./outputs/node_ccf_fixed
```

坐标模式由`station_coordinate_mode`选择，默认`fixed`：

- `fixed`：使用`station_csv`固定表；该路径省略或为null时使用`config/stations_fixed.csv`。
- `daily_gps`：忽略`station_csv`，每天从各台站DigiSolo.LOG读取对应日期GPS；缺失即报错。

切换到每日模式时，日期输出目录自动追加`_daily_gps`，不会混入固定几何结果。
每日模式的GPS日期依据计算区间中点的UTC日期；坐标和来源均保存到每日结果。

固定坐标模式在任务开始时只读取、解析一次CSV并计算SHA256，
所有日期共享同一份内存台站记录。每天的预检、计算和结果校验也复用该缓存，
不再读取CSV或GPS。运行期间更改CSV不会改变当前任务的几何；下次启动才会重新读取。
波形目录和SAC头仍按当天范围检查，保证逐天加载正确数据。

### 当前多日叠加入口

`./run_mohe_day_stack.sh`已统一使用`config/stack_config.jsonc`，
默认递归读取`outputs/node_ccf_fixed/`，全部文件合为`outputs/stack/stack_final.h5`。
不再要求10个文件；每天的30分钟、1小时或2小时结果按H5内时间筛选。

```bash
./run_mohe_day_stack.sh --list-only
./run_mohe_day_stack.sh
./run_mohe_day_stack.sh --start-date 2026-07-08 --end-date 2026-07-11
./run_mohe_day_stack.sh --first-n-days 3
./run_mohe_day_stack.sh --start-time 08:00 --end-time 18:00
./run_mohe_day_stack.sh --stack-count 20
```

配置条件与命令行条件组合，命令行覆盖同名配置。用`--output-dir`为不同筛选保存独立结果。
`selection.json`记录选中文件数、天数、路径和被排除的边界文件。
波形已在每个H5内叠加，选择每日时段只包含完整落入该时段的H5。

### Park相移法频散

```bash
# 原F-J方法
./run_mohe_day_plot.sh
# Park相移法；仍只读取最终叠加H5
 dispersion_method=park ./run_mohe_day_plot.sh
# 可选负延迟反转或两分支线性平均
 dispersion_method=park park_branch=negative ./run_mohe_day_plot.sh
 dispersion_method=park park_branch=symmetric ./run_mohe_day_plot.sh
```

参数集中在shell开头：`dispersion_method`可选`fj`（默认）/`park`；
`park_branch`可选`positive`（默认）/`negative`/`symmetric`。
正负分支都以零延迟为时间起点，负分支先反转；相移法保留复频谱相位，
逐道逐频率归一化后计算`|sum(U/|U| * exp(+i*2π*f*r/c))|`。
NumPy使用负指数FFT，因此补偿相位使用正号。各有效道等权，不使用F-J的J0核和r dr权重。
全配对谱按有效台站对等权，分箱谱按非空距离箱等权；它们是二维节点互相关的径向适配，
不是原文中的线性主动源炮集，方向性及介质不均匀仍可能影响成像。
仅有一侧±4秒时窗时FFT频率间隔约0.25Hz，扫描步长不会增加实际分辨率。

两种方法的H5分别写入`figures/dispersion/fj/`与`figures/dispersion/park/`，避免覆盖频散文件。
Park文件保存逐频率归一化的`energy`、未作显示归一化的`coherence`、有效频谱道数、分支和来源校验值。
图e/f自动显示实际方法名称。计算支持`torch_cuda`和`cpu`，每次用CPU参考核对部分频率。
参考：[Park, Miller & Xia (1998)](https://www.parkseismic.com/wp-content/uploads/2021/06/ParkEtAl1998.pdf)。

### SeisLib节点双站频散GUI

DAS主界面菜单栏新增“节点双站频散”。在本目录也可运行：

```bash
./two_station.sh
```

参数见`config/two_station_config.jsonc`。默认1–10Hz、500–4000m/s，只读取最终
`outputs/stack/stack_final.h5`。点击“加载 / 重开任务”，再计算/编辑并确认总体参考，
完成20个代表配对后人工复核，才启动全量。数据源可选`data`或存在的`data_raw`。
总体参考复用DAS自动跟踪，逐对曲线由SeisLib J0零交叉算法提取；关闭相邻炮融合。

输出根目录为`outputs/two_station/`，每个独立任务保存一个
`two_station_dispersion.h5`以及参考CSV、曲线汇总CSV、任务清单和日志。
支持候选谱缓存、续算、取消、人工改点和撤销；换参考不会覆盖人工编辑。
参数改变后点击重新加载，扫描范围不能超出源H5处理频带。
曲线中的质量标记区分短距离、延迟窗不足和补全点；通过只是可复核，不宣称已验证基阶。
完整说明见DAS仓库的`dasqt/features/node_two_station/README.md`。
