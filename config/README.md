# 节点配置

`cc_config.jsonc` 是独立节点参数，默认CPP CUDA、逐窗RMA、补零频谱白化互相关、
逐对PWS。`line_380_stations.csv` 给出台站顺序与真实distance_m；现有34节点561对。
路径相对04_node_ccf根目录，命令行显式路径覆盖配置。不继承03_ccf配置。
没有offset/sliding/dshot模式。采样率从SAC头获得，频带越界时报错。

用法：`CONFIG=/path/to/cc_config.jsonc bash run_node_ccf.sh`。
主机读取自动按可用内存50%估算选择preload或block；CUDA工作区按空闲显存
自动规划，cc_memory_budget_bytes>0只约束临时工作区，常驻统计量另计。
