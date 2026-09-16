# 节点 CCF 脚本

- `cli.py`：命令行入口，负责读取共享 DAS 配置和节点配置。
- `node_reader.py`：CSV 台站目录、SAC 分段索引和内存映射时间窗读取。
- `node_pipeline.py`：复用现有 DAS `compute_cc_shot` 和 pairwise stacking，执行两级时间叠加并保存 DAS 兼容的 MAT 文件。

通常直接运行上级目录的 `run_node_ccf.sh`，不需要手动调用这些文件。
