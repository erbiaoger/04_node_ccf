# 节点 CCF 脚本

- `cli.py`：命令行入口，负责读取共享 DAS 配置和节点配置。
- `node_reader.py`：CSV 台站目录、SAC 分段索引和内存映射读取；预加载模式下负责一次性组装共同时间段。
- `node_pipeline.py`：复用现有 DAS `compute_cc_shot` 和 pairwise stacking，执行两级时间叠加并保存 DAS 兼容的 MAT 文件。
- `stack_node_cc_mat.py`：对第一层 `cc_stack_*.mat` 做二次叠加，复用 DAS `stack_cc_chunks`，输出带 `profileX` 的 `restack_*.mat`。
- `plot_node_ccf.py`：读取二次叠加后的 MAT，使用其中的 `profileX` 真实节点坐标生成汇总图、逐源节点图和平均图。

通常直接运行上级目录的 `run_node_ccf.sh`，不需要手动调用这些文件。
后处理由上级目录的 `run_node_ccf_stack.sh` 调用本目录的 `stack_node_cc_mat.py` 完成二次叠加，再调用坐标感知绘图脚本；叠加内核仍复用 DAS 的 `stack_cc_chunks`。
