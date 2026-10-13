# 节点计算脚本

主入口：`cli.py`（配置与预检）、`node_pipeline.py`（整数时间窗、两级调度）。
`node_reader.py` 保留旧导入名，实际实现移至node_io。

功能模块和详细文件职责见：`node_io/README.md`、`node_compute/README.md`、
`node_output/README.md`。它们由主入口调用，不作为独立CLI。

`stack_node_cc_mat.py`逐文件流式二次叠加；`plot_node_ccf.py`按真实profileX
绘图，无效配对留空，Times New Roman；`benchmark_node_pairs.py`验证同一新定义的
原生CUDA/Torch精度与性能。每个CLI文件头包含用途、运行示例、输出说明，参数
通过--help查看。配套一键shell在上一级，激活项目.venv并使用uv。

测试位于主项目tests/node_ccf，方法与科学定义见上一级METHODS.md。

`audit_node_ccf.py`流式检查完成后的全部时间组、SHA256、精确计数、有效掩码、
延迟互易性和总叠加文件来源；配套上一级`run_node_ccf_audit.sh`，生成JSON报告。
