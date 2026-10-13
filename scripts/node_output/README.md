# 节点结果和续算

`artifacts.py` 提供原子 MAT/JSON 写入、SHA256 校验、任务清单和配对稠密化。
`result_fields` 保留 source×lag×receiver，反转延迟填充下三角，未计算对角线
置零；valid_mask 区分无效配对，计数采用 uint64，不以浮点编码。

由 `node_pipeline.py` 和 `stack_node_cc_mat.py` 使用。续算身份由SAC文件属性、
头信息、台站顺序、时间范围和科学参数组成；读取方式与GPU批次可变化。
损坏文件重新计算。参数/输入变化需使用新目录。生产目录不要同时运行两个
写入任务；计算结束后运行二次叠加。
