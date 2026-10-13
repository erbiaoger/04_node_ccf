# 节点 SAC 输入

- `catalog.py`：CSV 顺序、SAC 头校验、参考时间+b、半开区间、缺失节点和输入指纹。
- `reader.py`：`read_samples(start_timestamp, npts)` 按范围读取 memmap，返回
  `(time, station)` float32 波形和同形状 bool 有效掩码，不进行信号处理。
- `__init__.py`：提供公共导入；旧 `node_reader.py` 是兼容入口。

由上级 `cli.py` 调用；无需独立运行。CSV 至少两列 station_id、distance_m，
距离严格递增。缺测节点保留位置，间隙置无效，重叠文件和多组件歧义报错。
