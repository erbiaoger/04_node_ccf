# 节点 SAC 输入

- `catalog.py`：CSV 顺序、SAC 头校验、参考时间+b、半开区间、缺失节点和输入指纹。
- `reader.py`：`read_samples(start_timestamp, npts)` 按范围显式读取（每线程最多4 MiB输入临时缓冲），返回
  `(time, station)` float32 波形和同形状 bool 有效掩码，不进行信号处理。`workers=4` 可并行读取独立台站列；所有读取异常传播给调用方。
  库存扫描先筛选台站文件名，再查询元数据，减少NAS往返。
- `__init__.py`：提供公共导入；旧 `node_reader.py` 是兼容入口。

由上级 `cli.py` 调用；无需独立运行。CSV 至少两列 station_id、distance_m，
距离严格递增。缺测节点保留位置，间隙置无效，重叠文件和多组件歧义报错。

跨片段复用每台站一个文件句柄，减少CIFS反复打开开销；任务结束调用close关闭。

数字编号子目录也可作为SAC输入。按CSV顺序建立目录，只对任务范围重叠的已识别日文件读取头。
地理布局CSV支持 `station_id,latitude_deg,longitude_deg`，不要求虚构一维距离。
`gps.py`从DigiSolo日志的GPS分节读取实际Fix UTC日期和成对坐标，生成每日中位数、数量及离散程度。
可选显式标称采样率仅在float32 SAC头间隔一致时采用，记录原间隔并按准确采样网格安排日边界。
