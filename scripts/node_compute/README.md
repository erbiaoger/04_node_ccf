# 节点对计算

- `parameters.py`：独立默认值、采样网格/频带/后端校验和计算版本。
- `schedule.py`：全局整数采样窗口、分钟归属、输出组及降采样边界。
- `backend.py`：严格 C++/CUDA 边界，检查扩展的节点计算能力；无隐式回退。
- `reference.py`：独立 CPU/Torch 科学参考，逐对在线 PWS，包含一致 FIR。

由 `cli.py` 调用。生产 `cc_backend=cpp_cuda,cc_device=cuda`；显式 CPU 试算
需同时指定 `cc_backend=cpu,cc_device=cpu`。白化在至少2N−1补零后的频谱
上进行，随后直接构造互相关谱，不做白化后时域截断或再次FFT。

原生短窗batch支持跨分钟；CPU/Torch参考维持同一逐分钟PWS定义，用于核对数值和计数。
