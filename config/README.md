# 节点 CCF 配置

`cc_config.jsonc` 是节点工作流的完整配置，包含输入路径、台站对模式、读取模式、频带、短窗、归一化、后端、降采样、滑动几何和 TSI 兼容参数。默认 `read_mode=preload`，会将共同时间段一次性加载到内存，再按 `cc_batch_chunks` 将短窗组成 batch 交给配置的 CPU/GPU 后端；内存不足时可改为 `read_mode=window`。运行时仍会先读取 `examples/03_ccf/config/cc_config.jsonc`，再用本文件覆盖同名字段，因此两套 DAS CCF 参数保持兼容；命令行 `--config` 可以继续覆盖本文件。
