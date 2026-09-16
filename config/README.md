# 节点 CCF 配置

`cc_config.jsonc` 是节点工作流的完整配置，包含输入路径、台站对模式、频带、短窗、归一化、后端、降采样、滑动几何和 TSI 兼容参数。运行时仍会先读取 `examples/03_ccf/config/cc_config.jsonc`，再用本文件覆盖同名字段，因此两套 DAS CCF 参数保持兼容；命令行 `--config` 可以继续覆盖本文件。
