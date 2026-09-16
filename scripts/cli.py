#!/usr/bin/env python3
"""Command line entrypoint for ordered-node SAC cross correlation.

Purpose
-------
Read a station-order CSV, map each station to its SAC file(s), and reuse the
DAS CCF implementation.  By default the common time range is preloaded into
RAM. Short-window correlations are batched into 60-second blocks, followed by a
second stack every ``save_every`` blocks, and written as MAT files.

Usage
-----
The recommended entrypoint is ``bash examples/04_node_ccf/run_node_ccf.sh``.
For direct use, pass ``--config`` plus optional ``--csv``, ``--data-dir`` and
``--output-dir`` overrides.  Correlation and grouping overrides are available
through the corresponding ``--read-mode``, ``--cc-backend``, ``--cc-device``,
``--cc-batch-chunks``, ``--cc-len``, ``--step-s``, ``--minute-stack-s``,
``--save-every`` and time-range options; ``--help`` prints the full list.

Output
------
The output directory contains ``cc_stack_XXXX_nN.mat`` files.  Each file has a
DAS-compatible ``data`` array with shape ``(source, lag, receiver)`` and
metadata including ``profileX``, station IDs, pair indices, and the actual
number of one-minute and short-window stacks.

Example::

    uv run python examples/04_node_ccf/scripts/cli.py --csv stations.csv \
      --data-dir /Volumes/CSIM/2026SaErTuoHai_passive --output-dir outputs/node
"""

from __future__ import annotations

import argparse
from pathlib import Path

from node_pipeline import NodeCCFConfig, run_node_ccf

from dasqt.features.dispersion.backend.components.ccf.folder_pipeline import (
    load_config_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 CSV 顺序读取 SAC 节点并计算 DAS 兼容互相关"
    )
    parser.add_argument("--config", type=Path, help="复用 DAS 的 cc_config.jsonc")
    parser.add_argument(
        "--csv", type=Path, help="排序 CSV；也可在配置中设置 station_csv"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="SAC 节点目录；也可在配置中设置 node_data_dir"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="输出目录；也可在配置中设置 node_output_dir"
    )
    parser.add_argument(
        "--read-mode",
        choices=("preload", "window"),
        help="数据读取模式：preload 一次性读入内存，window 逐窗读取",
    )
    parser.add_argument(
        "--include-autocorr",
        action="store_true",
        help="额外计算台站自相关；默认只计算不同台站的无序对",
    )
    parser.add_argument("--pair-mode", choices=("all_pairs", "sliding"))
    parser.add_argument("--offset-m", type=float, help="滑动模式接收范围（米）")
    parser.add_argument("--dshot-m", type=float, help="滑动模式震源步长（米）")
    parser.add_argument("--cc-len", type=float, help="互相关短窗长度（秒）")
    parser.add_argument("--step-s", type=float, help="短窗步长（秒）")
    parser.add_argument(
        "--cc-backend",
        choices=("auto", "cpu", "torch", "cupy", "cpp"),
        help="CC 后端；GPU 可选 torch 或 cupy",
    )
    parser.add_argument(
        "--cc-device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="计算设备，例如 cuda",
    )
    parser.add_argument(
        "--cc-batch-chunks", type=int, help="一次送入后端的短窗数量"
    )
    parser.add_argument(
        "--minute-stack-s", type=float, help="先将多少秒短窗叠加为一个分钟段"
    )
    parser.add_argument("--maxlag", type=float, help="最大延迟（秒）")
    parser.add_argument("--save-every", type=int, help="每多少分钟保存一个 MAT 文件")
    parser.add_argument("--start-utc", help="UTC 起始时间，例如 2026-07-09T11:00:00Z")
    parser.add_argument("--end-utc", help="UTC 结束时间，例如 2026-07-10T11:00:00Z")
    args = parser.parse_args()
    base_config = Path("examples/03_ccf/config/cc_config.jsonc")
    config_path = args.config or Path("examples/04_node_ccf/config/cc_config.jsonc")
    params = load_config_json(base_config) if base_config.exists() else {}
    if config_path.exists():
        params.update(load_config_json(config_path))
    if args.cc_backend:
        params["cc_backend"] = args.cc_backend
    if args.cc_device:
        params["cc_device"] = args.cc_device
    if args.cc_batch_chunks is not None:
        params["cc_batch_chunks"] = args.cc_batch_chunks
    run_node_ccf(
        NodeCCFConfig(
            csv_path=args.csv or Path(params.get("station_csv", "stations.csv")),
            data_dir=args.data_dir
            or Path(
                params.get("node_data_dir", "/Volumes/CSIM/2026SaErTuoHai_passive")
            ),
            output_dir=args.output_dir
            or Path(
                params.get("node_output_dir", "examples/04_node_ccf/outputs/node_ccf")
            ),
            read_mode=args.read_mode or params.get("read_mode", "preload"),
            include_autocorr=bool(
                args.include_autocorr or params.get("include_autocorr", False)
            ),
            pair_mode=args.pair_mode or params.get("pair_mode", "all_pairs"),
            cc_len=args.cc_len or float(params.get("cc_len", 5.0)),
            step_s=args.step_s
            or float(params.get("cc_step_s", params.get("cc_len", 5.0))),
            minute_stack_s=args.minute_stack_s
            or float(params.get("minute_stack_s", 60.0)),
            maxlag=args.maxlag or float(params.get("maxlag", 0.4)),
            offset_m=args.offset_m or float(params.get("offset", 50.0)),
            dshot_m=args.dshot_m or float(params.get("dshot", 2.5)),
            dtype=str(params.get("cc_dtype", "float32")),
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            cc_params=params,
            save_every=args.save_every
            or int(params.get("save_every", params.get("next_data", 30))),
        )
    )


if __name__ == "__main__":
    main()
