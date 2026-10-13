#!/usr/bin/env python3
"""节点 SAC 全配对互相关 CLI（C++/CUDA 默认，不含 offset 模式）。

用途：按 CSV 台站顺序读取 SAC，补零FFT后白化并直接频域互相关，逐对PWS，
短窗→分钟→30分钟MAT。默认校验任务清单并续算。
运行：bash examples/04_node_ccf/run_node_ccf.sh [CSV] [SAC目录] [输出目录]
直接运行：在项目 .venv 激活后，uv run --no-sync python
examples/04_node_ccf/scripts/cli.py --config examples/04_node_ccf/config/cc_config.jsonc
--start-utc 2026-07-09T11:00:00Z --end-utc 2026-07-09T12:00:00Z
输出：cc_stack_XXXX_nN.mat、task_manifest.json，终端输出库存/进度/后端日志。
--preflight 仅检查输入、参数和时间范围，不读取全部波形或创建输出。
参数说明见各 add_argument(help=...)；--help 显示完整说明。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from node_compute.parameters import DEFAULTS
from node_pipeline import NodeCCFConfig, choose_read_mode, prepare, run_node_ccf

from dasqt.features.dispersion.backend.components.ccf.folder_pipeline import (
    load_config_json,
)

LOG = logging.getLogger(__name__)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(
        description="节点全配对频域白化互相关，逐对PWS与原生CUDA"
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config/cc_config.jsonc",
        help="独立节点JSONC配置，不继承DAS几何参数",
    )
    for key, help_text in [
        ("csv", "台站排序CSV"),
        ("data-dir", "SAC目录"),
        ("output-dir", "MAT和任务清单输出目录"),
    ]:
        parser.add_argument("--" + key, type=Path, help=help_text)
    parser.add_argument(
        "--read-mode",
        choices=("auto", "preload", "block"),
        help="auto按可用内存50%%选择读取策略",
    )
    parser.add_argument(
        "--cc-backend",
        choices=("cpp_cuda", "cpu", "torch"),
        help="cpp_cuda默认；cpu/torch是显式参考实现",
    )
    parser.add_argument(
        "--cc-device", choices=("cpu", "cuda"), help="cpp_cuda必须cuda；cpu必须cpu"
    )
    for name, key, type_, help_text in [
        ("cc-len", "cc_len", float, "短窗秒数"),
        ("step-s", "cc_step_s", float, "短窗步长秒数"),
        ("minute-stack-s", "minute_stack_s", float, "第一层时间块秒数"),
        ("save-every", "save_every", int, "每多少第一层时间块保存"),
        ("maxlag", "maxlag", float, "最大延迟秒数"),
        ("freqmin", "freqmin", float, "带通下界Hz"),
        ("freqmax", "freqmax", float, "带通上界Hz"),
        ("time-downsample", "time_downsample", int, "GPU抗混叠降采样整数因子"),
        ("cc-batch-chunks", "cc_batch_chunks", int, "最大窗口批次数"),
        ("cc-pair-batch", "cc_pair_batch", int, "最大配对批次；0自动"),
        (
            "cc-memory-budget-bytes",
            "cc_memory_budget_bytes",
            int,
            "临时GPU工作区预算字节；0自动，常驻统计量另计",
        ),
        ("cc-cuda-device", "cc_cuda_device", int, "CUDA设备编号"),
    ]:
        parser.add_argument("--" + name, dest=key, type=type_, help=help_text)
    parser.add_argument(
        "--stacking-method",
        choices=("linear", "pws"),
        help="每层叠加方式；PWS指数2，独立节点对",
    )
    parser.add_argument("--start-utc", help="带时区起点，例如2026-07-09T11:00:00Z")
    parser.add_argument("--end-utc", help="带时区终点，半开区间")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="校验并跳过完整输出；默认启用",
    )
    parser.add_argument(
        "--preflight", action="store_true", help="只检查SAC头和时间范围，输出JSON摘要"
    )
    parser.add_argument(
        "--print-output-dir", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    values = load_config_json(args.config)
    paths = {}
    for key, arg, default in [
        ("station_csv", "csv", "config/line_380_stations.csv"),
        ("node_data_dir", "data_dir", "/mnt/nas-changbai/2026SaErTuoHai_passive"),
        ("node_output_dir", "output_dir", "outputs/node_ccf"),
    ]:
        path = getattr(args, arg)
        if path is None:
            path = Path(values.get(key, default)).expanduser()
            if not path.is_absolute():
                path = root / path
        paths[arg] = path.expanduser().resolve()
    removed = {
        "offset",
        "dshot",
        "xshot",
        "receiver_side",
        "min_branch_traces",
        "cc_threads",
        "cc_memory_budget_mb",
    }
    for key in removed.intersection(values):
        LOG.warning("Ignoring removed node parameter %s", key)
    params = dict(DEFAULTS)
    params.update(
        {
            key: value
            for key, value in values.items()
            if key in DEFAULTS or key in {"pair_mode", "include_autocorr", "fk_enabled"}
        }
    )
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    return NodeCCFConfig(
        paths["csv"],
        paths["data_dir"],
        paths["output_dir"],
        params,
        args.start_utc,
        args.end_utc,
    ), "output" if args.print_output_dir else args.preflight


def main():
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    config, preflight = parse_config()
    if preflight == "output":
        print(config.output_dir)
    elif preflight:
        catalog, params, start, end, total = prepare(config)
        print(
            json.dumps(
                {
                    "stations": len(catalog.stations),
                    "pairs": len(catalog.stations) * (len(catalog.stations) - 1) // 2,
                    "raw_fs": catalog.sampling_rate,
                    "samples": total,
                    "start_unix": start,
                    "end_unix": end,
                    "duration_hours": (end - start) / 3600,
                    "selected_read_mode": choose_read_mode(
                        params, total, len(catalog.stations)
                    ),
                    "output_dir": str(config.output_dir),
                },
                indent=2,
            )
        )
    else:
        run_node_ccf(config)


if __name__ == "__main__":
    main()
