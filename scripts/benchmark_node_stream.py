#!/usr/bin/env python3
"""Compare serial and pipelined real-SAC CUDA runs with a warm filesystem cache.

Each repetition writes to a fresh directory; order alternates to reduce bias.
Reports end-to-end wall time, exposed I/O wait, CUDA timings and numerical parity.
Catalog scanning is included in total wall time. NAS traffic from other jobs can
still affect measurements; this benchmark does not drop system caches.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from cli import parse_config
from node_compute.schedule import block_bounds, groups
from node_io import NodeSACReader
from node_output.artifacts import atomic_json
from node_output.artifacts import load_result as loadmat
from node_pipeline import prepare, run_node_ccf

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="真实SAC串行/并行流水线对照；预热缓存、交替运行"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/cc_config.jsonc",
    )
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="新的独立基准目录"
    )
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.output_dir.exists():
        parser.error("repeats>=1且output-dir必须不存在，禁止复用基准结果")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    config, _ = parse_config(
        [
            "--config",
            str(args.config),
            "--output-dir",
            str(args.output_dir),
            "--start-utc",
            args.start_utc,
            "--end-utc",
            args.end_utc,
            "--read-mode",
            "block",
        ]
    )
    catalog, params, start, _, total = prepare(config)
    reader = NodeSACReader(catalog, workers=8)
    LOG.info("Warming filesystem cache in bounded blocks")
    for group in groups(total, reader.fs, params):
        lo, hi = block_bounds(group, total, reader.fs, params)
        data, valid = reader.read_samples(start + lo / reader.fs, hi - lo)
        del data, valid
    reader.close()
    report = {
        "cache_policy": "warm; alternating order; other NAS jobs may interfere",
        "runs": {},
    }
    outputs = {}
    for repeat in range(args.repeats):
        order = ("serial", "parallel") if repeat % 2 == 0 else ("parallel", "serial")
        for name in order:
            directory = args.output_dir / f"{name}_{repeat}"
            runtime = dict(
                config.cc_params,
                io_workers=1 if name == "serial" else 8,
                prefetch_blocks=0 if name == "serial" else 1,
                read_hours=0 if name == "serial" else config.cc_params["read_hours"],
                cc_upload_block=0 if name == "serial" else 1,
                async_save=0 if name == "serial" else 1,
            )
            current = replace(config, output_dir=directory, cc_params=runtime)
            before = time.perf_counter()
            run_node_ccf(current)
            elapsed = time.perf_counter() - before
            diag = json.loads((directory / "task_manifest.json").read_text())[
                "diagnostics"
            ]
            diag["total_wall_seconds"] = elapsed
            report["runs"].setdefault(name, []).append(diag)
            outputs[name] = directory
    error = 0.0
    for a, b in zip(
        sorted(outputs["serial"].glob("*.h5")),
        sorted(outputs["parallel"].glob("*.h5")),
        strict=True,
    ):
        aa, bb = loadmat(a), loadmat(b)
        for key in (
            "valid_mask",
            "valid_short_window_count",
            "valid_minute_count",
            "task_id",
        ):
            np.testing.assert_array_equal(aa[key], bb[key])
        np.testing.assert_allclose(aa["data"], bb["data"], rtol=1e-3, atol=2e-4)
        error = max(error, float(np.max(np.abs(aa["data"] - bb["data"]))))
    report["max_abs_error"] = error
    report["median_wall_seconds"] = {
        name: statistics.median(run["wall_seconds"] for run in runs)
        for name, runs in report["runs"].items()
    }
    report["serial_to_parallel_ratio"] = (
        report["median_wall_seconds"]["serial"]
        / report["median_wall_seconds"]["parallel"]
    )
    atomic_json(args.output_dir / "results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
