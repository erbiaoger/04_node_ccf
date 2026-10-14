#!/usr/bin/env python3
"""Tune CUDA short-window batch sizes using the same real SAC block in RAM.

Includes upload, preprocessing, correlation, per-minute and per-file PWS and
result download. Excludes SAC I/O, H5 writing and session construction. Reuses one session per
candidate, warming up before timing; this matches consecutive output groups. Verifies
counts exactly and CCFs within the established native numerical tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np
from cli import parse_config
from node_compute.backend import NativeSession
from node_compute.schedule import block_bounds, groups
from node_io import NodeSACReader
from node_output.artifacts import atomic_json
from node_pipeline import prepare

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="同一真实SAC数据块的跨分钟CUDA batch对照"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/cc_config.jsonc",
    )
    parser.add_argument("--start-utc")
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[12, 32, 64, 128, 256, 512]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pair-batch", type=int, default=0)
    parser.add_argument("--reuse-workspace", type=int, choices=(0, 1), default=1)
    parser.add_argument("--memory-budget-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.batches) < 1 or args.seconds <= 0:
        parser.error("repeats, batches and seconds must be positive")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    argv = [
        "--config",
        str(args.config),
        "--cc-backend",
        "cpp_cuda",
        "--cc-device",
        "cuda",
    ]
    if args.start_utc:
        argv += ["--start-utc", args.start_utc]
    config, _ = parse_config(argv)
    catalog, params, start, _, total = prepare(config)
    reader = NodeSACReader(catalog, workers=params["io_workers"])
    total = min(total, round(args.seconds * reader.fs))
    group = next(groups(total, reader.fs, params))
    lo, hi = block_bounds(group, total, reader.fs, params)
    try:
        data, valid = reader.read_samples(start + lo / reader.fs, hi - lo)
    finally:
        reader.close()
    reference = None
    report = {
        "stations": len(catalog.stations),
        "windows": len(group.starts),
        "runs": [],
    }
    for batch in args.batches:
        timings = []
        error = 0.0
        session = NativeSession(
            dict(
                params,
                cc_batch_chunks=batch,
                cc_pair_batch=args.pair_batch,
                cc_memory_budget_bytes=args.memory_budget_bytes,
                cc_upload_block=1,
                cc_reuse_workspace=args.reuse_workspace,
            ),
            len(catalog.stations),
        )
        for repeat in range(args.repeats + 1):
            session.clear_group()
            previous_diag = session.diagnostics()
            before = time.perf_counter()
            session.add_block(data, valid, group.starts - lo, group.minute_ids)
            result = session.finish_group()
            elapsed = time.perf_counter() - before
            diag = session.diagnostics()
            diag["compute_batches_per_run"] = (
                diag["compute_batches"] - previous_diag["compute_batches"]
            )
            if reference is None:
                reference = result
            for key in ("valid_count", "short_count"):
                np.testing.assert_array_equal(reference[key], result[key])
            np.testing.assert_allclose(
                reference["pairs"], result["pairs"], rtol=1e-3, atol=2e-4
            )
            error = max(error, float(abs(reference["pairs"] - result["pairs"]).max()))
            if repeat:
                timings.append(elapsed)
        del session
        record = {
            "requested_batch": batch,
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "max_abs_error": error,
            "diagnostics": diag,
        }
        report["runs"].append(record)
        LOG.info(
            "requested=%d actual_max=%d pairs_last=%d batches=%d median=%.6fs error=%.6g",
            batch,
            diag["window_batch_max"],
            diag["pair_batch_last"],
            diag["compute_batches_per_run"],
            record["median_seconds"],
            error,
        )
    report["fastest_requested_batch"] = min(
        report["runs"], key=lambda row: row["median_seconds"]
    )["requested_batch"]
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
