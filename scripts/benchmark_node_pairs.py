#!/usr/bin/env python3
"""Benchmark the identical node-pair definition on native CUDA and Torch CUDA.

Purpose: include input upload, preprocessing, correlation, two-level PWS and
finished-output download; exclude SAC I/O and MAT writing. Results are checked
before performance is reported. Every repetition creates a new session and
recomputes all signals; no waveform/result caching or backend fallback.
Run in project .venv: uv run --no-sync python
examples/04_node_ccf/scripts/benchmark_node_pairs.py --output outputs/node_benchmark.json
Arguments and Chinese descriptions live in main() below; --help lists them.
Outputs: terminal progress and JSON containing medians, parity errors and actual
device diagnostics. Exit nonzero for missing CUDA or failed numerical parity.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np
from node_compute.backend import create_session
from node_compute.parameters import validate
from node_output.artifacts import atomic_json

LOG = logging.getLogger(__name__)


def run(params, data, valid, starts, minutes):
    import torch

    torch.cuda.synchronize()
    begin = time.perf_counter()
    session = create_session(params, data.shape[1])
    session.add_block(data, valid, starts, minutes)
    result = session.finish_group()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    diagnostics = session.diagnostics()
    del session
    return result, seconds, diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="节点全配对原生CUDA/Torch数值与性能验证，含传输"
    )
    parser.add_argument("--stations", type=int, default=34, help="节点数")
    parser.add_argument("--seconds", type=int, default=120, help="模拟数据时长秒数")
    parser.add_argument("--sampling-rate", type=int, default=250, help="原始采样率Hz")
    parser.add_argument(
        "--repeats", type=int, default=3, help="稳态重复次数，报告中位数"
    )
    parser.add_argument("--output", type=Path, required=True, help="输出JSON路径")
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds < 10:
        parser.error("repeats>=1 and seconds>=10 required")
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    data = (
        np.random.default_rng(20261014)
        .normal(size=(args.seconds * args.sampling_rate, args.stations))
        .astype(np.float32)
    )
    valid = np.ones_like(data, dtype=bool)
    starts = np.arange(
        0,
        len(data) - 10 * args.sampling_rate + 1,
        5 * args.sampling_rate,
        dtype=np.int64,
    )
    minutes = starts // (60 * args.sampling_rate)
    results = {}
    for backend in ("cpp_cuda", "torch"):
        params = validate(
            {"cc_backend": backend, "cc_device": "cuda"}, float(args.sampling_rate)
        )
        _, warmup, _ = run(params, data, valid, starts, minutes)
        timings = []
        for index in range(args.repeats):
            result, elapsed, diag = run(params, data, valid, starts, minutes)
            timings.append(elapsed)
            LOG.info("%s repetition %d: %.6f s", backend, index + 1, elapsed)
        results[backend] = {
            "warmup_seconds": warmup,
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "diagnostics": diag,
        }
        results[backend]["_result"] = result
    native, reference = (
        results["cpp_cuda"].pop("_result"),
        results["torch"].pop("_result"),
    )
    np.testing.assert_array_equal(native["short_count"], reference["short_count"])
    np.testing.assert_array_equal(native["valid_count"], reference["valid_count"])
    np.testing.assert_allclose(
        native["pairs"], reference["pairs"], rtol=1e-3, atol=2e-4
    )
    results.update(
        stations=args.stations,
        sampling_rate=args.sampling_rate,
        seconds=args.seconds,
        max_abs_error=float(abs(native["pairs"] - reference["pairs"]).max()),
        native_to_torch_ratio=results["cpp_cuda"]["median_seconds"]
        / results["torch"]["median_seconds"],
    )
    atomic_json(args.output, results)
    LOG.info("Verified benchmark: %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
