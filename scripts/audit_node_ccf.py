#!/usr/bin/env python3
"""Audit completed node CCF outputs and their final restack without recomputing.

Purpose: verify all scheduled groups, manifest checksums, dimensions, finite
values, reciprocal lag reversal, validity masks, exact count limits and final
restack provenance. Streams one H5 at a time; no full history is loaded.
Run in the project .venv: uv run --no-sync python
examples/04_node_ccf/scripts/audit_node_ccf.py --input-dir examples/04_node_ccf/outputs/node_ccf_spectral_v1
--restack-dir examples/04_node_ccf/outputs/node_ccf_spectral_v1_restack
--output-json outputs/node_ccf_audit.json
Outputs: terminal progress and an atomic JSON quality report. Failure raises an
error and exits nonzero. Parameters have Chinese help text in main() below.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from node_compute.parameters import VERSION
from node_compute.schedule import groups
from node_output.artifacts import atomic_json, digest
from node_output.artifacts import load_result as loadmat

LOG = logging.getLogger(__name__)


def scalar(mat, key):
    return np.asarray(mat[key]).reshape(-1)[0].item()


def check_cube(mat, stations, lags):
    data = mat["data"]
    if (
        data.shape != (stations, lags, stations)
        or data.dtype != np.float32
        or not np.isfinite(data).all()
    ):
        raise ValueError("Invalid CCF cube shape/dtype or nonfinite values")
    if not np.array_equal(data, data.transpose(2, 1, 0)[:, ::-1, :]):
        raise ValueError("Source/receiver lag reversal differs")
    mask = np.asarray(mat["valid_mask"], dtype=bool)
    if (
        mask.shape != (stations, stations)
        or not np.array_equal(mask, mask.T)
        or np.diag(mask).any()
    ):
        raise ValueError("Invalid node validity mask")
    if np.any(data[np.arange(stations), :, np.arange(stations)]) or np.any(
        data.transpose(0, 2, 1)[~mask]
    ):
        raise ValueError("Diagonal/invalid pairs must have zero data")
    if str(scalar(mat, "computation_version")) != VERSION:
        raise ValueError("Wrong node scientific definition version")


def audit(input_dir, restack_dir, verify_checksums=True):
    manifest = json.loads((input_dir / "task_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Node calculation has not completed")
    identity = manifest["identity"]
    params = identity["parameters"]
    total = round((identity["end"] - identity["start"]) * params["samp_freq"])
    stations = len(identity["inputs"])
    source, receiver = np.triu_indices(stations, 1)
    lags = (
        2 * round(params["maxlag"] * params["samp_freq"] / params["time_downsample"])
        + 1
    )
    short = np.zeros(len(source), dtype=np.uint64)
    minutes = np.zeros_like(short)
    files = []
    hashes = []
    windows = 0
    valid_pair_min = len(source)
    for group in groups(total, params["samp_freq"], params):
        record = manifest["completed"].get(str(group.index))
        path = input_dir / (
            record["file"]
            if record
            else f"cc_stack_{group.index:04d}_n{group.minute_count}.h5"
        )
        if not record or record["file"] != path.name or not path.is_file():
            raise ValueError(f"Missing completed group: {path}")
        if verify_checksums and digest(path) != record["sha256"]:
            raise ValueError(f"H5 checksum differs: {path}")
        mat = loadmat(path)
        check_cube(mat, stations, lags)
        if str(scalar(mat, "task_id")) != manifest["task_id"] or scalar(
            mat, "actual_short_window_count"
        ) != len(group.starts):
            raise ValueError(f"Wrong task/window count: {path}")
        sc = np.asarray(mat["valid_short_window_count"], dtype=np.uint64)[
            source, receiver
        ]
        mc = np.asarray(mat["valid_minute_count"], dtype=np.uint64)[source, receiver]
        if (
            (sc > len(group.starts)).any()
            or (mc > group.minute_count).any()
            or not np.array_equal(mat["valid_mask"][source, receiver] > 0, mc > 0)
        ):
            raise ValueError(f"Invalid effective counts: {path}")
        short += sc
        minutes += mc
        windows += len(group.starts)
        valid_pair_min = min(valid_pair_min, int((mc > 0).sum()))
        files.append(path.name)
        hashes.append(record["sha256"])
        if group.index % 50 == 0:
            LOG.info("Audited %d groups", group.index)
    if len(manifest["completed"]) != len(files) or len(
        list(input_dir.glob("cc_stack_*.h5"))
    ) != len(files):
        raise ValueError("Unexpected extra output groups")
    restacks = sorted(restack_dir.glob("restack_*.h5"))
    if len(restacks) != 1:
        raise ValueError("Expected exactly one full-period restack")
    final = loadmat(restacks[0])
    check_cube(final, stations, lags)
    names = [
        str(np.asarray(x).reshape(-1)[0]) for x in final["source_files"].reshape(-1)
    ]
    recorded_hashes = [
        str(np.asarray(x).reshape(-1)[0])
        for x in final["source_file_sha256"].reshape(-1)
    ]
    if (
        names != files
        or recorded_hashes != hashes
        or str(scalar(final, "task_id")) != manifest["task_id"]
    ):
        raise ValueError("Final restack provenance differs")
    if not np.array_equal(
        short, final["valid_short_window_count"][source, receiver]
    ) or not np.array_equal(minutes, final["valid_minute_count"][source, receiver]):
        raise ValueError("Final restack effective counts differ")
    counts = final["valid_file_count"][source, receiver]
    if (counts > len(files)).any() or not np.array_equal(
        final["valid_mask"][source, receiver] > 0, counts > 0
    ):
        raise ValueError("Invalid final file counts")
    return {
        "status": "passed",
        "task_id": manifest["task_id"],
        "stations": stations,
        "pairs": len(source),
        "lags": lags,
        "groups": len(files),
        "scheduled_short_windows": windows,
        "valid_pairs_final": int((counts > 0).sum()),
        "minimum_valid_pairs_per_group": valid_pair_min,
        "valid_short_count_min": int(short.min()),
        "valid_short_count_max": int(short.max()),
        "valid_minute_count_min": int(minutes.min()),
        "valid_minute_count_max": int(minutes.max()),
        "start_unix": identity["start"],
        "end_unix": identity["end"],
        "duration_hours": (identity["end"] - identity["start"]) / 3600,
        "restack_file": str(restacks[0]),
        "checksums_verified": verify_checksums,
        "computation_diagnostics": manifest["diagnostics"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="流式审计节点全部H5、有效计数、互易性及总叠加来源"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="完成计算的H5与task_manifest目录"
    )
    parser.add_argument(
        "--restack-dir", type=Path, required=True, help="全时段总叠加H5目录"
    )
    parser.add_argument("--output-json", type=Path, required=True, help="质量报告JSON")
    parser.add_argument(
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否逐文件校验SHA256；默认启用",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    result = audit(args.input_dir, args.restack_dir, args.verify_checksums)
    atomic_json(args.output_json, result)
    LOG.info("Node audit passed: %s", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
