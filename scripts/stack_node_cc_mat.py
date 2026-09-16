#!/usr/bin/env python3
"""Second-level stack for node CCF MAT files.

This is the standalone-node counterpart of DAS ``stack_saved_cc_mat.py``. It
uses the shared DAS ``stack_cc_chunks`` backend, preserves ``profileX`` and
writes ``restack_*.mat``. Plotting is intentionally handled by
``plot_node_ccf.py`` so irregular node coordinates are drawn correctly.

Example
-------
    uv run python scripts/stack_node_cc_mat.py \
      --input-dir ./outputs/node_ccf \
      --output-dir ./outputs/node_ccf_restack \
      --glob 'cc_stack_*.mat' --stack-size None \
      --stack-method pws --cc-backend torch --cc-device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from dasqt.features.dispersion.backend.components.ccf import stack_cc_chunks
from scipy.io import loadmat, savemat

LOG = logging.getLogger(__name__)


def _optional_int(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    result = int(text)
    if result <= 0:
        raise ValueError("stack-size/interval must be positive or None")
    return result


def _scalar(mat: dict[str, Any], key: str, default: float) -> float:
    if key not in mat:
        return float(default)
    value = np.asarray(mat[key]).squeeze()
    return float(value.reshape(-1)[0]) if value.size else float(default)


def load_mat(path: Path) -> tuple[np.ndarray, float, float, float | None, np.ndarray]:
    mat = loadmat(path)
    if "data" not in mat or "dt" not in mat or "dx" not in mat:
        raise ValueError(f"{path.name} must contain data, dt and dx")
    data = np.asarray(mat["data"], dtype=np.float32).squeeze()
    if data.ndim == 2:
        data = data[None, :, :]
    if data.ndim != 3:
        raise ValueError(f"{path.name} data must be 2D/3D, got {data.shape}")
    dt = _scalar(mat, "dt", 1.0)
    dx = _scalar(mat, "dx", 1.0)
    dshot = _scalar(mat, "dshot", dx) if "dshot" in mat else None
    profile = np.asarray(mat.get("profileX", []), dtype=float).squeeze()
    if profile.ndim != 1 or profile.size != data.shape[-1]:
        profile = np.arange(data.shape[-1], dtype=float) * dx
    return data, dt, dx, dshot, profile


def build_windows(
    files: list[Path], stack_size: int | None, interval: int | None
) -> list[tuple[int, int, list[Path]]]:
    size = len(files) if stack_size is None else stack_size
    if size <= 0 or len(files) < size:
        raise ValueError(f"Not enough files for stack-size={size}: found {len(files)}")
    if interval is None:
        return [(0, size - 1, files[:size])]
    windows = []
    start = 0
    while start + size <= len(files):
        end = start + size - 1
        windows.append((start, end, files[start : end + 1]))
        start += interval
    return windows


def stack_window(
    files: list[Path],
    *,
    stack_method: str,
    cc_backend: str,
    cc_device: str,
    cc_dtype: str,
) -> tuple[np.ndarray, float, float, float | None, np.ndarray]:
    loaded = [load_mat(path) for path in files]
    reference = loaded[0]
    data = np.stack([item[0] for item in loaded], axis=0)
    if any(item[0].shape != reference[0].shape for item in loaded[1:]):
        raise ValueError("MAT files in one stack window have different data shapes")
    if any(not np.isclose(item[1], reference[1]) for item in loaded[1:]):
        raise ValueError("MAT files in one stack window have different dt")
    if any(not np.isclose(item[2], reference[2]) for item in loaded[1:]):
        raise ValueError("MAT files in one stack window have different dx")
    if stack_method == "mean":
        stacked = np.mean(data, axis=0)
    else:
        # DAS stack_cc_chunks stacks its second axis. Move source shots there
        # so each source is stacked independently across input MAT files.
        chunks = np.transpose(data, (1, 0, 2, 3))
        stacked = stack_cc_chunks(
            chunks,
            dt=reference[1],
            smethod=stack_method,
            params={
                "cc_backend": cc_backend,
                "cc_device": cc_device,
                "cc_dtype": cc_dtype,
            },
        )
    return (
        np.asarray(stacked, dtype=np.float32),
        reference[1],
        reference[2],
        reference[3],
        reference[4],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="节点 CCF MAT 二次叠加。")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="cc_stack_*.mat")
    parser.add_argument("--stack-size", default="None")
    parser.add_argument("--interval", default="None")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--stack-method", choices=("pws", "robust", "linear", "mean"), default="pws"
    )
    parser.add_argument(
        "--cc-backend", choices=("auto", "cpu", "torch"), default="torch"
    )
    parser.add_argument(
        "--cc-device", choices=("auto", "cpu", "mps", "cuda"), default="cuda"
    )
    parser.add_argument("--cc-dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    files = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched: {input_dir}/{args.glob}")
    if args.start_index < 0 or args.start_index >= len(files):
        raise ValueError("start-index is outside the input MAT list")
    selected = files[args.start_index :]
    windows = build_windows(
        selected, _optional_int(args.stack_size), _optional_int(args.interval)
    )
    LOG.info("input=%s files=%d windows=%d", input_dir, len(selected), len(windows))
    LOG.info(
        "stack method=%s backend=%s device=%s",
        args.stack_method,
        args.cc_backend,
        args.cc_device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (start, end, window_files) in enumerate(windows, start=1):
        data, dt, dx, dshot, profile = stack_window(
            window_files,
            stack_method=args.stack_method,
            cc_backend=args.cc_backend,
            cc_device=args.cc_device,
            cc_dtype=args.cc_dtype,
        )
        output = (
            output_dir
            / f"restack_{index:04d}_n{len(window_files)}_idx{start:04d}-{end:04d}.mat"
        )
        savemat(
            output,
            {
                "data": data,
                "dt": dt,
                "dx": dx,
                "dshot": dx if dshot is None else dshot,
                "profileX": profile,
                "source_files": np.asarray(
                    [path.name for path in window_files], dtype=object
                ),
            },
        )
        LOG.info("saved %s", output.name)


if __name__ == "__main__":
    main()
