#!/usr/bin/env python3
"""Plot restacked node CCF MAT files with the recorded station coordinates.

Purpose
-------
The DAS ``stack_saved_cc_mat.py`` script performs the second-level stack and
writes ``restack_*.mat`` files. This companion plotter uses the ``profileX``
array stored by the node workflow instead of assuming a uniform ``dx``.

Usage
-----
    uv run python scripts/plot_node_ccf.py \
      --input-dir ./outputs/node_ccf_restack \
      --plot-name node_cc_preview.png

Output
------
The input directory receives the summary plot named by ``--plot-name``. Each
MAT file also gets a ``shots_fig/<mat-stem>/`` directory containing one PNG per
source station and a ``*_merged_by_position.png`` average over source stations.
All figures use Times New Roman, matching the project plotting convention.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]

LOG = logging.getLogger(__name__)


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=float).ravel()
    if centers.size == 0:
        return np.array([0.0, 1.0])
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = 0.5 * (centers[:-1] + centers[1:])
    left = centers[0] - 0.5 * (centers[1] - centers[0])
    right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([left], mids, [right]))


def _scalar(mat: dict, key: str, default: float) -> float:
    if key not in mat:
        return float(default)
    value = np.asarray(mat[key]).squeeze()
    if value.size == 0:
        return float(default)
    return float(value.reshape(-1)[0])


def _load_node_mat(
    path: Path, fallback_profile: np.ndarray | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    mat = loadmat(path)
    if "data" not in mat:
        raise ValueError(f"{path.name} missing data")
    data = np.asarray(mat["data"], dtype=np.float32).squeeze()
    if data.ndim == 2:
        data = data[None, :, :]
    if data.ndim != 3:
        raise ValueError(f"{path.name} data must be 2D/3D, got {data.shape}")
    dt = _scalar(mat, "dt", 1.0)
    dx = _scalar(mat, "dx", 1.0)
    profile = np.asarray(mat.get("profileX", []), dtype=float).squeeze()
    if profile.ndim != 1 or profile.size != data.shape[-1]:
        profile = np.asarray(fallback_profile if fallback_profile is not None else [])
    if profile.ndim != 1 or profile.size != data.shape[-1]:
        profile = np.arange(data.shape[-1], dtype=float) * dx
    return data, dt, profile


def _plot_matrix(
    matrix: np.ndarray,
    dt: float,
    profile_x: np.ndarray,
    output: Path,
    *,
    title: str,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    nt, _ = matrix.shape
    t = (np.arange(nt, dtype=float) - nt // 2) * float(dt)
    vmax = float(np.percentile(np.abs(matrix), 99)) if matrix.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(9, 4))
    image = ax.pcolormesh(
        _centers_to_edges(profile_x),
        _centers_to_edges(t),
        matrix,
        cmap="RdBu_r",
        shading="auto",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_aspect("auto")
    ax.set_title(title)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time lag (s)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _load_fallback_profile(source_dir: Path | None) -> np.ndarray | None:
    if source_dir is None:
        return None
    files = sorted(path for path in source_dir.glob("cc_stack_*.mat") if path.is_file())
    if not files:
        return None
    mat = loadmat(files[0])
    profile = np.asarray(mat.get("profileX", []), dtype=float).squeeze()
    return profile if profile.ndim == 1 else None


def plot_node_results(
    input_dir: Path,
    plot_name: str,
    save_shot_figures: bool,
    source_dir: Path | None,
) -> None:
    files = sorted(path for path in input_dir.glob("restack_*.mat") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No restack MAT files found in {input_dir}")
    fallback_profile = _load_fallback_profile(source_dir)
    items = []
    for path in files:
        data, dt, profile_x = _load_node_mat(path, fallback_profile)
        items.append((path, data, dt, profile_x))

    if save_shot_figures:
        for path, data, dt, profile_x in items:
            shots_dir = input_dir / "shots_fig" / path.stem
            for source_index, gather in enumerate(data):
                _plot_matrix(
                    gather,
                    dt,
                    profile_x,
                    shots_dir / f"{path.stem}_shot_{source_index:04d}.png",
                    title=f"{path.stem} | source {source_index}",
                )
            _plot_matrix(
                np.mean(data, axis=0),
                dt,
                profile_x,
                shots_dir / f"{path.stem}_merged_by_position.png",
                title=f"{path.stem} | mean over source positions",
            )
            LOG.info("saved source figures: %s", shots_dir)

    ncols = min(3, len(items))
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    for index, (path, data, dt, profile_x) in enumerate(items):
        ax = axes.flat[index]
        matrix = data[0]
        nt = matrix.shape[0]
        t = (np.arange(nt, dtype=float) - nt // 2) * float(dt)
        vmax = float(np.percentile(np.abs(matrix), 99)) if matrix.size else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        image = ax.pcolormesh(
            _centers_to_edges(profile_x),
            _centers_to_edges(t),
            matrix,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"{path.stem} | source 0")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time lag (s)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.flat[len(items) :]:
        ax.axis("off")
    fig.tight_layout()
    output = input_dir / plot_name
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    LOG.info("saved summary plot: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 MAT 中 profileX 真实节点坐标绘制节点 CCF 结果。"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="restack MAT 所在目录。"
    )
    parser.add_argument(
        "--profile-source-dir",
        type=Path,
        help="可选的第一层 cc_stack_*.mat 目录，用于恢复 profileX。",
    )
    parser.add_argument(
        "--plot-name", default="node_cc_preview.png", help="汇总图文件名。"
    )
    parser.add_argument(
        "--no-save-shot-figures",
        action="store_true",
        help="只生成汇总图，不保存每个源节点图。",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    source_dir = (
        args.profile_source_dir.expanduser().resolve()
        if args.profile_source_dir
        else None
    )
    plot_node_results(
        input_dir, args.plot_name, not args.no_save_shot_figures, source_dir
    )


if __name__ == "__main__":
    main()
