"""Node-only parameters and scientific-definition validation."""

from __future__ import annotations

import numpy as np
from node_io.catalog import sample_index

VERSION = "node_spectral_pair_pws_v1"
DEFAULTS = {
    "read_mode": "auto",
    "save_every": 30,
    "minute_stack_s": 60.0,
    "cc_len": 10.0,
    "cc_step_s": 5.0,
    "maxlag": 4.0,
    "freqmin": 2.0,
    "freqmax": 30.0,
    "time_norm": "rma",
    "freq_norm": "rma",
    "cc_method": "xcorr",
    "stacking_method": "pws",
    "smooth_N": 5,
    "smoothspect_N": 5,
    "max_over_std": 1.0e9,
    "time_downsample": 1,
    "space_downsample": 1,
    "cc_backend": "cpp_cuda",
    "cc_device": "cuda",
    "cc_dtype": "float32",
    "cc_batch_chunks": 32,
    "cc_pair_batch": 0,
    "cc_cuda_device": 0,
    "cc_memory_safety_fraction": 0.70,
    "cc_memory_budget_bytes": 0,
    "resume": True,
}


def validate(params: dict, raw_fs: float):
    p = dict(DEFAULTS, **params)
    ints = (
        "save_every",
        "time_downsample",
        "space_downsample",
        "smooth_N",
        "smoothspect_N",
        "cc_batch_chunks",
    )
    for key in ints:
        if not np.isfinite(p[key]) or int(p[key]) != p[key] or int(p[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
        p[key] = int(p[key])
    for key in ("cc_pair_batch", "cc_cuda_device", "cc_memory_budget_bytes"):
        if not np.isfinite(p[key]) or int(p[key]) != p[key] or p[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        p[key] = int(p[key])
    for key in (
        "cc_len",
        "cc_step_s",
        "minute_stack_s",
        "maxlag",
        "freqmin",
        "freqmax",
        "max_over_std",
        "cc_memory_safety_fraction",
    ):
        p[key] = float(p[key])
        if not np.isfinite(p[key]):
            raise ValueError(f"{key} must be finite")
    fs = raw_fs / p["time_downsample"]
    n = sample_index(p["cc_len"], fs, "cc_len")
    radius = sample_index(p["maxlag"], fs, "maxlag")
    if n < 2 or radius < 0 or radius >= n:
        raise ValueError("maxlag must lie within cc_len - one sample")
    for key in ("cc_step_s", "minute_stack_s"):
        if sample_index(p[key], fs, key) < 1:
            raise ValueError(f"{key} must contain at least one sample")
    # Preserve the requested flat passband: do not silently invoke the old
    # high-frequency shoulder fallback which would move the upper band edge.
    allowed = 0.45 * fs / 1.1
    if not 0 <= p["freqmin"] < p["freqmax"] <= allowed:
        raise ValueError(
            f"Requested band {p['freqmin']}..{p['freqmax']} Hz is invalid; freqmax <= {allowed:.6g} Hz at fs={fs:.9g}"
        )
    if (
        p["space_downsample"] != 1
        or p.get("fk_enabled", False)
        or p.get("include_autocorr", False)
    ):
        raise ValueError(
            "Node all-pairs requires space_downsample=1, no FK and no autocorrelation"
        )
    if p.get("pair_mode", "all_pairs") != "all_pairs":
        raise ValueError("Only all_pairs is supported; sliding/offset mode was removed")
    if p["cc_method"] != "xcorr" or p["stacking_method"] not in {"pws", "linear"}:
        raise ValueError("Node pairs support xcorr and linear/PWS stacking")
    if p["time_norm"] not in {"no", "one_bit", "rma"} or p["freq_norm"] not in {
        "no",
        "phase_only",
        "rma",
    }:
        raise ValueError("Unsupported node normalization")
    if p["read_mode"] not in {"auto", "preload", "block"} or p["cc_dtype"] != "float32":
        raise ValueError("Use auto/preload/block reads and float32")
    if not 0.05 <= p["cc_memory_safety_fraction"] <= 1 or p["max_over_std"] <= 0:
        raise ValueError("Invalid memory fraction or quality threshold")
    backend, device = p["cc_backend"], p["cc_device"]
    if (
        backend not in {"cpp_cuda", "cpu", "torch"}
        or (backend == "cpp_cuda" and device != "cuda")
        or (backend == "cpu" and device != "cpu")
        or device not in {"cpu", "cuda"}
    ):
        raise ValueError("Select cpp_cuda/cuda, cpu/cpu, or torch/cpu|cuda explicitly")
    p["samp_freq"] = raw_fs
    return p
