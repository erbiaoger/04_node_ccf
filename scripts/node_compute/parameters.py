"""Node-only parameters and scientific-definition validation."""

from __future__ import annotations

import numpy as np
from node_io.catalog import sample_index

VERSION = "node_spectral_pair_pws_v1"
DEFAULTS = {
    "read_mode": "auto",
    "read_hours": 24.0,
    "io_workers": 8,
    "prefetch_blocks": 1,
    "async_save": 1,
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
    "cc_batch_chunks": 12,
    "cc_upload_block": 1,
    "cc_reuse_workspace": 1,
    "cc_pair_batch": 0,
    "cc_cuda_device": 0,
    "cc_memory_safety_fraction": 0.70,
    "cc_memory_budget_bytes": 0,
    "resume": True,
}

TSI_DEFAULTS = {
    "tsi_files_per_group": 0,
    "tsi_iterations": 4,
    "tsi_stop_rel_l1": 0.005,
    "tsi_mode": "geographic_all_pairs",
    "tsi_backend": "torch",
    "tsi_device": "cuda",
    "tsi_pair_batch_size": 32,
    "tsi_geometry_max_path_error_m": 10.0,
    "tsi_min_common_k": 1,
    "tsi_show_progress": True,
    "tsi_taper_enabled": True,
    "tsi_taper_vmin": 500.0,
    "tsi_taper_vmax": 4000.0,
    "tsi_taper_outside_scale": 0.25,
    "tsi_taper_shoulder_fraction": 0.25,
}
OPTIONAL_PARAMS = {
    "file_stacking_method",
    "cc_enhance_method",
    "cc_memory_budget_gib",
    "sac_sampling_rate_hz",
    *TSI_DEFAULTS,
}


def validate(params: dict, raw_fs: float):
    p = dict(DEFAULTS, **params)
    if "cc_memory_budget_gib" in p:
        gib = p["cc_memory_budget_gib"]
        if gib is not None and (not np.isfinite(gib) or gib <= 0):
            raise ValueError("cc_memory_budget_gib must be positive or null")
        budget = 0 if gib is None else round(gib * 1024**3)
        if p.get("cc_memory_budget_bytes", 0) not in (0, budget):
            raise ValueError("Conflicting GiB and byte memory budgets")
        p["cc_memory_budget_bytes"] = budget
    if p["cc_device"] == "auto":
        p["cc_device"] = "cpu" if p["cc_backend"] == "cpu" else "cuda"
    if "file_stacking_method" in p and p["file_stacking_method"] not in {
        "linear",
        "pws",
    }:
        raise ValueError("Node file_stacking_method supports linear/pws")
    if p.get("cc_enhance_method", "raw") not in {"raw", "tsi"}:
        raise ValueError("cc_enhance_method must be raw or tsi")
    if p.get("cc_enhance_method") == "tsi":
        p = dict(TSI_DEFAULTS, **p)
        for key in ("tsi_taper_enabled", "tsi_show_progress"):
            if not isinstance(p[key], (bool, int)) or p[key] not in (0, 1):
                raise ValueError(f"{key} must be a boolean")
            p[key] = bool(p[key])
        if p["tsi_files_per_group"] == 0:
            p["tsi_files_per_group"] = p["save_every"]
        for key in (
            "tsi_files_per_group",
            "tsi_iterations",
            "tsi_pair_batch_size",
            "tsi_min_common_k",
        ):
            if not np.isfinite(p[key]) or p[key] < 1 or int(p[key]) != p[key]:
                raise ValueError(f"{key} must be a positive integer")
            p[key] = int(p[key])
        if p["save_every"] % p["tsi_files_per_group"]:
            raise ValueError(
                "save_every must be a multiple of tsi_files_per_group; these count minute_stack_s blocks"
            )
        if p["tsi_mode"] != "geographic_all_pairs":
            raise ValueError(
                "Node GPS TSI requires tsi_mode=geographic_all_pairs; DAS index-based modes are not applicable"
            )
        if p["tsi_backend"] not in {"torch", "cpu"} or p["tsi_device"] not in {
            "cuda",
            "cpu",
        }:
            raise ValueError("Node TSI supports torch/cuda|cpu or cpu/cpu")
        if p["tsi_backend"] == "cpu":
            p["tsi_device"] = "cpu"
        for key in (
            "tsi_stop_rel_l1",
            "tsi_geometry_max_path_error_m",
            "tsi_taper_vmin",
            "tsi_taper_vmax",
            "tsi_taper_outside_scale",
            "tsi_taper_shoulder_fraction",
        ):
            p[key] = float(p[key])
            if not np.isfinite(p[key]):
                raise ValueError(f"{key} must be finite")
        if (
            p["tsi_stop_rel_l1"] < 0
            or p["tsi_geometry_max_path_error_m"] < 0
            or not 0 < p["tsi_taper_vmin"] < p["tsi_taper_vmax"]
            or not 0 <= p["tsi_taper_outside_scale"] <= 1
            or p["tsi_taper_shoulder_fraction"] < 0
        ):
            raise ValueError(
                "Invalid node TSI convergence, geometry or velocity taper parameters"
            )
    if "sac_sampling_rate_hz" in p:
        nominal = float(p["sac_sampling_rate_hz"])
        if not np.isfinite(nominal) or nominal <= 0 or nominal != raw_fs:
            raise ValueError(
                "sac_sampling_rate_hz must agree with the validated SAC sample grid"
            )
        p["sac_sampling_rate_hz"] = nominal
    ints = (
        "save_every",
        "time_downsample",
        "space_downsample",
        "smooth_N",
        "smoothspect_N",
        "cc_batch_chunks",
        "io_workers",
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
        "read_hours",
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
    if p["prefetch_blocks"] not in (0, 1):
        raise ValueError("prefetch_blocks must be 0 or 1 (bounded double buffering)")
    p["prefetch_blocks"] = int(p["prefetch_blocks"])
    if p["cc_upload_block"] not in (0, 1):
        raise ValueError("cc_upload_block must be 0 or 1")
    p["cc_upload_block"] = int(p["cc_upload_block"])
    if p["cc_reuse_workspace"] not in (0, 1):
        raise ValueError("cc_reuse_workspace must be 0 or 1")
    p["cc_reuse_workspace"] = int(p["cc_reuse_workspace"])
    if p["async_save"] not in (0, 1):
        raise ValueError("async_save must be 0 or 1")
    p["async_save"] = int(p["async_save"])
    fs = raw_fs / p["time_downsample"]
    n = sample_index(p["cc_len"], fs, "cc_len")
    radius = sample_index(p["maxlag"], fs, "maxlag")
    if n < 2 or radius < 0 or radius >= n:
        raise ValueError("maxlag must lie within cc_len - one sample")
    for key in ("cc_step_s", "minute_stack_s"):
        if sample_index(p[key], fs, key) < 1:
            raise ValueError(f"{key} must contain at least one sample")
    if p["read_hours"] < 0:
        raise ValueError("read_hours must be nonnegative; 0 reads one output group")
    if p["read_hours"]:
        ratio = p["read_hours"] * 3600 / (p["minute_stack_s"] * p["save_every"])
        if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-7):
            raise ValueError(
                "read_hours must be a positive multiple of the output group duration"
            )
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
