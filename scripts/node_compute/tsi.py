"""GPS-aware direct-wave three-station interferometry for node CCF cubes.

Triplets are selected by sum/difference path-length residual, never station
number. Causal/acausal branches are routed using directed paths. FFTs are
zero-padded for linear convolution/correlation. Unsupported targets retain CC.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.fft import next_fast_len

from .backend import create_session


def triplet_geometry(distance, observed, tolerance_m):
    distance = np.asarray(distance, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    n = len(distance)
    if (
        distance.shape != (n, n)
        or observed.shape != (n, n)
        or not np.isfinite(distance).all()
        or np.any(distance < 0)
        or not np.allclose(distance, distance.T)
        or not np.allclose(np.diag(distance), 0)
        or not np.array_equal(observed, observed.T)
        or not np.isfinite(tolerance_m)
        or tolerance_m < 0
    ):
        raise ValueError("Invalid TSI geographic distance matrix or path tolerance")
    a, b = np.triu_indices(n, 1)
    ik = distance[a, :]
    jk = distance[b, :]
    ij = distance[a, b, None]
    between = (ik + jk - ij) <= tolerance_m
    outside = (ij - abs(ik - jk)) <= tolerance_m
    usable = (
        (between | outside)
        & observed[a, :]
        & observed[b, :]
        & observed[a, b, None]
        & (ij > 0)
        & (ik > 0)
        & (jk > 0)
    )
    usable[np.arange(len(a)), a] = False
    usable[np.arange(len(a)), b] = False
    kind = np.where(between, 0, np.where(ik < jk, 1, 2))
    return a, b, usable, kind


def geographic_tsi(cube, distance, observed, dt, params):
    import torch

    cube = np.asarray(cube, dtype=np.float32)
    n, _, lag = cube.shape
    if (
        cube.shape != (n, n, lag)
        or n < 3
        or lag % 2 != 1
        or not np.isfinite(cube).all()
        or dt <= 0
    ):
        raise ValueError("TSI needs >=3 stations and finite odd-length full-lag CCFs")
    backend = params.get("tsi_backend", "torch")
    device = params.get("tsi_device", "cuda") if backend == "torch" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Node TSI CUDA requested but unavailable")
    if device == "cuda":
        device = f"cuda:{params.get('cc_cuda_device', 0)}"
    dtype = torch.float64 if backend == "cpu" else torch.float32
    a, b, eligible, kind = triplet_geometry(
        distance, observed, params.get("tsi_geometry_max_path_error_m", 10.0)
    )
    support = eligible.sum(axis=1).astype(np.uint64)
    applied = support >= params.get("tsi_min_common_k", 1)
    half = lag // 2
    one = half + 1
    nfft = next_fast_len(2 * one - 1)
    batch = params.get("tsi_pair_batch_size", 32)
    gate = np.ones((n, n, one), dtype=np.float32)
    if params.get("tsi_taper_enabled", True):
        times = np.arange(one) * dt
        vmin = params.get("tsi_taper_vmin", 500.0)
        vmax = params.get("tsi_taper_vmax", 4000.0)
        outside = params.get("tsi_taper_outside_scale", 0.25)
        shoulder = params.get("tsi_taper_shoulder_fraction", 0.25)
        fast = np.asarray(distance) / vmax
        slow = np.asarray(distance) / vmin
        width = np.maximum((slow - fast) * shoulder, dt)
        start = np.maximum(fast - width, 0)
        t = times[None, None, :]
        gate.fill(outside)
        middle = (t >= fast[:, :, None]) & (t <= slow[:, :, None])
        gate[middle] = 1
        low = (t >= start[:, :, None]) & (t < fast[:, :, None])
        ramp = np.clip(
            (t - start[:, :, None]) / np.maximum(fast - start, dt)[:, :, None], 0, 1
        )
        gate[low] = (outside + (1 - outside) * 0.5 * (1 - np.cos(np.pi * ramp)))[low]
        high = (t > slow[:, :, None]) & (t <= slow[:, :, None] + width[:, :, None])
        ramp = np.clip((t - slow[:, :, None]) / width[:, :, None], 0, 1)
        gate[high] = (outside + (1 - outside) * 0.5 * (1 + np.cos(np.pi * ramp)))[high]
        gate[np.arange(n), np.arange(n)] = 1
    history = []
    started = time.perf_counter()
    with torch.inference_mode():
        current = torch.as_tensor(cube, device=device, dtype=dtype).clone()
        gate_gpu = torch.as_tensor(gate, device=device, dtype=dtype)
        ai = torch.as_tensor(a, device=device)
        bi = torch.as_tensor(b, device=device)
        support_gpu = torch.as_tensor(support.astype(np.int64), device=device)
        mask_gpu = torch.as_tensor(eligible, device=device)
        kind_gpu = torch.as_tensor(kind, device=device)
        ever_applied = torch.zeros(len(a), device=device, dtype=torch.bool)
        for _ in range(params.get("tsi_iterations", 4)):
            waves = torch.stack(
                (
                    current[:, :, half:],
                    torch.flip(current[:, :, : half + 1], dims=[-1]),
                ),
                dim=2,
            )
            spec = torch.fft.rfft(waves * gate_gpu[:, :, None, :], n=nfft, dim=-1)
            updated = current.clone()
            for begin in range(0, len(a), batch):
                end = min(begin + batch, len(a))
                i = ai[begin:end]
                j = bi[begin:end]
                # Interior k: directed i→k and k→j convolution.
                internal = spec[i, :, :, :] * spec[:, j, :, :].permute(1, 0, 2, 3)
                ki = spec[:, i, :, :].permute(1, 0, 2, 3)
                kj = spec[:, j, :, :].permute(1, 0, 2, 3)
                # Exterior near i: k→i and k→j correlation.
                near_i = torch.conj(ki) * kj
                # Exterior near j: reverse branches to retain i→j orientation.
                near_j = torch.flip(torch.conj(kj) * ki, dims=[2])
                modes = kind_gpu[begin:end, :, None, None]
                product = torch.where(
                    modes == 0, internal, torch.where(modes == 1, near_i, near_j)
                )
                wave = torch.fft.irfft(product, n=nfft, dim=-1)[..., :one]
                peaks = wave.abs().amax(dim=-1, keepdim=True)
                # Do not promote cropped-out FFT roundoff into a unit-amplitude arrival.
                wave = torch.where(peaks > 1e-6, wave / peaks.clamp_min(1e-6), 0)
                wave = wave * mask_gpu[begin:end, :, None, None]
                average = (
                    wave.sum(dim=1) / support_gpu[begin:end].clamp_min(1)[:, None, None]
                )
                zero = (average[:, 0, 0] + average[:, 1, 0]) * 0.5
                full = torch.cat(
                    (
                        torch.flip(average[:, 1, 1:], dims=[-1]),
                        zero[:, None],
                        average[:, 0, 1:],
                    ),
                    dim=-1,
                )
                full_peak = full.abs().amax(dim=-1, keepdim=True)
                full = full / full_peak.clamp_min(1e-6)
                active = torch.as_tensor(applied[begin:end], device=device) & (
                    full_peak[:, 0] > 1e-6
                )
                ever_applied[begin:end] |= active
                full = torch.where(active[:, None], full, current[i, j])
                updated[i, j] = full
                updated[j, i] = torch.flip(full, dims=[-1])
            change = (updated - current).abs().mean() / current.abs().mean().clamp_min(
                1e-12
            )
            rel = float(change.cpu())
            history.append(rel)
            current = updated
            if rel <= params.get("tsi_stop_rel_l1", 0.005):
                break
            if params.get("tsi_show_progress", False):
                import logging

                logging.getLogger(__name__).info(
                    "TSI iteration %d/%d: rel_l1=%.5g, supported %d/%d pairs",
                    len(history),
                    params.get("tsi_iterations", 4),
                    rel,
                    int(applied.sum()),
                    len(a),
                )
        result = current.cpu().numpy().astype(np.float32)
        applied = ever_applied.cpu().numpy()
    report = {
        "method": "geographic_direct_wave_tsi_v1",
        "backend": backend,
        "device": device,
        "iterations_completed": len(history),
        "rel_l1_history": history,
        "converged": bool(history[-1] <= params.get("tsi_stop_rel_l1", 0.005)),
        "updated_pairs": int(applied.sum()),
        "retained_raw_pairs": int(np.asarray(observed)[a, b].sum() - applied.sum()),
        "path_tolerance_m": params.get("tsi_geometry_max_path_error_m", 10.0),
        "minimum_common_stations": params.get("tsi_min_common_k", 1),
        "seconds": time.perf_counter() - started,
        "linear_fft_size": nfft,
        "reference": "https://doi.org/10.3389/feart.2022.952410",
    }
    return result, support, applied, report


def enhance_output(fields, group, data, valid, lo, params):
    """Enhance subgroups independently, then stack into one requested save slice."""
    n = fields["data"].shape[0]
    a, b = np.triu_indices(n, 1)
    interval = params["tsi_files_per_group"]
    ids = group.minute_ids // interval
    chunks = []
    if group.minute_count <= interval:
        chunks = [(fields["data"], fields["valid_mask"], len(group.starts))]
    else:
        session = create_session(params, n)
        factor = params["time_downsample"]
        halo = 20 * factor if factor > 1 else 0
        points = round(params["cc_len"] * params["samp_freq"] / factor)
        for chunk in np.unique(ids):
            selected = ids == chunk
            starts = group.starts[selected]
            left = max(lo, int(starts[0]) - halo)
            right = min(
                lo + len(data), int(starts[-1]) + (points - 1) * factor + halo + 1
            )
            session.add_block(
                data[left - lo : right - lo],
                valid[left - lo : right - lo],
                starts - left,
                group.minute_ids[selected],
            )
            result = session.finish_group()
            from node_output.artifacts import dense_pairs

            chunks.append(
                (
                    dense_pairs(result["pairs"], n),
                    dense_pairs(result["valid_count"] > 0, n, mirror_lag=False),
                    len(starts),
                )
            )
            session.clear_group()
        del session
    stacked = create_session(params, n) if len(chunks) > 1 else None
    support_min = np.full(len(a), n, dtype=np.uint64)
    applied_count = np.zeros(len(a), dtype=np.uint64)
    diagnostics = []
    raw = fields["data"].copy()
    for cube, observed, windows in chunks:
        enhanced, support, applied, diag = geographic_tsi(
            cube.transpose(0, 2, 1),
            fields["station_distance_m"],
            observed,
            float(fields["dt"]),
            params,
        )
        support_min = np.minimum(support_min, support)
        applied_count += applied.astype(np.uint64)
        diagnostics.append(dict(diag, short_windows=windows))
        if stacked is not None:
            stacked.stack_pairs(enhanced[a, b], observed[a, b])
    from node_output.artifacts import dense_pairs

    output = (
        dense_pairs(stacked.finish_group()["pairs"], n)
        if stacked is not None
        else enhanced.transpose(0, 2, 1)
    )
    unchanged = applied_count == 0
    output[a[unchanged], :, b[unchanged]] = raw[a[unchanged], :, b[unchanged]]
    output[b[unchanged], :, a[unchanged]] = raw[b[unchanged], :, a[unchanged]]
    fields.update(
        data=output.astype(np.float32),
        data_raw=raw,
        cc_enhance_method="tsi",
        tsi_group_count=len(chunks),
        tsi_files_per_group=interval,
        tsi_group_duration_s=interval * params["minute_stack_s"],
        tsi_applied_mask=dense_pairs(applied_count > 0, n, mirror_lag=False),
        tsi_applied_group_count=dense_pairs(applied_count, n, mirror_lag=False),
        tsi_common_station_count_min=dense_pairs(support_min, n, mirror_lag=False),
        tsi_diagnostics_json=json.dumps(diagnostics, sort_keys=True),
        tsi_original_count_semantics="valid counts describe original CC input windows/minutes; TSI support is separate",
    )
    return fields
