"""Zero-order frequency–Bessel imaging of GPS node CCFs at actual offsets.

FFT zero lag is moved to the origin. The real cross spectrum (even CCF part)
is integrated against J0(2*pi*f*r/c)*r by the trapezoid rule. Each frequency
column is normalized after integration; squared magnitude is display energy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0


def distance_gathers(mat, bin_width_m=50):
    if not np.isfinite(bin_width_m) or bin_width_m <= 0:
        raise ValueError("Distance bin width must be positive")
    a, b = mat["pair_source_index"], mat["pair_receiver_index"]
    good = mat["valid_mask"][a, b].astype(bool)
    distance = np.asarray(mat["pair_distance_m"], dtype=float)[good]
    traces = np.asarray(mat["data"][a, :, b], dtype=float)[good]
    if len(distance) < 2 or not np.isfinite(distance).all() or np.any(distance <= 0):
        raise ValueError("Need at least two valid positive pair distances")
    if not np.isfinite(traces).all():
        raise ValueError("Nonfinite CCF samples")
    peak = np.max(np.abs(traces), axis=1)
    traces = np.divide(
        traces, peak[:, None], out=np.zeros_like(traces), where=peak[:, None] > 0
    )
    upper = (np.floor(distance.max() / bin_width_m) + 1) * bin_width_m
    edges = np.arange(0, upper + bin_width_m / 2, bin_width_m)
    group = np.minimum(
        np.searchsorted(edges, distance, side="right") - 1, len(edges) - 2
    )
    counts = np.bincount(group, minlength=len(edges) - 1)
    summed = np.zeros((len(counts), traces.shape[1]), dtype=float)
    np.add.at(summed, group, traces)
    mean = np.divide(
        summed, counts[:, None], out=np.zeros_like(summed), where=counts[:, None] > 0
    )
    occupied = counts > 0
    return {
        "all_pair_distance_m": distance,
        "all_pair_ccf": traces,
        "bin_edges_m": edges,
        "bin_centers_m": (edges[:-1] + edges[1:]) / 2,
        "bin_counts": counts,
        "binned_distance_m": ((edges[:-1] + edges[1:]) / 2)[occupied],
        "binned_ccf": mean[occupied],
        "bin_occupied": occupied,
    }


def frequency_bessel(
    traces,
    offsets,
    dt,
    frequencies,
    velocities,
    *,
    zero_lag_index,
    backend="torch_cuda",
    frequency_batch=8,
    cuda_device=0,
):
    traces = np.asarray(traces, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    f = np.asarray(frequencies, dtype=np.float64)
    c = np.asarray(velocities, dtype=np.float64)
    if traces.ndim != 2 or offsets.ndim != 1 or traces.shape[0] != len(offsets):
        raise ValueError("Expected pair×lag traces and matching distance vector")
    if (
        not np.isfinite(traces).all()
        or not np.isfinite(offsets).all()
        or np.any(offsets <= 0)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid input traces, offsets or sampling interval")
    if (
        f.ndim != 1
        or c.ndim != 1
        or not len(f)
        or not len(c)
        or not np.isfinite(f).all()
        or not np.isfinite(c).all()
        or np.any(f <= 0)
        or np.any(f > 0.5 / dt)
        or np.any(c <= 0)
        or np.any(np.diff(f) <= 0)
        or np.any(np.diff(c) <= 0)
    ):
        raise ValueError("Invalid frequency or velocity axis")
    if not 0 <= zero_lag_index < traces.shape[1] or frequency_batch < 1:
        raise ValueError("Invalid zero lag or frequency batch")
    spectrum = np.fft.rfft(np.roll(traces, -int(zero_lag_index), axis=1), axis=1)
    fft_f = np.fft.rfftfreq(traces.shape[1], dt)
    if f[-1] > fft_f[-1]:
        raise ValueError("Frequency exceeds actual FFT support")
    spectral_real = np.asarray([np.interp(f, fft_f, row.real) for row in spectrum]).T
    distance, inverse = np.unique(offsets, return_inverse=True)
    if len(distance) < 2:
        raise ValueError("Need at least two distinct distances for radial integration")
    pooled = np.zeros((len(f), len(distance)), dtype=np.float64)
    for index, group in enumerate(inverse):
        pooled[:, group] += spectral_real[:, index]
    pooled /= np.bincount(inverse)[None, :]
    dr = np.diff(distance)
    weights = np.r_[dr[0] / 2, (dr[:-1] + dr[1:]) / 2, dr[-1] / 2] * distance
    raw = np.empty((len(c), len(f)), dtype=np.float64)
    if backend == "cpu":
        for begin in range(0, len(f), frequency_batch):
            end = min(begin + frequency_batch, len(f))
            kr = (
                2
                * np.pi
                * f[None, begin:end, None]
                * distance[None, None, :]
                / c[:, None, None]
            )
            raw[:, begin:end] = np.einsum(
                "vfr,fr->vf", j0(kr), pooled[begin:end] * weights, optimize=True
            )
    elif backend == "torch_cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda F-J requested but CUDA is unavailable")
        device = torch.device(f"cuda:{cuda_device}")
        with torch.inference_mode():
            rd = torch.as_tensor(distance, device=device, dtype=torch.float64)
            cd = torch.as_tensor(c, device=device, dtype=torch.float64)
            fd = torch.as_tensor(f, device=device, dtype=torch.float64)
            sw = torch.as_tensor(pooled * weights, device=device, dtype=torch.float64)
            for begin in range(0, len(f), frequency_batch):
                end = min(begin + frequency_batch, len(f))
                kr = (
                    2
                    * torch.pi
                    * fd[None, begin:end, None]
                    * rd[None, None, :]
                    / cd[:, None, None]
                )
                block = torch.einsum(
                    "vfr,fr->vf", torch.special.bessel_j0(kr), sw[begin:end]
                )
                raw[:, begin:end] = block.cpu().numpy()
    else:
        raise ValueError("Select cpu or torch_cuda explicitly")
    column_scale = np.max(np.abs(raw), axis=0)
    amplitude = np.divide(
        raw,
        column_scale[None, :],
        out=np.zeros_like(raw),
        where=column_scale[None, :] > 0,
    )
    return {
        "energy": amplitude**2,
        "signed_amplitude": amplitude,
        "raw_integral": raw,
        "column_scale": column_scale,
        "distance_m": distance,
        "frequency_hz": f,
        "phase_velocity_m_s": c,
        "fft_spacing_hz": float(fft_f[1] - fft_f[0]),
    }


def park_phase_shift(
    traces,
    offsets,
    dt,
    frequencies,
    velocities,
    *,
    zero_lag_index,
    backend="torch_cuda",
    frequency_batch=8,
    cuda_device=0,
    branch="positive",
):
    """Park phase-only coherent stack; NumPy negative-exponent FFT convention.

    Image = |mean(U/|U| * exp(+i*2*pi*f*r/c))|^2. Each valid trace
    contributes equally. Select one CCF branch before FFT to preserve phase.
    """
    traces = np.asarray(traces, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    f, c = np.asarray(frequencies, dtype=float), np.asarray(velocities, dtype=float)
    if traces.ndim != 2 or offsets.shape != (traces.shape[0],) or traces.shape[0] < 2:
        raise ValueError("Need pair×lag traces and at least two matching offsets")
    if (
        not np.isfinite(traces).all()
        or not np.isfinite(offsets).all()
        or np.any(offsets <= 0)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid traces, distances or sampling interval")
    if (
        f.ndim != 1
        or c.ndim != 1
        or not len(f)
        or not len(c)
        or not np.isfinite(f).all()
        or not np.isfinite(c).all()
        or np.any(f <= 0)
        or np.any(c <= 0)
        or np.any(np.diff(f) <= 0)
        or np.any(np.diff(c) <= 0)
        or f[-1] > 0.5 / dt
    ):
        raise ValueError("Invalid frequency or velocity axis")
    if (
        not isinstance(zero_lag_index, (int, np.integer))
        or not 0 <= zero_lag_index < traces.shape[1]
        or frequency_batch < 1
    ):
        raise ValueError("Invalid zero lag or frequency batch")
    positive = traces[:, zero_lag_index:]
    negative = traces[:, : zero_lag_index + 1][:, ::-1]
    if branch == "positive":
        wave = positive
    elif branch == "negative":
        wave = negative
    elif branch == "symmetric":
        if positive.shape != negative.shape:
            raise ValueError(
                "Symmetric branch requires matching positive/negative lag lengths"
            )
        wave = (positive + negative) / 2
    else:
        raise ValueError("branch must be positive, negative or symmetric")
    if wave.shape[1] < 2:
        raise ValueError("Selected lag branch is too short")
    spec = np.fft.rfft(wave, axis=1)
    fft_f = np.fft.rfftfreq(wave.shape[1], dt)
    if f[-1] > fft_f[-1]:
        raise ValueError("Frequency exceeds actual FFT support")
    spectral = np.asarray(
        [
            np.interp(f, fft_f, row.real) + 1j * np.interp(f, fft_f, row.imag)
            for row in spec
        ]
    ).T
    magnitude = np.abs(spectral)
    mask = magnitude > np.maximum(
        magnitude.max(axis=1, keepdims=True) * 1e-12, np.finfo(float).tiny
    )
    phase = np.divide(spectral, magnitude, out=np.zeros_like(spectral), where=mask)
    valid_count = mask.sum(axis=1)
    raw = np.empty((len(c), len(f)), dtype=np.complex128)
    if backend == "cpu":
        for start in range(0, len(f), frequency_batch):
            end = min(start + frequency_batch, len(f))
            shift = np.exp(
                2j
                * np.pi
                * f[None, start:end, None]
                * offsets[None, None, :]
                / c[:, None, None]
            )
            raw[:, start:end] = np.einsum(
                "vfr,fr->vf", shift, phase[start:end], optimize=True
            )
    elif backend == "torch_cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda Park requested but CUDA is unavailable")
        device = torch.device(f"cuda:{cuda_device}")
        with torch.inference_mode():
            rd = torch.as_tensor(offsets, device=device, dtype=torch.float64)
            fd = torch.as_tensor(f, device=device, dtype=torch.float64)
            cd = torch.as_tensor(c, device=device, dtype=torch.float64)
            pd = torch.as_tensor(phase, device=device, dtype=torch.complex128)
            for start in range(0, len(f), frequency_batch):
                end = min(start + frequency_batch, len(f))
                shift = torch.exp(
                    2j
                    * torch.pi
                    * fd[None, start:end, None]
                    * rd[None, None, :]
                    / cd[:, None, None]
                )
                raw[:, start:end] = (
                    torch.einsum("vfr,fr->vf", shift, pd[start:end]).cpu().numpy()
                )
    else:
        raise ValueError("Select cpu or torch_cuda explicitly")
    coherence = np.divide(
        abs(raw),
        valid_count[None, :],
        out=np.zeros(raw.shape, dtype=float),
        where=valid_count[None, :] > 0,
    )
    scale = abs(raw).max(axis=0)
    amplitude = np.divide(
        abs(raw),
        scale[None, :],
        out=np.zeros(raw.shape, dtype=float),
        where=scale[None, :] > 0,
    )
    return {
        "energy": amplitude**2,
        "coherence": coherence,
        "signed_amplitude": amplitude,
        "raw_integral": raw,
        "column_scale": scale,
        "distance_m": offsets,
        "frequency_hz": f,
        "phase_velocity_m_s": c,
        "valid_trace_count": valid_count,
        "fft_spacing_hz": float(fft_f[1] - fft_f[0]),
        "lag_branch": branch,
    }
