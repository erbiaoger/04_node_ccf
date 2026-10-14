#!/usr/bin/env python3
"""Compute matched F-J or Park spectra from one final stacked H5."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from node_compute.dispersion import distance_gathers, frequency_bessel, park_phase_shift
from node_output.artifacts import atomic_json, digest, load_result, save_h5

LOG = logging.getLogger(__name__)


def compute(
    input_path,
    output_dir,
    *,
    fmin=2,
    fmax=30,
    df=0.1,
    cmin=500,
    cmax=4000,
    dc=10,
    bin_width_m=50,
    backend="torch_cuda",
    cuda_device=0,
    frequency_batch=8,
    method="fj",
    park_branch="positive",
):
    if not 0 < fmin < fmax or not 0 < cmin < cmax or df <= 0 or dc <= 0:
        raise ValueError("Invalid frequency/velocity bounds or increments")
    if method not in {"fj", "park"}:
        raise ValueError("method must be fj or park")
    transform = frequency_bessel if method == "fj" else park_phase_shift
    method_options = {} if method == "fj" else {"branch": park_branch}
    mat = load_result(input_path)
    source_hash = digest(input_path)
    zero = int(np.argmin(abs(mat["lags_s"])))
    dt = float(mat["dt"])
    if abs(float(mat["lags_s"][zero])) > dt * 1e-6:
        raise ValueError("No exact zero lag in input")
    if str(mat.get("geometry_type", "")) != "geographic":
        raise ValueError("Expected geographic H5 with actual pair distances")
    f = np.arange(fmin, fmax + df / 2, df)
    c = np.arange(cmin, cmax + dc / 2, dc)
    gathers = distance_gathers(mat, bin_width_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_h5_sha256": source_hash,
        "method": "F-J J0" if method == "fj" else "Park phase shift",
        "backend": backend,
        "frequency_range_hz": [fmin, fmax],
        "phase_velocity_range_m_s": [cmin, cmax],
        "frequency_step_hz": df,
        "velocity_step_m_s": dc,
        "distance_bin_width_m": bin_width_m,
        "zero_lag_index": zero,
        "lag_window_s": [float(mat["lags_s"][0]), float(mat["lags_s"][-1])],
        "preprocessing": "Per-pair peak normalization; no added spectral whitening, time derivative or strain correction.",
        "radial_integral": "Trapezoid integral of Re(FFT(CCF))*J0(2*pi*f*r/c)*r dr; zero lag rolled to FFT origin.",
        "energy": "Squared integral magnitude normalized per frequency; not absolute physical energy.",
        "references": ["https://doi.org/10.1029/2018JB016595"],
        "limitations": [
            "2D array interpreted by radial distance; directional noise and azimuthal heterogeneity can bias imaging.",
            "50m binning averages traces with different distances and can attenuate short wavelengths.",
            "Output frequency step interpolates FFT bins and does not increase resolution.",
            "Energy maxima are not verified fundamental-mode picks.",
        ],
        "results": {},
    }
    if method == "park":
        report.update(
            radial_integral="Equal-trace sum of phase-normalized FFT times exp(+i*2*pi*f*r/c).",
            lag_branch=park_branch,
            preprocessing="Per-pair peak normalization; selected lag branch, FFT, complex interpolation, per-trace/per-frequency phase-only normalization; no time derivative or strain correction.",
            energy="Squared phase coherence normalized per frequency; coherence also stored before display normalization.",
            references=["https://doi.org/10.1190/1.1820161"],
        )
    for name, traces, offsets in (
        ("all_pairs", gathers["all_pair_ccf"], gathers["all_pair_distance_m"]),
        ("binned", gathers["binned_ccf"], gathers["binned_distance_m"]),
    ):
        started = time.perf_counter()
        result = transform(
            traces,
            offsets,
            dt,
            f,
            c,
            **method_options,
            zero_lag_index=zero,
            backend=backend,
            frequency_batch=frequency_batch,
            cuda_device=cuda_device,
        )
        # A handful of frequency columns cross-check actual GPU imaging against SciPy.
        checks = np.unique(np.linspace(0, len(f) - 1, 7, dtype=int))
        reference = transform(
            traces,
            offsets,
            dt,
            f[checks],
            c,
            **method_options,
            zero_lag_index=zero,
            backend="cpu",
            frequency_batch=frequency_batch,
        )
        scale = np.maximum(reference["column_scale"], 1e-12)
        # CUDA J0 is approximate even with float64; compare on a column scale
        # rather than unstable relative error at individual integral zero crossings.
        np.testing.assert_allclose(
            result["raw_integral"][:, checks] / scale,
            reference["raw_integral"] / scale,
            rtol=0,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            result["energy"][:, checks], reference["energy"], rtol=0, atol=2e-5
        )
        if not np.isfinite(result["energy"]).all():
            raise ValueError("Nonfinite dispersion spectrum")
        path = output_dir / f"dispersion_{name}.h5"
        fields = dict(
            result,
            method="frequency_bessel_J0" if method == "fj" else "park_phase_shift",
            backend=backend,
            source_h5_sha256=source_hash,
            source_task_id=mat["task_id"],
            input_pair_count=len(gathers["all_pair_distance_m"]),
            input_offset_count=len(offsets),
            bin_width_m=bin_width_m if name == "binned" else 0,
            trace_normalization="individual_pair_abs_peak_before_binning",
            bin_edges_m=gathers["bin_edges_m"],
            bin_counts=gathers["bin_counts"],
        )
        save_h5(
            path,
            fields,
            format_name="node_fj_dispersion_v1"
            if method == "fj"
            else "node_park_dispersion_v1",
            data_axes="phase_velocity,frequency",
        )
        report["results"][name] = {
            "file": path.name,
            "sha256": digest(path),
            "input_traces": len(offsets),
            "shape": list(result["energy"].shape),
            "seconds": time.perf_counter() - started,
            "fft_spacing_hz": result["fft_spacing_hz"],
            "cpu_check_max_abs_error": float(
                np.max(
                    abs(result["raw_integral"][:, checks] - reference["raw_integral"])
                )
            ),
            "cpu_check_passed": True,
            "cpu_check_max_energy_error": float(
                np.max(abs(result["energy"][:, checks] - reference["energy"]))
            ),
            "cpu_check_scaled_tolerance": 2e-5,
        }
        LOG.info(
            "Computed %s: %d offsets, image %s",
            name,
            len(offsets),
            result["energy"].shape,
        )
    if digest(input_path) != source_hash:
        raise ValueError("Input H5 changed during computation")
    atomic_json(output_dir / "dispersion_validation.json", dict(report, passed=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in (
        ("fmin", 2),
        ("fmax", 30),
        ("df", 0.1),
        ("cmin", 500),
        ("cmax", 4000),
        ("dc", 10),
        ("distance-bin-m", 50),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument(
        "--backend", choices=("torch_cuda", "cpu"), default="torch_cuda"
    )
    parser.add_argument("--method", choices=("fj", "park"), default="fj")
    parser.add_argument(
        "--park-branch",
        choices=("positive", "negative", "symmetric"),
        default="positive",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--frequency-batch", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    compute(
        args.input,
        args.output_dir,
        fmin=args.fmin,
        fmax=args.fmax,
        df=args.df,
        cmin=args.cmin,
        cmax=args.cmax,
        dc=args.dc,
        bin_width_m=args.distance_bin_m,
        backend=args.backend,
        cuda_device=args.cuda_device,
        frequency_batch=args.frequency_batch,
        method=args.method,
        park_branch=args.park_branch,
    )


if __name__ == "__main__":
    main()
