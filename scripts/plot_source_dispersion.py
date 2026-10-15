"""Image a selected single-source CCF gather using its true irregular offsets."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from node_compute.dispersion import frequency_bessel, park_phase_shift

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/stack/line_source19/source_19_gather.h5"),
    )
    p.add_argument("--fmin", type=float, default=1)
    p.add_argument("--fmax", type=float, default=10)
    p.add_argument("--df", type=float, default=0.1)
    p.add_argument("--vmin", type=float, default=500)
    p.add_argument("--vmax", type=float, default=4000)
    p.add_argument("--dv", type=float, default=10)
    args = p.parse_args()
    if not (
        0 < args.fmin < args.fmax
        and 0 < args.vmin < args.vmax
        and args.df > 0
        and args.dv > 0
    ):
        raise ValueError("Invalid frequency/velocity grid")
    with h5py.File(args.input) as h:
        traces = h["data"][:].T.astype(float)
        distances = h["receiver_distance_m"][:]
        lags = h["lags_s"][:]
        dt = float(h["dt"][()])
        source = h["source_station_id"].asstr()[()]
        mute_metadata = (
            h["mute_parameters_json"].asstr()[()] if "mute_parameters_json" in h else ""
        )
        receivers = h["receiver_station_id"].asstr()[:].tolist()
        if not h["valid_mask"][:].all():
            raise ValueError("Gather includes invalid traces")
    assert np.isfinite(traces).all() and np.all(np.diff(distances) > 0)
    peaks = np.max(abs(traces), axis=1, keepdims=True)
    if np.any(peaks == 0):
        raise ValueError("Zero-amplitude trace")
    traces /= peaks
    zero = int(np.argmin(abs(lags)))
    np.testing.assert_allclose(lags, (np.arange(len(lags)) - zero) * dt, atol=1e-8)
    frequencies = np.arange(args.fmin, args.fmax + args.df / 2, args.df)
    velocities = np.arange(args.vmin, args.vmax + args.dv / 2, args.dv)
    results = {}
    for branch in ("positive", "negative", "symmetric"):
        results[f"park_{branch}"] = park_phase_shift(
            traces,
            distances,
            dt,
            frequencies,
            velocities,
            zero_lag_index=zero,
            backend="cpu",
            branch=branch,
        )
    results["fj"] = frequency_bessel(
        traces,
        distances,
        dt,
        frequencies,
        velocities,
        zero_lag_index=zero,
        backend="cpu",
    )
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    stem = args.input.with_name(f"source_{source}_dispersion")
    with h5py.File(stem.with_suffix(".h5.tmp"), "w") as h:
        h.attrs.update(
            input_file=str(args.input.resolve()),
            input_sha256=digest,
            source_station_id=source,
            preprocessing="Per-trace peak normalization; no binning or resampling",
            data_axes="phase_velocity,frequency",
            display="Energy normalized per frequency; no automatic picks",
            input_mute_parameters_json=mute_metadata,
        )
        h.create_dataset("receiver_station_id", data=np.asarray(receivers, dtype="S"))
        for name, result in results.items():
            group = h.create_group(name)
            for key, value in result.items():
                group.create_dataset(key, data=value)
            assert result["energy"].shape == (len(velocities), len(frequencies))
            assert np.isfinite(result["energy"]).all()
    stem.with_suffix(".h5.tmp").replace(stem.with_suffix(".h5"))

    # Diagnostic comparison: all selected receivers, actual distances, identical grids.
    # No mode identification, interpolation beyond FFT support, or uncertainty claim.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 6.5), layout="constrained", sharex=True, sharey=True
    )
    titles = [
        "Park: positive lag",
        "Park: negative lag (reversed)",
        "Park: symmetric CCF",
        "F-J: full CCF, J0",
    ]
    for ax, (name, result), title in zip(axes.flat, results.items(), titles):
        im = ax.pcolormesh(
            frequencies,
            velocities,
            result["energy"],
            shading="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlim(args.fmin, args.fmax)
        ax.set_ylim(args.vmin, args.vmax)
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Phase velocity (m/s)")
    fig.colorbar(im, ax=axes, label="Normalized spectral energy", shrink=0.85)
    fig.suptitle(
        f"Virtual source {source} | {len(receivers)} receivers | {distances[0]:.0f}–{distances[-1]:.0f} m"
        + ("\nEarly-lag cosine mute applied" if mute_metadata else "")
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    report = {
        "source": source,
        "receivers": receivers,
        "input_sha256": digest,
        "input_mute_parameters": json.loads(mute_metadata) if mute_metadata else None,
        "frequency_hz": [args.fmin, args.fmax, args.df],
        "velocity_m_s": [args.vmin, args.vmax, args.dv],
        "fft_spacing_hz": {name: r["fft_spacing_hz"] for name, r in results.items()},
        "notes": "Selected irregularly spaced traces only; no distance bins. Grid interpolation does not improve FFT resolution. Sparse spatial sampling can cause aliasing. Peaks are not confirmed modes.",
        "figure_qa": "Diagnostic PNG at 300 dpi; editable PDF/SVG. Four panels share scales and per-frequency normalization. No uncertainty estimates applicable.",
    }
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
