"""Apply a symmetric early-lag cosine mute, preserving original CCF samples."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.io import savemat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def early_lag_weights(lags, distances, velocity, shoulder_fraction):
    if not np.isfinite(velocity) or velocity <= 0 or not 0 < shoulder_fraction < 1:
        raise ValueError("Positive velocity and shoulder fraction in (0,1) required")
    if np.any(distances <= 0) or not np.isfinite(distances).all():
        raise ValueError("Distances must be finite and positive")
    end = distances / velocity
    start = end * (1 - shoulder_fraction)
    phase = np.clip((abs(lags[:, None]) - start) / (end - start), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * phase)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/stack/line_source15/source_15_gather.h5"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stack/line_source15_muted")
    )
    p.add_argument(
        "--velocity",
        type=float,
        default=4000,
        help="Earliest retained velocity boundary in m/s",
    )
    p.add_argument(
        "--shoulder-fraction",
        type=float,
        default=0.2,
        help="Cosine rise occupies last fraction of distance/velocity",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.input.name
    if output.resolve() == args.input.resolve():
        raise ValueError("Output must differ from original input")
    with h5py.File(args.input) as f:
        raw = f["data"][:]
        lags = f["lags_s"][:]
        distances = f["receiver_distance_m"][:]
        ids = f["receiver_station_id"].asstr()[:].tolist()
        source = f["source_station_id"].asstr()[()]
        payload = {key: f[key][()] for key in f}
    weights = early_lag_weights(lags, distances, args.velocity, args.shoulder_fraction)
    data = (raw * weights).astype(np.float32)
    assert np.isfinite(data).all()
    np.testing.assert_array_equal(data[weights == 1], raw[weights == 1])
    assert np.all(data[weights == 0] == 0)
    np.testing.assert_allclose(weights, weights[::-1], atol=1e-12)
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "input_file": str(args.input.resolve()),
        "input_sha256": digest,
        "method": "symmetric early-lag cosine mute; not zero-phase frequency filtering",
        "velocity_m_s": args.velocity,
        "shoulder_fraction": args.shoulder_fraction,
        "source": source,
        "receivers": ids,
        "zero_until_s": (
            distances / args.velocity * (1 - args.shoulder_fraction)
        ).tolist(),
        "fully_retained_from_s": (distances / args.velocity).tolist(),
        "removed_energy_fraction": (
            1
            - np.sum(data.astype(float) ** 2, axis=0)
            / np.sum(raw.astype(float) ** 2, axis=0)
        ).tolist(),
        "limitations": "Distance/velocity prior imposed by mute; improvement in imaging alone is not evidence of a real propagation mode.",
        "figure_qa": "All six traces, full lags, same raw per-trace amplitude scale on both panels. PNG at 300 dpi for diagnosis; no uncertainty estimates applicable.",
    }
    payload.update(
        data=data,
        data_raw=raw,
        mute_weights=weights.astype(np.float32),
        mute_parameters_json=json.dumps(report),
        normalization="No amplitude normalization; data = data_raw * mute_weights",
    )
    temporary = output.with_suffix(".h5.tmp")
    with h5py.File(temporary, "w") as f:
        for key, value in payload.items():
            f.create_dataset(key, data=value)
    temporary.replace(output)
    # Convert HDF5 variable-length strings into MATLAB-readable strings.
    savemat(
        output.with_suffix(".mat"),
        {k: v.decode() if isinstance(v, bytes) else v for k, v in payload.items()},
    )
    with h5py.File(output) as f:
        np.testing.assert_array_equal(f["data_raw"][:], raw)
        np.testing.assert_array_equal(f["data"][:], data)

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
        1, 2, figsize=(7.2, 5), layout="constrained", sharex=True, sharey=True
    )
    scale = np.max(abs(raw), axis=0)
    amplitude = np.min(np.diff(np.r_[0, distances])) * 0.35
    for ax, matrix, title in zip(
        axes, [raw, data], ["Original", "After early-lag mute"]
    ):
        for j, distance in enumerate(distances):
            trace = matrix[:, j] / scale[j]
            x = distance + trace * amplitude
            ax.plot(x, lags, color="#24587a", linewidth=0.65)
            ax.fill_betweenx(
                lags, distance, x, where=trace > 0, color="#24587a", alpha=0.5
            )
        ax.axhline(0, color="0.7", linewidth=0.6)
        ax.set_xticks(distances, ids)
        ax.set_xlabel("Receiver station (true offset spacing)")
        ax.set_title(title)
        ax.set_ylim(lags[-1], lags[0])
    axes[1].plot(
        distances, distances / args.velocity, "--", color="#b76a26", linewidth=0.8
    )
    axes[1].plot(
        distances, -distances / args.velocity, "--", color="#b76a26", linewidth=0.8
    )
    axes[0].set_ylabel("Correlation lag (s)")
    fig.suptitle(
        f"Source {source} | identical amplitude scale before/after\nDashed lines: fully retained boundary at {args.velocity:.0f} m/s"
    )
    stem = args.output_dir / "mute_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    (args.output_dir / "mute_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
