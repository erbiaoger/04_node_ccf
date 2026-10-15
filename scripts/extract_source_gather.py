"""Extract an irregular receiver gather without changing stacked CCF samples."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from scipy.io import savemat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/stack/stack_final.h5")
    )
    parser.add_argument(
        "--stations", nargs="+", default=["15", "4", "65", "13", "25", "42", "19"]
    )
    parser.add_argument("--source", default="19")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/stack/line_source19")
    )
    args = parser.parse_args()
    if (
        len(set(args.stations)) != len(args.stations)
        or args.source not in args.stations
    ):
        raise ValueError("Stations must be unique and include the source")
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as f:
        ids = f["station_id"].asstr()[:].tolist()
        s = ids.index(args.source)
        lat = f["station_latitude_deg"][:]
        lon = f["station_longitude_deg"][:]
        receivers = [ids.index(i) for i in args.stations if i != args.source]
        distances = {
            i: gps2dist_azimuth(lat[s], lon[s], lat[i], lon[i])[0] for i in receivers
        }
        receivers.sort(key=distances.get)
        offsets = np.array([distances[i] for i in receivers])
        data = np.column_stack([f["data"][s, :, i] for i in receivers])
        lags = f["lags_s"][:]
        payload = {
            "data": data,
            "dt": f["dt"][()],
            "lags_s": lags,
            "time_axis_s": lags,
            "profile_x_m": offsets,
            "receiver_distance_m": offsets,
            "source_station_id": args.source,
            "receiver_station_id": np.array([ids[i] for i in receivers], dtype="S"),
            "source_latitude_deg": lat[s],
            "source_longitude_deg": lon[s],
            "receiver_latitude_deg": lat[receivers],
            "receiver_longitude_deg": lon[receivers],
            "original_source_index": s,
            "original_receiver_index": receivers,
        }
        for key in (
            "valid_mask",
            "valid_file_count",
            "valid_minute_count",
            "valid_short_window_count",
        ):
            payload[key] = np.array([f[key][s, i] for i in receivers])
        for key in ("parameters_json", "correlation_definition", "coordinate_system"):
            if key in f:
                payload[key] = f[key].asstr()[()]
        assert payload["valid_mask"].all() and np.isfinite(data).all()
        assert np.all(np.diff(offsets) > 0) and np.all(np.diff(lags) > 0)
        for col, i in enumerate(receivers):
            np.testing.assert_array_equal(data[:, col], f["data"][s, :, i])
            np.testing.assert_allclose(
                f["station_distance_m"][s, i], offsets[col], atol=0.01
            )
            np.testing.assert_allclose(
                data[:, col], f["data"][i, :, s][::-1], atol=1e-6
            )
    with args.input.open("rb") as stream:
        payload["input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    payload["input_file"] = str(args.input.resolve())
    payload["axis_order"] = "lag,receiver"
    payload["geometry_type"] = "single_source_geographic_offsets"
    payload["normalization"] = "none; exact input samples"
    stem = args.output / f"source_{args.source}_gather"
    temporary = stem.with_suffix(".h5.tmp")
    with h5py.File(temporary, "w") as f:
        for key, value in payload.items():
            f.create_dataset(key, data=value)
    temporary.replace(stem.with_suffix(".h5"))
    savemat(stem.with_suffix(".mat"), payload)

    # Show every selected trace at its actual radial offset, over the full lag axis.
    # Per-trace peak normalization affects this preview only; no uncertainty claim.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 6), layout="constrained")
    amplitude = min(np.diff(np.r_[0, offsets])) * 0.35
    for col, distance in enumerate(offsets):
        trace = data[:, col] / max(float(np.max(np.abs(data[:, col]))), 1e-30)
        x = distance + trace * amplitude
        ax.plot(x, lags, color="#24587a", linewidth=0.65)
        ax.fill_betweenx(lags, distance, x, where=trace > 0, color="#24587a", alpha=0.5)
    ax.axvline(0, color="0.65", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="0.8", linewidth=0.6)
    ax.set_xticks(
        np.r_[0, offsets],
        [f"{args.source}\n0"] + [f"{ids[i]}\n{distances[i]:.0f}" for i in receivers],
    )
    ax.set_xlabel("Station ID / source-receiver distance (m)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("Correlation lag (s)")
    ax.set_ylim(lags[-1], lags[0])
    ax.set_title(
        f"Virtual source {args.source}: stacked CCF gather\nPer-trace peak normalization for display only"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    report = {
        "source": args.source,
        "receivers": [ids[i] for i in receivers],
        "distance_m": offsets.tolist(),
        "shape": list(data.shape),
        "dt": float(payload["dt"]),
        "input_sha256": payload["input_sha256"],
        "verified": "exact samples, valid pairs, geodesic offsets, reverse-lag symmetry",
        "display": "all selected receivers; per-trace peak normalization; no resampling, binning or interpolation",
    }
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
