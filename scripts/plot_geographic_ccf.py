#!/usr/bin/env python3
"""Inspect geographic node CCFs using real pair distances and hour stability.

Display normalization is per trace; original H5 samples remain unchanged.
All valid pairs and all available hours are retained. Distance bins are linear
means of normalized traces, without imposing positive/negative lag symmetry.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from node_output.artifacts import atomic_json, digest, load_result
from obspy.geodetics.base import gps2dist_azimuth

LOG = logging.getLogger(__name__)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def normalized(traces):
    peak = np.max(np.abs(traces), axis=1)
    return np.divide(
        traces, peak[:, None], out=np.zeros_like(traces), where=peak[:, None] > 0
    )


def edges(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - 0.0005, centers[0] + 0.0005])
    if np.any(np.diff(centers) <= 0):
        raise ValueError("Plot coordinate centers must be strictly increasing")
    middle = (centers[:-1] + centers[1:]) / 2
    return np.r_[
        centers[0] - (middle[0] - centers[0]),
        middle,
        centers[-1] + (centers[-1] - middle[-1]),
    ]


def distance_image(ax, traces, distance_m, lags, limit, title, *, cmap="RdBu_r"):
    # Exactly coincident distances are displayed as their linear mean; no pair is dropped.
    unique, inverse = np.unique(distance_m, return_inverse=True)
    summed = np.zeros((len(unique), len(lags)), dtype=np.float32)
    np.add.at(summed, inverse, traces)
    summed /= np.bincount(inverse)[:, None]
    image = ax.pcolormesh(
        edges(lags),
        edges(unique / 1000),
        summed,
        cmap=cmap,
        norm=Normalize(-limit, limit),
        rasterized=True,
    )
    ax.axvline(0, color="0.3", lw=0.4, ls=":")
    ax.set(
        xlabel="Lag (s)",
        ylabel="Pair distance (km)",
        title=title,
        xlim=(lags[0], lags[-1]),
        ylim=(0, max(0.1, unique[-1] / 1000 * 1.025)),
    )
    return image


def save(fig, output, name, dpi, vector=True, formats=("png", "pdf", "svg")):
    if "png" in formats or not vector:
        fig.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight")
    if vector and "pdf" in formats:
        fig.savefig(output / f"{name}.pdf", dpi=dpi, bbox_inches="tight")
    if vector and "svg" in formats:
        fig.savefig(output / f"{name}.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_result(
    path,
    hourly_dir,
    output,
    *,
    bin_width_m=50,
    clip_percentile=99.5,
    dpi=300,
    save_sources=True,
    dispersion_dir=None,
    ccf_cmap="RdBu_r",
    dispersion_cmap="magma",
    figure_width_in=7.2,
    figure_height_in=None,
    font_family="DejaVu Sans",
    font_size=8,
    export_formats=("png", "pdf", "svg"),
    fmin=None,
    fmax=None,
    vmin=None,
    vmax=None,
):
    if (
        ccf_cmap not in matplotlib.colormaps
        or dispersion_cmap not in matplotlib.colormaps
    ):
        raise ValueError("Unknown plotting colormap")
    if (
        figure_width_in <= 0
        or (figure_height_in is not None and figure_height_in <= 0)
        or font_size < 7
        or not export_formats
        or not set(export_formats) <= {"png", "pdf", "svg"}
    ):
        raise ValueError("Invalid figure dimensions, font size or export formats")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
        }
    )
    mat = load_result(path)
    if str(mat.get("geometry_type", "")) != "geographic":
        raise ValueError("Expected geographic GPS node results")
    ids = mat["station_id"].astype(str)
    data, mask, lags = mat["data"], mat["valid_mask"].astype(bool), mat["lags_s"]
    source, receiver = mat["pair_source_index"], mat["pair_receiver_index"]
    distance = mat["pair_distance_m"]
    if data.shape != (len(ids), len(lags), len(ids)) or not np.isfinite(data).all():
        raise ValueError("Invalid CCF cube")
    valid = mask[source, receiver]
    if not valid.any():
        raise ValueError("No valid pair traces")
    traces = data[source, :, receiver]
    display = normalized(traces)
    limit = max(float(np.percentile(np.abs(display[valid]), clip_percentile)), 1e-6)
    peak_lags = lags[np.argmax(np.abs(traces), axis=1)]
    central = int(np.argmin(mat["station_distance_m"].mean(axis=1)))
    latitude, longitude = mat["station_latitude_deg"], mat["station_longitude_deg"]
    lat0, lon0 = float(np.mean(latitude)), float(np.mean(longitude))
    coordinates = [
        gps2dist_azimuth(lat0, lon0, lat, lon)[:2]
        for lat, lon in zip(latitude, longitude, strict=True)
    ]
    radial, azimuth = np.asarray(coordinates).T
    east, north = (
        radial * np.sin(np.deg2rad(azimuth)) / 1000,
        radial * np.cos(np.deg2rad(azimuth)) / 1000,
    )
    upper = (np.floor(distance.max() / bin_width_m) + 1) * bin_width_m
    bins = np.arange(0, upper + bin_width_m / 2, bin_width_m)
    bin_index = np.minimum(
        np.searchsorted(bins, distance, side="right") - 1, len(bins) - 2
    )
    counts = np.bincount(bin_index[valid], minlength=len(bins) - 1)
    binned = np.zeros((len(counts), len(lags)), dtype=np.float32)
    np.add.at(binned, bin_index[valid], display[valid])
    binned = np.divide(
        binned,
        counts[:, None],
        out=np.full_like(binned, np.nan),
        where=counts[:, None] > 0,
    )

    start = datetime.fromtimestamp(
        float(mat["source_start_unix"]), ZoneInfo("Asia/Shanghai")
    )
    end = datetime.fromtimestamp(
        float(mat["source_end_unix"]), ZoneInfo("Asia/Shanghai")
    )
    end_label = end.strftime("%H:%M" if start.date() == end.date() else "%Y-%m-%d %H:%M")
    time_label = f"{start:%Y-%m-%d %H:%M}–{end_label} (UTC+8)"
    hourly_rows = []
    correlations = []
    hours = []
    input_hashes = []
    for name, expected_hash in zip(
        mat.get("source_files", []) if hourly_dir is not None else [],
        mat.get("source_file_sha256", []) if hourly_dir is not None else [],
        strict=True,
    ):
        hour_path = hourly_dir / str(name)
        if digest(hour_path) != str(expected_hash):
            raise ValueError(f"Hourly input checksum changed: {hour_path}")
        hour = load_result(hour_path)
        if not np.array_equal(hour["station_id"], ids) or not np.array_equal(
            hour["lags_s"], lags
        ):
            raise ValueError("Hourly station/lag geometry differs")
        hourly_valid = hour["valid_mask"][source, receiver].astype(bool) & valid
        h = hour["data"][source, :, receiver]
        denominator = np.linalg.norm(h, axis=1) * np.linalg.norm(traces, axis=1)
        cosine = np.divide(
            np.sum(h * traces, axis=1),
            denominator,
            out=np.full(len(source), np.nan),
            where=hourly_valid & (denominator > 0),
        )
        finite = cosine[np.isfinite(cosine)]
        if not len(finite):
            raise ValueError("No comparable hourly pair traces")
        q25, median, q75 = np.percentile(finite, [25, 50, 75])
        local = datetime.fromtimestamp(
            float(hour["source_start_unix"]), ZoneInfo("Asia/Shanghai")
        )
        hours.append(local.hour + local.minute / 60)
        correlations.append((q25, median, q75))
        hourly_rows.append(
            {
                "file": str(name),
                "local_start": local.isoformat(),
                "valid_pairs": int(hourly_valid.sum()),
                "cosine_q25": q25,
                "cosine_median": median,
                "cosine_q75": q75,
            }
        )
        input_hashes.append(str(expected_hash))

    output.mkdir(parents=True, exist_ok=True)
    dispersion = {}
    if dispersion_dir is not None:
        for name in ("binned", "all_pairs"):
            item = load_result(Path(dispersion_dir) / f"dispersion_{name}.h5")
            if str(item["source_h5_sha256"]) != digest(path):
                raise ValueError("Dispersion source H5 differs from displayed CCF")
            if name == "binned" and float(item["bin_width_m"]) != bin_width_m:
                raise ValueError("Dispersion and CCF distance bin widths differ")
            if (
                item["energy"].shape
                != (len(item["phase_velocity_m_s"]), len(item["frequency_hz"]))
                or not np.isfinite(item["energy"]).all()
            ):
                raise ValueError("Invalid dispersion image")
            dispersion[name] = item
        for key in ("frequency_hz", "phase_velocity_m_s"):
            np.testing.assert_array_equal(
                dispersion["binned"][key], dispersion["all_pairs"][key]
            )
    dispersion_label = (
        "Park phase shift"
        if dispersion
        and str(dispersion["all_pairs"].get("method")) == "park_phase_shift"
        else "F–J"
    )
    if dispersion and dispersion["binned"].get("method") != dispersion["all_pairs"].get(
        "method"
    ):
        raise ValueError("Dispersion methods differ between panels")
    dispersion_view = None
    if dispersion:
        frequency = dispersion["all_pairs"]["frequency_hz"]
        velocity = dispersion["all_pairs"]["phase_velocity_m_s"]
        bounds = np.array(
            [
                frequency[0] if fmin is None else fmin,
                frequency[-1] if fmax is None else fmax,
                velocity[0] if vmin is None else vmin,
                velocity[-1] if vmax is None else vmax,
            ],
            dtype=float,
        )
        if (
            not np.isfinite(bounds).all()
            or not 0 < bounds[0] < bounds[1]
            or not 0 < bounds[2] < bounds[3]
        ):
            raise ValueError("Require finite 0 < fmin < fmax and 0 < vmin < vmax (m/s)")
        if (
            bounds[0] < frequency[0] - 1e-8
            or bounds[1] > frequency[-1] + 1e-8
            or bounds[2] < velocity[0] - 1e-8
            or bounds[3] > velocity[-1] + 1e-8
        ):
            raise ValueError(
                "Dispersion display range exceeds computed H5 support; expand the scan with run_mohe_day_dispersion.sh first"
            )
        dispersion_view = {
            "fmin_hz": float(bounds[0]),
            "fmax_hz": float(bounds[1]),
            "vmin_m_s": float(bounds[2]),
            "vmax_m_s": float(bounds[3]),
        }
    overview_height = figure_height_in or (9.75 if dispersion else 6.5)
    atomic_json(
        output / "figure_contract.json",
        {
            "question": "What spatial coverage, lag-distance structure and dispersion energy does the final CCF stack show?",
            "claim": "Descriptive inspection of spatial coverage, lag-distance structure and dispersion energy; no verified mode picks.",
            "archetype": "quantitative grid",
            "backend": "python/matplotlib",
            "size_inches": [figure_width_in, overview_height],
            "panels": {
                "a": "Spatial coverage, all stations; full IDs in separate station map",
                "c": "Distance bins, linear mean of peak-normalized traces",
                "b": "Hourly cosine similarity to final stack; median and IQR across valid pairs",
                "d": "All valid pair CCFs at real distances",
                **(
                    {
                        "e": f"{dispersion_label} spectrum of binned CCFs in c",
                        "f": f"{dispersion_label} spectrum of all pair CCFs in d",
                    }
                    if dispersion
                    else {}
                ),
            },
            "display_normalization": "Each trace divided by its full-lag absolute peak; no lag symmetrization, smoothing, filtering or resampling.",
            "uncertainty": "IQR is across-pair dispersion, not a confidence interval. Final stack includes the compared hour.",
            "exports": list(export_formats),
            "source_h5_sha256": digest(path),
            "pairs_total": len(source),
            "pairs_displayed": int(valid.sum()),
            "invalid_pairs_excluded": int((~valid).sum()),
        },
    )

    fig = plt.figure(figsize=(figure_width_in, overview_height), layout="constrained")
    grid = fig.add_gridspec(3 if dispersion else 2, 2, width_ratios=(1, 1))
    ax_map = fig.add_subplot(grid[0, 0])
    ax_bins = fig.add_subplot(grid[1, 0])
    ax_hours = fig.add_subplot(grid[0, 1])
    ax_pairs = fig.add_subplot(grid[1, 1])
    ax_map.scatter(east, north, s=12, color="#547C96", zorder=2)
    ax_map.scatter(east[central], north[central], s=40, color="#C36C43", zorder=3)
    ax_map.annotate(
        ids[central],
        (east[central], north[central]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
    )
    ax_map.set(
        xlabel="East (km)", ylabel="North (km)", title=f"a  {len(ids)} GPS stations"
    )
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.margins(0.15)
    image = distance_image(
        ax_pairs,
        display[valid],
        distance[valid],
        lags,
        limit,
        f"d  All {valid.sum()} pairs\nPeak-normalized CCFs",
        cmap=ccf_cmap,
    )
    fig.colorbar(
        image,
        ax=ax_pairs,
        label=f"CCF (clipped at ±{limit:.2f})",
        fraction=0.035,
        pad=0.02,
    )
    bin_limit = max(float(np.nanpercentile(np.abs(binned), 99.5)), 1e-6)
    bimage = ax_bins.pcolormesh(
        edges(lags),
        bins / 1000,
        binned,
        cmap=ccf_cmap,
        norm=Normalize(-bin_limit, bin_limit),
        rasterized=True,
    )
    ax_bins.set(
        xlabel="Lag (s)",
        ylabel="Pair distance (km)",
        title=f"c  {bin_width_m:g} m distance bins\nLinear mean of normalized CCFs",
    )
    ax_bins.axvline(0, color="0.3", lw=0.4, ls=":")
    fig.colorbar(
        bimage, ax=ax_bins, label="Mean normalized CCF", fraction=0.035, pad=0.02
    )
    q25 = median = q75 = np.array([])
    if correlations:
        q25, median, q75 = np.asarray(correlations).T
        ax_hours.errorbar(
            hours,
            median,
            yerr=[median - q25, q75 - median],
            fmt="o-",
            ms=3,
            lw=1,
            capsize=2,
            color="#547C96",
        )
        ax_hours.set(
            xlabel="Hour (UTC+8)",
            ylabel="Cosine similarity to final stack",
            title="b  Hourly consistency\nMedian and IQR across pairs",
            ylim=(-0.05, 1.05),
        )
        ax_hours.set_xticks(hours[::2])
    else:
        ax_hours.bar(
            (bins[:-1] + bin_width_m / 2) / 1000,
            counts,
            width=bin_width_m / 1000,
            color="#547C96",
        )
        ax_hours.set(
            xlabel="Pair distance (km)",
            ylabel="Valid pair count",
            title=f"b  Pair distance distribution\n{bin_width_m:g} m bins",
        )

    def plot_dispersion(ax, item, title, binned_input=False):
        frequency = item["frequency_hz"]
        velocity = item["phase_velocity_m_s"] / 1000
        image = ax.pcolormesh(
            edges(frequency),
            edges(velocity),
            item["energy"],
            cmap=dispersion_cmap,
            norm=Normalize(0, 1),
            rasterized=True,
        )
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel="Phase velocity (km/s)",
            title=title,
            xlim=(dispersion_view["fmin_hz"], dispersion_view["fmax_hz"]),
            ylim=(
                dispersion_view["vmin_m_s"] / 1000,
                dispersion_view["vmax_m_s"] / 1000,
            ),
        )
        if binned_input:
            guide = 2 * bin_width_m * frequency / 1000
            selected = (guide >= velocity[0]) & (guide <= velocity[-1])
            ax.plot(
                frequency[selected],
                guide[selected],
                color="white",
                lw=0.7,
                ls="--",
                label="2 samples / wavelength guide",
            )
            ax.legend(
                loc="upper right",
                fontsize=6,
                frameon=True,
                facecolor="0.92",
                framealpha=0.9,
            )
        return image

    if dispersion:
        ax_e = fig.add_subplot(grid[2, 0])
        ax_f = fig.add_subplot(grid[2, 1])
        for ax, name, title in (
            (ax_e, "binned", f"e  {dispersion_label}: distance bins (from c)"),
            (ax_f, "all_pairs", f"f  {dispersion_label}: all pairs (from d)"),
        ):
            image = plot_dispersion(
                ax, dispersion[name], title, binned_input=name == "binned"
            )
            fig.colorbar(
                image,
                ax=ax,
                label=f"Normalized {dispersion_label} energy",
                fraction=0.035,
                pad=0.02,
            )
    fig.suptitle(f"Mohe Z–Z CCF | {time_label}", fontsize=10)
    save(fig, output, "ccf_overview", dpi, formats=export_formats)
    if dispersion:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), layout="constrained")
        for ax, name, title in (
            (axes[0], "binned", f"{bin_width_m:g} m distance bins"),
            (axes[1], "all_pairs", f"All {valid.sum()} pairs"),
        ):
            image = plot_dispersion(
                ax, dispersion[name], title, binned_input=name == "binned"
            )
            fig.colorbar(
                image,
                ax=ax,
                label=f"Normalized {dispersion_label} energy",
                fraction=0.035,
                pad=0.02,
            )
        save(fig, output, "dispersion_comparison", dpi, formats=export_formats)

    fig, ax = plt.subplots(figsize=(7.2, 6), layout="constrained")
    ax.scatter(east, north, s=24, color="#547C96", zorder=2)
    ax.scatter(
        east[central],
        north[central],
        s=65,
        color="#C36C43",
        zorder=3,
        label=f"Central source {ids[central]}",
    )
    ax.set(
        xlabel="East relative to mean GPS position (km)",
        ylabel="North relative to mean GPS position (km)",
        title=f"All {len(ids)} GPS stations | {time_label}",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.15)
    ax.legend(loc="upper right", fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    offsets = [
        (4, 4),
        (4, -12),
        (-16, 4),
        (-16, -12),
        (12, 12),
        (12, -20),
        (-24, 12),
        (-24, -20),
        (4, 24),
        (4, -30),
    ]
    for x, y, station in zip(east, north, ids, strict=True):
        annotation = ax.annotate(
            station,
            (x, y),
            xytext=offsets[0],
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "0.6"},
        )
        for offset in offsets:
            annotation.set_position(offset)
            # Text extent alone; leader lines can legitimately cross nearby points.
            bbox = plt.Text.get_window_extent(annotation, renderer).expanded(1.08, 1.1)
            if not any(bbox.overlaps(previous) for previous in occupied):
                break
        occupied.append(bbox)
    save(fig, output, "station_map", dpi, formats=export_formats)

    fig, ax = plt.subplots(figsize=(7.2, 5.5), layout="constrained")
    image = distance_image(
        ax,
        display[valid],
        distance[valid],
        lags,
        limit,
        f"{valid.sum()} node pairs | {time_label}",
        cmap=ccf_cmap,
    )
    fig.colorbar(image, ax=ax, label="Peak-normalized CCF", pad=0.02)
    save(fig, output, "ccf_all_pairs", dpi, formats=export_formats)

    def source_plot(index, vector=False):
        receivers = np.flatnonzero(mask[index])
        gather = normalized(data[index][:, receivers].T)
        distances = mat["station_distance_m"][index, receivers]
        fig, ax = plt.subplots(figsize=(7.2, 5.5), layout="constrained")
        image = distance_image(
            ax,
            gather,
            distances,
            lags,
            limit,
            f"Source {ids[index]} → {len(receivers)} receivers | {time_label}",
            cmap=ccf_cmap,
        )
        fig.colorbar(image, ax=ax, label="Peak-normalized CCF", pad=0.02)
        stem = f"ccf_source_{ids[index]}"
        save(fig, output / "sources", stem, dpi, vector=vector, formats=export_formats)

    (output / "sources").mkdir(exist_ok=True)
    if save_sources:
        for index in range(len(ids)):
            source_plot(index, vector=index == central)
            LOG.info("Source plot %d/%d: station %s", index + 1, len(ids), ids[index])
    else:
        source_plot(central, vector=True)

    write_csv(
        output / "pair_diagnostics.csv",
        [
            "source_id",
            "receiver_id",
            "distance_m",
            "valid_hours",
            "valid_short_windows",
            "valid_minutes",
            "absolute_peak",
            "global_peak_lag_s",
            "valid",
        ],
        (
            {
                "source_id": ids[a],
                "receiver_id": ids[b],
                "distance_m": distance[i],
                "valid_hours": int(mat["valid_file_count"][a, b]),
                "valid_short_windows": int(mat["valid_short_window_count"][a, b]),
                "valid_minutes": int(mat["valid_minute_count"][a, b]),
                "absolute_peak": float(np.max(abs(traces[i]))),
                "global_peak_lag_s": float(peak_lags[i]),
                "valid": bool(valid[i]),
            }
            for i, (a, b) in enumerate(zip(source, receiver, strict=True))
        ),
    )
    if hourly_rows:
        write_csv(output / "hourly_consistency.csv", list(hourly_rows[0]), hourly_rows)
    np.savez_compressed(
        output / "distance_bin_source_data.npz",
        lags_s=lags,
        distance_edges_m=bins,
        normalized_ccf_mean=binned,
        pair_count=counts,
    )
    summary = {
        "stations": len(ids),
        "valid_pairs": int(valid.sum()),
        "hours": len(hours),
        "source_sha256": digest(path),
        "hourly_sha256": input_hashes,
        "time_range_local": [start.isoformat(), end.isoformat()],
        "distance_range_m": [
            float(distance[valid].min()),
            float(distance[valid].max()),
        ],
        "central_source_id": str(ids[central]),
        "color_clip_percentile": clip_percentile,
        "display_color_limit": limit,
        "distance_bin_width_m": bin_width_m,
        "hourly_cosine_medians": median.tolist(),
        "hourly_cosine_IQR": np.column_stack([q25, q75]).tolist(),
        "near_zero_global_peak_fraction": float(np.mean(abs(peak_lags[valid]) <= 0.02)),
        "near_zero_definition_s": 0.02,
        "per_source_figures": len(ids) if save_sources else 1,
        "plot_settings": {
            "ccf_cmap": ccf_cmap,
            "dispersion_cmap": dispersion_cmap,
            "figure_width_in": figure_width_in,
            "figure_height_in": overview_height,
            "font_family": font_family,
            "font_size": font_size,
            "dpi": dpi,
            "export_formats": list(export_formats),
            "dispersion_display_range": dispersion_view,
        },
        "dispersion_method": dispersion_label if dispersion else None,
        "panel_order": [
            "map",
            "hourly_consistency" if correlations else "pair_distance_distribution",
            "binned_ccf",
            "all_pair_ccf",
        ]
        + (["binned_dispersion", "all_pair_dispersion"] if dispersion else []),
        "interpretation_limits": [
            "Display-normalized amplitudes are not physical amplitude comparisons.",
            "All-pair distance sorting mixes source azimuths; per-source views preserve source orientation.",
            "Global peak lags are not picked arrivals or fitted velocities.",
            "Hour-to-final similarity is descriptive; the reference includes that hour.",
        ],
    }
    atomic_json(output / "plot_diagnostics.json", summary)
    (output / "figure_notes.md").write_text(
        "# Geographic CCF inspection\n\n"
        f"{time_label}; {len(ids)} stations, {int(valid.sum())}/{len(valid)} valid pairs; 2–30 Hz, ±4 s.\n\n"
        "a: GPS positions relative to the mean coordinate; highlight: central source. All station IDs are shown in station_map.\n"
        f"c: {bin_width_m:g} m bins; linear mean of normalized CCFs. Empty bins are blank. Counts and matrices are exported. No lag symmetrization.\n"
        "b (without hourly inputs): Valid pair counts in the same distance bins as c.\n"
        "b (with hourly inputs): Cosine similarity of each hourly CCF to the final PWS stack. Symbols show the median; bars show the 25th–75th percentiles across valid station pairs, not confidence intervals. The final stack includes each compared hour; this is not independent validation.\n"
        "d: All valid pair traces at WGS84 pair distance. Each trace is divided by its absolute peak. Color clipping is global. Exactly equal distances are averaged only for display.\n"
        f"e/f (when available): {dispersion_label} spectra of c/d. Squared imaging magnitude normalized per frequency. No verified mode picks. The white dashed line in e is the 2-samples-per-wavelength guide for distance bins, not a guaranteed resolution limit.\n\n"
        "All 60 source plots retain the receiver geometry and directed lag convention; self/invalid pairs are excluded according to the stored validity mask.\n"
        "Original H5 data and metadata remain unchanged. Source data: input H5, pair_diagnostics.csv, hourly_consistency.csv, distance_bin_source_data.npz.\n",
        encoding="utf-8",
    )
    LOG.info("Plots saved: %s", output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--hourly-dir",
        type=Path,
        default=None,
        help="Optional hourly consistency comparison; omitted: plot pair distance distribution",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dispersion-dir",
        type=Path,
        help="对应的全配对/分箱频散H5目录；指定后生成六子图",
    )
    parser.add_argument("--distance-bin-m", type=float, default=50)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--fmin", type=float, help="频散显示最低频率，Hz；默认采用H5范围"
    )
    parser.add_argument(
        "--fmax", type=float, help="频散显示最高频率，Hz；默认采用H5范围"
    )
    parser.add_argument(
        "--vmin", type=float, help="频散显示最低相速度，m/s；默认采用H5范围"
    )
    parser.add_argument(
        "--vmax", type=float, help="频散显示最高相速度，m/s；默认采用H5范围"
    )
    parser.add_argument("--ccf-cmap", default="RdBu_r")
    parser.add_argument("--dispersion-cmap", default="magma")
    parser.add_argument("--figure-width-in", type=float, default=7.2)
    parser.add_argument("--figure-height-in", type=float)
    parser.add_argument("--font-family", default="DejaVu Sans")
    parser.add_argument("--font-size", type=float, default=8)
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png", "pdf", "svg"],
    )
    parser.add_argument(
        "--save-source-figures", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.distance_bin_m <= 0 or not 0 < args.clip_percentile <= 100 or args.dpi < 72:
        parser.error("Invalid bin width, color percentile or DPI")
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    report = plot_result(
        args.input,
        args.hourly_dir,
        args.output_dir,
        bin_width_m=args.distance_bin_m,
        clip_percentile=args.clip_percentile,
        dpi=args.dpi,
        save_sources=args.save_source_figures,
        dispersion_dir=args.dispersion_dir,
        ccf_cmap=args.ccf_cmap,
        dispersion_cmap=args.dispersion_cmap,
        figure_width_in=args.figure_width_in,
        figure_height_in=args.figure_height_in,
        font_family=args.font_family,
        font_size=args.font_size,
        export_formats=args.export_formats,
        fmin=args.fmin,
        fmax=args.fmax,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
