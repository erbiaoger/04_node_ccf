#!/usr/bin/env python3
"""Run and verify one local daytime of GPS-located Mohe Z-node pairs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from node_compute.backend import NativeSession
from node_compute.parameters import DEFAULTS
from node_compute.reference import ReferenceSession
from node_compute.schedule import groups
from node_io import NodeSACReader
from node_io.gps import discover_station_gps, write_station_table
from node_output.artifacts import atomic_json, digest, load_result
from node_pipeline import NodeCCFConfig, prepare, run_node_ccf

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def daytime_bounds(date, start_hour=8, end_hour=18):
    if not 0 <= start_hour < end_hour <= 23:
        raise ValueError("Require 0 <= start_hour < end_hour <= 23")
    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return (
        day.replace(hour=start_hour).astimezone(UTC),
        day.replace(hour=end_hour).astimezone(UTC),
    )


def validate_hour(path, catalog, params, group, task_start, task_end):
    mat = load_result(path)
    stations = len(catalog.stations)
    radius = round(params["maxlag"] * params["samp_freq"] / params["time_downsample"])
    data = mat["data"]
    if data.shape != (stations, radius * 2 + 1, stations) or data.dtype != np.float32:
        raise ValueError(f"Invalid cube shape/dtype: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite CCF: {path}")
    np.testing.assert_array_equal(mat["station_id"], catalog.station_ids)
    np.testing.assert_allclose(
        mat["station_latitude_deg"],
        [s.latitude_deg for s in catalog.stations],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        mat["station_longitude_deg"],
        [s.longitude_deg for s in catalog.stations],
        rtol=0,
        atol=0,
    )
    source, receiver = np.triu_indices(stations, 1)
    np.testing.assert_array_equal(mat["pair_source_index"], source)
    np.testing.assert_array_equal(mat["pair_receiver_index"], receiver)
    np.testing.assert_array_equal(
        data[source, :, receiver], data[receiver, ::-1, source]
    )
    np.testing.assert_array_equal(data[np.arange(stations), :, np.arange(stations)], 0)
    np.testing.assert_allclose(
        mat["lags_s"], np.arange(-radius, radius + 1) * float(mat["dt"]), atol=1e-12
    )
    np.testing.assert_allclose(
        float(mat["dt"]),
        params["time_downsample"] / params["samp_freq"],
        atol=0,
        rtol=0,
    )
    for key, upper in (
        ("valid_short_window_count", len(group.starts)),
        ("valid_minute_count", group.minute_count),
    ):
        counts = mat[key]
        if counts.shape != (stations, stations) or counts.dtype != np.uint64:
            raise ValueError(f"Invalid count matrix: {key}")
        np.testing.assert_array_equal(counts, counts.T)
        np.testing.assert_array_equal(np.diag(counts), 0)
        if np.any(counts > upper):
            raise ValueError(f"Excess effective count: {key}")
    np.testing.assert_array_equal(mat["valid_mask"], mat["valid_minute_count"] > 0)
    if np.any(mat["valid_minute_count"] > mat["valid_short_window_count"]):
        raise ValueError("More valid minutes than valid windows")
    begin = task_start + group.begin / catalog.sampling_rate
    end = task_start + group.end / catalog.sampling_rate
    if float(mat["source_start_unix"]) != begin or float(mat["source_end_unix"]) != end:
        raise ValueError("Incorrect hourly time range")
    if (
        float(mat["data_end_unix"]) > task_end
        or float(mat["data_start_unix"]) < task_start
    ):
        raise ValueError("Waveform read extends outside daytime")
    if (
        int(mat["actual_short_window_count"]) != len(group.starts)
        or int(mat["actual_minute_count"]) != group.minute_count
    ):
        raise ValueError("Incorrect schedule metadata")
    distances = mat["pair_distance_m"]
    if (
        distances.shape != source.shape
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
    ):
        raise ValueError("Invalid geographic distances")
    np.testing.assert_array_equal(
        mat["station_distance_m"][source, receiver], distances
    )
    np.testing.assert_array_equal(
        mat["station_distance_m"], mat["station_distance_m"].T
    )
    if params.get("cc_enhance_method") == "tsi":
        raw = mat["data_raw"]
        if (
            raw.shape != data.shape
            or raw.dtype != np.float32
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Invalid original CCF in TSI output")
        np.testing.assert_array_equal(raw, raw.transpose(2, 1, 0)[:, ::-1, :])
        applied = mat["tsi_applied_mask"].astype(bool)
        count = mat["tsi_applied_group_count"]
        support = mat["tsi_common_station_count_min"]
        group_count = int(mat["tsi_group_count"])
        for values in (applied, count, support):
            if (
                values.shape != (stations, stations)
                or not np.array_equal(values, values.T)
                or np.diag(values).any()
            ):
                raise ValueError("Invalid TSI support matrices")
        if (
            np.any(count > group_count)
            or np.any(support > stations - 2)
            or np.any(applied & ~mat["valid_mask"].astype(bool))
        ):
            raise ValueError("Invalid TSI application counts")
        np.testing.assert_array_equal(applied, count > 0)
        untouched = ~applied[source, receiver]
        np.testing.assert_array_equal(
            data[source[untouched], :, receiver[untouched]],
            raw[source[untouched], :, receiver[untouched]],
        )
        diagnostics = json.loads(str(mat["tsi_diagnostics_json"]))
        if len(diagnostics) != group_count or sum(
            d["short_windows"] for d in diagnostics
        ) != len(group.starts):
            raise ValueError("Wrong TSI group/window metadata")
    for key, stamp in (
        ("source_start_utc", begin),
        ("source_end_utc", end),
        ("source_start_local", begin),
        ("source_end_local", end),
    ):
        if datetime.fromisoformat(str(mat[key])).timestamp() != stamp:
            raise ValueError(f"Incorrect timestamp: {key}")
    return {
        "file": path.name,
        "shape": list(data.shape),
        "scheduled_windows": len(group.starts),
        "valid_pairs": int(mat["valid_mask"][source, receiver].sum()),
        "min_valid_windows": int(
            mat["valid_short_window_count"][source, receiver].min()
        ),
        "max_valid_windows": int(
            mat["valid_short_window_count"][source, receiver].max()
        ),
        "source_start_local": str(mat["source_start_local"]),
        "source_end_local": str(mat["source_end_local"]),
        "passed": True,
    }


def real_cuda_reference_check(catalog, params, start):
    """Check five real 10s windows of three stations, including a masked gap."""
    reader = NodeSACReader(catalog, workers=8)
    try:
        raw, valid = reader.read_samples(start, 15040)
    finally:
        reader.close()
    data = np.ascontiguousarray(raw[:, :3])
    mask = np.ascontiguousarray(valid[:, :3])
    mask[7000, 1] = False
    starts = np.arange(40, 10041, 2500)
    minutes = np.array([0, 0, 1, 1, 2])
    native = NativeSession(params, 3)
    reference = ReferenceSession(dict(params, cc_backend="cpu", cc_device="cpu"), 3)
    native.add_block(data, mask, starts, minutes)
    reference.add_block(data, mask, starts, minutes)
    actual, expected = native.finish_group(), reference.finish_group()
    for key in ("short_count", "valid_count"):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_allclose(actual["pairs"], expected["pairs"], rtol=1e-3, atol=2e-4)
    return {
        "passed": True,
        "stations": list(catalog.station_ids[:3]),
        "windows": 5,
        "masked_gap": True,
        "max_abs_error": float(np.max(np.abs(actual["pairs"] - expected["pairs"]))),
        "rtol": 1e-3,
        "atol": 2e-4,
    }


def verify_task(config):
    catalog, params, start, end, total = prepare(config)
    manifest = json.loads((config.output_dir / "task_manifest.json").read_text())
    reports = []
    for group in groups(total, catalog.sampling_rate, params):
        path = (
            config.output_dir / f"cc_stack_{group.index:04d}_n{group.minute_count}.h5"
        )
        record = manifest["completed"].get(str(group.index), {})
        if record.get("file") != path.name or record.get("sha256") != digest(path):
            raise ValueError(f"Incomplete output or checksum mismatch: {path}")
        reports.append(validate_hour(path, catalog, params, group, start, end))
        if str(load_result(path, ["task_id"])["task_id"]) != manifest["task_id"]:
            raise ValueError(f"Incompatible task ID: {path}")
    if manifest["status"] != "complete" or len(
        list(config.output_dir.glob("cc_stack_*.h5"))
    ) != len(reports):
        raise ValueError("Unexpected task status or hourly file count")
    return {
        "passed": True,
        "task_id": manifest["task_id"],
        "stations": len(catalog.stations),
        "pairs": len(catalog.stations) * (len(catalog.stations) - 1) // 2,
        "hours": len(reports),
        "files": reports,
        "diagnostics": manifest["diagnostics"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-07-08")
    parser.add_argument("--start-hour", type=int, default=8)
    parser.add_argument("--end-hour", type=int, default=18)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/mnt/nas-downstairs1/mohe_cut/z_cut")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    start, end = daytime_bounds(args.date, args.start_hour, args.end_hour)
    output = (
        args.output_dir
        or ROOT
        / "outputs"
        / f"mohe_{args.date.replace('-', '')}_day_{args.start_hour:02d}00_{args.end_hour:02d}00"
    ).resolve()
    data_dir = args.data_dir.resolve()
    csv_path = output / "stations_gps.csv"
    params = dict(
        DEFAULTS,
        read_mode="block",
        read_hours=24,
        prefetch_blocks=0,
        async_save=0,
        save_every=60,
        time_downsample=2,
        cc_batch_chunks=12,
        sac_sampling_rate_hz=500,
    )
    config = NodeCCFConfig(
        csv_path, data_dir, output, params, start.isoformat(), end.isoformat()
    )
    if args.verify_only:
        report = verify_task(config)
        atomic_json(output / "validation.json", report)
        print(json.dumps(report, indent=2))
        return
    rows = discover_station_gps(data_dir, start.strftime("%Y/%m/%d"))
    write_station_table(csv_path, rows)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    atomic_json(
        output / "run_config.json",
        dict(
            params,
            station_csv=str(csv_path),
            node_data_dir=str(data_dir),
            node_output_dir=str(output),
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
        ),
    )
    catalog, validated, begin, finish, total = prepare(config)
    if any(not station.segments for station in catalog.stations):
        raise ValueError("Missing selected-day waveforms for a discovered station")
    schedule = list(groups(total, catalog.sampling_rate, validated))
    preflight = {
        "stations": len(catalog.stations),
        "pairs": len(catalog.stations) * (len(catalog.stations) - 1) // 2,
        "raw_sampling_rate": catalog.sampling_rate,
        "processing_sampling_rate": 250,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "hours": len(schedule),
        "host_cache_gib": total * len(catalog.stations) * 5 / 1024**3,
        "station_ids": list(catalog.station_ids),
        "gps": rows,
    }
    atomic_json(output / "preflight.json", preflight)
    LOG.info(
        "Preflight: %d stations, %d pairs, %d hours, %.3f GiB host cache",
        preflight["stations"],
        preflight["pairs"],
        preflight["hours"],
        preflight["host_cache_gib"],
    )
    if args.preflight:
        print(json.dumps(preflight, indent=2))
        return
    parity = real_cuda_reference_check(catalog, validated, begin)
    atomic_json(output / "cuda_reference_validation.json", parity)
    LOG.info("Real CUDA/CPU check passed: max error %.6g", parity["max_abs_error"])
    by_begin = {
        begin + group.begin / catalog.sampling_rate: group for group in schedule
    }

    def check(path, fields):
        group = by_begin[fields["source_start_unix"]]
        report = validate_hour(path, catalog, validated, group, begin, finish)
        if group.index == 1:
            atomic_json(output / "first_hour_validation.json", report)
            LOG.info("FIRST HOUR VERIFIED; proceeding to remaining hours")
        LOG.info(
            "Hourly validation passed: %s, %d valid pairs",
            path.name,
            report["valid_pairs"],
        )

    run_node_ccf(config, group_validator=check)
    report = verify_task(config)
    atomic_json(output / "validation.json", dict(report, cuda_reference=parity))
    LOG.info(
        "ALL HOURS VERIFIED: %d H5 files, %d stations, %d pairs",
        report["hours"],
        report["stations"],
        report["pairs"],
    )


if __name__ == "__main__":
    main()
