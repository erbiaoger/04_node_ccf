#!/usr/bin/env python3
"""JSONC-configured GPS node CC/TSI entry point; one H5 per save interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dasqt.features.dispersion.backend.components.ccf.folder_pipeline import (
    load_config_json,
)
from node_compute.parameters import DEFAULTS, OPTIONAL_PARAMS
from node_compute.schedule import groups
from node_io.catalog import _read_csv_rows, filename_interval
from node_io.gps import discover_station_gps, write_station_table
from node_output.artifacts import atomic_json, digest
from node_pipeline import NodeCCFConfig, prepare, run_node_ccf
from run_mohe_day import validate_hour, verify_task

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def config_from_values(
    values, *, root=ROOT, date=None, start_hour=None, end_hour=None, output_dir=None
):
    date = date or values.get(
        "processing_start_date", values.get("processing_date", "2026-07-08")
    )
    timezone = values.get("local_timezone", "Asia/Shanghai")
    if timezone != "Asia/Shanghai":
        raise ValueError("This daytime entry point uses Asia/Shanghai")

    def clock(key, legacy, override, default):
        if override is not None:
            if not 0 <= override <= 23:
                raise ValueError("Hour override must be between 0 and 23")
            return f"{override:02d}:00"
        value = values.get(key, f"{values.get(legacy, default):02d}:00")
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a quoted HH:MM string")
        parsed = datetime.strptime(value, "%H:%M")
        if parsed.strftime("%H:%M") != value:
            raise ValueError(f"{key} must use HH:MM format")
        return value

    full_day = time_mode(values) == "all_date_time"
    first = (
        "00:00" if full_day else clock("start_time", "day_start_hour", start_hour, 8)
    )
    last = "24:00" if full_day else clock("end_time", "day_end_hour", end_hour, 18)
    if first >= last:
        raise ValueError("Require start_time < end_time within the same day")
    zone = ZoneInfo(timezone)
    start = (
        datetime.fromisoformat(f"{date}T{first}").replace(tzinfo=zone).astimezone(UTC)
    )
    end = (
        (start + timedelta(days=1))
        if full_day
        else datetime.fromisoformat(f"{date}T{last}")
        .replace(tzinfo=zone)
        .astimezone(UTC)
    )
    if full_day and values.get("_available_bounds"):
        lo, hi = values["_available_bounds"]
        start = max(start, datetime.fromtimestamp(lo, UTC))
        end = min(end, datetime.fromtimestamp(hi, UTC))

    def path(value):
        result = Path(value).expanduser()
        return (result if result.is_absolute() else root / result).resolve()

    params = dict(DEFAULTS)
    params.update(
        {
            k: v
            for k, v in values.items()
            if k in DEFAULTS
            or k in OPTIONAL_PARAMS
            or k in {"fk_enabled", "include_autocorr", "pair_mode"}
        }
    )
    declared = [
        float(values[k])
        for k in ("sac_sampling_rate_hz", "sps", "samp_freq")
        if k in values
    ]
    if declared and any(v != declared[0] for v in declared):
        raise ValueError(
            "sac_sampling_rate_hz, sps and samp_freq must agree on the raw sampling rate"
        )
    if declared:
        params["sac_sampling_rate_hz"] = declared[0]
    mode = params.get("cc_enhance_method", "raw")
    output = (
        path(output_dir)
        if output_dir is not None
        else path(values.get("node_output_dir", "outputs/mohe_node_ccf"))
        / f"{date.replace('-', '')}_day_{first.replace(':', '')}_{last.replace(':', '')}_{mode}"
    )
    coordinate_mode = values.get("station_coordinate_mode", "fixed")
    if coordinate_mode not in {"fixed", "daily_gps"}:
        raise ValueError("station_coordinate_mode must be fixed or daily_gps")
    if coordinate_mode == "daily_gps":
        output = output.with_name(output.name + "_daily_gps")
        csv = output / "stations_gps.csv"
    else:
        csv = path(values.get("station_csv") or "config/stations_fixed.csv")
    return NodeCCFConfig(
        csv,
        path(values["node_data_dir"]),
        output,
        params,
        start.isoformat(),
        end.isoformat(),
    ), coordinate_mode == "fixed"


def time_mode(values):
    mode = values.get("processing_time_mode", "date_range")
    if mode not in {"date_range", "all_date", "all_date_time"}:
        raise ValueError(
            "processing_time_mode must be date_range, all_date or all_date_time"
        )
    return mode


def available_intervals(values, root=ROOT):
    from obspy.io.sac.sactrace import SACTrace

    directory = Path(values["node_data_dir"]).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    intervals = set()
    for path in directory.rglob("*"):
        if path.suffix.lower() != ".sac":
            continue
        interval = filename_interval(path)
        if interval is None:
            header = SACTrace.read(str(path), headonly=True)
            start = float(header.reftime.timestamp) + float(header.b or 0)
            interval = (start, start + header.npts * float(header.delta))
        if interval[1] > interval[0]:
            intervals.add(interval)
    if not intervals:
        raise ValueError(f"No SAC waveform dates found: {directory}")
    return sorted(intervals)


def processing_dates(values, override=None):
    mode = time_mode(values)
    if override:
        return [datetime.strptime(override, "%Y-%m-%d").strftime("%Y-%m-%d")]
    if mode != "date_range":
        intervals = values.get("_available_intervals") or available_intervals(values)
        dates = set()
        zone = ZoneInfo(values.get("local_timezone", "Asia/Shanghai"))
        for lo, hi in intervals:
            day = datetime.fromtimestamp(lo, zone).date()
            last = datetime.fromtimestamp(hi - 0.001, zone).date()
            while day <= last:
                date = day.isoformat()
                config, _ = config_from_values(values, date=date)
                if (
                    datetime.fromisoformat(config.start_utc).timestamp() < hi
                    and datetime.fromisoformat(config.end_utc).timestamp() > lo
                ):
                    dates.add(date)
                day += timedelta(days=1)
        return sorted(dates)
    first = values.get(
        "processing_start_date", values.get("processing_date", "2026-07-08")
    )
    last = values.get("processing_end_date", first)
    start, end = (datetime.strptime(v, "%Y-%m-%d") for v in (first, last))
    if end < start:
        raise ValueError("processing_end_date must be >= processing_start_date")
    return [
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end - start).days + 1)
    ]


def run_day(config, explicit_csv, args):
    if not explicit_csv and not args.verify_only:
        date = (
            datetime.fromisoformat(config.start_utc)
            + (
                datetime.fromisoformat(config.end_utc)
                - datetime.fromisoformat(config.start_utc)
            )
            / 2
        ).strftime("%Y/%m/%d")
        rows = discover_station_gps(
            config.data_dir, date, workers=int(config.cc_params["io_workers"])
        )
        write_station_table(config.csv_path, rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.output_dir / "run.log"),
        ],
    )
    if explicit_csv:
        LOG.info(
            "Fixed station geometry: %d cached stations, SHA256=%s",
            len(config.station_records),
            config.station_csv_sha256,
        )
    catalog, p, start, end, total = prepare(config)
    if any(not s.segments for s in catalog.stations):
        raise ValueError("Selected day is missing waveforms for a discovered station")
    schedule = list(groups(total, catalog.sampling_rate, p))
    summary = {
        "station_coordinate_mode": "fixed" if explicit_csv else "daily_gps",
        "station_csv": str(config.csv_path),
        "station_csv_sha256": config.station_csv_sha256
        if explicit_csv
        else digest(config.csv_path),
        "stations": len(catalog.stations),
        "pairs": len(catalog.stations) * (len(catalog.stations) - 1) // 2,
        "raw_sampling_rate": catalog.sampling_rate,
        "processing_sampling_rate": catalog.sampling_rate / p["time_downsample"],
        "start_utc": config.start_utc,
        "end_utc": config.end_utc,
        "save_interval_minutes": p["save_every"] * p["minute_stack_s"] / 60,
        "tsi_interval_minutes": p.get("tsi_files_per_group", 0)
        * p["minute_stack_s"]
        / 60,
        "cc_enhance_method": p.get("cc_enhance_method", "raw"),
        "output_files": len(schedule),
        "output_dir": str(config.output_dir),
        "host_cache_gib": total * len(catalog.stations) * 5 / 1024**3,
        "station_ids": list(catalog.station_ids),
    }
    atomic_json(config.output_dir / "preflight.json", summary)
    atomic_json(
        config.output_dir / "run_config.json",
        dict(
            p,
            station_csv=str(config.csv_path),
            node_data_dir=str(config.data_dir),
            node_output_dir=str(config.output_dir),
            start_utc=config.start_utc,
            end_utc=config.end_utc,
        ),
    )
    LOG.info("Node preflight: %s", json.dumps(summary, ensure_ascii=False))
    if args.preflight:
        return summary
    if args.verify_only:
        report = verify_task(config)
    else:
        by_start = {
            start + group.begin / catalog.sampling_rate: group for group in schedule
        }

        def check(path, fields):
            # This workflow can also support an explicit one-dimensional station CSV.
            if all(s.latitude_deg is not None for s in catalog.stations):
                report = validate_hour(
                    path, catalog, p, by_start[fields["source_start_unix"]], start, end
                )
                LOG.info(
                    "Output verified: %s, %d valid pairs",
                    path.name,
                    report["valid_pairs"],
                )
            if p.get("cc_enhance_method") == "tsi":
                import numpy as np

                np.testing.assert_array_equal(
                    fields["data"], fields["data"].transpose(2, 1, 0)[:, ::-1, :]
                )
                if not np.isfinite(fields["data_raw"]).all():
                    raise ValueError("Nonfinite original CCF in TSI output")
                LOG.info(
                    "TSI supported %d pairs in %d subgroups",
                    int(fields["tsi_applied_mask"].sum() // 2),
                    fields["tsi_group_count"],
                )

        run_node_ccf(config, group_validator=check)
        report = (
            verify_task(config)
            if all(s.latitude_deg is not None for s in catalog.stations)
            else {"passed": True, "outputs": len(schedule)}
        )
    atomic_json(config.output_dir / "validation.json", report)
    LOG.info("CC/TSI task complete: %s", config.output_dir)
    return summary


def cache_fixed_stations(configurations):
    cache = {}
    result = []
    for config, fixed in configurations:
        if fixed:
            key = config.csv_path.resolve()
            if key not in cache:
                content = key.read_bytes()
                records = tuple(_read_csv_rows(key, text=content.decode("utf-8-sig")))
                cache[key] = (records, hashlib.sha256(content).hexdigest())
                print(
                    f"Fixed station table loaded once: {key}, {len(records)} stations",
                    flush=True,
                )
            records, checksum = cache[key]
            config = replace(
                config, station_records=records, station_csv_sha256=checksum
            )
        result.append((config, fixed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/cc_config.jsonc")
    parser.add_argument("--date", help="Override date range and run this single day")
    parser.add_argument("--start-hour", type=int)
    parser.add_argument("--end-hour", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root for multiple days, exact directory for a single day",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    values = load_config_json(args.config)
    if time_mode(values) != "date_range":
        intervals = available_intervals(values)
        values = dict(
            values,
            _available_intervals=intervals,
            _available_bounds=(
                min(x[0] for x in intervals),
                max(x[1] for x in intervals),
            ),
        )
    dates = processing_dates(values, args.date)
    if not dates:
        raise ValueError("No waveform dates overlap the configured daily time interval")
    # Validate every date/time before performing any work.
    configurations = []
    for date in dates:
        daily_values = dict(values)
        if args.output_dir is not None and len(dates) > 1:
            daily_values["node_output_dir"] = str(args.output_dir)
        config, explicit_csv = config_from_values(
            daily_values,
            date=date,
            start_hour=args.start_hour,
            end_hour=args.end_hour,
            output_dir=args.output_dir if len(dates) == 1 else None,
        )
        configurations.append((config, explicit_csv))
    configurations = cache_fixed_stations(configurations)
    for index, (config, explicit_csv) in enumerate(configurations, 1):
        print(
            f"Day {index}/{len(dates)}: {dates[index - 1]} → {config.output_dir}",
            flush=True,
        )
        run_day(config, explicit_csv, args)


if __name__ == "__main__":
    main()
