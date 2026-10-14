"""Nested GPS node layouts, exact SAC clocks and hourly daytime ownership."""

import json
from dataclasses import replace

import numpy as np
import pytest
from cli import parse_config
from node_compute.parameters import validate
from node_compute.reference import fir_taps
from node_compute.schedule import block_bounds, groups
from node_io import NodeSACReader, load_node_catalog
from node_io.gps import discover_station_gps, station_gps, write_station_table
from node_output.artifacts import load_result, save_result
from node_pipeline import NodeCCFConfig, prepare, run_node_ccf
from obspy import UTCDateTime
from obspy.io.sac.sactrace import SACTrace
from run_mohe_day import daytime_bounds, validate_hour, verify_task
from scipy.signal import fftconvolve


def gps_log(date="2026/07/08"):
    return (
        f'[GPS1]\nLatest Fix UTC Time = "{date},00:01:00"\n'
        "Longitude = 122.2\nLatitude = 53.0\nAltitude = 500\n"
        f'[GPS2]\nLatest Fix UTC Time = "{date},00:02:00"\n'
        "Longitude = 122.20002\nLatitude = 53.00002\nAltitude = 502\n"
        '[GPS3]\nLatest Fix UTC Time = "2026/07/09,00:01:00"\n'
        "Longitude = 100\nLatitude = 40\n"
    )


def fixture(tmp_path):
    rng = np.random.default_rng(140)
    for station in ("10", "2", "1"):
        directory = tmp_path / station
        directory.mkdir()
        (directory / "DigiSolo.LOG").write_text(gps_log())
        sac = SACTrace(data=rng.normal(size=6500).astype(np.float32), delta=0.002)
        sac.reftime = UTCDateTime("2026-07-08T00:00:00Z")
        sac.b = 0
        sac.write(str(directory / f"{station}.a.sac"))
    rows = discover_station_gps(tmp_path, "2026/07/08")
    csv = tmp_path / "gps.csv"
    write_station_table(csv, rows)
    p = {
        "cc_backend": "cpu",
        "cc_device": "cpu",
        "read_mode": "block",
        "read_hours": 24,
        "async_save": 0,
        "sac_sampling_rate_hz": 500,
        "time_downsample": 2,
        "cc_len": 1,
        "cc_step_s": 0.5,
        "minute_stack_s": 2,
        "save_every": 2,
        "maxlag": 0.3,
    }
    return NodeCCFConfig(
        csv,
        tmp_path,
        tmp_path / "out",
        p,
        "2026-07-08T00:00:00Z",
        "2026-07-08T00:00:13Z",
    )


def test_gps_day_median_and_numeric_order(tmp_path):
    config = fixture(tmp_path)
    catalog, p, start, end, total = prepare(config)
    assert catalog.station_ids == ("1", "2", "10")
    assert catalog.sampling_rate == 500 and total == 6500 and end - start == 13
    assert p["time_downsample"] == 2
    for station in catalog.stations:
        assert station.distance_m is None
        assert station.gps_info["gps_fix_count"] == "2"
        assert station.latitude_deg == pytest.approx(53.00001)
        assert station.longitude_deg == pytest.approx(122.20001)
        assert station.segments[0].header_delta == float(np.float32(0.002))
    fingerprint = catalog.fingerprint()
    assert fingerprint[0]["distance_m"] is None
    assert fingerprint[0]["segments"][0]["header_delta"] != 1 / 500


def test_gps_missing_or_invalid_fixes_rejected(tmp_path):
    (tmp_path / "DigiSolo.LOG").write_text(gps_log("2026/07/07"))
    with pytest.raises(ValueError, match="No valid GPS fixes"):
        station_gps(tmp_path, "2026/07/08")
    (tmp_path / "DigiSolo.LOG").write_text(
        gps_log().replace("Latitude = 53", "Latitude = 153")
    )
    with pytest.raises(ValueError, match="No valid GPS fixes"):
        station_gps(tmp_path, "2026/07/08")


def test_nominal_rate_rejects_real_mismatch(tmp_path):
    config = fixture(tmp_path)
    with pytest.raises(ValueError, match="disagrees"):
        load_node_catalog(config.csv_path, config.data_dir, sampling_rate_hz=250)
    with pytest.raises(ValueError, match="positive"):
        load_node_catalog(config.csv_path, config.data_dir, sampling_rate_hz=0)


def test_date_filter_ignores_other_day_and_reads_selected_samples(tmp_path):
    config = fixture(tmp_path)
    for station in ("1", "2", "10"):
        directory = tmp_path / station
        (directory / f"{station}.a.sac").rename(
            directory
            / f"{station}.2026.07.08.00.00.00.000-2026.07.09.00.00.00.000.z.sac"
        )
        (
            directory
            / f"{station}.2026.07.09.00.00.00.000-2026.07.10.00.00.00.000.z.sac"
        ).write_bytes(b"not selected; intentionally invalid header")
    catalog, _, start, _, _ = prepare(config)
    assert all(len(station.segments) == 1 for station in catalog.stations)
    reader = NodeSACReader(catalog)
    try:
        data, valid = reader.read_samples(start + 2, 1000)
    finally:
        reader.close()
    assert data.shape == (1000, 3) and valid.all()
    original = SACTrace.read(str(catalog.stations[0].segments[0].path)).data
    np.testing.assert_array_equal(data[:, 0], original[1000:2000])


def test_nominal_500hz_adjacent_days_no_float_drift(tmp_path):
    config = fixture(tmp_path)
    # Sparse files simulate full days without allocating their waveforms.
    for station in ("1", "2", "10"):
        for day in (8, 9):
            path = tmp_path / station / f"{station}.{day}.sac"
            sac = SACTrace(data=np.array([1], dtype=np.float32), delta=0.002)
            sac.reftime = UTCDateTime(f"2026-07-{day:02d}T00:00:00Z")
            sac.b = 0
            sac.write(str(path))
            with path.open("r+b") as handle:
                handle.seek(280 + 9 * 4)
                handle.write(np.array([43200000], dtype="<i4").tobytes())
                handle.truncate(632 + 43200000 * 4)
        (tmp_path / station / f"{station}.a.sac").unlink()
    catalog = load_node_catalog(config.csv_path, config.data_dir, sampling_rate_hz=500)
    assert catalog.end_timestamp - catalog.start_timestamp == 48 * 3600
    assert all(
        s.segments[0].end_timestamp == s.segments[1].start_timestamp
        for s in catalog.stations
    )


def test_daytime_hour_schedule_and_no_night_samples():
    start, end = daytime_bounds("2026-07-08")
    assert start.isoformat() == "2026-07-08T00:00:00+00:00"
    assert end.isoformat() == "2026-07-08T10:00:00+00:00"
    p = validate(
        {"sac_sampling_rate_hz": 500, "save_every": 60, "time_downsample": 2}, 500
    )
    scheduled = list(groups(18000000, 500, p))
    assert len(scheduled) == 10
    assert [len(g.starts) for g in scheduled] == [720] * 9 + [719]
    starts = np.concatenate([g.starts for g in scheduled])
    np.testing.assert_array_equal(starts, np.arange(0, 18000000 - 5000 + 1, 2500))
    for group in scheduled:
        lo, hi = block_bounds(group, 18000000, 500, p)
        assert 0 <= lo < hi <= 18000000
    assert block_bounds(scheduled[0], 18000000, 500, p)[1] > 1800000


def test_factor_two_fir_suppresses_high_frequency_alias_into_passband():
    # At 250 Hz, an unfiltered 225 Hz input aliases directly to the target 25 Hz.
    t = np.arange(10000) / 500
    taps = fir_taps(2)

    def rms_after(frequency):
        signal = np.sin(2 * np.pi * frequency * t)
        decimated = fftconvolve(signal, taps, mode="same")[200:-200:2]
        return np.sqrt(np.mean(decimated**2))

    assert rms_after(25) > 0.70
    assert rms_after(225) < rms_after(25) * 1e-4


def test_geographic_h5_validation_resume_and_corruption(tmp_path):
    config = fixture(tmp_path)
    catalog, p, start, end, total = prepare(config)
    schedule = list(groups(total, 500, p))
    validated = []

    def check(path, fields):
        group = schedule[len(validated)]
        validated.append(validate_hour(path, catalog, p, group, start, end))

    run_node_ccf(config, group_validator=check)
    assert len(validated) == 4
    report = verify_task(config)
    assert report["passed"] and report["diagnostics"]["host_reads"] == 1
    files = sorted(config.output_dir.glob("*.h5"))
    data = load_result(files[0])
    assert not {"profileX", "dx", "dshot"} & data.keys()
    assert data["geometry_type"] == "geographic"
    assert data["data"].shape == (3, 151, 3)
    mtimes = [f.stat().st_mtime_ns for f in files]
    run_node_ccf(config)
    assert mtimes == [f.stat().st_mtime_ns for f in files]
    assert verify_task(config)["diagnostics"]["groups_resumed"] == 4
    broken = dict(data, source_end_unix=float(data["source_end_unix"]) + 1)
    save_result(files[0], broken)
    with pytest.raises(ValueError, match="time range"):
        validate_hour(files[0], catalog, p, schedule[0], start, end)
    run_node_ccf(config)
    assert verify_task(config)["passed"]
    assert mtimes[1:] == [f.stat().st_mtime_ns for f in files[1:]]
    from stack_node_cc_mat import stack_window

    stack = stack_window(files, cc_backend="cpu", cc_device="cpu")
    np.testing.assert_array_equal(stack["pair_distance_m"], data["pair_distance_m"])
    assert "profileX" not in stack


def test_validation_gate_stops_before_second_group(tmp_path):
    config = fixture(tmp_path)

    def fail(*args):
        raise ValueError("first-hour acceptance failed")

    with pytest.raises(ValueError, match="acceptance failed"):
        run_node_ccf(config, group_validator=fail)
    assert len(list(config.output_dir.glob("*.h5"))) == 1
    assert not json.loads((config.output_dir / "task_manifest.json").read_text())[
        "completed"
    ]


def test_cli_reuses_task_config_and_explicit_override(tmp_path):
    config = fixture(tmp_path)
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            dict(
                config.cc_params,
                station_csv=str(config.csv_path),
                node_data_dir=str(config.data_dir),
                node_output_dir=str(config.output_dir),
                start_utc=config.start_utc,
                end_utc=config.end_utc,
            )
        )
    )
    parsed, _ = parse_config(["--config", str(path)])
    assert parsed.start_utc == config.start_utc and parsed.end_utc == config.end_utc
    assert parsed.cc_params["sac_sampling_rate_hz"] == 500
    parsed, _ = parse_config(
        ["--config", str(path), "--start-utc", "2026-07-08T00:00:02Z"]
    )
    assert parsed.start_utc.endswith("02Z")
    with pytest.raises(ValueError, match="outside"):
        prepare(replace(config, start_utc="2026-07-07T23:00:00Z"))
