"""Geographic TSI invariants, independent stacking and timed grouping."""

from dataclasses import replace

import numpy as np
import pytest
from node_compute.backend import NativeSession
from node_compute.parameters import validate
from node_compute.reference import ReferenceSession
from node_compute.tsi import geographic_tsi, triplet_geometry
from node_output.artifacts import load_result
from node_pipeline import run_node_ccf
from run_mohe_day import verify_task
from run_node_ccf_config import config_from_values
from scipy.signal import fftconvolve
from test_mohe import fixture


def cubes(n=5, lag=201):
    x = np.array([0, 100, 230, 400, 900.0])[:n]
    distance = abs(x[:, None] - x[None, :])
    cube = np.zeros((n, n, lag), np.float32)
    a, b = np.triu_indices(n, 1)
    cube[a, b] = np.random.default_rng(771).normal(size=(len(a), lag))
    cube[b, a] = cube[a, b, ::-1]
    observed = np.ones((n, n), bool)
    np.fill_diagonal(observed, False)
    return cube, distance, observed


def test_geometry_excludes_noncollinear_triplets_and_uses_distances():
    points = np.array([[0, 0], [100, 0], [230, 0], [50, 300.0]])
    d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    observed = np.ones((4, 4), bool)
    a, b, good, kind = triplet_geometry(d, observed, 1)
    pair = int(np.flatnonzero((a == 0) & (b == 2))[0])
    assert good[pair, 1] and kind[pair, 1] == 0
    assert not good[pair, 3] and not good[pair, 0] and not good[pair, 2]


@pytest.mark.parametrize("gate", [False, True])
def test_tsi_station_permutation_invariant(gate):
    cube, d, obs = cubes()
    p = {
        "tsi_backend": "cpu",
        "tsi_iterations": 2,
        "tsi_stop_rel_l1": 0,
        "tsi_taper_enabled": gate,
        "tsi_taper_vmin": 500,
        "tsi_taper_vmax": 4000,
    }
    expected, _, _, _ = geographic_tsi(cube, d, obs, 0.01, p)
    order = np.array([3, 0, 4, 1, 2])
    inverse = np.argsort(order)
    actual, _, _, _ = geographic_tsi(
        cube[order][:, order], d[order][:, order], obs[order][:, order], 0.01, p
    )
    np.testing.assert_allclose(
        actual[inverse][:, inverse], expected, rtol=1e-5, atol=2e-6
    )
    np.testing.assert_array_equal(actual, actual.transpose(1, 0, 2)[:, :, ::-1])


def test_tsi_gpu_matches_cpu_with_masked_missing_pairs():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cube, d, obs = cubes()
    obs[0, 3] = obs[3, 0] = False
    cube[0, 3] = cube[3, 0] = 0
    p = {"tsi_iterations": 2, "tsi_stop_rel_l1": 0, "tsi_taper_enabled": True}
    expected, support, applied, _ = geographic_tsi(
        cube, d, obs, 0.01, dict(p, tsi_backend="cpu")
    )
    actual, support2, applied2, diag = geographic_tsi(
        cube, d, obs, 0.01, dict(p, tsi_backend="torch", tsi_device="cuda")
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=2e-5)
    np.testing.assert_array_equal(support, support2)
    np.testing.assert_array_equal(applied, applied2)
    assert not actual[0, 3].any() and diag["iterations_completed"] == 2


def test_tsi_matches_linear_convolution_for_between_station():
    cube, d, obs = cubes(n=3)
    p = {"tsi_backend": "cpu", "tsi_iterations": 1, "tsi_taper_enabled": False}
    actual, _, _, _ = geographic_tsi(cube, d, obs, 0.01, p)
    one = 101
    pos = fftconvolve(cube[0, 1, 100:], cube[1, 2, 100:], mode="full")[:one]
    neg = fftconvolve(cube[0, 1, 100::-1], cube[1, 2, 100::-1], mode="full")[:one]
    pos /= abs(pos).max()
    neg /= abs(neg).max()
    expected = np.r_[neg[:0:-1], (neg[0] + pos[0]) / 2, pos[1:]]
    expected /= abs(expected).max()
    np.testing.assert_allclose(actual[0, 2], expected, atol=1e-6)


def test_tsi_no_geographic_support_retains_original():
    cube, _, obs = cubes(n=3)
    d = np.array([[0, 100, 100], [100, 0, 100], [100, 100, 0.0]])
    actual, support, applied, diag = geographic_tsi(
        cube, d, obs, 0.01, {"tsi_backend": "cpu", "tsi_iterations": 4}
    )
    np.testing.assert_array_equal(actual, cube)
    assert not support.any() and not applied.any() and diag["iterations_completed"] == 1


@pytest.mark.parametrize(
    "file_method,save_method", [("linear", "pws"), ("pws", "linear")]
)
def test_independent_minute_and_output_stack_cpu_cuda(file_method, save_method):
    p = validate(
        {
            "file_stacking_method": file_method,
            "stacking_method": save_method,
            "cc_len": 1,
            "maxlag": 0.3,
            "freqmax": 15,
        },
        100,
    )
    x = np.random.default_rng(11).normal(size=(600, 4)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    starts = [0, 100, 200, 300, 400, 500]
    minutes = [0, 0, 1, 1, 2, 2]
    native = NativeSession(p, 4)
    ref = ReferenceSession(dict(p, cc_backend="cpu", cc_device="cpu"), 4)
    native.add_block(x, mask, starts, minutes)
    ref.add_block(x, mask, starts, minutes)
    a, b = native.finish_group(), ref.finish_group()
    np.testing.assert_allclose(a["pairs"], b["pairs"], rtol=1e-3, atol=2e-4)
    np.testing.assert_array_equal(a["short_count"], b["short_count"])


@pytest.mark.parametrize("tsi_interval,expected_groups", [(1, 2), (2, 1)])
def test_tsi_timing_raw_preservation_and_resume(
    tmp_path, tsi_interval, expected_groups
):
    config = fixture(tmp_path)
    # Existing GPS fixture has repeated coordinates; actual missing support must retain CC.
    params = dict(
        config.cc_params,
        cc_enhance_method="tsi",
        tsi_files_per_group=tsi_interval,
        tsi_backend="cpu",
        tsi_device="cpu",
        tsi_iterations=1,
        tsi_show_progress=False,
    )
    enriched = replace(config, cc_params=params, output_dir=tmp_path / "tsi")
    run_node_ccf(config)
    run_node_ccf(enriched)
    assert verify_task(enriched)["passed"]
    raw = sorted(config.output_dir.glob("*.h5"))
    files = sorted(enriched.output_dir.glob("*.h5"))
    for i, (a, b) in enumerate(zip(raw, files, strict=True)):
        aa, bb = load_result(a), load_result(b)
        np.testing.assert_array_equal(aa["data"], bb["data_raw"])
        assert "tsi_diagnostics_json" in bb and "tsi_applied_mask" in bb
        assert int(bb["tsi_group_count"]) == (expected_groups if i < 3 else 1)
    times = [f.stat().st_mtime_ns for f in files]
    run_node_ccf(enriched)
    assert times == [f.stat().st_mtime_ns for f in files]


def test_config_aliases_and_30_60_minute_cadences(tmp_path):
    from pathlib import Path

    from dasqt.features.dispersion.backend.components.ccf.folder_pipeline import (
        load_config_json,
    )

    values = load_config_json(
        Path(__file__).resolve().parents[1] / "config/cc_config.jsonc"
    )
    values = dict(values, cc_enhance_method="tsi")
    for save, tsi in [(60, 60), (60, 30), (30, 30)]:
        c, _ = config_from_values(
            dict(values, save_every=save, tsi_files_per_group=tsi), root=tmp_path
        )
        p = validate(c.cc_params, 500)
        assert p["save_every"] == save and p["tsi_files_per_group"] == tsi
        assert p["cc_memory_budget_bytes"] == 2 * 1024**3 and p["cc_device"] == "cuda"
    with pytest.raises(ValueError, match="multiple"):
        validate(dict(c.cc_params, save_every=30, tsi_files_per_group=60), 500)
    with pytest.raises(ValueError, match="must agree"):
        config_from_values(dict(values, sps=50))
    assert str(c.output_dir).endswith("20260708_day_0800_1800_tsi")


def test_reused_upload_workspace_tail_uses_actual_input_length():
    # Force local uploads and a one-window tail after a full 12-window batch.
    p = validate(
        {
            "cc_len": 1,
            "cc_step_s": 0.5,
            "maxlag": 0.3,
            "freqmax": 15,
            "cc_batch_chunks": 12,
            "cc_upload_block": 0,
            "cc_reuse_workspace": 1,
        },
        100,
    )
    x = np.random.default_rng(751).normal(size=(700, 6)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    starts = np.arange(0, 601, 50)
    minutes = np.arange(len(starts)) // 12
    native = NativeSession(p, 6)
    reference = ReferenceSession(dict(p, cc_backend="cpu", cc_device="cpu"), 6)
    native.add_block(x, mask, starts, minutes)
    reference.add_block(x, mask, starts, minutes)
    a, b = native.finish_group(), reference.finish_group()
    np.testing.assert_allclose(a["pairs"], b["pairs"], rtol=1e-3, atol=2e-4)
    np.testing.assert_array_equal(a["short_count"], b["short_count"])
    assert native.diagnostics()["window_batch_last"] == 1
    assert native.diagnostics()["input_transferred_bytes"] == (650 + 100) * 6 * 5


def test_date_range_daily_times_and_legacy(tmp_path):
    from run_node_ccf_config import processing_dates

    values = {
        "node_data_dir": str(tmp_path),
        "processing_start_date": "2026-07-08",
        "processing_end_date": "2026-07-11",
        "start_time": "08:30",
        "end_time": "18:15",
    }
    assert processing_dates(values) == [f"2026-07-{i:02d}" for i in range(8, 12)]
    assert processing_dates(values, "2026-07-09") == ["2026-07-09"]
    for date in processing_dates(values):
        c, _ = config_from_values(values, root=tmp_path, date=date)
        assert c.start_utc == f"{date}T00:30:00+00:00"
        assert c.end_utc == f"{date}T10:15:00+00:00"
        assert c.output_dir.name == f"{date.replace('-', '')}_day_0830_1815_raw"
    legacy = {
        "node_data_dir": str(tmp_path),
        "processing_date": "2026-07-08",
        "day_start_hour": 8,
        "day_end_hour": 18,
    }
    assert processing_dates(legacy) == ["2026-07-08"]
    assert config_from_values(legacy, root=tmp_path)[0].start_utc.endswith(
        "T00:00:00+00:00"
    )
    with pytest.raises(ValueError):
        processing_dates(dict(values, processing_end_date="2026-07-07"))
    with pytest.raises(ValueError):
        config_from_values(dict(values, start_time="18:30"), root=tmp_path)
    with pytest.raises(TypeError):
        config_from_values(dict(values, start_time=8), root=tmp_path)


def test_all_date_modes_and_midnight_edges(tmp_path):
    from datetime import UTC, datetime

    from run_node_ccf_config import processing_dates

    lo = datetime(2026, 7, 8, tzinfo=UTC).timestamp()
    hi = lo + 2 * 86400
    values = {
        "node_data_dir": str(tmp_path),
        "processing_time_mode": "all_date",
        "start_time": "08:00",
        "end_time": "18:00",
        "_available_intervals": [(lo, hi)],
        "_available_bounds": (lo, hi),
    }
    assert processing_dates(values) == ["2026-07-08", "2026-07-09"]
    values["processing_time_mode"] = "all_date_time"
    assert processing_dates(values) == ["2026-07-08", "2026-07-09", "2026-07-10"]
    configs = [
        config_from_values(values, root=tmp_path, date=d)[0]
        for d in processing_dates(values)
    ]
    assert configs[0].start_utc == "2026-07-08T00:00:00+00:00"
    assert configs[-1].end_utc == "2026-07-10T00:00:00+00:00"
    assert configs[1].start_utc == configs[0].end_utc
    assert configs[2].start_utc == configs[1].end_utc
    assert (
        sum(
            (
                datetime.fromisoformat(c.end_utc) - datetime.fromisoformat(c.start_utc)
            ).total_seconds()
            for c in configs
        )
        == 2 * 86400
    )
    with pytest.raises(ValueError, match="processing_time_mode"):
        processing_dates(dict(values, processing_time_mode="invalid"))


def test_fixed_station_csv_is_reused_across_dates(tmp_path):
    values = {
        "node_data_dir": str(tmp_path),
        "station_csv": "config/fixed.csv",
        "processing_start_date": "2026-07-08",
        "processing_end_date": "2026-07-13",
    }
    first, explicit_first = config_from_values(values, root=tmp_path, date="2026-07-08")
    last, explicit_last = config_from_values(values, root=tmp_path, date="2026-07-13")
    assert explicit_first and explicit_last
    assert first.csv_path == last.csv_path == tmp_path / "config/fixed.csv"
    assert first.output_dir != last.output_dir


def test_station_coordinate_modes(tmp_path):
    values = {"node_data_dir": str(tmp_path), "station_csv": "config/fixed.csv"}
    fixed, use_fixed = config_from_values(values, root=tmp_path)
    assert use_fixed and fixed.csv_path == tmp_path / "config/fixed.csv"
    default, use_default = config_from_values(
        dict(values, station_csv=None), root=tmp_path
    )
    assert use_default and default.csv_path == tmp_path / "config/stations_fixed.csv"
    daily, use_daily = config_from_values(
        dict(values, station_coordinate_mode="daily_gps"), root=tmp_path
    )
    assert not use_daily and daily.csv_path == daily.output_dir / "stations_gps.csv"
    assert daily.output_dir.name == fixed.output_dir.name + "_daily_gps"
    with pytest.raises(ValueError, match="station_coordinate_mode"):
        config_from_values(
            dict(values, station_coordinate_mode="invalid"), root=tmp_path
        )
