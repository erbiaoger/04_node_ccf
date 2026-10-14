"""Node-to-SeisLib numerics, selection, quality and resumable edit ownership."""

from __future__ import annotations

import json
import threading

import h5py
import numpy as np
import pytest
from dasqt.features.node_two_station.engine import (
    Task,
    extract_curve,
    pair_spectrum,
    process_pair,
    quality,
    representative_pairs,
    validate_options,
)
from scipy.special import j0


def fixture_input(tmp_path):
    dt = 0.002
    lags = np.arange(-2000, 2001) * dt
    distances = np.array([500.0, 1200.0, 2300.0])
    n = 3
    f = np.fft.rfftfreq(len(lags), dt)
    spectrum = j0(2 * np.pi * f[None, :] * distances[:, None] / 2000.0) * np.exp(
        -((f[None, :] / 15) ** 4)
    )
    traces = np.fft.fftshift(np.fft.irfft(spectrum, n=len(lags), axis=1), axes=1)
    cube = np.zeros((n, len(lags), n), np.float32)
    a, b = np.triu_indices(n, 1)
    cube[a, :, b] = traces
    cube[b, :, a] = traces[:, ::-1]
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as h:
        for k, v in {
            "data": cube,
            "dt": dt,
            "lags_s": lags,
            "pair_distance_m": distances,
            "pair_source_index": a,
            "pair_receiver_index": b,
            "station_latitude_deg": [53.0, 53.005, 53.02],
            "station_longitude_deg": [122.0, 122.005, 122.02],
            "valid_mask": ~np.eye(3, dtype=bool),
        }.items():
            h[k] = v
        h.create_dataset("station_id", data=["1", "2", "3"], dtype=h5py.string_dtype())
        h["parameters_json"] = json.dumps({"freqmin": 1.0, "freqmax": 10.0})
    return path


def make_task(tmp_path, **kw):
    return Task(
        dict(
            input_file=str(fixture_input(tmp_path)),
            output_dir=str(tmp_path / "results"),
            overall_backend="cpu",
            workers=1,
            velocity_window=False,
            **kw,
        )
    )


def test_spectrum_preserves_odd_length_physical_phase():
    p = validate_options({"velocity_window": False})
    lags = np.arange(-200, 201) * 0.01
    wave = np.exp(-(((lags - 0.7) / 0.04) ** 2))
    freq, spec, window = pair_spectrum(wave, lags, 1000, p)
    for i in [1, 7, 13]:
        oracle = np.sum(wave * np.exp(-2j * np.pi * freq[i] * lags))
        np.testing.assert_allclose(spec[i], oracle, atol=1e-11)
    assert len(freq) == 202 and window.all()
    neg = pair_spectrum(wave[::-1], lags, 1000, p)[1]
    np.testing.assert_allclose(neg, spec.conj(), atol=1e-10)


def test_quality_marks_short_distance_time_and_interpolation():
    p = validate_options({})
    curve = np.array([[1.0, 2000], [8, 500], [8, 2000], [10, 5000]])
    q = quality(curve, 2500, [-4, 4], p, [1, 1, 2, 3])
    assert q["quality_flags"][0] & 2
    assert q["quality_flags"][1] & 4
    assert q["quality_flags"][2] & 8
    assert q["quality_flags"][3] & 1
    assert not q["reviewable"]


def test_seislib_recovers_synthetic_j0_branch(tmp_path):
    task = make_task(tmp_path)
    pair = task.meta["pairs"][2]
    with h5py.File(task.path, "r") as h:
        trace = h["data"][pair["source"], :, pair["receiver"]]
    f, spec, _ = pair_spectrum(trace, task.meta["lags"], pair["distance_m"], task.p)
    ref = np.array([[1, 2000], [10, 2000.0]])
    curve = extract_curve(f, spec, pair["distance_m"], ref, task.p)
    assert len(curve) >= 5
    np.testing.assert_allclose(curve[:, 1], 2000, atol=100)
    assert curve[:, 0].min() >= 1 and curve[:, 0].max() <= 10


def test_process_zero_signal_and_tiny_window_fail_clearly(tmp_path):
    task = make_task(tmp_path)
    ref = np.array([[1, 2000.0], [10, 2000.0]])
    with h5py.File(task.path, "a") as h:
        h["data"][0, :, 1] = 0
    result = process_pair((str(task.path), task.meta["pairs"][0], task.p, ref, None))
    assert result["status"] == "failed" and "zero CCF" in result["reason"]
    with pytest.raises(ValueError, match="too few"):
        pair_spectrum(
            np.ones(401), np.arange(-200, 201) * 0.01, 0.001, validate_options({})
        )


def test_reference_edit_resume_and_manual_protection(tmp_path):
    task = make_task(tmp_path)
    task.set_reference([[1, 2000], [10, 2000]], True)
    task.batch([2])
    before = task.pair(2)
    assert before["status"] in ("reviewable", "needs_review")
    assert before["candidate_density"].shape == (91, 351)
    manual = np.array([[2, 2050], [3, 2060], [4, 2070], [5, 2080], [6, 2090]])
    task.edit_curve(2, manual)
    task.batch([2])
    np.testing.assert_array_equal(task.pair(2)["curve"], manual)
    task.set_reference([[1, 1900], [10, 1900]], True)
    # Changed reference re-picks while preserving manual points and candidates.
    old_density = task.pair(2)["candidate_density"].copy()
    task.batch([2])
    np.testing.assert_array_equal(task.pair(2)["candidate_density"], old_density)
    np.testing.assert_array_equal(task.pair(2)["curve"], manual)
    assert task.pair(2)["reference_hash"] == task.reference_hash
    task.undo_edit(2)
    assert not task.pair(2)["manual_edited"]
    task.edit_curve(2, manual)
    task.reset_edit(2)
    assert not task.pair(2)["manual_edited"]
    reopened = Task(task.p)
    assert reopened.confirmed and reopened.reference_hash == task.reference_hash
    assert (
        json.loads((task.directory / "task_manifest.json").read_text())["total_pairs"]
        == 3
    )


def test_cancel_and_reference_gate(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(ValueError, match="Confirm"):
        task.batch([0])
    task.set_reference([[1, 2000], [10, 2000]], True)
    event = threading.Event()
    event.set()
    task.batch([0, 1, 2], event)
    assert all(task.pair(i)["status"] == "pending" for i in range(3))
    assert representative_pairs(task.meta["pairs"], 2) == [0, 2]
    with pytest.raises(ValueError, match="band"):
        Task(dict(task.p, fmax=11))


def test_atomic_staged_record_recovery(tmp_path):
    task = make_task(tmp_path)
    with h5py.File(task.file, "a") as h:
        h.copy("pairs/0000", "_pending")
        h["_pending"].attrs.update(
            complete=True, status="failed", reason="persisted before interruption"
        )
        del h["pairs/0000"]
    reopened = Task(task.p)
    assert reopened.pair(0)["reason"] == "persisted before interruption"


def test_reference_versions_and_manual_source_provenance(tmp_path):
    task = make_task(tmp_path)
    task.set_reference([[1, 2000], [10, 2000]], True)
    old_hash = task.reference_hash
    manual = np.array([[2, 2000], [4, 2100]])
    task.edit_curve(2, manual, [1, 3])
    task.set_reference([[1, 1800], [10, 1800]], True)
    result = process_pair(
        (str(task.path), task.meta["pairs"][2], task.p, task.reference, None)
    )
    task.store_pair(2, result)
    r = task.pair(2)
    np.testing.assert_array_equal(r["curve_source"], [1, 3])
    assert r["curve_reference_hash"] == old_hash
    assert r["auto_reference_hash"] == task.reference_hash
    with h5py.File(task.file, "r") as h:
        assert old_hash in h["references"] and task.reference_hash in h["references"]


def test_candidate_only_stage_never_confirms_reference(tmp_path):
    task = make_task(tmp_path)
    task.batch([2], candidates_only=True)
    assert not task.confirmed and task.reference is None
    r = task.pair(2)
    assert r["status"] == "awaiting_reference" and "candidate_density" in r
    assert not len(r["curve"])
