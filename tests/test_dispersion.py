"""F-J zero-lag alignment, radial quadrature and synthetic phase velocity."""

import numpy as np
import pytest
from dasqt.features.dispersion.backend.algorithms.calculate_disp.mmfj.core import (
    mmfj_from_egf,
)
from node_compute.dispersion import distance_gathers, frequency_bessel
from scipy.special import j0


def test_fj_matches_existing_zero_order_reference():
    offsets = np.array([90, 220, 390, 800, 1300.0])
    traces = np.random.default_rng(4).normal(size=(5, 401))
    f = np.array([3.0, 6.0, 12.0])
    c = np.array([600.0, 1500.0, 3000.0])
    actual = frequency_bessel(
        traces, offsets, 0.004, f, c, zero_lag_index=200, backend="cpu"
    )
    expected, _, _ = mmfj_from_egf(
        traces.T,
        offsets,
        0.004,
        f,
        c,
        kernel="j0",
        integration="trapezoid",
        part="real",
        polarity="signed",
        normalize=False,
        zero_lag_index=200,
    )
    np.testing.assert_allclose(actual["raw_integral"], expected, rtol=1e-12, atol=1e-7)


def test_fj_recovers_known_synthetic_velocity():
    n = 2001
    dt = 0.004
    offsets = np.linspace(20, 3000, 240)
    target = 2200.0
    f0 = np.fft.rfftfreq(n, dt)[80]
    lags = (np.arange(n) - 1000) * dt
    traces = (
        j0(2 * np.pi * f0 * offsets / target)[:, None]
        * np.cos(2 * np.pi * f0 * lags)[None, :]
    )
    c = np.arange(1500, 3001, 10.0)
    result = frequency_bessel(
        traces, offsets, dt, np.array([f0]), c, zero_lag_index=1000, backend="cpu"
    )
    peak = c[np.argmax(result["energy"][:, 0])]
    assert abs(peak - target) <= 30
    assert result["energy"].max() == pytest.approx(1)


def test_fj_cuda_matches_cpu_and_duplicate_offsets():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    offsets = np.array([100.0, 200.0, 200.0, 700.0, 1800.0])
    traces = np.random.default_rng(41).normal(size=(5, 501))
    args = (
        traces,
        offsets,
        0.004,
        np.array([2.0, 10.0, 30.0]),
        np.array([500.0, 2200.0, 4000.0]),
    )
    a = frequency_bessel(*args, zero_lag_index=250, backend="cpu")
    b = frequency_bessel(*args, zero_lag_index=250, backend="torch_cuda")
    scale = np.maximum(a["column_scale"], 1e-12)
    np.testing.assert_allclose(
        b["raw_integral"] / scale, a["raw_integral"] / scale, rtol=0, atol=2e-5
    )
    np.testing.assert_allclose(b["energy"], a["energy"], rtol=0, atol=2e-5)
    assert len(b["distance_m"]) == 4


def test_distance_bins_keep_pair_counts_and_coherent_cancellation():
    a, b = np.triu_indices(3, 1)
    cube = np.zeros((3, 5, 3), dtype=np.float32)
    cube[a[0], :, b[0]] = [1, 2, 3, 2, 1]
    cube[a[1], :, b[1]] = [-2, -4, -6, -4, -2]
    cube[a[2], :, b[2]] = [0, 0, 1, 0, 0]
    mat = {
        "data": cube,
        "pair_source_index": a,
        "pair_receiver_index": b,
        "pair_distance_m": np.array([112.0, 128.0, 250.0]),
        "valid_mask": np.ones((3, 3), bool),
    }
    result = distance_gathers(mat, 50)
    assert result["bin_counts"].sum() == 3
    assert result["bin_counts"][2] == 2
    np.testing.assert_array_equal(result["binned_ccf"][0], 0)
    assert result["binned_distance_m"].tolist() == [125.0, 275.0]
    mat["valid_mask"][a[1], b[1]] = False
    result = distance_gathers(mat, 50)
    assert result["bin_counts"].sum() == 2
    np.testing.assert_allclose(result["binned_ccf"][0], [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3])


@pytest.mark.parametrize("branch", ["positive", "negative", "symmetric"])
def test_park_recovers_velocity_and_cuda_matches_cpu(branch):
    from node_compute.dispersion import park_phase_shift

    dt = 0.004
    count = 1001
    t = np.arange(count) * dt
    frequency = 20 / (count * dt)
    offsets = np.linspace(80, 1200, 31)
    speed = 2000.0
    positive = np.cos(2 * np.pi * frequency * (t[None, :] - offsets[:, None] / speed))
    full = np.concatenate((positive[:, 1:][:, ::-1], positive), axis=1)
    kwargs = {"zero_lag_index": count - 1, "branch": branch}
    c = np.arange(1000, 3001, 10.0)
    cpu = park_phase_shift(full, offsets, dt, [frequency], c, backend="cpu", **kwargs)
    assert c[np.argmax(cpu["energy"][:, 0])] == speed
    assert cpu["coherence"].max() > 0.999
    scaled = park_phase_shift(
        full * np.arange(1, 32)[:, None],
        offsets,
        dt,
        [frequency],
        c,
        backend="cpu",
        **kwargs,
    )
    np.testing.assert_allclose(cpu["energy"], scaled["energy"], atol=1e-12)
    import torch

    if torch.cuda.is_available():
        gpu = park_phase_shift(
            full, offsets, dt, [frequency], c, backend="torch_cuda", **kwargs
        )
        np.testing.assert_allclose(cpu["raw_integral"], gpu["raw_integral"], atol=1e-10)


def test_park_zero_signals_do_not_create_energy():
    from node_compute.dispersion import park_phase_shift

    r = park_phase_shift(
        np.zeros((3, 201)),
        [50, 100, 150],
        0.01,
        [2, 3],
        [500, 1000],
        zero_lag_index=100,
        backend="cpu",
    )
    assert not r["energy"].any() and not r["coherence"].any()
