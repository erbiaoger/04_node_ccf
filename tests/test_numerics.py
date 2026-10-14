"""Independent direct correlation, per-pair Hilbert and two-level weights."""

import numpy as np
import pytest
from node_compute.parameters import validate
from node_compute.reference import Ops, PairAccumulator, ReferenceSession, linear_pairs
from node_output.artifacts import dense_pairs
from scipy import fft, signal
from scipy.fftpack import next_fast_len


def settings(**overrides):
    return validate(
        dict(
            cc_backend="cpu",
            cc_device="cpu",
            cc_len=1.0,
            cc_step_s=0.5,
            maxlag=0.4,
            freqmin=2.0,
            freqmax=20.0,
            **overrides,
        ),
        100.0,
    )


def test_padded_fft_matches_direct_linear_and_delay_sign():
    # End-to-start pulses would correlate at a wrapped small lag in the old path.
    traces = np.zeros((1, 3, 100), dtype=np.float32)
    traces[0, 0, 90], traces[0, 1, 95], traces[0, 2, 2] = 1, 2, 3
    a, b = np.triu_indices(3, 1)
    nfft = next_fast_len(199)
    actual = linear_pairs(fft.rfft(traces, n=nfft), a, b, 40, Ops("cpu", "cpu"), nfft)[
        0
    ]
    lags = signal.correlation_lags(100, 100)
    for pair, (i, j) in enumerate(zip(a, b, strict=True)):
        expected = signal.correlate(traces[0, j], traces[0, i], method="direct")
        np.testing.assert_allclose(actual[pair], expected[abs(lags) <= 40], atol=1e-6)
    assert np.argmax(actual[0]) - 40 == 5
    assert abs(actual[1]).max() < 1e-6
    dense = dense_pairs(actual, 3)
    np.testing.assert_array_equal(dense[1, :, 0], dense[0, ::-1, 1])
    assert not dense[np.arange(3), :, np.arange(3)].any()


def independent_stack(data, valid):
    output = np.zeros(data.shape[1:], dtype=np.float64)
    for pair in range(data.shape[1]):
        values = data[valid[:, pair], pair]
        if not len(values):
            continue
        analytic = signal.hilbert(values, N=next_fast_len(values.shape[-1]), axis=-1)[
            ..., : values.shape[-1]
        ]
        phase = np.divide(
            analytic,
            abs(analytic),
            out=np.zeros_like(analytic),
            where=abs(analytic) > 0,
        )
        result = values.mean(0) * abs(phase.mean(0)) ** 2
        scale = abs(result).max() or 1
        result /= scale
        output[pair] = result - result.mean()
    return output


def test_independent_pws_masks_zero_phase_and_other_pairs():
    rng = np.random.default_rng(14)
    data = rng.normal(size=(5, 3, 81)).astype(np.float32)
    data[0, 0] = 0
    valid = np.array(
        [[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool
    )
    acc = PairAccumulator(3, 81, Ops("cpu", "cpu"), "pws")
    for d, v in zip(data, valid, strict=True):
        acc.add(d, v)
    result = acc.finish()
    np.testing.assert_allclose(result, independent_stack(data, valid), atol=2e-6)
    np.testing.assert_array_equal(acc.count, [3, 3, 0])
    altered = data.copy()
    altered[:, 1] *= np.arange(1, 6)[:, None] * 100
    other = PairAccumulator(3, 81, Ops("cpu", "cpu"), "pws")
    other.add(altered, valid)
    np.testing.assert_array_equal(result[0], other.finish()[0])


def test_bad_station_affects_only_its_pairs():
    p = settings()
    s = ReferenceSession(p, 4)
    x = np.random.default_rng(1).normal(size=(250, 4)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    mask[30, 0] = False
    x[:, 3] = 0
    s.add_block(x, mask, [0, 50, 100], [0, 0, 1])
    r = s.finish_group()
    # upper pairs: 01,02,03,12,13,23
    np.testing.assert_array_equal(r["short_count"], [2, 2, 0, 3, 0, 0])
    np.testing.assert_array_equal(r["valid_count"], [2, 2, 0, 2, 0, 0])
    assert not r["pairs"][[2, 4, 5]].any()


def test_each_valid_minute_has_equal_weight():
    p = settings()
    x = np.random.default_rng(2).normal(size=(300, 2)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    full = ReferenceSession(p, 2)
    full.add_block(x, mask, [0, 50, 100], [0, 0, 1])
    parts = []
    for starts in ([0, 50], [100]):
        s = ReferenceSession(p, 2)
        s.add_block(x, mask, starts, [0] * len(starts))
        parts.append(s.minute.finish().copy())
    acc = PairAccumulator(1, 81, Ops("cpu", "cpu"), "pws")
    for part in parts:
        acc.add(part, np.ones(1, dtype=bool))
    np.testing.assert_allclose(full.finish_group()["pairs"], acc.finish(), atol=2e-6)


def test_torch_cpu_is_independent_reference_parity():
    pytest.importorskip("torch")
    p = settings()
    x = np.random.default_rng(14).normal(size=(250, 4)).astype(np.float32)
    results = []
    for backend in ("cpu", "torch"):
        s = ReferenceSession(dict(p, cc_backend=backend), 4)
        s.add_block(x, np.ones_like(x, dtype=bool), [0, 50, 100], [0, 0, 1])
        results.append(s.finish_group())
    np.testing.assert_allclose(
        results[0]["pairs"], results[1]["pairs"], rtol=1e-3, atol=2e-4
    )
