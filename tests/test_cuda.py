"""Real sm_89 node CUDA parity; REQUIRE_NODE_CUDA=1 prohibits skips."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
from node_compute.backend import NativeSession
from node_compute.parameters import VERSION, validate
from node_compute.reference import ReferenceSession


def native_available():
    try:
        m = importlib.import_module("_dasqt_cc_cuda")
        return (
            getattr(m, "node_pairwise_version", None) == VERSION
            and m.runtime_available()
        )
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def require_cuda():
    if not native_available():
        if os.environ.get("REQUIRE_NODE_CUDA") == "1":
            pytest.fail("New node CUDA extension or sm_89 device unavailable")
        pytest.skip("Node CUDA requires rebuilt extension and NVIDIA sm_89")


def params(**overrides):
    return validate(
        dict(
            cc_backend="cpp_cuda",
            cc_device="cuda",
            cc_len=1.0,
            cc_step_s=0.5,
            maxlag=0.3,
            freqmin=2.0,
            freqmax=15.0,
            **overrides,
        ),
        100.0,
    )


@pytest.mark.parametrize("norm", ["no", "phase_only", "rma"])
@pytest.mark.parametrize("method", ["linear", "pws"])
@pytest.mark.parametrize("factor", [1, 2])
def test_native_matches_reference_across_tiles(norm, method, factor):
    p = params(freq_norm=norm, stacking_method=method, time_downsample=factor)
    x = np.random.default_rng(14).normal(size=(700, 6)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    mask[155, 1] = False
    x[350, 2] = np.nan
    x[:, 5] = 0
    starts = [40, 140, 240, 340, 440, 540]
    minutes = [0, 0, 1, 1, 2, 2]
    ref = ReferenceSession(dict(p, cc_backend="cpu", cc_device="cpu"), 6)
    ref.add_block(x, mask, starts, minutes)
    expected = ref.finish_group()
    for windows, pairs, budget in [(32, 0, 0), (1, 1, 0), (2, 3, 3 * 1024**2)]:
        actual = NativeSession(
            dict(
                p,
                cc_batch_chunks=windows,
                cc_pair_batch=pairs,
                cc_memory_budget_bytes=budget,
            ),
            6,
        )
        actual.add_block(x, mask, starts, minutes)
        result = actual.finish_group()
        np.testing.assert_array_equal(result["short_count"], expected["short_count"])
        np.testing.assert_array_equal(result["valid_count"], expected["valid_count"])
        np.testing.assert_allclose(
            result["pairs"], expected["pairs"], rtol=1e-3, atol=2e-4
        )
        assert actual.diagnostics()["backend"] == "cpp_cuda"
        assert actual.diagnostics()["input_transfers"] > 0


def test_restack_masks_and_pair_independence():
    p = params()
    native = NativeSession(p, 4)
    reference = ReferenceSession(dict(p, cc_backend="cpu", cc_device="cpu"), 4)
    x = np.random.default_rng(14).normal(size=(5, 6, 61)).astype(np.float32)
    valid = np.ones((5, 6), dtype=bool)
    valid[:, 2] = False
    valid[2, 0] = False
    x[0, 1] = 0
    for a, b in zip(x, valid, strict=True):
        native.stack_pairs(a, b)
        reference.stack_pairs(a, b)
    actual, expected = native.finish_group(), reference.finish_group()
    np.testing.assert_allclose(actual["pairs"], expected["pairs"], rtol=1e-3, atol=2e-4)
    np.testing.assert_array_equal(actual["valid_count"], expected["valid_count"])
    native.clear_group()
    native.stack_pairs(x[0], np.zeros(6, dtype=bool))
    assert not native.finish_group()["pairs"].any()


def test_native_rejects_mismatched_shapes_and_cancel():
    session = NativeSession(params(), 4)
    with pytest.raises(ValueError):
        session.add_block(
            np.zeros((200, 3), dtype=np.float32),
            np.ones((200, 3), dtype=bool),
            [0],
            [0],
        )
    with pytest.raises(ValueError):
        session.stack_pairs(np.ones((6, 60), dtype=np.float32), np.ones(6, dtype=bool))
    session.cancel()
    with pytest.raises(RuntimeError, match="cancel"):
        session.finish_group()


@pytest.mark.parametrize("samples", [1400, 100000])
def test_complete_block_upload_parity_and_budget_fallback(samples):
    x = np.random.default_rng(2026).normal(size=(samples, 4)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    mask[155, 1] = False
    starts = np.arange(0, 1301, 50)
    minutes = np.arange(len(starts)) // 4
    outputs = []
    transfers = []
    for upload in (0, 1):
        session = NativeSession(
            params(
                cc_upload_block=upload,
                cc_batch_chunks=4,
                cc_memory_budget_bytes=3 * 1024**2,
            ),
            4,
        )
        session.add_block(x, mask, starts, minutes)
        outputs.append(session.finish_group())
        transfers.append(session.diagnostics()["input_transfers"])
    for key in ("valid_count", "short_count"):
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
    np.testing.assert_allclose(
        outputs[0]["pairs"], outputs[1]["pairs"], rtol=1e-3, atol=2e-4
    )
    assert transfers[0] >= 7
    if samples == 1400:
        assert transfers[1] == 1
    else:
        assert (
            transfers[1] == transfers[0]
        )  # staging would consume too much of the 3MiB budget


@pytest.mark.parametrize("method", ["linear", "pws"])
def test_large_cross_minute_batches_match_reference_and_streaming(method):
    p = params(stacking_method=method)
    data = np.random.default_rng(202607).normal(size=(7000, 6)).astype(np.float32)
    mask = np.ones_like(data, dtype=bool)
    mask[501, 1] = False
    data[1301, 2] = np.nan
    data[:, 5] = 0
    starts = np.arange(0, 6901, 37)
    minutes = starts // 230
    ref = ReferenceSession(dict(p, cc_backend="cpu", cc_device="cpu"), 6)
    ref.add_block(data, mask, starts, minutes)
    expected = ref.finish_group()
    for batch in (1, 7, 12, 32, 64, 128, 256, 512):
        native = NativeSession(dict(p, cc_batch_chunks=batch), 6)
        # Split one minute between calls as well as between internal batches.
        native.add_block(data, mask, starts[:9], minutes[:9])
        native.add_block(data, mask, starts[9:], minutes[9:])
        actual = native.finish_group()
        for key in ("valid_count", "short_count"):
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_allclose(
            actual["pairs"], expected["pairs"], rtol=1e-3, atol=2e-4
        )
        diag = native.diagnostics()
        assert diag["window_batch_max"] <= batch
        if batch >= 32:
            assert diag["window_batch_max"] > 12
            assert diag["minute_span_max"] > 1
    # A large request must automatically fit a small temporary-workspace budget.
    native = NativeSession(
        dict(p, cc_batch_chunks=512, cc_memory_budget_bytes=3 * 1024**2), 6
    )
    native.add_block(data, mask, starts, minutes)
    actual = native.finish_group()
    for key in ("valid_count", "short_count"):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_allclose(actual["pairs"], expected["pairs"], rtol=1e-3, atol=2e-4)
    assert native.diagnostics()["window_batch_max"] < len(starts)


@pytest.mark.parametrize("reuse", [0, 1])
def test_fft_plan_cache_eviction_and_workspace_reuse(reuse):
    p = params(cc_pair_batch=8, cc_reuse_workspace=reuse)
    data = np.random.default_rng(811).normal(size=(6200, 8)).astype(np.float32)
    valid = np.ones_like(data, dtype=bool)
    valid[155, 1] = False
    starts = np.arange(0, 6101, 100)
    minutes = np.arange(len(starts)) // 4
    ref = ReferenceSession(dict(p, cc_backend="cpu", cc_device="cpu"), 8)
    ref.add_block(data, valid, starts, minutes)
    expected = ref.finish_group()
    # Fill and evict the eight-entry inverse-plan cache; 28 pairs leave a tail of 4.
    for batch in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        native = NativeSession(dict(p, cc_batch_chunks=batch), 8)
        for _ in range(2):
            native.add_block(data, valid, starts, minutes)
            actual = native.finish_group()
            for key in ("valid_count", "short_count"):
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_allclose(
                actual["pairs"], expected["pairs"], rtol=1e-3, atol=2e-4
            )
            if reuse:
                assert native.diagnostics()["workspace_cached_bytes"] > 0
            else:
                assert native.diagnostics()["workspace_cached_bytes"] == 0
            native.clear_group()
