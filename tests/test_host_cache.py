"""Host read duration is independent of the 30-minute output/GPU blocks."""

from dataclasses import replace

import numpy as np
import pytest
from node_compute.parameters import validate
from node_io import NodeSACReader
from node_output.artifacts import load_result
from node_pipeline import run_node_ccf
from test_pipeline import configuration


@pytest.mark.parametrize("read_hours", [8 / 3600, 12 / 3600, 24, 48])
@pytest.mark.parametrize("factor", [1, 2])
def test_host_cache_boundaries_match_per_group_reads(
    tmp_path, read_hours, factor, monkeypatch
):
    config = configuration(tmp_path)
    config = replace(
        config, cc_params=dict(config.cc_params, time_downsample=factor, freqmax=15.0)
    )
    run_node_ccf(config)
    calls = []
    original = NodeSACReader.read_samples

    def read(reader, start, npts):
        calls.append((start, npts))
        return original(reader, start, npts)

    monkeypatch.setattr(NodeSACReader, "read_samples", read)
    cached = replace(
        config,
        output_dir=tmp_path / "cached",
        cc_params=dict(config.cc_params, read_hours=read_hours),
    )
    run_node_ccf(cached)
    assert len(calls) == (2 if read_hours < 24 else 1)
    for a, b in zip(
        sorted(config.output_dir.glob("*.h5")),
        sorted(cached.output_dir.glob("*.h5")),
        strict=True,
    ):
        aa, bb = load_result(a), load_result(b)
        for key in (
            "data",
            "valid_mask",
            "valid_short_window_count",
            "valid_minute_count",
            "task_id",
        ):
            np.testing.assert_array_equal(aa[key], bb[key])
    # Read duration changes do not invalidate existing scientific outputs.
    calls.clear()
    run_node_ccf(replace(cached, cc_params=dict(cached.cc_params, read_hours=48)))
    assert not calls


def test_invalid_host_read_duration_rejected():
    base = {"cc_backend": "cpu", "cc_device": "cpu", "freqmax": 20.0}
    for hours in (-1, 0.25, 0.75, float("nan")):
        with pytest.raises(ValueError):
            validate(dict(base, read_hours=hours), 100.0)
    assert validate(dict(base, read_hours=24), 100.0)["read_hours"] == 24
    assert validate(dict(base, read_hours=48), 100.0)["read_hours"] == 48


def test_host_memory_limit_checked_before_read(tmp_path, monkeypatch):
    import node_pipeline

    config = configuration(tmp_path)
    config = replace(config, cc_params=dict(config.cc_params, read_hours=24))
    monkeypatch.setattr(node_pipeline, "available_memory", lambda: 256 * 1024**2)

    def forbidden(*args):
        raise AssertionError("Read must not start when requested memory is unavailable")

    monkeypatch.setattr(NodeSACReader, "read_samples", forbidden)
    with pytest.raises(MemoryError, match="Requested host caches"):
        run_node_ccf(config)
