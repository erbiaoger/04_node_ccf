"""Concurrency, numerical parity and resume checks for bounded streaming."""

from dataclasses import replace
from threading import Event

import node_pipeline
import numpy as np
from node_io import NodeSACReader, load_node_catalog
from node_output.artifacts import load_result as loadmat
from test_pipeline import configuration
from test_reader import csv_file, make_sac


def test_parallel_reader_preserves_masks_and_endian(tmp_path):
    make_sac(tmp_path, "1", "a", [1, np.nan, 3, np.inf], endian="big")
    make_sac(tmp_path, "2", "a", [5, 6, 7, 8])
    catalog = load_node_catalog(csv_file(tmp_path), tmp_path)
    serial = NodeSACReader(catalog, workers=1).read_samples(catalog.start_timestamp, 4)
    parallel = NodeSACReader(catalog, workers=4).read_samples(
        catalog.start_timestamp, 4
    )
    for a, b in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(a, b)


def test_prefetch_overlaps_compute_and_is_bounded(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config = replace(config, cc_params=dict(config.cc_params, prefetch_blocks=1))
    computing, next_read = Event(), Event()
    reads = []
    original_read = NodeSACReader.read_samples
    original_create = node_pipeline.create_session

    def read(self, start, npts):
        reads.append(start)
        if len(reads) == 2:
            assert computing.wait(5), "second read must overlap first computation"
            next_read.set()
        return original_read(self, start, npts)

    def create(params, stations):
        session = original_create(params, stations)
        original_add = session.add_block
        calls = 0

        def add(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                computing.set()
                assert next_read.wait(5), "next read blocked by computation"
                assert len(reads) == 2, "prefetch queued more than one block"
            return original_add(*args)

        session.add_block = add
        return session

    monkeypatch.setattr(NodeSACReader, "read_samples", read)
    monkeypatch.setattr(node_pipeline, "create_session", create)
    node_pipeline.run_node_ccf(config)


def test_streaming_serial_parity_and_resume(tmp_path):
    config = configuration(tmp_path)
    serial = replace(
        config,
        output_dir=tmp_path / "serial",
        cc_params=dict(config.cc_params, io_workers=1, prefetch_blocks=0, async_save=0),
    )
    node_pipeline.run_node_ccf(serial)
    node_pipeline.run_node_ccf(config)
    for a, b in zip(
        sorted(serial.output_dir.glob("*.h5")),
        sorted(config.output_dir.glob("*.h5")),
        strict=True,
    ):
        aa, bb = loadmat(a), loadmat(b)
        for key in (
            "data",
            "valid_mask",
            "valid_short_window_count",
            "valid_minute_count",
            "task_id",
        ):
            np.testing.assert_array_equal(aa[key], bb[key])
    stamps = [p.stat().st_mtime_ns for p in sorted(config.output_dir.glob("*.h5"))]
    node_pipeline.run_node_ccf(
        replace(
            config,
            cc_params=dict(
                config.cc_params, io_workers=1, prefetch_blocks=0, async_save=0
            ),
        )
    )
    assert stamps == [
        p.stat().st_mtime_ns for p in sorted(config.output_dir.glob("*.h5"))
    ]


def test_parallel_read_propagates_truncation(tmp_path):
    import pytest

    make_sac(tmp_path, "1", "a", np.arange(8))
    make_sac(tmp_path, "2", "a", np.arange(8))
    catalog = load_node_catalog(csv_file(tmp_path), tmp_path)
    # Input changes after header inspection: never silently accept a short read.
    with (tmp_path / "1.a.sac").open("r+b") as handle:
        handle.truncate(632 + 4 * 3)
    with pytest.raises(OSError, match="Short SAC read"):
        NodeSACReader(catalog, workers=4).read_samples(catalog.start_timestamp, 8)


def test_async_save_overlaps_next_compute(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    saving, second_compute = Event(), Event()
    original_save = node_pipeline.save_result
    original_create = node_pipeline.create_session
    saves = 0

    def save(path, fields):
        nonlocal saves
        saves += 1
        if saves == 1:
            saving.set()
            assert second_compute.wait(5), "writer prevented next computation"
        original_save(path, fields)

    def create(params, stations):
        session = original_create(params, stations)
        original_add = session.add_block
        calls = 0

        def add(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                assert saving.wait(5)
                second_compute.set()
            return original_add(*args)

        session.add_block = add
        return session

    monkeypatch.setattr(node_pipeline, "save_result", save)
    monkeypatch.setattr(node_pipeline, "create_session", create)
    node_pipeline.run_node_ccf(config)


def test_async_write_failure_never_completes_task(tmp_path, monkeypatch):
    import json

    import pytest

    config = configuration(tmp_path)

    def fail(*args):
        raise OSError("simulated full disk")

    monkeypatch.setattr(node_pipeline, "save_result", fail)
    with pytest.raises(OSError, match="simulated full disk"):
        node_pipeline.run_node_ccf(config)
    manifest = json.loads((config.output_dir / "task_manifest.json").read_text())
    assert manifest["status"] == "running"
    assert not manifest["completed"]


def test_reader_reuses_and_closes_station_handles(tmp_path, monkeypatch):
    from pathlib import Path

    make_sac(tmp_path, "1", "a", np.arange(8))
    make_sac(tmp_path, "2", "a", np.arange(8))
    catalog = load_node_catalog(csv_file(tmp_path), tmp_path)
    original_open = Path.open
    opened = []

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.suffix == ".sac":
            opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", open_file)
    reader = NodeSACReader(catalog, workers=4)
    first, _ = reader.read_samples(catalog.start_timestamp, 4)
    second, _ = reader.read_samples(catalog.start_timestamp + 4 / reader.fs, 4)
    np.testing.assert_array_equal(first[:, 0], np.arange(4))
    np.testing.assert_array_equal(second[:, 0], np.arange(4, 8))
    assert len(opened) == 2
    reader.close()
    assert all(handle.closed for handle in opened)
