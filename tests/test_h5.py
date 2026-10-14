"""Native HDF5 layout and lossless legacy migration."""

import json

import h5py
import numpy as np
from migrate_node_h5 import migrate
from node_output.artifacts import TaskManifest, load_result, save_mat
from node_pipeline import run_node_ccf
from test_pipeline import configuration


def test_one_h5_contains_all_stations_and_metadata(tmp_path):
    config = configuration(tmp_path)
    run_node_ccf(config)
    files = sorted(config.output_dir.glob("*.h5"))
    assert len(files) == 4 and not list(config.output_dir.glob("*.mat"))
    with h5py.File(files[0], "r") as file:
        assert file.attrs["format"] == "node_ccf_h5_v1"
        assert file.attrs["data_axes"] == "source,lag,receiver"
        assert file["data"].shape == (3, 61, 3)
        assert file["data"].dtype == np.float32
        assert file["data"].chunks == (1, 61, 3)
        assert file["station_id"].asstr()[()].tolist() == ["1", "2", "3"]
        assert file["valid_short_window_count"].dtype == np.uint64
        assert float(file["source_end_unix"][()]) > float(file["source_start_unix"][()])


def test_lossless_legacy_migration_and_resume(tmp_path):
    path = tmp_path / "cc_stack_0001_n30.mat"
    fields = {
        "data": np.arange(24, dtype=np.float32).reshape(2, 6, 2),
        "station_id": np.array(["节点1", "节点2"], dtype=object),
        "valid_short_window_count": np.array([[0, 100], [100, 0]], dtype=np.uint64),
        "parameters_json": '{"stacking_method":"pws"}',
        "source_start_unix": 123.0,
    }
    task = TaskManifest(tmp_path, {"version": "test"})
    save_mat(path, fields)
    task.record(1, path)
    task.finish({})
    task_id = task.task_id
    assert migrate(tmp_path) == 1
    archived = tmp_path / "legacy_mat" / path.name
    assert archived.exists() and not path.exists()
    h5 = path.with_suffix(".h5")
    restored = load_result(h5)
    legacy = load_result(archived)
    np.testing.assert_array_equal(restored["data"], legacy["data"])
    np.testing.assert_array_equal(
        restored["valid_short_window_count"], legacy["valid_short_window_count"]
    )
    assert restored["station_id"].reshape(-1).tolist() == ["节点1", "节点2"]
    assert (
        json.loads((tmp_path / "task_manifest.json").read_text())["task_id"] == task_id
    )
    assert TaskManifest(tmp_path, {"version": "test"}).complete(1, h5)
    assert migrate(tmp_path) == 1
