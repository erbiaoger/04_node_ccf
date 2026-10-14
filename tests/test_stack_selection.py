"""Multi-day CCF selection, provenance and compatibility."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from node_output.artifacts import load_result, save_result
from node_pipeline import run_node_ccf
from run_node_stack_config import select_files
from stack_node_cc_mat import stack_window
from test_pipeline import configuration


def test_date_time_day_and_count_filters(tmp_path):
    for day in (8, 9, 10):
        for hour in (7, 8, 9, 17, 18):
            start = datetime(2026, 7, day, hour, tzinfo=UTC).timestamp()
            save_result(
                tmp_path / str(day) / f"cc_stack_{hour}.h5",
                {"source_start_unix": start, "source_end_unix": start + 3600},
            )
    base = {"local_timezone": "UTC"}
    all_rows, _ = select_files(tmp_path, base)
    assert len(all_rows) == 15
    selected, _ = select_files(
        tmp_path,
        dict(
            base,
            start_date="2026-07-09",
            end_date="2026-07-10",
            start_time="08:00",
            end_time="18:00",
        ),
    )
    assert len(selected) == 6
    selected, _ = select_files(tmp_path, dict(base, first_n_days=2, stack_count=7))
    assert len(selected) == 7 and {r["date"] for r in selected} == {
        "2026-07-08",
        "2026-07-09",
    }
    selected, skipped = select_files(
        tmp_path, dict(base, start_time="08:30", end_time="18:00")
    )
    assert len(selected) == 6 and len(skipped) == 3
    with pytest.raises(ValueError, match="only"):
        select_files(tmp_path, dict(base, stack_count=100))
    with pytest.raises(ValueError, match="overlap"):
        save_result(
            tmp_path / "duplicate" / "cc_stack_8.h5",
            load_result(tmp_path / "8" / "cc_stack_8.h5"),
        )
        select_files(tmp_path, base)


def test_multiday_restack_accepts_task_ids_rejects_science_changes(tmp_path):
    config = configuration(tmp_path)
    run_node_ccf(config)
    files = sorted(config.output_dir.glob("*.h5"))[:2]
    original = load_result(files[1])
    changed = dict(original, task_id="another-day")
    save_result(files[1], changed)
    with pytest.raises(ValueError, match="task_id"):
        stack_window(files, cc_backend="cpu", cc_device="cpu")
    result = stack_window(
        files, cc_backend="cpu", cc_device="cpu", allow_multi_day=True
    )
    assert len(result["source_file_paths"]) == 2
    assert json.loads(result["source_geometry_json"])[1]["task_id"] == "another-day"
    np.testing.assert_array_equal(
        result["data"], result["data"].transpose(2, 1, 0)[:, ::-1, :]
    )
    params = json.loads(str(changed["parameters_json"]))
    params["freqmax"] = 10
    save_result(files[1], dict(changed, parameters_json=json.dumps(params)))
    with pytest.raises(ValueError, match="freqmax"):
        stack_window(files, cc_backend="cpu", cc_device="cpu", allow_multi_day=True)


def test_multiday_gps_displacement_limit(tmp_path):
    from test_mohe import fixture

    config = fixture(tmp_path)
    run_node_ccf(config)
    files = sorted(config.output_dir.glob("*.h5"))[:2]
    mat = load_result(files[1])
    shifted = dict(
        mat, task_id="day-two", station_latitude_deg=mat["station_latitude_deg"] + 1e-6
    )
    save_result(files[1], shifted)
    result = stack_window(
        files, cc_backend="cpu", cc_device="cpu", allow_multi_day=True
    )
    np.testing.assert_array_equal(
        result["station_latitude_deg"], load_result(files[0])["station_latitude_deg"]
    )
    shifted["station_latitude_deg"] = mat["station_latitude_deg"] + 0.001
    save_result(files[1], shifted)
    with pytest.raises(ValueError, match="movement"):
        stack_window(files, cc_backend="cpu", cc_device="cpu", allow_multi_day=True)
