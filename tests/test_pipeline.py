"""Read-mode parity, global overlap ownership, atomic output and resume."""

from dataclasses import replace

import numpy as np
import pytest
from node_compute.schedule import groups
from node_output.artifacts import load_result as loadmat
from node_pipeline import NodeCCFConfig, choose_read_mode, run_node_ccf
from test_reader import csv_file, make_sac


def configuration(tmp_path):
    rng = np.random.default_rng(14)
    for station in ("1", "2"):
        make_sac(tmp_path, station, "a", rng.normal(size=1300))
    return NodeCCFConfig(
        csv_file(tmp_path),
        tmp_path,
        tmp_path / "result",
        {
            "cc_backend": "cpu",
            "cc_device": "cpu",
            "read_mode": "block",
            "read_hours": 0,
            "cc_len": 1.0,
            "cc_step_s": 0.5,
            "minute_stack_s": 2.0,
            "save_every": 2,
            "maxlag": 0.3,
            "freqmax": 20.0,
        },
    )


def test_window_ownership_tail_and_arbitrary_step():
    p = {
        "time_downsample": 1,
        "cc_len": 10.0,
        "cc_step_s": 7.0,
        "minute_stack_s": 60.0,
        "save_every": 2,
    }
    scheduled = list(groups(25700, 100.0, p))
    actual = np.concatenate([g.starts for g in scheduled])
    np.testing.assert_array_equal(actual, np.arange(0, 25700 - 1000 + 1, 700))
    assert len(set(actual)) == len(actual)
    for g in scheduled:
        assert ((g.starts >= g.begin) & (g.starts < g.end)).all()


def test_preload_block_outputs_masks_and_resume(tmp_path):
    config = configuration(tmp_path)
    run_node_ccf(config)
    first = sorted(config.output_dir.glob("*.h5"))
    other = replace(
        config,
        output_dir=tmp_path / "preload",
        cc_params=dict(config.cc_params, read_mode="preload"),
    )
    run_node_ccf(other)
    for a, b in zip(first, sorted(other.output_dir.glob("*.h5")), strict=True):
        aa, bb = loadmat(a), loadmat(b)
        for key in (
            "data",
            "valid_mask",
            "valid_short_window_count",
            "valid_minute_count",
        ):
            np.testing.assert_array_equal(aa[key], bb[key])
        assert aa["data"].shape == (3, 61, 3)
        assert not aa["valid_mask"][2].any()
    timestamps = [p.stat().st_mtime_ns for p in first]
    run_node_ccf(config)
    assert timestamps == [p.stat().st_mtime_ns for p in first]
    # A corrupt completed file is rebuilt, leaving other completed groups intact.
    first[0].write_bytes(b"interrupted/corrupt")
    run_node_ccf(config)
    assert loadmat(first[0])["data"].shape == (3, 61, 3)
    assert timestamps[1:] == [p.stat().st_mtime_ns for p in first[1:]]
    with pytest.raises(ValueError, match="Resume rejected"):
        run_node_ccf(replace(config, cc_params=dict(config.cc_params, maxlag=0.2)))
    with pytest.raises(FileExistsError):
        run_node_ccf(replace(config, cc_params=dict(config.cc_params, resume=False)))


def test_input_file_change_rejects_resume(tmp_path):
    config = configuration(tmp_path)
    run_node_ccf(config)
    make_sac(tmp_path, "1", "a", np.zeros(1300))
    with pytest.raises(ValueError, match="Resume rejected"):
        run_node_ccf(config)


def test_auto_memory_policy():
    p = {
        "read_mode": "auto",
        "samp_freq": 100.0,
        "minute_stack_s": 60.0,
        "save_every": 30,
        "cc_len": 10.0,
    }
    assert choose_read_mode(p, 100000, 34, available=100 * 1024**3) == "preload"
    assert choose_read_mode(p, 100000000, 34, available=1024**3) == "block"


def test_downsampling_preload_block_parity(tmp_path):
    config = configuration(tmp_path)
    params = dict(config.cc_params, time_downsample=2, freqmax=15.0)
    config = replace(config, cc_params=params)
    run_node_ccf(config)
    preload = replace(
        config,
        output_dir=tmp_path / "preload",
        cc_params=dict(params, read_mode="preload"),
    )
    run_node_ccf(preload)
    for a, b in zip(
        sorted(config.output_dir.glob("*.h5")),
        sorted(preload.output_dir.glob("*.h5")),
        strict=True,
    ):
        np.testing.assert_array_equal(loadmat(a)["data"], loadmat(b)["data"])


def test_stream_restack_preserves_station_identity_and_valid_counts(tmp_path):
    from node_compute.reference import Ops, PairAccumulator
    from stack_node_cc_mat import stack_window

    config = configuration(tmp_path)
    run_node_ccf(config)
    files = sorted(config.output_dir.glob("*.h5"))
    result = stack_window(files, cc_backend="cpu", cc_device="cpu")
    expected = PairAccumulator(3, 61, Ops("cpu", "cpu"), "pws")
    a, b = np.triu_indices(3, 1)
    for path in files:
        mat = loadmat(path)
        expected.add(mat["data"][a, :, b], mat["valid_mask"][a, b].astype(bool))
    np.testing.assert_allclose(result["data"][a, :, b], expected.finish(), atol=2e-6)
    np.testing.assert_array_equal(result["valid_file_count"][a, b], expected.count)
    assert not result["valid_mask"][2].any()
    np.testing.assert_array_equal(result["station_id"], loadmat(files[0])["station_id"])


def test_plot_masks_without_changing_layout(tmp_path):
    from plot_node_ccf import _load_node_mat

    config = configuration(tmp_path)
    run_node_ccf(config)
    data, _dt, profile = _load_node_mat(min(config.output_dir.glob("*.h5")))
    assert data.shape == (3, 61, 3) and profile.shape == (3,)
    assert np.isnan(data[2]).all()
    assert np.isnan(data[0, :, 0]).all()
    assert np.isfinite(data[0, :, 1]).all()


def test_performance_knobs_can_change_on_resume_and_restack(tmp_path):
    from node_output.artifacts import save_result as save_mat
    from stack_node_cc_mat import stack_window

    config = configuration(tmp_path)
    run_node_ccf(config)
    files = sorted(config.output_dir.glob("*.h5"))
    # Runtime read/batch changes are permitted by the task identity.
    import json

    mat = loadmat(files[-1])
    params = json.loads(str(mat["parameters_json"].reshape(-1)[0]))
    params.update(read_mode="preload", cc_batch_chunks=1)
    mat["parameters_json"] = json.dumps(params)
    save_mat(files[-1], {k: v for k, v in mat.items() if not k.startswith("__")})
    result = stack_window(files, cc_backend="cpu", cc_device="cpu")
    assert result["valid_file_count"][0, 1] == len(files)


def test_cli_help_and_output_path():
    from cli import parse_config

    with pytest.raises(SystemExit) as exc:
        parse_config(["--help"])
    assert exc.value.code == 0
    config, mode = parse_config(["--print-output-dir"])
    assert mode == "output" and config.output_dir.name == "node_ccf_spectral_v1"


def test_audit_checks_all_groups_and_final_provenance(tmp_path):
    from audit_node_ccf import audit
    from node_output.artifacts import save_result as save_mat
    from stack_node_cc_mat import stack_window

    config = configuration(tmp_path)
    run_node_ccf(config)
    files = sorted(config.output_dir.glob("*.h5"))
    output = tmp_path / "restack"
    save_mat(
        output / "restack_0001.h5",
        stack_window(files, cc_backend="cpu", cc_device="cpu"),
    )
    report = audit(config.output_dir, output)
    assert report["groups"] == len(files) and report["valid_pairs_final"] == 1
    final = loadmat(output / "restack_0001.h5")
    final["valid_short_window_count"][0, 1] += 1
    save_mat(
        output / "restack_0001.h5",
        {k: v for k, v in final.items() if not k.startswith("__")},
    )
    with pytest.raises(ValueError, match="effective counts"):
        audit(config.output_dir, output)
