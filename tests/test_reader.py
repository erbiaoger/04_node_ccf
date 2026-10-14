"""SAC inventory and masked sample boundary regression tests."""

import numpy as np
import pytest
from node_io import NodeSACReader, load_node_catalog
from obspy import UTCDateTime
from obspy.io.sac.sactrace import SACTrace


def make_sac(
    root, station, name, data, b=0, delta=0.01, endian="little", component="Z"
):
    sac = SACTrace(
        data=np.asarray(data, dtype=np.float32), delta=delta, b=b, kcmpnm=component
    )
    sac.reftime = UTCDateTime("2026-07-09T00:00:00")
    sac.b = b
    sac.write(str(root / f"{station}.{name}.sac"), byteorder=endian)


def csv_file(root, stations=("1", "2", "3")):
    path = root / "stations.csv"
    path.write_text(
        "station_id,distance_m\n"
        + "".join(f"{s},{i * 40}\n" for i, s in enumerate(stations))
    )
    return path


def test_b_big_endian_gaps_missing_and_nan(tmp_path):
    make_sac(tmp_path, "1", "a", [1, 2, np.nan, 4], b=2, endian="big")
    make_sac(tmp_path, "1", "b", [9, 10], b=2.06)
    make_sac(tmp_path, "2", "a", np.arange(8), b=2)
    catalog = load_node_catalog(csv_file(tmp_path), tmp_path)
    reader = NodeSACReader(catalog)
    data, valid = reader.read_samples(catalog.start_timestamp, 8)
    np.testing.assert_array_equal(valid[:, 0], [1, 1, 0, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(data[:, 0], [1, 2, 0, 4, 0, 0, 9, 10])
    assert not valid[:, 2].any()
    assert catalog.start_timestamp == UTCDateTime("2026-07-09T00:00:02").timestamp
    assert round((catalog.end_timestamp - catalog.start_timestamp) * reader.fs) == 8


@pytest.mark.parametrize("kind", ["overlap", "component", "rate", "grid", "truncated"])
def test_reject_invalid_inventory(tmp_path, kind):
    make_sac(tmp_path, "1", "a", np.arange(10))
    make_sac(
        tmp_path,
        "2",
        "a",
        np.arange(10),
        delta=0.02 if kind == "rate" else 0.01,
        b=0.005 if kind == "grid" else 0,
    )
    if kind in {"overlap", "component"}:
        make_sac(
            tmp_path,
            "1",
            "b",
            np.arange(10),
            b=0.05,
            component="N" if kind == "component" else "Z",
        )
    if kind == "truncated":
        with (tmp_path / "1.a.sac").open("r+b") as handle:
            handle.truncate(640)
    with pytest.raises(ValueError):
        load_node_catalog(csv_file(tmp_path), tmp_path)
