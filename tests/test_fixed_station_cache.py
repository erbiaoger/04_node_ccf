"""Fixed geometry is parsed once and reused without further CSV reads."""

from dataclasses import replace
from pathlib import Path

from node_pipeline import prepare, run_node_ccf
from run_node_ccf_config import cache_fixed_stations
from test_pipeline import configuration


def test_fixed_table_read_once_and_survives_removal(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    reads = []
    original = Path.read_bytes

    def read(path):
        if path == config.csv_path.resolve():
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    cached = cache_fixed_stations(
        [(config, True), (replace(config, output_dir=tmp_path / "second"), True)]
    )
    assert len(reads) == 1
    first, second = (row[0] for row in cached)
    assert first.station_records is second.station_records
    assert first.station_csv_sha256 == second.station_csv_sha256
    config.csv_path.unlink()
    for item in (first, second):
        assert prepare(item)[0].station_ids == ("1", "2", "3")
        run_node_ccf(item)
    assert len(reads) == 1


def test_daily_gps_mode_is_not_cached(tmp_path):
    config = configuration(tmp_path)
    assert cache_fixed_stations([(config, False)]) == [(config, False)]
    assert config.station_records is None
