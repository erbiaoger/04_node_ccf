"""CSV ordered station catalog and memory-mapped SAC time-window reader.

The large passive node files are never loaded in full.  SAC headers are read
with ObsPy in head-only mode and the contiguous float32 data block is exposed
through ``numpy.memmap``.  The reader returns ``(nt, nstation)`` arrays in the
exact CSV order so the existing CCF code can consume them unchanged.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
from obspy.io.sac.sactrace import SACTrace

LOG = logging.getLogger(__name__)
SAC_HEADER_BYTES = 632


@dataclass(frozen=True)
class SACSegment:
    path: Path
    start_timestamp: float
    sampling_rate: float
    npts: int
    byteorder: str


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    distance_m: float
    segments: tuple[SACSegment, ...]

    @property
    def start_timestamp(self) -> float:
        return self.segments[0].start_timestamp

    @property
    def end_timestamp(self) -> float:
        segment = self.segments[-1]
        return segment.start_timestamp + (segment.npts - 1) / segment.sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.segments[0].sampling_rate


@dataclass(frozen=True)
class NodeCatalog:
    stations: tuple[StationRecord, ...]

    @property
    def distances_m(self) -> np.ndarray:
        return np.asarray([s.distance_m for s in self.stations], dtype=float)

    @property
    def station_ids(self) -> tuple[str, ...]:
        return tuple(s.station_id for s in self.stations)

    @property
    def start_timestamp(self) -> float:
        return max(s.start_timestamp for s in self.stations)

    @property
    def end_timestamp(self) -> float:
        return min(s.end_timestamp for s in self.stations)


def _clean_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("station_id cannot be empty")
    return text


def _candidate_ids(station_id: str) -> set[str]:
    """Accept full filename ids and SAC's six/seven digit station id."""
    digits = re.sub(r"\D", "", station_id)
    out = {station_id}
    if digits:
        out.add(digits)
        out.add(digits[-7:])
    return out


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    fields = {str(k).strip().lower() for k in rows[0]}
    id_field = next(
        (x for x in ("station_id", "station", "sta", "id") if x in fields), None
    )
    distance_field = next(
        (x for x in ("distance_m", "distance", "x_m", "x") if x in fields), None
    )
    if id_field is None or distance_field is None:
        raise ValueError(
            "CSV must contain station_id (or station/sta/id) and distance_m (or distance/x_m/x)"
        )
    normalized = []
    for row in rows:
        lower = {str(k).strip().lower(): v for k, v in row.items()}
        normalized.append(
            {
                "station_id": _clean_id(lower[id_field]),
                "distance_m": str(lower[distance_field]).strip(),
            }
        )
    return normalized


def _index_sac_files(data_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("*.sac")) + sorted(data_dir.glob("*.SAC")):
        stem_id = path.name.split(".", 1)[0]
        for key in _candidate_ids(stem_id):
            result.setdefault(key, []).append(path)
    if not result:
        raise FileNotFoundError(f"No SAC files found in {data_dir}")
    return result


def load_node_catalog(csv_path: str | Path, data_dir: str | Path) -> NodeCatalog:
    """Build a validated CSV-ordered catalog from SAC headers."""
    csv_path, data_dir = Path(csv_path).expanduser(), Path(data_dir).expanduser()
    index = _index_sac_files(data_dir)
    stations: list[StationRecord] = []
    seen: set[str] = set()
    for row in _read_csv_rows(csv_path):
        station_id = row["station_id"]
        if station_id in seen:
            raise ValueError(f"Duplicate station_id in CSV: {station_id}")
        seen.add(station_id)
        paths = next(
            (
                index[candidate]
                for candidate in _candidate_ids(station_id)
                if candidate in index
            ),
            None,
        )
        if not paths:
            raise FileNotFoundError(f"No SAC file matches CSV station_id={station_id}")
        distance_m = float(row["distance_m"])
        if not np.isfinite(distance_m):
            raise ValueError(f"Invalid distance_m for {station_id}")
        segments = []
        for path in paths:
            header = SACTrace.read(str(path), headonly=True)
            if header.npts <= 1 or header.delta <= 0:
                raise ValueError(f"Invalid SAC header: {path}")
            segments.append(
                SACSegment(
                    path,
                    float(header.reftime.timestamp),
                    1.0 / float(header.delta),
                    int(header.npts),
                    str(header.byteorder),
                )
            )
        segments.sort(key=lambda segment: segment.start_timestamp)
        if any(
            not np.isclose(
                segment.sampling_rate, segments[0].sampling_rate, rtol=1e-6, atol=1e-5
            )
            for segment in segments
        ):
            raise ValueError(f"Sampling rates differ within station {station_id}")
        for previous, current in pairwise(segments):
            gap = current.start_timestamp - (
                previous.start_timestamp + previous.npts / previous.sampling_rate
            )
            if gap > 1.0 / previous.sampling_rate + 1e-5:
                raise ValueError(
                    f"Time gap between SAC segments for {station_id}: {gap:.6g} s"
                )
        stations.append(StationRecord(station_id, distance_m, tuple(segments)))
    rates = np.asarray([s.sampling_rate for s in stations])
    if not np.allclose(rates, rates[0], rtol=1e-6, atol=1e-5):
        raise ValueError(
            f"Sampling rates differ across nodes: {rates.min()}..{rates.max()}"
        )
    distances = np.asarray([s.distance_m for s in stations])
    if np.any(np.diff(distances) <= 0):
        raise ValueError("CSV distances must be strictly increasing")
    LOG.info(
        "Loaded %d stations, fs=%.6g Hz, distance=%.3f..%.3f m",
        len(stations),
        rates[0],
        distances[0],
        distances[-1],
    )
    return NodeCatalog(tuple(stations))


class NodeSACReader:
    """Read aligned windows from a :class:`NodeCatalog`."""

    def __init__(self, catalog: NodeCatalog, dtype: str = "float32"):
        self.catalog = catalog
        self.dtype = np.dtype(dtype)
        self.fs = catalog.stations[0].sampling_rate

    def read_window(
        self, start_timestamp: float, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        npts = round(duration_s * self.fs)
        start_index = []
        for station in self.catalog.stations:
            index = round((start_timestamp - station.start_timestamp) * self.fs)
            if index < 0 or start_timestamp + duration_s > station.end_timestamp + 1e-6:
                raise ValueError(
                    f"Window outside {station.station_id} at {datetime.fromtimestamp(start_timestamp, UTC).isoformat()}"
                )
            start_index.append(index)
        data = np.empty((npts, len(self.catalog.stations)), dtype=self.dtype)
        valid = np.ones(len(self.catalog.stations), dtype=bool)
        for station_index, (station, index) in enumerate(
            zip(self.catalog.stations, start_index)
        ):
            pieces: list[np.ndarray] = []
            remaining_start, remaining = index, npts
            for segment in station.segments:
                segment_offset = round(
                    (segment.start_timestamp - station.start_timestamp) * self.fs
                )
                segment_end = segment_offset + segment.npts
                if (
                    remaining_start >= segment_end
                    or remaining_start + remaining <= segment_offset
                ):
                    continue
                local_start = max(remaining_start, segment_offset) - segment_offset
                take = min(segment.npts - local_start, remaining)
                endian = "<" if segment.byteorder == "little" else ">"
                values = np.memmap(
                    segment.path,
                    mode="r",
                    dtype=np.dtype(endian + "f4"),
                    offset=SAC_HEADER_BYTES,
                    shape=(segment.npts,),
                )
                pieces.append(
                    np.asarray(
                        values[local_start : local_start + take], dtype=self.dtype
                    )
                )
                del values
                remaining_start += take
                remaining -= take
                if remaining == 0:
                    break
            if remaining:
                raise ValueError(
                    f"Window crosses a missing SAC segment for {station.station_id}"
                )
            data[:, station_index] = np.concatenate(pieces)
        return data, valid

    def iter_windows(
        self,
        start_timestamp: float,
        end_timestamp: float,
        duration_s: float,
        step_s: float,
    ):
        cursor = float(start_timestamp)
        while cursor + duration_s <= end_timestamp + 1e-6:
            yield cursor, self.read_window(cursor, duration_s)
            cursor += step_s
