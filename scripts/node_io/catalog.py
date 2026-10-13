"""CSV-ordered SAC inventory, header validation and reproducible input identity.

Used by cli.py; reads headers only and retains missing stations in the output
geometry. Start times include SAC b. Segments use half-open sample intervals.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from obspy.io.sac.sactrace import SACTrace

LOG = logging.getLogger(__name__)
SAC_HEADER_BYTES = 632
ALIGN_TOLERANCE = 0.01  # samples; accommodates timestamp floating-point roundoff


@dataclass(frozen=True)
class SACSegment:
    path: Path
    start_timestamp: float
    sampling_rate: float
    npts: int
    byteorder: str
    component: str = ""

    @property
    def end_timestamp(self):
        return self.start_timestamp + self.npts / self.sampling_rate


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    distance_m: float
    segments: tuple[SACSegment, ...]


@dataclass(frozen=True)
class NodeCatalog:
    stations: tuple[StationRecord, ...]
    sampling_rate: float
    start_timestamp: float
    end_timestamp: float

    @property
    def distances_m(self):
        return np.asarray([s.distance_m for s in self.stations], dtype=np.float64)

    @property
    def station_ids(self):
        return tuple(s.station_id for s in self.stations)

    def fingerprint(self):
        return [
            {
                "station_id": station.station_id,
                "distance_m": station.distance_m,
                "segments": [
                    {
                        "path": str(s.path.resolve()),
                        "size": s.path.stat().st_size,
                        "mtime_ns": s.path.stat().st_mtime_ns,
                        "start": s.start_timestamp,
                        "fs": s.sampling_rate,
                        "npts": s.npts,
                        "byteorder": s.byteorder,
                        "component": s.component,
                    }
                    for s in station.segments
                ],
            }
            for station in self.stations
        ]


def sample_index(seconds: float, fs: float, label: str = "time") -> int:
    value = seconds * fs
    index = round(value)
    if abs(value - index) > ALIGN_TOLERANCE:
        raise ValueError(f"{label} is not on the sampling grid: {value:.6f} samples")
    return index


def _read_csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"CSV has no station rows: {path}")
    rows = [
        {str(k).strip().lower(): str(v).strip() for k, v in row.items()} for row in raw
    ]
    fields = rows[0]
    id_key = next(
        (k for k in ("station_id", "station", "sta", "id") if k in fields), None
    )
    x_key = next(
        (k for k in ("distance_m", "distance", "x_m", "x") if k in fields), None
    )
    if id_key is None or x_key is None:
        raise ValueError("CSV requires station_id and distance_m")
    result = [(row[id_key], float(row[x_key])) for row in rows]
    ids = [row[0] for row in result]
    x = np.asarray([row[1] for row in result])
    if len(ids) < 2 or any(not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("CSV requires at least two unique nonempty station IDs")
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("CSV distances must be finite and strictly increasing")
    return result


def load_node_catalog(csv_path: str | Path, data_dir: str | Path) -> NodeCatalog:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SAC directory does not exist: {root}")
    index: dict[str, list[Path]] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() == ".sac":
            index.setdefault(path.name.split(".", 1)[0], []).append(path)
    stations = []
    all_segments = []
    for station_id, distance in _read_csv_rows(Path(csv_path)):
        paths = index.get(station_id, [])
        if not paths:
            candidates = [key for key in index if key[-7:] == station_id[-7:]]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous filename IDs for station {station_id}")
            paths = index[candidates[0]] if candidates else []
        segments = []
        for path in paths:
            h = SACTrace.read(str(path), headonly=True)
            if (
                h.npts < 1
                or not np.isfinite(h.delta)
                or h.delta <= 0
                or h.iftype != "itime"
                or not h.leven
            ):
                raise ValueError(f"Not a regularly sampled SAC time series: {path}")
            if path.stat().st_size < SAC_HEADER_BYTES + 4 * h.npts:
                raise ValueError(f"Truncated SAC data: {path}")
            start = float(h.reftime.timestamp) + float(h.b or 0)
            if not np.isfinite(start):
                raise ValueError(f"Invalid SAC start time: {path}")
            segments.append(
                SACSegment(
                    path,
                    start,
                    1 / float(h.delta),
                    int(h.npts),
                    h.byteorder,
                    h.kcmpnm or "",
                )
            )
        segments.sort(key=lambda s: s.start_timestamp)
        if len({s.component for s in segments if s.component}) > 1:
            raise ValueError(f"Multiple SAC components for station {station_id}")
        if not segments:
            LOG.warning(
                "Missing station %s: preserving geometry with invalid samples",
                station_id,
            )
        stations.append(StationRecord(station_id, distance, tuple(segments)))
        all_segments.extend(segments)
    if not all_segments:
        raise FileNotFoundError(f"No selected station has SAC data in {root}")
    fs = all_segments[0].sampling_rate
    if any(
        not np.isclose(s.sampling_rate, fs, rtol=1e-7, atol=1e-7) for s in all_segments
    ):
        raise ValueError("SAC sampling rates differ; explicit resampling is required")
    present = [s for s in stations if s.segments]
    start = max(s.segments[0].start_timestamp for s in present)
    end = min(s.segments[-1].end_timestamp for s in present)
    if end <= start:
        raise ValueError("Stations have no common outer time interval")
    for station in present:
        previous_end = None
        for segment in station.segments:
            origin = sample_index(
                segment.start_timestamp - start, fs, str(segment.path)
            )
            if previous_end is not None and origin < previous_end:
                raise ValueError(f"Overlapping SAC segments for {station.station_id}")
            previous_end = origin + segment.npts
    sample_index(end - start, fs, "common end")
    LOG.info(
        "Inventory: %d stations, %.9g Hz, common interval %.3f hours",
        len(stations),
        fs,
        (end - start) / 3600,
    )
    return NodeCatalog(tuple(stations), fs, start, end)
