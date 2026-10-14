"""CSV-ordered SAC inventory, header validation and reproducible input identity.

Used by cli.py; reads headers only and retains missing stations in the output
geometry. Start times include SAC b. Segments use half-open sample intervals.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
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
    header_delta: float | None = None

    @property
    def end_timestamp(self):
        return self.start_timestamp + self.npts / self.sampling_rate


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    distance_m: float | None
    segments: tuple[SACSegment, ...]
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    gps_info: dict | None = None


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
                **(
                    {
                        "latitude_deg": station.latitude_deg,
                        "longitude_deg": station.longitude_deg,
                        "gps_info": station.gps_info,
                    }
                    if station.latitude_deg is not None
                    else {}
                ),
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
                        **(
                            {"header_delta": s.header_delta}
                            if s.header_delta is not None
                            else {}
                        ),
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


def _read_csv_rows(path: Path, *, text=None):
    with (
        path.open(encoding="utf-8-sig", newline="")
        if text is None
        else io.StringIO(text, newline="")
    ) as handle:
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
    geo = "latitude_deg" in fields and "longitude_deg" in fields
    if id_key is None or (x_key is None and not geo):
        raise ValueError(
            "CSV requires station_id and distance_m or latitude_deg/longitude_deg"
        )
    result = []
    for row in rows:
        lat = float(row["latitude_deg"]) if geo else None
        lon = float(row["longitude_deg"]) if geo else None
        if geo and (
            not np.isfinite([lat, lon]).all()
            or not -90 <= lat <= 90
            or not -180 <= lon <= 180
        ):
            raise ValueError("Invalid WGS84 station coordinates")
        info = {
            k: row[k]
            for k in (
                "gps_source",
                "gps_fix_count",
                "gps_max_offset_m",
                "gps_median_offset_m",
                "gps_date_utc",
                "elevation_m",
            )
            if k in row
        } or None
        result.append(
            StationRecord(
                row[id_key], float(row[x_key]) if x_key else None, (), lat, lon, info
            )
        )
    ids = [row.station_id for row in result]
    x = np.asarray([row.distance_m for row in result]) if x_key else None
    if len(ids) < 2 or any(not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("CSV requires at least two unique nonempty station IDs")
    if x is not None and (not np.isfinite(x).all() or np.any(np.diff(x) <= 0)):
        raise ValueError("CSV distances must be finite and strictly increasing")
    return result


def filename_interval(path):
    """Recognize daily cut filenames; unknown naming falls back to SAC headers."""
    match = re.search(
        r"\.(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{3})-(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{3})\.",
        path.name,
    )
    if not match:
        return None
    try:
        return tuple(
            datetime.strptime(v, "%Y.%m.%d.%H.%M.%S.%f").replace(tzinfo=UTC).timestamp()
            for v in match.groups()
        )
    except ValueError:
        return None


def load_node_catalog(
    csv_path: str | Path,
    data_dir: str | Path,
    *,
    sampling_rate_hz=None,
    start_timestamp=None,
    end_timestamp=None,
    station_records=None,
) -> NodeCatalog:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SAC directory does not exist: {root}")
    rows = (
        _read_csv_rows(Path(csv_path)) if station_records is None else station_records
    )
    if sampling_rate_hz is not None and (
        not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0
    ):
        raise ValueError("sac_sampling_rate_hz must be positive and finite")
    selected_suffixes = {row.station_id[-7:] for row in rows}
    index: dict[str, list[Path]] = {}
    paths_to_scan = list(root.iterdir())
    for row in rows:
        directory = root / row.station_id
        if directory.is_dir():
            paths_to_scan.extend(directory.iterdir())
    for path in sorted(paths_to_scan):
        # CIFS stat calls are remote round trips: filter names before querying metadata.
        if (
            path.suffix.lower() == ".sac"
            and path.name.split(".", 1)[0][-7:] in selected_suffixes
            and path.is_file()
        ):
            interval = filename_interval(path)
            if interval is not None and (
                (start_timestamp is not None and interval[1] <= start_timestamp)
                or (end_timestamp is not None and interval[0] >= end_timestamp)
            ):
                continue
            index.setdefault(path.name.split(".", 1)[0], []).append(path)
    stations = []
    all_segments = []
    for row in rows:
        station_id, distance = row.station_id, row.distance_m
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
            fs = 1 / float(h.delta)
            if sampling_rate_hz is not None:
                if np.float32(h.delta) != np.float32(1 / sampling_rate_hz):
                    raise ValueError(
                        f"SAC delta disagrees with sac_sampling_rate_hz={sampling_rate_hz}: {path}"
                    )
                fs = float(sampling_rate_hz)
            if (
                start_timestamp is not None
                and start + int(h.npts) / fs <= start_timestamp
            ) or (end_timestamp is not None and start >= end_timestamp):
                continue
            segments.append(
                SACSegment(
                    path,
                    start,
                    fs,
                    int(h.npts),
                    h.byteorder,
                    h.kcmpnm or "",
                    float(h.delta) if sampling_rate_hz is not None else None,
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
        stations.append(
            StationRecord(
                station_id,
                distance,
                tuple(segments),
                row.latitude_deg,
                row.longitude_deg,
                row.gps_info,
            )
        )
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
