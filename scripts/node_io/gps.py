"""DigiSolo GPS fixes grouped by section and actual UTC fix date."""

from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth


def station_gps(directory, date_utc):
    directory = Path(directory)
    path = directory / "DigiSolo.LOG"
    fixes = []
    block = {}

    def keep():
        if block.get("date") != date_utc or not {"lat", "lon"} <= block.keys():
            return
        lat, lon = block["lat"], block["lon"]
        if (
            np.isfinite([lat, lon]).all()
            and -90 <= lat <= 90
            and -180 <= lon <= 180
            and (lat != 0 or lon != 0)
        ):
            fixes.append((lat, lon, block.get("alt", np.nan)))

    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("["):
                keep()
                block = {}
            elif line.startswith("Latest Fix UTC Time"):
                match = re.search(r"\d{4}/\d{2}/\d{2}", line)
                if match:
                    block["date"] = match.group()
            else:
                for key, prefix in (
                    ("lat", "Latitude"),
                    ("lon", "Longitude"),
                    ("alt", "Altitude"),
                ):
                    if line.startswith(prefix):
                        try:
                            block[key] = float(line.split("=", 1)[1])
                        except (ValueError, IndexError):
                            pass
                        break
    keep()
    if not fixes:
        raise ValueError(f"No valid GPS fixes on {date_utc}: {path}")
    points = np.asarray(fixes)
    lat, lon = np.median(points[:, :2], axis=0)
    offsets = np.asarray(
        [gps2dist_azimuth(lat, lon, a, b)[0] for a, b in points[:, :2]]
    )
    altitude = points[np.isfinite(points[:, 2]), 2]
    return {
        "station_id": directory.name,
        "latitude_deg": float(lat),
        "longitude_deg": float(lon),
        "elevation_m": float(np.median(altitude)) if len(altitude) else "",
        "gps_source": str(path.resolve()),
        "gps_date_utc": date_utc,
        "gps_fix_count": len(fixes),
        "gps_max_offset_m": float(offsets.max()),
        "gps_median_offset_m": float(np.median(offsets)),
    }


def discover_station_gps(root, date_utc, workers=8):
    directories = sorted(
        (p for p in Path(root).iterdir() if p.name.isdigit() and p.is_dir()),
        key=lambda p: int(p.name),
    )
    if len(directories) < 2:
        raise ValueError("Expected at least two numbered station directories")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: station_gps(p, date_utc), directories))


def write_station_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
