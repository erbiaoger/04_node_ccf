"""Bounded SAC sample reads with an explicit per-sample validity mask.

No signal processing is performed here. Missing, NaN and Inf samples are
stored as zero with validity=False, so they never become valid zero padding.
"""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)

from .catalog import SAC_HEADER_BYTES, NodeCatalog, sample_index


class NodeSACReader:
    def __init__(self, catalog: NodeCatalog, dtype: str = "float32"):
        self.catalog = catalog
        self.fs = catalog.sampling_rate
        self.dtype = np.dtype(dtype)

    def read_samples(self, start_timestamp: float, npts: int):
        if npts < 1:
            raise ValueError("read_samples requires a positive length")
        data = np.zeros((npts, len(self.catalog.stations)), dtype=self.dtype)
        valid = np.zeros(data.shape, dtype=bool)
        for channel, station in enumerate(self.catalog.stations):
            if npts * len(self.catalog.stations) * self.dtype.itemsize > 1024**3:
                LOG.info(
                    "Preload station %d/%d: %s",
                    channel + 1,
                    len(self.catalog.stations),
                    station.station_id,
                )
            for segment in station.segments:
                origin = sample_index(
                    segment.start_timestamp - start_timestamp, self.fs, "SAC segment"
                )
                lo, hi = max(0, origin), min(npts, origin + segment.npts)
                if hi <= lo:
                    continue
                endian = "<" if segment.byteorder == "little" else ">"
                values = np.memmap(
                    segment.path,
                    mode="r",
                    dtype=endian + "f4",
                    offset=SAC_HEADER_BYTES,
                    shape=(segment.npts,),
                )
                piece = values[lo - origin : hi - origin]
                finite = np.isfinite(piece)
                data[lo:hi, channel] = np.where(finite, piece, 0)
                valid[lo:hi, channel] = finite
                del piece, values
        return data, valid

    def read_window(self, start_timestamp: float, duration_s: float):
        return self.read_samples(
            start_timestamp, sample_index(duration_s, self.fs, "duration")
        )
