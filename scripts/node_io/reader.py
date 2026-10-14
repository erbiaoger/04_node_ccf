"""Bounded SAC sample reads with an explicit per-sample validity mask.

No signal processing is performed here. Missing, NaN and Inf samples are
stored as zero with validity=False, so they never become valid zero padding.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np

LOG = logging.getLogger(__name__)

from .catalog import SAC_HEADER_BYTES, NodeCatalog, sample_index


class NodeSACReader:
    def __init__(self, catalog: NodeCatalog, dtype: str = "float32", workers: int = 1):
        self.catalog = catalog
        self.fs = catalog.sampling_rate
        self.dtype = np.dtype(dtype)
        self.workers = max(1, int(workers))
        self._handles = {}
        self._locks = [Lock() for _ in catalog.stations]

    def close(self):
        for channel, lock in enumerate(self._locks):
            with lock:
                entry = self._handles.pop(channel, None)
                if entry is not None:
                    entry[1].close()

    def __del__(self):
        if hasattr(self, "_locks"):
            self.close()

    def _handle(self, channel, path):
        entry = self._handles.get(channel)
        if entry is None or entry[0] != path:
            if entry is not None:
                entry[1].close()
            entry = (path, path.open("rb"))
            self._handles[channel] = entry
        return entry[1]

    def read_samples(self, start_timestamp: float, npts: int):
        if npts < 1:
            raise ValueError("read_samples requires a positive length")
        data = np.zeros((npts, len(self.catalog.stations)), dtype=self.dtype)
        valid = np.zeros(data.shape, dtype=bool)

        def read_station(item):
            channel, station = item
            # One retained handle per station avoids reopening/revalidating CIFS
            # leases for every time block. Lock protects seek+read across callers.
            with self._locks[channel]:
                for segment in station.segments:
                    origin = sample_index(
                        segment.start_timestamp - start_timestamp,
                        self.fs,
                        "SAC segment",
                    )
                    lo, hi = max(0, origin), min(npts, origin + segment.npts)
                    if hi <= lo:
                        continue
                    endian = "<" if segment.byteorder == "little" else ">"
                    # Explicit bounded reads avoid CIFS mmap page-fault I/O and keep
                    # temporary arrays bounded even for an explicit full preload.
                    handle = self._handle(channel, segment.path)
                    handle.seek(SAC_HEADER_BYTES + (lo - origin) * 4)
                    for begin in range(lo, hi, 1024 * 1024):
                        stop = min(begin + 1024 * 1024, hi)
                        piece = np.fromfile(
                            handle, dtype=endian + "f4", count=stop - begin
                        )
                        if piece.size != stop - begin:
                            raise OSError(f"Short SAC read: {segment.path}")
                        finite = np.isfinite(piece)
                        data[begin:stop, channel] = np.where(finite, piece, 0)
                        valid[begin:stop, channel] = finite

        items = enumerate(self.catalog.stations)
        if self.workers == 1:
            for item in items:
                read_station(item)
        else:
            with ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="sac"
            ) as pool:
                # Exhaust map so exceptions propagate before consuming any block.
                for _ in pool.map(read_station, items):
                    pass
        return data, valid

    def read_window(self, start_timestamp: float, duration_s: float):
        return self.read_samples(
            start_timestamp, sample_index(duration_s, self.fs, "duration")
        )
