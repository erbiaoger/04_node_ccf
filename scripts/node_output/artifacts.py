"""Atomic MAT output, exact pair metadata and checksum-verified task resume."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from node_compute.parameters import VERSION
from scipy.io import savemat


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def atomic_json(path, value):
    _atomic(
        path,
        lambda handle: handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()
        ),
    )


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".node-", suffix=path.suffix, delete=False
        ) as handle:
            name = handle.name
            write(handle.file)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def save_mat(path, fields):
    _atomic(path, lambda handle: savemat(handle, fields, appendmat=False))


def dense_pairs(pairs, stations, *, mirror_lag=True):
    a, b = np.triu_indices(stations, 1)
    values = np.asarray(pairs)
    if values.shape[0] != len(a):
        raise ValueError("Pair count does not match station geometry")
    if mirror_lag:
        output = np.zeros((stations, values.shape[1], stations), dtype=values.dtype)
        output[a, :, b] = values
        output[b, :, a] = values[:, ::-1]
    else:
        output = np.zeros((stations, stations), dtype=values.dtype)
        output[a, b] = values
        output[b, a] = values
    return output


def result_fields(result, catalog, params):
    fs = params["samp_freq"] / params["time_downsample"]
    radius = round(params["maxlag"] * fs)
    stations = len(catalog.stations)
    source, receiver = np.triu_indices(stations, 1)
    dx = float(np.median(np.diff(catalog.distances_m)))
    return {
        "data": dense_pairs(result["pairs"], stations),
        "dt": 1 / fs,
        "dx": dx,
        "dshot": dx,
        "profileX": catalog.distances_m,
        "station_id": np.asarray(catalog.station_ids, dtype=object),
        "pair_source_index": source,
        "pair_receiver_index": receiver,
        "pair_index_base": 0,
        "lags_s": np.arange(-radius, radius + 1) / fs,
        "valid_mask": dense_pairs(
            np.asarray(result["valid_count"]) > 0, stations, mirror_lag=False
        ),
        "valid_short_window_count": dense_pairs(
            np.asarray(result["short_count"], dtype=np.uint64),
            stations,
            mirror_lag=False,
        ),
        "valid_minute_count": dense_pairs(
            np.asarray(result["valid_count"], dtype=np.uint64),
            stations,
            mirror_lag=False,
        ),
        "computation_version": VERSION,
        "pair_mode": "all_pairs",
        "include_autocorr": False,
        "correlation_definition": "padded_fft_whiten_cross_spectrum",
        "pws_axis": "per_pair_lag",
        "pws_power": 2,
        "weighting": "equal_valid_inputs_each_level",
        "cc_backend": params["cc_backend"],
        "cc_device": params["cc_device"],
        "raw_sampling_rate": params["samp_freq"],
        "time_downsample": params["time_downsample"],
        "parameters_json": json.dumps(params, sort_keys=True, ensure_ascii=False),
    }


class TaskManifest:
    def __init__(self, directory, identity, resume=True):
        self.directory = Path(directory)
        self.path = self.directory / "task_manifest.json"
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.task_id = hashlib.sha256(encoded.encode()).hexdigest()
        if self.path.exists():
            existing = json.loads(self.path.read_text())
            if not resume:
                raise FileExistsError(
                    "Output task already exists; choose a new output directory or enable resume"
                )
            if (
                existing.get("task_id") != self.task_id
                or existing.get("identity") != identity
            ):
                raise ValueError(
                    "Resume rejected: inputs, station order, parameters or computation version differ"
                )
            self.value = existing
        else:
            if self.directory.exists() and any(self.directory.glob("cc_stack_*.mat")):
                raise ValueError(
                    "Existing MAT outputs lack a matching task manifest; choose a new output directory"
                )
            self.value = {
                "task_id": self.task_id,
                "identity": identity,
                "completed": {},
                "status": "running",
            }
            atomic_json(self.path, self.value)

    def complete(self, index, path):
        record = self.value["completed"].get(str(index))
        return bool(
            record
            and record["file"] == path.name
            and path.is_file()
            and digest(path) == record["sha256"]
        )

    def record(self, index, path):
        self.value["completed"][str(index)] = {
            "file": path.name,
            "sha256": digest(path),
        }
        atomic_json(self.path, self.value)

    def finish(self, diagnostics):
        self.value["status"] = "complete"
        self.value["diagnostics"] = diagnostics
        atomic_json(self.path, self.value)
