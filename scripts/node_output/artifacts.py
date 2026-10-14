"""Atomic H5 output (with legacy MAT support), exact pair metadata and checksum-verified task resume."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from node_compute.parameters import VERSION
from obspy.geodetics.base import gps2dist_azimuth
from scipy.io import loadmat, savemat


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


def save_h5(
    path, fields, *, format_name="node_ccf_h5_v1", data_axes="source,lag,receiver"
):
    """One complete time-slice cube, native numeric dtypes and UTF-8 metadata."""

    def write(handle):
        with h5py.File(handle, "w") as file:
            file.attrs["format"] = format_name
            file.attrs["data_axes"] = data_axes
            for key, value in fields.items():
                if key.startswith("__"):
                    continue
                array = np.asarray(value)
                if array.dtype.kind in "OUS":
                    strings = np.asarray(
                        [
                            str(np.asarray(item).reshape(-1)[0])
                            for item in array.reshape(-1)
                        ],
                        dtype=object,
                    ).reshape(array.shape)
                    file.create_dataset(
                        key, data=strings, dtype=h5py.string_dtype("utf-8")
                    )
                else:
                    options = {}
                    if key == "data" and array.ndim == 3:
                        options["chunks"] = (1, array.shape[1], array.shape[2])
                    file.create_dataset(key, data=array, **options)

    _atomic(path, write)


def save_result(path, fields):
    return (
        save_h5(path, fields) if Path(path).suffix == ".h5" else save_mat(path, fields)
    )


def load_result(path, variable_names=None):
    if Path(path).suffix != ".h5":
        return loadmat(path, variable_names=variable_names)
    with h5py.File(path, "r") as file:
        result = {}
        for key in file:
            if variable_names is not None and key not in variable_names:
                continue
            dataset = file[key]
            if h5py.check_string_dtype(dataset.dtype) is not None:
                result[key] = np.asarray(dataset.asstr()[()], dtype=str)
            else:
                result[key] = np.asarray(dataset[()])
        return result


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


@lru_cache(maxsize=8)
def geographic_fields(coordinates):
    latitude, longitude = np.asarray(coordinates, dtype=np.float64).T
    source, receiver = np.triu_indices(len(latitude), 1)
    distance = np.asarray(
        [
            gps2dist_azimuth(latitude[a], longitude[a], latitude[b], longitude[b])[0]
            for a, b in zip(source, receiver, strict=True)
        ]
    )
    return {
        "station_latitude_deg": latitude,
        "station_longitude_deg": longitude,
        "pair_distance_m": distance,
        "station_distance_m": dense_pairs(distance, len(latitude), mirror_lag=False),
        "coordinate_system": "WGS84",
        "geometry_type": "geographic",
    }


def result_fields(result, catalog, params):
    fs = params["samp_freq"] / params["time_downsample"]
    radius = round(params["maxlag"] * fs)
    stations = len(catalog.stations)
    source, receiver = np.triu_indices(stations, 1)
    fields = {
        "data": dense_pairs(result["pairs"], stations),
        "dt": 1 / fs,
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
    if all(s.distance_m is not None for s in catalog.stations):
        dx = float(np.median(np.diff(catalog.distances_m)))
        fields.update(dx=dx, dshot=dx, profileX=catalog.distances_m)
    if all(s.latitude_deg is not None for s in catalog.stations):
        fields.update(
            geographic_fields(
                tuple((s.latitude_deg, s.longitude_deg) for s in catalog.stations)
            )
        )
        fields["gps_metadata_json"] = json.dumps(
            [s.gps_info for s in catalog.stations], ensure_ascii=False, sort_keys=True
        )
    return fields


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
            if self.directory.exists() and any(self.directory.glob("cc_stack_*.*")):
                raise ValueError(
                    "Existing outputs lack a matching task manifest; choose a new output directory"
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
        self.value["output_format"] = path.suffix.lstrip(".")
        self.value["completed"][str(index)] = {
            "file": path.name,
            "sha256": digest(path),
        }
        atomic_json(self.path, self.value)

    def finish(self, diagnostics):
        self.value["status"] = "complete"
        self.value["diagnostics"] = diagnostics
        atomic_json(self.path, self.value)
