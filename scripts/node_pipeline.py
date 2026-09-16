"""Two-level node SAC CCF pipeline.

The reader supplies CSV-ordered node traces while the correlation and stacking
kernels are reused from the DAS implementation.  In the default ``preload``
mode, the common time range is loaded once into RAM, then short windows within
each ``minute_stack_s`` block (60 s by default) are batched for the configured
backend.  Consecutive minute blocks are then stacked in groups of ``save_every``
(30 by default) and written as DAS-compatible ``.mat`` files.

The output ``data`` array has shape ``(source, lag, receiver)``.  For
``pair_mode=all_pairs`` only the upper-triangle pairs are computed; the lower
triangle is filled by reversing the lag axis, and the diagonal contains the
autocorrelation of each node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from dasqt.features.dispersion.backend.components.ccf.folder_pipeline import (
    save_stack_shots_mat,
)
from dasqt.features.dispersion.backend.components.ccf.stacking import (
    stack_pairwise_cubes,
)
from dasqt.features.dispersion.backend.components.processors.cc_processor import (
    compute_cc_shot,
)

try:  # Supports both ``python -m`` and direct CLI execution.
    from .node_reader import NodeCatalog, NodeSACReader, load_node_catalog
except ImportError:  # pragma: no cover - exercised by the shell entrypoint
    from node_reader import NodeCatalog, NodeSACReader, load_node_catalog

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeCCFConfig:
    csv_path: Path
    data_dir: Path
    output_dir: Path
    read_mode: str = "preload"
    pair_mode: str = "all_pairs"
    cc_len: float = 5.0
    step_s: float = 5.0
    minute_stack_s: float = 60.0
    maxlag: float = 0.4
    offset_m: float = 50.0
    dshot_m: float = 2.5
    dtype: str = "float32"
    start_utc: str | None = None
    end_utc: str | None = None
    cc_params: dict | None = None
    save_every: int = 30  # number of one-minute blocks per MAT file


def _pair_indices(
    catalog: NodeCatalog, config: NodeCCFConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = catalog.distances_m
    if config.pair_mode == "all_pairs":
        return (
            np.asarray([i for i in range(len(x)) for j in range(i, len(x))]),
            np.asarray([j for i in range(len(x)) for j in range(i, len(x))]),
            np.arange(len(x)),
        )
    if config.pair_mode != "sliding":
        raise ValueError("pair_mode must be all_pairs or sliding")
    sources, receivers = [], []
    shot_positions = np.arange(x[0], x[-1] + 1e-9, config.dshot_m)
    used_sources: set[int] = set()
    for shot_position in shot_positions:
        source = int(np.argmin(np.abs(x - shot_position)))
        if source in used_sources:
            continue
        used_sources.add(source)
        selected = np.flatnonzero(
            (x >= x[source]) & (x <= x[source] + config.offset_m + 1e-9)
        )
        sources.extend([source] * len(selected))
        receivers.extend(selected.tolist())
    return (
        np.asarray(sources, dtype=int),
        np.asarray(receivers, dtype=int),
        np.arange(len(x)),
    )


def _compute_dense_gather(data: np.ndarray, meta: dict, params: dict) -> np.ndarray:
    """Compute source×receiver gathers, including the diagonal autocorrelation."""
    nchan = data.shape[1]
    output = None
    for source_index in range(nchan):
        gather = compute_cc_shot(
            data, meta, np.arange(data.shape[1]), iiS=source_index, cc_par=params
        )
        if output is None:
            output = np.empty((nchan, nchan, gather.shape[0]), dtype=np.float32)
        for receiver_index in range(source_index, nchan):
            trace = np.asarray(gather[:, receiver_index], dtype=np.float32)
            output[source_index, receiver_index, :] = trace
            if receiver_index != source_index:
                output[receiver_index, source_index, :] = trace[::-1]
    return output


def _stack_gathers(gathers: list[np.ndarray], dt: float, params: dict) -> np.ndarray:
    return np.asarray(
        stack_pairwise_cubes(
            np.asarray(gathers),
            dt=dt,
            smethod=str(params.get("stacking_method", "pws")),
            params=params,
        ),
        dtype=np.float32,
    )


def run_node_ccf(config: NodeCCFConfig) -> Path:
    """Run the two-level time-block calculation and write grouped MAT output."""
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    catalog = load_node_catalog(config.csv_path, config.data_dir)
    if config.minute_stack_s <= 0:
        raise ValueError("minute_stack_s must be positive")
    if config.save_every <= 0:
        raise ValueError("save_every must be a positive number of minutes")
    if config.cc_len <= 0 or config.step_s <= 0:
        raise ValueError("cc_len and step_s must be positive")
    read_mode = str(config.read_mode).strip().lower()
    if read_mode not in {"preload", "window"}:
        raise ValueError("read_mode must be preload or window")
    source, receiver, _ = _pair_indices(catalog, config)
    if config.pair_mode != "all_pairs":
        raise ValueError("MAT output currently supports pair_mode=all_pairs only")
    reader = NodeSACReader(catalog, config.dtype)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dt": 1.0 / reader.fs,
        "dx": float(np.median(np.diff(catalog.distances_m))),
        "profileX": catalog.distances_m,
    }
    cc_params = {
        "cc_len": config.cc_len,
        "maxlag": config.maxlag,
        "freqmin": 5.0,
        "freqmax": 40.0,
        "time_norm": "rma",
        "freq_norm": "rma",
        "cc_method": "xcorr",
        "cc_backend": "cpu",
        "cc_dtype": config.dtype,
        "max_over_std": 1.0e30,
        "smooth_N": 5,
        "smoothspect_N": 5,
        "fk_enabled": False,
    }
    if config.cc_params:
        cc_params.update(config.cc_params)
    cc_params.update(
        {"cc_len": config.cc_len, "maxlag": config.maxlag, "cc_dtype": config.dtype}
    )
    time_downsample = max(int(cc_params.get("time_downsample", 1)), 1)
    if time_downsample > 1:
        LOG.info(
            "applying DAS config time_downsample=%d with anti-alias filtering",
            time_downsample,
        )
    start = (
        catalog.start_timestamp
        if config.start_utc is None
        else datetime.fromisoformat(config.start_utc).astimezone(UTC).timestamp()
    )
    end = (
        catalog.end_timestamp
        if config.end_utc is None
        else datetime.fromisoformat(config.end_utc).astimezone(UTC).timestamp()
    )
    if start < catalog.start_timestamp or end > catalog.end_timestamp or start >= end:
        raise ValueError("Requested UTC range is outside the common catalog interval")
    minute_block: list[np.ndarray] = []
    minute_start = None
    group_block: list[np.ndarray] = []
    group_start = None
    group_short_count = 0
    file_index = 0

    def save_group(
        values: list[np.ndarray], start_time: float, end_time: float, short_count: int
    ) -> None:
        nonlocal file_index
        file_index += 1
        stacked = _stack_gathers(values, meta["dt"], cc_params).transpose(0, 2, 1)
        output = config.output_dir / f"cc_stack_{file_index:04d}_n{len(values)}.mat"
        save_stack_shots_mat(
            output,
            stacked,
            dt=meta["dt"],
            dx=meta["dx"],
            dshot=config.dshot_m,
            cc_meta={
                "station_id": np.asarray(catalog.station_ids, dtype=object),
                "profileX": catalog.distances_m,
                "pair_source_index": source,
                "pair_receiver_index": receiver,
                "source_start_unix": start_time,
                "source_end_unix": end_time,
                "actual_minute_count": len(values),
                "actual_short_window_count": short_count,
                "minute_stack_s": config.minute_stack_s,
                "save_every": config.save_every,
                "cc_len": config.cc_len,
                "step_s": config.step_s,
                "read_mode": config.read_mode,
                "cc_backend": str(cc_params.get("cc_backend", "auto")),
                "cc_device": str(cc_params.get("cc_device", "auto")),
                "pair_mode": config.pair_mode,
            },
        )
        LOG.info(
            "saved %s (%d one-minute segments, %d short windows)",
            output.name,
            len(values),
            short_count,
        )

    def append_minute(
        minute_start_time: float,
        minute_gather: np.ndarray,
        short_count: int,
        end_time: float,
    ) -> None:
        nonlocal group_block, group_start, group_short_count
        group_block.append(minute_gather)
        group_short_count += short_count
        if group_start is None:
            group_start = minute_start_time
        if len(group_block) >= config.save_every:
            save_group(group_block, group_start, end_time, group_short_count)
            group_block, group_start, group_short_count = [], None, 0

    if read_mode == "preload":
        # Read the complete common interval once.  The following minute loops
        # only slice this in-memory array, so SAC I/O is not repeated.
        full_data, _ = reader.read_window(start, end - start)
        LOG.info(
            "preloaded %.3f GiB (%d samples x %d stations) into RAM",
            full_data.nbytes / 1024**3,
            full_data.shape[0],
            full_data.shape[1],
        )
        if time_downsample > 1:
            full_data = resample_poly(
                full_data, 1, time_downsample, axis=0
            ).astype(config.dtype, copy=False)
            meta["dt"] = time_downsample / reader.fs
        else:
            meta["dt"] = 1.0 / reader.fs
        meta["profileX"] = catalog.distances_m
        effective_fs = 1.0 / meta["dt"]
        window_npts = round(config.cc_len * effective_fs)
        cursor = start
        while cursor + config.cc_len <= end + 1e-6:
            block_end = min(cursor + config.minute_stack_s, end)
            window_starts: list[float] = []
            window_cursor = cursor
            while (
                window_cursor < block_end - 1e-6
                and window_cursor + config.cc_len <= end + 1e-6
            ):
                window_starts.append(window_cursor)
                window_cursor += config.step_s
            if window_starts:
                pieces = []
                for window_start in window_starts:
                    start_index = round((window_start - start) * effective_fs)
                    pieces.append(
                        full_data[start_index : start_index + window_npts, :]
                    )
                # Concatenating independent windows lets the established DAS
                # kernel batch them without changing the configured step/overlap.
                minute_data = np.concatenate(pieces, axis=0)
                minute_gather = _compute_dense_gather(
                    minute_data, meta, cc_params
                )
                append_minute(cursor, minute_gather, len(window_starts), block_end)
            cursor = block_end
    else:
        # Low-memory compatibility path: read one short window at a time.
        for window_start, (data, _) in reader.iter_windows(
            start, end, config.cc_len, config.step_s
        ):
            if time_downsample > 1:
                data = resample_poly(data, 1, time_downsample, axis=0).astype(
                    config.dtype, copy=False
                )
                meta["dt"] = time_downsample / reader.fs
            else:
                meta["dt"] = 1.0 / reader.fs
            meta["profileX"] = catalog.distances_m
            if (
                minute_start is not None
                and window_start >= minute_start + config.minute_stack_s
            ):
                minute_gather = _stack_gathers(minute_block, meta["dt"], cc_params)
                append_minute(
                    minute_start, minute_gather, len(minute_block), window_start
                )
                minute_block, minute_start = [], None
            minute_block.append(_compute_dense_gather(data, meta, cc_params))
            if minute_start is None:
                minute_start = window_start
    if minute_block:
        append_minute(
            minute_start,
            _stack_gathers(minute_block, meta["dt"], cc_params),
            len(minute_block),
            end,
        )
    if group_block:
        save_group(group_block, group_start, end, group_short_count)
    if file_index == 0:
        raise ValueError("No complete time windows were available for the catalog")
    return config.output_dir
