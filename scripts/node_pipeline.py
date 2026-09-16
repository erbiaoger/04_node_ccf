"""Two-level node SAC CCF pipeline.

The reader supplies CSV-ordered node traces while the correlation and stacking
kernels are reused from the DAS implementation.  In the default ``preload``
mode, the common time range is loaded once into RAM, then short windows within
each ``minute_stack_s`` block (60 s by default) are batched for the configured
backend.  Consecutive minute blocks are then stacked in groups of ``save_every``
(30 by default) and written as DAS-compatible ``.mat`` files.

The output ``data`` array has shape ``(source, lag, receiver)``.  For
``pair_mode=all_pairs`` only the upper-triangle pairs are computed; the lower
triangle is filled by reversing the lag axis.  The diagonal is zero by default
and is computed only when ``include_autocorr`` is enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from dasqt.features.dispersion.backend.components.backends import (
    cc_backend,
    torch_cc_backend,
)
from dasqt.features.dispersion.backend.components.ccf.folder_pipeline import (
    save_stack_shots_mat,
)
from dasqt.features.dispersion.backend.components.ccf.spectrum_cache import (
    build_spectrum_cache,
)
from dasqt.features.dispersion.backend.components.ccf.stacking import (
    stack_pairwise_cubes,
)
from dasqt.features.dispersion.backend.components.processors.cc_processor import (
    _build_prepro_params,
    compute_cc_shot,
)

try:  # Supports both ``python -m`` and direct CLI execution.
    from .node_reader import NodeCatalog, NodeSACReader, load_node_catalog
except ImportError:  # pragma: no cover - exercised by the shell entrypoint
    from node_reader import NodeCatalog, NodeSACReader, load_node_catalog

LOG = logging.getLogger(__name__)
_GPU_PATH_LOGGED = False
_GPU_STACK_LOGGED = False


@dataclass(frozen=True)
class NodeCCFConfig:
    csv_path: Path
    data_dir: Path
    output_dir: Path
    read_mode: str = "preload"
    include_autocorr: bool = False
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
        first_receiver = 0 if config.include_autocorr else 1
        return (
            np.asarray(
                [i for i in range(len(x)) for j in range(i + first_receiver, len(x))]
            ),
            np.asarray(
                [j for i in range(len(x)) for j in range(i + first_receiver, len(x))]
            ),
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
        receiver_start = (
            source_index if params.get("include_autocorr", False) else source_index + 1
        )
        for receiver_index in range(receiver_start, nchan):
            trace = np.asarray(gather[:, receiver_index], dtype=np.float32)
            output[source_index, receiver_index, :] = trace
            if receiver_index != source_index:
                output[receiver_index, source_index, :] = trace[::-1]
    return output


def _torch_stack_pairwise_cubes(cubes, method: str):
    """Stack ``(window, source, receiver, lag)`` cubes on the accelerator."""
    import torch

    if cubes.ndim != 4:
        raise ValueError(f"Torch pairwise cubes must be 4D, got {tuple(cubes.shape)}")
    method = str(method).lower()
    if method not in {"linear", "pws"}:
        raise ValueError(
            "GPU pairwise stacking supports linear and pws; "
            f"got stacking_method={method!r}"
        )
    # The established DAS stacker treats each source gather independently and
    # flattens (lag, receiver) while calculating PWS.  A 4D batch preserves
    # that exact layout while running all sources in one Torch call.
    gathers = cubes.permute(1, 0, 3, 2).contiguous()
    stacked = torch_cc_backend.stack_chunks(
        gathers,
        method,
        device=str(cubes.device),
        dtype="float64" if cubes.dtype == torch.float64 else "float32",
    ).permute(0, 2, 1)
    # Match DAS ``stack_cc`` post-processing: normalize each receiver trace
    # along lag, then remove its lag mean.
    scale = torch.amax(torch.abs(stacked), dim=-1, keepdim=True)
    stacked = stacked / torch.where(scale == 0, torch.ones_like(scale), scale)
    return stacked - torch.mean(stacked, dim=-1, keepdim=True)


def _compute_dense_gather_batched(
    data: np.ndarray,
    meta: dict,
    params: dict,
    *,
    include_autocorr: bool,
) -> np.ndarray:
    """Compute one minute gather with one GPU prewhitening pass.

    The DAS spectrum cache preprocesses each short window once, keeps the
    spectra on the accelerator, and correlates only the upper-triangle pairs.
    The lower triangle is filled by lag reversal.  This avoids repeating the
    same receiver preprocessing for every source node.
    """
    import torch

    if (
        str(params.get("_runtime_cc_backend", params.get("cc_backend", "cpu")))
        != "torch"
    ):
        return _compute_dense_gather(data, meta, params)
    global _GPU_PATH_LOGGED
    if not _GPU_PATH_LOGGED:
        LOG.info(
            "GPU pairwise correlation active: device=%s, short_windows=%d, stations=%d",
            params.get("_runtime_cc_device", params.get("cc_device", "auto")),
            int(data.shape[0]),
            int(data.shape[1]),
        )
        _GPU_PATH_LOGGED = True

    nchan = int(data.shape[1])
    prepro = _build_prepro_params(meta, params, np.arange(nchan, dtype=int))
    chunk_npts = int(prepro["npts_chunk"])
    nwin = int(data.shape[0] // chunk_npts)
    if nwin <= 0:
        raise ValueError("Data length is too short for one correlation window")
    data = np.asarray(
        data[: nwin * chunk_npts], dtype=params.get("cc_dtype", "float32")
    )
    spectra, valid_masks, nfft = build_spectrum_cache(
        data,
        prepro,
        mm=nwin,
        chunk_npts=chunk_npts,
        max_over_std=float(prepro.get("max_over_std", 1.0e30)),
    )
    if not all(np.all(np.asarray(mask)) for mask in valid_masks):
        invalid = sum(int(np.count_nonzero(~np.asarray(mask))) for mask in valid_masks)
        LOG.error("GPU preprocessing rejected %d window/channel values", invalid)
        if str(params.get("_runtime_cc_device", "cpu")) in {"cuda", "mps"}:
            raise RuntimeError(
                "GPU CCF refused to fall back to CPU because preprocessing found "
                f"{invalid} invalid window/channel values"
            )
        return _compute_dense_gather(data, meta, params)

    batch_chunks = max(int(params.get("cc_batch_chunks", 8)), 1)
    cube_batches = []
    for batch_start in range(0, nwin, batch_chunks):
        batch_stop = min(nwin, batch_start + batch_chunks)
        spectra_batch = torch.stack(spectra[batch_start:batch_stop], dim=0)
        batch_size = batch_stop - batch_start
        pair_source, pair_receiver = np.triu_indices(
            nchan, k=0 if include_autocorr else 1
        )
        if pair_source.size == 0:
            raise ValueError("No station pairs were selected")
        source_spectra = spectra_batch[:, pair_source, :]
        receiver_spectra = spectra_batch[:, pair_receiver, :]
        with torch.inference_mode():
            pair_spectra = torch.conj(source_spectra) * receiver_spectra
            full_spectra = torch_cc_backend._hermitian_complete(pair_spectra, nfft)
            full_spectra[..., 0] = 0
            corr_device = torch.real(
                torch.fft.ifftshift(
                    torch.fft.ifft(full_spectra, n=nfft, dim=-1), dim=-1
                )
            )
        lag_axis = np.arange(-nfft // 2, nfft // 2) / float(prepro["samp_freq"])
        lag_indices = np.where(np.abs(lag_axis) <= float(prepro["maxlag"]))[0]
        corr = corr_device[..., lag_indices]
        pair_source_t = torch.as_tensor(pair_source, device=corr.device)
        pair_receiver_t = torch.as_tensor(pair_receiver, device=corr.device)
        cube = torch.zeros(
            (batch_size, nchan, nchan, int(corr.shape[-1])),
            dtype=corr.dtype,
            device=corr.device,
        )
        cube[:, pair_source_t, pair_receiver_t, :] = corr
        cross_pair = pair_source != pair_receiver
        cross_pair_t = torch.as_tensor(cross_pair, device=corr.device)
        cube[:, pair_receiver_t[cross_pair_t], pair_source_t[cross_pair_t], :] = corr[
            :, cross_pair_t, :
        ].flip(-1)
        cube_batches.append(cube)
        del spectra_batch
    if not cube_batches:
        raise ValueError("No station pairs were selected")
    stacked = _torch_stack_pairwise_cubes(
        torch.cat(cube_batches, dim=0), str(params.get("stacking_method", "pws"))
    )
    if stacked.device.type == "cuda":
        torch.cuda.synchronize(stacked.device)
    return stacked.detach().cpu().numpy().astype(np.float32, copy=False)


def _stack_gathers(gathers: list[np.ndarray], dt: float, params: dict) -> np.ndarray:
    global _GPU_STACK_LOGGED
    runtime_device = str(params.get("_runtime_cc_device", "cpu"))
    if (
        str(params.get("_runtime_cc_backend", params.get("cc_backend", "cpu")))
        == "torch"
        and runtime_device != "cpu"
    ):
        import torch

        cubes = torch.as_tensor(
            np.asarray(gathers), device=runtime_device, dtype=torch.float32
        )
        stacked = _torch_stack_pairwise_cubes(
            cubes, str(params.get("stacking_method", "pws"))
        )
        if stacked.device.type == "cuda":
            torch.cuda.synchronize(stacked.device)
        if not _GPU_STACK_LOGGED:
            LOG.info(
                "GPU pairwise stacking active: device=%s, cubes=%s, method=%s",
                runtime_device,
                tuple(cubes.shape),
                params.get("stacking_method", "pws"),
            )
            _GPU_STACK_LOGGED = True
        return stacked.detach().cpu().numpy().astype(np.float32, copy=False)
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
    LOG.info(
        "selected %d stations, computing %d unique station pairs%s",
        len(catalog.stations),
        len(source),
        " including autocorrelation" if config.include_autocorr else "",
    )
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
        "include_autocorr": config.include_autocorr,
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
    runtime = cc_backend.diagnose_runtime(
        str(cc_params.get("cc_backend", "auto")),
        str(cc_params.get("cc_device", "auto")),
        config.dtype,
    )
    cc_params["_runtime_cc_backend"] = runtime.actual_backend
    cc_params["_runtime_cc_device"] = runtime.actual_device
    LOG.info(
        "CC runtime backend=%s device=%s (requested %s/%s)",
        runtime.actual_backend,
        runtime.actual_device,
        runtime.requested_backend,
        runtime.requested_device,
    )
    if runtime.actual_backend == "torch" and runtime.actual_device == "cuda":
        import torch

        LOG.info(
            "Torch CUDA device=%d: %s (capability %s); CPU fallback is disabled",
            torch.cuda.current_device(),
            torch.cuda.get_device_name(),
            torch.cuda.get_device_capability(),
        )
    if (
        str(cc_params.get("cc_device", "auto")) == "cuda"
        and runtime.actual_device != "cuda"
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable; check the DAS Python environment and NVIDIA driver"
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
                "include_autocorr": config.include_autocorr,
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
            full_data = resample_poly(full_data, 1, time_downsample, axis=0).astype(
                config.dtype, copy=False
            )
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
                    pieces.append(full_data[start_index : start_index + window_npts, :])
                # Concatenating independent windows lets the established DAS
                # kernel batch them without changing the configured step/overlap.
                minute_data = np.concatenate(pieces, axis=0)
                minute_gather = _compute_dense_gather_batched(
                    minute_data,
                    meta,
                    cc_params,
                    include_autocorr=config.include_autocorr,
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
