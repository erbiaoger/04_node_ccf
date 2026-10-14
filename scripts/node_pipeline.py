"""Two-level, masked node-pair pipeline with strict CUDA and bounded SAC I/O.

Run via cli.py or run_node_ccf.sh; configuration and CLI examples are documented
in ../README.md. Outputs are atomic cc_stack_XXXX_nN.h5 and task_manifest.json.
Window starts are global integer sample indices; overlap is never concatenated
on the host. Scientific kernels live in node_compute and the CUDA extension.
"""

from __future__ import annotations

import logging
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from zoneinfo import ZoneInfo

from node_compute.backend import create_session
from node_compute.parameters import VERSION, validate
from node_compute.schedule import block_bounds, groups
from node_io import NodeSACReader, load_node_catalog
from node_output.artifacts import TaskManifest, result_fields, save_result

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeCCFConfig:
    csv_path: Path
    data_dir: Path
    output_dir: Path
    cc_params: dict | None = None
    start_utc: str | None = None
    end_utc: str | None = None
    station_records: tuple | None = None
    station_csv_sha256: str | None = None


def _utc(value, fallback):
    if value is None:
        return fallback
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("UTC range must include a timezone, e.g. 2026-07-09T11:00:00Z")
    return result.astimezone(UTC).timestamp()


def available_memory():
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        import os

        if Path("/proc/meminfo").exists():
            fields = dict(
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
            return int(fields["MemAvailable"].split()[0]) * 1024
        # macOS fallback: free+inactive pages, obtained without changing state.
        import subprocess

        output = subprocess.check_output(["vm_stat"], text=True)
        page = int(output.splitlines()[0].split("page size of ")[1].split()[0])
        fields = dict(
            line.split(":", 1) for line in output.splitlines()[1:] if ":" in line
        )
        free = sum(
            int(fields.get(key, "0").strip().rstrip("."))
            for key in ("Pages free", "Pages inactive")
        )
        return max(page * free, os.sysconf("SC_PAGE_SIZE"))


def choose_read_mode(params, total, stations, available=None):
    if params["read_mode"] != "auto":
        return params["read_mode"]
    if params.get("read_hours", 0):
        return "block"
    # raw float32 + mask + a full conversion-copy allowance + current block.
    group = round(
        params["samp_freq"]
        * (params["minute_stack_s"] * params["save_every"] + params["cc_len"])
    )
    estimate = total * stations * 9 + group * stations * 9
    available = available_memory() if available is None else available
    LOG.info(
        "Auto reader: estimated peak %.3f GiB; available %.3f GiB",
        estimate / 1024**3,
        available / 1024**3,
    )
    return "preload" if estimate <= min(available * 0.5, 512 * 1024**2) else "block"


def prepare(config):
    catalog = load_node_catalog(
        config.csv_path,
        config.data_dir,
        station_records=config.station_records,
        sampling_rate_hz=(config.cc_params or {}).get("sac_sampling_rate_hz"),
        start_timestamp=_utc(config.start_utc, None),
        end_timestamp=_utc(config.end_utc, None),
    )
    params = validate(config.cc_params or {}, catalog.sampling_rate)
    if params.get("cc_enhance_method") == "tsi" and len(catalog.stations) < 3:
        raise ValueError("Node TSI requires at least three stations")
    start = _utc(config.start_utc, catalog.start_timestamp)
    end = _utc(config.end_utc, catalog.end_timestamp)
    fs = catalog.sampling_rate
    if (
        start < catalog.start_timestamp - 0.01 / fs
        or end > catalog.end_timestamp + 0.01 / fs
        or end <= start
    ):
        raise ValueError("Requested range lies outside the common SAC outer interval")
    # Human UTC bounds need not fall exactly on the float32 SAC delta grid.
    # Snap task bounds to the nearest existing sample; station grids remain strict.
    begin_index = round((start - catalog.start_timestamp) * fs)
    end_index = round((end - catalog.start_timestamp) * fs)
    snapped_start = catalog.start_timestamp + begin_index / fs
    snapped_end = catalog.start_timestamp + end_index / fs
    if abs(snapped_start - start) * fs > 0.01 or abs(snapped_end - end) * fs > 0.01:
        LOG.info(
            "UTC bounds snapped to nearest sample (start %.6f ms, end %.6f ms)",
            (snapped_start - start) * 1000,
            (snapped_end - end) * 1000,
        )
    start, end, total = snapped_start, snapped_end, end_index - begin_index
    if total < round(params["cc_len"] * fs):
        raise ValueError("Requested interval contains no complete correlation window")
    return catalog, params, start, end, total


def run_node_ccf(config, *, group_validator=None):
    catalog, params, start, end, total = prepare(config)
    stations = len(catalog.stations)
    reader = NodeSACReader(catalog, workers=params["io_workers"])
    mode = choose_read_mode(params, total, stations)
    # Performance knobs may change on resume; scientific/schedule parameters may not.
    excluded = {
        "io_workers",
        "prefetch_blocks",
        "async_save",
        "read_mode",
        "read_hours",
        "resume",
        "cc_batch_chunks",
        "cc_upload_block",
        "cc_reuse_workspace",
        "cc_pair_batch",
        "cc_memory_safety_fraction",
        "cc_memory_budget_bytes",
        "cc_memory_budget_gib",
        "tsi_pair_batch_size",
        "tsi_show_progress",
    }
    identity = {
        "version": VERSION,
        "inputs": catalog.fingerprint(),
        "start": start,
        "end": end,
        "parameters": {k: v for k, v in params.items() if k not in excluded},
    }
    task = TaskManifest(config.output_dir, identity, params["resume"])
    # Instantiate before touching large waveforms: fail fast on unavailable CUDA.
    session = create_session(params, stations)
    LOG.info(
        "Node computation: %d stations, %d pairs, backend=%s/%s, read=%s, duration=%.3f h",
        stations,
        stations * (stations - 1) // 2,
        params["cc_backend"],
        params["cc_device"],
        mode,
        (end - start) / 3600,
    )
    started = time.perf_counter()
    read_seconds = save_seconds = compute_seconds = read_wait_seconds = 0.0
    save_wait_seconds = tsi_seconds = 0.0
    completed = skipped = host_reads = 0

    def pending_groups():
        nonlocal skipped
        for group in groups(total, reader.fs, params):
            path = (
                config.output_dir
                / f"cc_stack_{group.index:04d}_n{group.minute_count}.h5"
            )
            if task.complete(group.index, path):
                skipped += 1
                LOG.info("Verified resume: %s", path.name)
                continue
            lo, hi = block_bounds(group, total, reader.fs, params)
            yield group, path, lo, hi

    output_seconds = params["minute_stack_s"] * params["save_every"]
    cache_groups = max(1, round(params["read_hours"] * 3600 / output_seconds))

    def batch_items():
        if mode == "preload":
            batch = list(pending_groups())
            if batch:
                yield batch
        else:
            for _, batch in groupby(
                pending_groups(), key=lambda item: (item[0].index - 1) // cache_groups
            ):
                yield list(batch)

    def read_cache(batch):
        nonlocal host_reads
        lo = (
            0
            if mode == "preload"
            else min(min(item[2], item[0].begin) for item in batch)
        )
        hi = (
            total
            if mode == "preload"
            else max(max(item[3], item[0].end) for item in batch)
        )
        LOG.info(
            "Reading host cache: groups %04d..%04d, %.3f h, %.3f GiB, %d workers",
            batch[0][0].index,
            batch[-1][0].index,
            (hi - lo) / reader.fs / 3600,
            (hi - lo) * stations * 5 / 1024**3,
            params["io_workers"],
        )
        before = time.perf_counter()
        data, valid = reader.read_samples(start + lo / reader.fs, hi - lo)
        elapsed = time.perf_counter() - before
        host_reads += 1
        LOG.info(
            "Host cache ready: groups %04d..%04d, read %.3fs",
            batch[0][0].index,
            batch[-1][0].index,
            elapsed,
        )
        return batch, lo, data, valid, elapsed

    def cached_batches():
        batches = iter(batch_items())
        first = next(batches, None)
        if first is None:
            return
        span = (
            total
            if mode == "preload"
            else min(
                total,
                round(cache_groups * output_seconds * reader.fs)
                + round(params["cc_len"] * reader.fs)
                + 40 * params["time_downsample"],
            )
        )
        buffers = (
            2 if mode != "preload" and params["prefetch_blocks"] and span < total else 1
        )
        estimate = (
            span * stations * 5 * buffers
            + params["io_workers"] * 9 * 1024**2
            + 512 * 1024**2
        )
        available = available_memory()
        LOG.info(
            "Host cache plan: read_hours=%.3f, up to %d buffers, estimated peak %.3f GiB, available %.3f GiB",
            params["read_hours"],
            buffers,
            estimate / 1024**3,
            available / 1024**3,
        )
        if estimate > available * 0.8:
            raise MemoryError(
                "Requested host caches exceed 80% of available RAM; reduce read_hours or set prefetch_blocks=0"
            )
        if mode == "preload" or not params["prefetch_blocks"]:
            batch = first
            while batch is not None:
                result = read_cache(batch)
                yield (*result, result[-1])
                del result
                batch = next(batches, None)
        else:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="prefetch"
            ) as pool:
                future = pool.submit(read_cache, first)
                try:
                    while future is not None:
                        before = time.perf_counter()
                        result = future.result()
                        wait = time.perf_counter() - before
                        batch = next(batches, None)
                        future = (
                            pool.submit(read_cache, batch)
                            if batch is not None
                            else None
                        )
                        yield (*result, wait)
                        del result
                finally:
                    if future is not None:
                        future.cancel()

    def blocks():
        caches = cached_batches()
        try:
            for batch, cache_lo, cache_data, cache_valid, elapsed, wait in caches:
                for item in batch:
                    _, _, lo, hi = item
                    yield (
                        item,
                        cache_data[lo - cache_lo : hi - cache_lo],
                        cache_valid[lo - cache_lo : hi - cache_lo],
                        elapsed,
                        wait,
                    )
                    elapsed = wait = 0.0
                del cache_data, cache_valid
        finally:
            caches.close()

    def persist(path, fields, index, windows, valid_pairs):
        before = time.perf_counter()
        save_result(path, fields)
        if group_validator is not None:
            group_validator(path, fields)
        task.record(index, path)
        LOG.info(
            "Saved %s: %d windows, %d/%d valid pairs",
            path.name,
            windows,
            valid_pairs,
            stations * (stations - 1) // 2,
        )
        return time.perf_counter() - before

    writer = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="h5-writer")
        if params["async_save"]
        else None
    )
    pending_save = None
    stream = blocks()
    try:
        for (group, path, lo, hi), data, valid, elapsed, wait in stream:
            read_seconds += elapsed
            read_wait_seconds += wait
            before = time.perf_counter()
            session.add_block(data, valid, group.starts - lo, group.minute_ids)
            result = session.finish_group()
            group_compute = time.perf_counter() - before
            compute_seconds += group_compute
            if params["cc_backend"] == "cpp_cuda":
                batch_diag = session.diagnostics()
                LOG.info(
                    "CUDA batch: max_windows=%d, last_windows=%d, last_pairs=%d, batches=%d, max_minute_span=%d",
                    batch_diag["window_batch_max"],
                    batch_diag["window_batch_last"],
                    batch_diag["pair_batch_last"],
                    batch_diag["compute_batches"],
                    batch_diag["minute_span_max"],
                )
            fields = result_fields(result, catalog, params)
            if params.get("cc_enhance_method") == "tsi":
                from node_compute.tsi import enhance_output

                if "station_distance_m" not in fields:
                    coordinates = catalog.distances_m
                    fields["station_distance_m"] = abs(
                        coordinates[:, None] - coordinates[None, :]
                    )
                tsi_started = time.perf_counter()
                fields = enhance_output(fields, group, data, valid, lo, params)
                group_tsi = time.perf_counter() - tsi_started
                tsi_seconds += group_tsi
                compute_seconds += group_tsi
                group_compute += group_tsi
                LOG.info(
                    "Node TSI: %d groups, interval %.1f minutes, %.3fs",
                    fields["tsi_group_count"],
                    params["tsi_files_per_group"] * params["minute_stack_s"] / 60,
                    group_tsi,
                )
            fields.update(
                task_id=task.task_id,
                source_start_unix=start + group.begin / reader.fs,
                source_end_unix=start + group.end / reader.fs,
                data_start_unix=start + lo / reader.fs,
                data_end_unix=start + hi / reader.fs,
                actual_minute_count=group.minute_count,
                actual_short_window_count=len(group.starts),
                minute_stack_s=params["minute_stack_s"],
                save_every=params["save_every"],
                cc_len=params["cc_len"],
                step_s=params["cc_step_s"],
                read_mode=mode,
                source_start_utc=datetime.fromtimestamp(
                    start + group.begin / reader.fs, UTC
                ).isoformat(),
                source_end_utc=datetime.fromtimestamp(
                    start + group.end / reader.fs, UTC
                ).isoformat(),
                source_start_local=datetime.fromtimestamp(
                    start + group.begin / reader.fs, ZoneInfo("Asia/Shanghai")
                ).isoformat(),
                source_end_local=datetime.fromtimestamp(
                    start + group.end / reader.fs, ZoneInfo("Asia/Shanghai")
                ).isoformat(),
                local_timezone="Asia/Shanghai",
            )
            if pending_save is not None:
                before = time.perf_counter()
                save_seconds += pending_save.result()
                save_wait_seconds += time.perf_counter() - before
                pending_save = None
            args = (
                path,
                fields,
                group.index,
                len(group.starts),
                int((result["valid_count"] > 0).sum()),
            )
            if writer is None:
                save_seconds += persist(*args)
            else:
                # At most one detached output cube queued; next CUDA group can run.
                pending_save = writer.submit(persist, *args)
            completed += 1
            LOG.info(
                "Group timing: read %.3fs, exposed read wait %.3fs, CUDA/compute %.3fs",
                elapsed,
                wait,
                group_compute,
            )
            session.clear_group()
            del data, valid, result, fields, args
    finally:
        try:
            stream.close()
            reader.close()
        finally:
            if writer is not None:
                before = time.perf_counter()
                writer.shutdown(wait=True)
                save_wait_seconds += time.perf_counter() - before
    if pending_save is not None:
        before = time.perf_counter()
        save_seconds += pending_save.result()
        save_wait_seconds += time.perf_counter() - before
    diagnostics = session.diagnostics()
    # ru_maxrss is bytes on macOS and KiB on Linux.
    import sys

    diagnostics.update(
        read_seconds=read_seconds,
        read_wait_seconds=read_wait_seconds,
        compute_seconds=compute_seconds,
        **(
            {"tsi_seconds": tsi_seconds}
            if params.get("cc_enhance_method") == "tsi"
            else {}
        ),
        io_workers=params["io_workers"],
        prefetch_blocks=params["prefetch_blocks"],
        save_seconds=save_seconds,
        save_wait_seconds=save_wait_seconds,
        async_save=params["async_save"],
        wall_seconds=time.perf_counter() - started,
        read_mode=mode,
        read_hours=params["read_hours"],
        host_reads=host_reads,
        groups_computed=completed,
        groups_resumed=skipped,
        peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    task.finish(diagnostics)
    LOG.info("Node calculation complete: %s", config.output_dir)
    return config.output_dir
