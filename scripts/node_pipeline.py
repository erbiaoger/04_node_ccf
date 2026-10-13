"""Two-level, masked node-pair pipeline with strict CUDA and bounded SAC I/O.

Run via cli.py or run_node_ccf.sh; configuration and CLI examples are documented
in ../README.md. Outputs are atomic cc_stack_XXXX_nN.mat and task_manifest.json.
Window starts are global integer sample indices; overlap is never concatenated
on the host. Scientific kernels live in node_compute and the CUDA extension.
"""

from __future__ import annotations

import logging
import resource
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from node_compute.backend import create_session
from node_compute.parameters import VERSION, validate
from node_compute.schedule import block_bounds, groups
from node_io import NodeSACReader, load_node_catalog
from node_output.artifacts import TaskManifest, result_fields, save_mat

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeCCFConfig:
    csv_path: Path
    data_dir: Path
    output_dir: Path
    cc_params: dict | None = None
    start_utc: str | None = None
    end_utc: str | None = None


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
    return "preload" if estimate <= available * 0.5 else "block"


def prepare(config):
    catalog = load_node_catalog(config.csv_path, config.data_dir)
    params = validate(config.cc_params or {}, catalog.sampling_rate)
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


def run_node_ccf(config):
    catalog, params, start, end, total = prepare(config)
    stations = len(catalog.stations)
    reader = NodeSACReader(catalog)
    mode = choose_read_mode(params, total, stations)
    # Performance knobs may change on resume; scientific/schedule parameters may not.
    excluded = {
        "read_mode",
        "resume",
        "cc_batch_chunks",
        "cc_pair_batch",
        "cc_memory_safety_fraction",
        "cc_memory_budget_bytes",
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
    full_data = full_valid = None
    started = time.perf_counter()
    read_seconds = save_seconds = 0.0
    completed = skipped = 0
    for group in groups(total, reader.fs, params):
        path = (
            config.output_dir / f"cc_stack_{group.index:04d}_n{group.minute_count}.mat"
        )
        if task.complete(group.index, path):
            skipped += 1
            LOG.info("Verified resume: %s", path.name)
            continue
        lo, hi = block_bounds(group, total, reader.fs, params)
        before = time.perf_counter()
        if mode == "preload":
            if full_data is None:
                LOG.info(
                    "Preloading %.3f GiB waveform + %.3f GiB validity",
                    total * stations * 4 / 1024**3,
                    total * stations / 1024**3,
                )
                full_data, full_valid = reader.read_samples(start, total)
            data, valid = full_data[lo:hi], full_valid[lo:hi]
        else:
            data, valid = reader.read_samples(start + lo / reader.fs, hi - lo)
        read_seconds += time.perf_counter() - before
        session.add_block(data, valid, group.starts - lo, group.minute_ids)
        result = session.finish_group()
        fields = result_fields(result, catalog, params)
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
        )
        before = time.perf_counter()
        save_mat(path, fields)
        task.record(group.index, path)
        save_seconds += time.perf_counter() - before
        completed += 1
        LOG.info(
            "Saved %s: %d windows, %d/%d valid pairs",
            path.name,
            len(group.starts),
            int((result["valid_count"] > 0).sum()),
            stations * (stations - 1) // 2,
        )
        session.clear_group()
    diagnostics = session.diagnostics()
    # ru_maxrss is bytes on macOS and KiB on Linux.
    import sys

    diagnostics.update(
        read_seconds=read_seconds,
        save_seconds=save_seconds,
        wall_seconds=time.perf_counter() - started,
        read_mode=mode,
        groups_computed=completed,
        groups_resumed=skipped,
        peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    task.finish(diagnostics)
    LOG.info("Node calculation complete: %s", config.output_dir)
    return config.output_dir
