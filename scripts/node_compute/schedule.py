"""Integer sample window/group scheduling shared by all readers/backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from node_io.catalog import sample_index


@dataclass(frozen=True)
class TimeGroup:
    index: int
    begin: int
    end: int
    starts: np.ndarray
    minute_ids: np.ndarray
    minute_count: int


def groups(total: int, raw_fs: float, params: dict):
    factor = int(params.get("time_downsample", 1))
    fs = raw_fs / factor
    n = sample_index(float(params["cc_len"]), fs, "cc_len")
    step = sample_index(float(params["cc_step_s"]), fs, "cc_step_s")
    minute = sample_index(float(params["minute_stack_s"]), fs, "minute_stack_s")
    size = minute * int(params["save_every"])
    samples = (total + factor - 1) // factor
    if min(n, step, minute, size) < 1 or n < 2 or samples < n:
        raise ValueError("Invalid schedule or no complete short window")
    # Generate only the current group's starts; memory independent of duration.
    for index, begin in enumerate(range(0, samples, size), 1):
        end = min(begin + size, samples)
        first = ((begin + step - 1) // step) * step
        stop = min(end, samples - n + 1)
        starts = np.arange(first, max(first, stop), step, dtype=np.int64)
        yield TimeGroup(
            index,
            begin * factor,
            min(end * factor, total),
            starts * factor,
            starts // minute,
            (end - begin + minute - 1) // minute,
        )


def block_bounds(group: TimeGroup, total: int, raw_fs: float, params: dict):
    factor = int(params.get("time_downsample", 1))
    n = sample_index(float(params["cc_len"]), raw_fs / factor, "cc_len")
    halo = 20 * factor if factor > 1 else 0
    if len(group.starts):
        lo = max(0, int(group.starts[0]) - halo)
        hi = min(total, int(group.starts[-1]) + (n - 1) * factor + halo + 1)
    else:
        lo, hi = group.begin, group.end
    return lo, hi
