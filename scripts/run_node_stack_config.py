#!/usr/bin/env python3
"""Select chronological node CCF files across dates and stack explicitly."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dasqt.features.dispersion.backend.components.ccf.folder_pipeline import (
    load_config_json,
)
from node_output.artifacts import atomic_json, load_result, save_result
from stack_node_cc_mat import stack_window

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def select_files(root, values):
    zone = ZoneInfo(values.get("local_timezone", "Asia/Shanghai"))
    start_date = values.get("start_date")
    end_date = values.get("end_date")
    for date in (start_date, end_date):
        if date is not None:
            datetime.strptime(date, "%Y-%m-%d")
    if start_date and end_date and end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    first, last = values.get("start_time"), values.get("end_time")
    if (first is None) != (last is None):
        raise ValueError("Set both start_time and end_time, or both null")
    if first is not None:
        datetime.strptime(first, "%H:%M")
        if last != "24:00":
            datetime.strptime(last, "%H:%M")
        if not "00:00" <= first < last <= "24:00":
            raise ValueError("Require HH:MM start_time < end_time within one day")
    selected = []
    boundary_skipped = []
    for path in root.rglob(values.get("glob", "cc_stack_*.h5")):
        mat = load_result(path, ["source_start_unix", "source_end_unix"])
        lo, hi = (
            float(np.asarray(mat[k]).reshape(-1)[0])
            for k in ("source_start_unix", "source_end_unix")
        )
        if hi <= lo:
            raise ValueError(f"Invalid input interval: {path}")
        begin, end = datetime.fromtimestamp(lo, zone), datetime.fromtimestamp(hi, zone)
        day = begin.strftime("%Y-%m-%d")
        if start_date and day < start_date or end_date and day > end_date:
            continue
        if first is not None:
            lower = (
                datetime.fromisoformat(f"{day}T{first}")
                .replace(tzinfo=zone)
                .timestamp()
            )
            upper = (
                (
                    datetime.fromisoformat(f"{day}T00:00")
                    .replace(tzinfo=zone)
                    .timestamp()
                    + 86400
                )
                if last == "24:00"
                else datetime.fromisoformat(f"{day}T{last}")
                .replace(tzinfo=zone)
                .timestamp()
            )
            if hi <= lower or lo >= upper:
                continue
            if lo < lower or hi > upper:
                boundary_skipped.append(str(path.resolve()))
                continue
        selected.append(
            {
                "path": str(path.resolve()),
                "local_start": begin.isoformat(),
                "local_end": end.isoformat(),
                "date": day,
                "start_unix": lo,
                "end_unix": hi,
            }
        )
    selected.sort(key=lambda row: (row["start_unix"], row["path"]))
    days = sorted({row["date"] for row in selected})
    count_days = values.get("first_n_days")
    count_files = values.get("stack_count")
    for name, value in (("first_n_days", count_days), ("stack_count", count_files)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or null")
    if count_days is not None:
        selected = [row for row in selected if row["date"] in days[:count_days]]
    if count_files is not None:
        if len(selected) < count_files:
            raise ValueError(
                f"Requested {count_files} files, only {len(selected)} matched"
            )
        selected = selected[:count_files]
    if not selected:
        raise ValueError("No CCF files match the selection")
    for previous, current in itertools.pairwise(selected):
        if current["start_unix"] < previous["end_unix"]:
            raise ValueError(
                "Selected intervals overlap: narrow input_dir or glob to one calculation task"
            )
    return selected, boundary_skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/stack_config.jsonc"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Save selected file manifest without computing",
    )
    parser.add_argument("--input-dir", type=Path, help="Override recursive input root")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--start-date", help="Inclusive first date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive last date, YYYY-MM-DD")
    parser.add_argument("--start-time", help="Daily start, HH:MM")
    parser.add_argument("--end-time", help="Daily end, HH:MM")
    parser.add_argument("--first-n-days", type=int, help="First N matching dates")
    parser.add_argument("--stack-count", type=int, help="First N matching files")
    args = parser.parse_args()
    values = load_config_json(args.config)
    for key in (
        "input_dir",
        "output_dir",
        "start_date",
        "end_date",
        "start_time",
        "end_time",
        "first_n_days",
        "stack_count",
    ):
        value = getattr(args, key)
        if value is not None:
            values[key] = str(value.resolve()) if isinstance(value, Path) else value

    def path(value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

    output = path(values["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    rows, skipped = select_files(path(values["input_dir"]), values)
    manifest = {
        "configuration": values,
        "files": rows,
        "selected_file_count": len(rows),
        "selected_day_count": len({r["date"] for r in rows}),
        "boundary_files_excluded": skipped,
        "status": "selected",
    }
    atomic_json(output / "selection.json", manifest)
    LOG.info(
        "Selected %d files across %d days; excluded %d boundary files; manifest: %s",
        len(rows),
        manifest["selected_day_count"],
        len(skipped),
        output / "selection.json",
    )
    if args.list_only:
        return
    fields = stack_window(
        [Path(r["path"]) for r in rows],
        stack_method=values.get("stack_method", "pws"),
        cc_backend=values.get("cc_backend", "cpp_cuda"),
        cc_device=values.get("cc_device", "cuda"),
        allow_multi_day=True,
        geometry_tolerance_m=float(values.get("geometry_tolerance_m", 10)),
    )
    fields["selection_json"] = json.dumps(manifest, ensure_ascii=False)
    final = output / "stack_final.h5"
    save_result(final, fields)
    checked = load_result(final)
    data = checked["data"]
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite final CCF")
    np.testing.assert_array_equal(data, data.transpose(2, 1, 0)[:, ::-1, :])
    manifest.update(
        status="complete",
        output_file=str(final),
        valid_pairs=int(checked["valid_mask"].sum() // 2),
    )
    atomic_json(output / "selection.json", manifest)
    LOG.info("Stack complete: %s", final)


if __name__ == "__main__":
    main()
