#!/usr/bin/env python3
"""Lossless conversion of completed node time slices from MAT to H5.

Verifies original manifest hashes and every numeric dataset after conversion.
Updates the manifest only after an H5 file passes verification. Legacy MAT files
are retained in legacy_mat/; no scientific computation is repeated. No active writer may share
this task directory during migration.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from node_output.artifacts import atomic_json, digest, load_result, save_h5

LOG = logging.getLogger(__name__)


def migrate(directory, workers=4):
    directory = Path(directory)
    manifest_path = directory / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a completed, inactive task can be migrated")

    def convert(item):
        index, record = item
        source = directory / record["file"]
        if digest(source) != record["sha256"]:
            raise ValueError(f"Source checksum differs: {source}")
        if source.suffix == ".h5":
            return index, record
        fields = load_result(source)
        target = source.with_suffix(".h5")
        save_h5(target, fields)
        restored = load_result(target)
        for key, value in fields.items():
            if key.startswith("__"):
                continue
            array = np.asarray(value)
            if array.dtype.kind not in "OUS":
                np.testing.assert_array_equal(
                    array, restored[key], err_msg=f"{source.name}: {key}"
                )
            else:
                expected = [
                    str(np.asarray(v).reshape(-1)[0]) for v in array.reshape(-1)
                ]
                assert expected == [str(v) for v in restored[key].reshape(-1)]
        return index, {"file": target.name, "sha256": digest(target)}

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="h5-convert"
    ) as pool:
        for count, (index, record) in enumerate(
            pool.map(convert, list(manifest["completed"].items())), 1
        ):
            manifest["completed"][index] = record
            if count % 16 == 0:
                atomic_json(manifest_path, manifest)
                LOG.info(
                    "Verified H5 time slices: %d/%d", count, len(manifest["completed"])
                )
    manifest["output_format"] = "h5"
    atomic_json(manifest_path, manifest)
    backup = directory / "legacy_mat"
    for record in manifest["completed"].values():
        source = (directory / record["file"]).with_suffix(".mat")
        if source.exists():
            backup.mkdir(exist_ok=True)
            target = backup / source.name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite archived MAT: {target}")
            source.rename(target)
    return len(manifest["completed"])


def main():
    parser = argparse.ArgumentParser(
        description="逐时间片段无损转换MAT为H5并更新续算清单"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    LOG.info(
        "Converted and verified %d complete time slices",
        migrate(args.input_dir, args.workers),
    )


if __name__ == "__main__":
    main()
