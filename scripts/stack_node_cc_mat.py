#!/usr/bin/env python3
"""节点 CCF 成品的流式二次叠加，逐对 PWS，默认 C++/CUDA。

用途：顺序读取新的 cc_stack H5，只保留一个输入文件和GPU逐对统计量，
按有效文件等权叠加。有效计数作质量元数据，不用于改变文件权重。
运行：bash examples/04_node_ccf/run_node_ccf_stack.sh
直接运行：项目 .venv 中 uv run --no-sync python
examples/04_node_ccf/scripts/stack_node_cc_mat.py --input-dir outputs/node_ccf
--output-dir outputs/node_ccf_restack --cc-backend cpp_cuda --cc-device cuda
输出：restack_XXXX_nN_idxXXXX-XXXX.h5，保留台站、坐标、有效掩码和输入来源。
参数详见 --help 与下方 argparse help；终端输出输入数量、后端和保存进度。
旧DAS展平PWS H5不能混入新节点结果，计算版本或台站顺序不一致会报错。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from node_compute.backend import create_session
from node_compute.parameters import VERSION, validate
from node_output.artifacts import dense_pairs, digest, save_result
from node_output.artifacts import load_result as loadmat

LOG = logging.getLogger(__name__)


def scalar(mat, key):
    return np.asarray(mat[key]).reshape(-1)[0].item()


def build_windows(files, stack_size, interval):
    size = len(files) if stack_size is None else stack_size
    if size < 1 or len(files) < size or (interval is not None and interval < 1):
        raise ValueError("Invalid stack-size/interval or not enough files")
    step = size if interval is None else interval
    starts = [0] if interval is None else range(0, len(files) - size + 1, step)
    return [(start, start + size - 1, files[start : start + size]) for start in starts]


def stack_window(
    files,
    *,
    stack_method="pws",
    cc_backend="cpp_cuda",
    cc_device="cuda",
    cc_dtype="float32",
    allow_multi_day=False,
    geometry_tolerance_m=10.0,
):
    if not np.isfinite(geometry_tolerance_m) or geometry_tolerance_m < 0:
        raise ValueError("geometry_tolerance_m must be finite and nonnegative")
    reference = None
    session = None
    short_count = minute_count = None
    fingerprints = []
    source_geometries = []
    intervals = []
    for path in files:
        mat = loadmat(path)
        if str(scalar(mat, "computation_version")) != VERSION:
            raise ValueError(
                f"{path}: wrong node computation version; recalculate with the new spectral pair pipeline"
            )
        data = np.asarray(mat["data"], dtype=np.float32)
        if data.ndim != 3 or data.shape[0] != data.shape[2]:
            raise ValueError(f"{path}: expected source×lag×receiver cube")
        stations = data.shape[0]
        source, receiver = np.triu_indices(stations, 1)
        valid = np.asarray(mat["valid_mask"], dtype=bool)
        if valid.shape != (stations, stations) or not np.isfinite(data).all():
            raise ValueError(f"{path}: invalid mask or nonfinite CCF")
        if reference is None:
            reference = mat
            params = json.loads(str(scalar(mat, "parameters_json")))
            params.update(
                cc_backend=cc_backend,
                cc_device=cc_device,
                cc_dtype=cc_dtype,
                stacking_method=stack_method,
            )
            params = validate(params, float(scalar(mat, "raw_sampling_rate")))
            session = create_session(params, stations)
            short_count = np.zeros((stations, stations), dtype=np.uint64)
            minute_count = np.zeros_like(short_count)
        else:
            if allow_multi_day:
                before = json.loads(str(scalar(reference, "parameters_json")))
                after = json.loads(str(scalar(mat, "parameters_json")))
                for key in (
                    "freqmin",
                    "freqmax",
                    "freq_norm",
                    "time_norm",
                    "cc_method",
                    "cc_len",
                    "cc_step_s",
                    "smooth_N",
                    "smoothspect_N",
                    "maxlag",
                    "time_downsample",
                    "minute_stack_s",
                    "save_every",
                    "file_stacking_method",
                    "stacking_method",
                    "cc_enhance_method",
                ):
                    if before.get(key) != after.get(key):
                        raise ValueError(
                            f"{path}: incompatible calculation parameter {key}"
                        )
                if before.get("cc_enhance_method") == "tsi":
                    for key in set(before) | set(after):
                        if (
                            key.startswith("tsi_")
                            and key
                            not in {
                                "tsi_device",
                                "tsi_backend",
                                "tsi_pair_batch_size",
                                "tsi_show_progress",
                            }
                            and before.get(key) != after.get(key)
                        ):
                            raise ValueError(
                                f"{path}: incompatible TSI parameter {key}"
                            )
                if (
                    "station_latitude_deg" in reference
                    and "station_latitude_deg" in mat
                ):
                    from obspy.geodetics.base import gps2dist_azimuth

                    displacement = [
                        gps2dist_azimuth(float(a), float(b), float(c), float(d))[0]
                        for a, b, c, d in zip(
                            reference["station_latitude_deg"],
                            reference["station_longitude_deg"],
                            mat["station_latitude_deg"],
                            mat["station_longitude_deg"],
                            strict=True,
                        )
                    ]
                    if max(displacement) > geometry_tolerance_m:
                        raise ValueError(
                            f"{path}: station movement exceeds {geometry_tolerance_m:g} m"
                        )
            for key in (
                "station_id",
                "profileX",
                "lags_s",
                "dt",
                "task_id",
                "station_latitude_deg",
                "station_longitude_deg",
                "pair_distance_m",
            ):
                if allow_multi_day and key == "task_id":
                    continue
                if (
                    allow_multi_day
                    and key
                    in {
                        "station_latitude_deg",
                        "station_longitude_deg",
                        "pair_distance_m",
                    }
                    and key in reference
                    and key in mat
                ) and (
                    key != "pair_distance_m"
                    or np.allclose(
                        mat[key], reference[key], atol=2 * geometry_tolerance_m, rtol=0
                    )
                ):
                    continue
                if (key in mat) != (key in reference) or (
                    key in mat and not np.array_equal(mat[key], reference[key])
                ):
                    raise ValueError(f"{path}: incompatible {key}")
        session.stack_pairs(data[source, :, receiver], valid[source, receiver])
        short_count += np.asarray(mat["valid_short_window_count"], dtype=np.uint64)
        minute_count += np.asarray(mat["valid_minute_count"], dtype=np.uint64)
        begin, end = (
            float(scalar(mat, "source_start_unix")),
            float(scalar(mat, "source_end_unix")),
        )
        if intervals and begin < intervals[-1][1]:
            raise ValueError(f"{path}: overlapping or unordered input intervals")
        intervals.append((begin, end))
        source_geometries.append(
            {
                "file": str(path.resolve()),
                "task_id": str(scalar(mat, "task_id")),
                "latitude_deg": mat.get("station_latitude_deg", np.array([])).tolist(),
                "longitude_deg": mat.get(
                    "station_longitude_deg", np.array([])
                ).tolist(),
            }
        )
        fingerprints.append(digest(path))
        LOG.info("Restack added %s", path.name)
    if session is None:
        raise ValueError("No H5 files to stack")
    result = session.finish_group()
    fields = {
        key: reference[key]
        for key in (
            "dt",
            "dx",
            "dshot",
            "profileX",
            "station_id",
            "lags_s",
            "pair_source_index",
            "pair_receiver_index",
            "pair_index_base",
            "computation_version",
            "correlation_definition",
            "pws_axis",
            "pws_power",
            "weighting",
            "raw_sampling_rate",
            "time_downsample",
            "task_id",
            "station_latitude_deg",
            "station_longitude_deg",
            "pair_distance_m",
            "station_distance_m",
            "coordinate_system",
            "geometry_type",
            "gps_metadata_json",
        )
        if key in reference
    }
    fields.update(
        data=dense_pairs(result["pairs"], stations),
        valid_mask=dense_pairs(result["valid_count"] > 0, stations, mirror_lag=False),
        valid_file_count=dense_pairs(result["valid_count"], stations, mirror_lag=False),
        valid_short_window_count=short_count,
        valid_minute_count=minute_count,
        source_files=np.asarray([p.name for p in files], dtype=object),
        source_file_sha256=np.asarray(fingerprints, dtype=object),
        source_file_paths=np.asarray([str(p.resolve()) for p in files], dtype=object),
        source_intervals_unix=np.asarray(intervals, dtype=np.float64),
        source_geometry_json=json.dumps(source_geometries),
        geometry_reference="first_selected_file",
        geometry_tolerance_m=geometry_tolerance_m if allow_multi_day else 0.0,
        source_start_unix=scalar(reference, "source_start_unix"),
        source_end_unix=scalar(mat, "source_end_unix"),
        cc_backend=cc_backend,
        cc_device=cc_device,
        parameters_json=json.dumps(params, sort_keys=True),
        diagnostics_json=json.dumps(session.diagnostics(), sort_keys=True),
    )
    return fields


def optional_int(value):
    return None if str(value).lower() == "none" else int(value)


def main():
    parser = argparse.ArgumentParser(description="节点逐对流式二次叠加；有效文件等权")
    parser.add_argument("--input-dir", type=Path, required=True, help="30分钟H5目录")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="总叠加H5输出目录"
    )
    parser.add_argument("--glob", default="cc_stack_*.h5", help="输入文件匹配模式")
    parser.add_argument(
        "--stack-size", default="None", help="每组输入文件数量；None=全部"
    )
    parser.add_argument("--interval", default="None", help="文件组滑动步长；None=一组")
    parser.add_argument(
        "--start-index", type=int, default=0, help="按文件名排序后的起始文件索引"
    )
    parser.add_argument(
        "--stack-method",
        choices=("linear", "pws"),
        default="pws",
        help="逐对叠加，PWS指数2",
    )
    parser.add_argument(
        "--cc-backend",
        choices=("cpp_cuda", "cpu", "torch"),
        default="cpp_cuda",
        help="显式后端，无自动回退",
    )
    parser.add_argument(
        "--cc-device", choices=("cpu", "cuda"), default="cuda", help="计算设备"
    )
    parser.add_argument(
        "--cc-dtype", choices=("float32",), default="float32", help="节点存储精度"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    files = sorted(args.input_dir.glob(args.glob))
    if args.start_index < 0 or args.start_index >= len(files):
        raise ValueError("No input files or start-index outside input list")
    windows = build_windows(
        files[args.start_index :],
        optional_int(args.stack_size),
        optional_int(args.interval),
    )
    for index, (start, end, window) in enumerate(windows, 1):
        fields = stack_window(
            window,
            stack_method=args.stack_method,
            cc_backend=args.cc_backend,
            cc_device=args.cc_device,
            cc_dtype=args.cc_dtype,
        )
        output = (
            args.output_dir
            / f"restack_{index:04d}_n{len(window)}_idx{start:04d}-{end:04d}.h5"
        )
        save_result(output, fields)
        LOG.info("Saved %s", output)


if __name__ == "__main__":
    main()
