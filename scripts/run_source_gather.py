"""Run extraction and dispersion with explicit station selection and a task log."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import h5py


def select_stations(available, source, selection):
    if source not in available:
        raise ValueError(f"震源编号不存在：{source}")
    receivers = (
        available
        if selection.strip().lower() == "all"
        else re.split(r"[,，\s]+", selection.strip())
    )
    receivers = list(dict.fromkeys(s for s in receivers if s and s != source))
    missing = [s for s in receivers if s not in available]
    if missing:
        raise ValueError(f"输入H5不存在台站：{', '.join(missing)}")
    if len(receivers) < 2:
        raise ValueError("频散计算至少需要两个接收台站")
    return [source, *receivers]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--receivers", default="all")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--wiggle-amplitude-m", type=float, default=20)
    for name, default in [
        ("fmin", 1),
        ("fmax", 10),
        ("df", 0.1),
        ("vmin", 500),
        ("vmax", 4000),
        ("dv", 10),
    ]:
        p.add_argument(f"--{name}", type=float, default=default)
    args = p.parse_args()
    with h5py.File(args.input) as f:
        stations = select_stations(
            f["station_id"].asstr()[:].tolist(), args.source, args.receivers
        )
    if not (
        0 < args.fmin < args.fmax
        and 0 < args.vmin < args.vmax
        and args.df > 0
        and args.dv > 0
    ):
        raise ValueError("频率/速度范围或步长无效")
    parameters = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    parameters["selected_stations"] = stations
    tag = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    output = args.output_root / f"source{args.source}_n{len(stations) - 1}_{tag}"
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    commands = [
        [
            sys.executable,
            str(scripts / "extract_source_gather.py"),
            "--input",
            str(args.input.resolve()),
            "--source",
            args.source,
            "--stations",
            *stations,
            "--output",
            str(output.resolve()),
            "--wiggle-amplitude-m",
            str(args.wiggle_amplitude_m),
        ],
        [
            sys.executable,
            str(scripts / "plot_source_dispersion.py"),
            "--input",
            str((output / f"source_{args.source}_gather.h5").resolve()),
        ],
    ]
    for key in ("fmin", "fmax", "df", "vmin", "vmax", "dv"):
        commands[1].extend([f"--{key}", str(getattr(args, key))])
    manifest = {
        "parameters": parameters,
        "status": "running",
        "output_dir": str(output.resolve()),
    }
    manifest_file = output / "task.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    try:
        with (output / "run.log").open("w") as log:
            for command in commands:
                result = subprocess.run(
                    command, check=False, stdout=log, stderr=subprocess.STDOUT
                )
                if result.returncode:
                    raise RuntimeError(f"计算失败，请查看 {output / 'run.log'}")
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        manifest_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    print(
        f"完成：源 {args.source}，{len(stations) - 1} 道；按WGS84炮检距排序。\n结果目录：{output.resolve()}"
    )
    print(
        f"互相关图：source_{args.source}_gather.png\n频散图：source_{args.source}_dispersion.png"
    )


if __name__ == "__main__":
    main()
