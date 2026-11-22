"""One command on the server: python mechanistic_analysis/run.py."""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechanistic_analysis.behavior import prepare
from pairwise_experiment.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, help="Resume an existing run")
    parser.add_argument("--phase", choices=("all", "prepare", "geometry", "analyze"), default="all")
    args = parser.parse_args()
    os.chdir(ROOT)
    saved = args.run_dir / "config_used.yaml" if args.run_dir else None
    path = args.config or (saved if saved and saved.exists() else Path(__file__).with_name("config.yaml"))
    cfg = yaml.safe_load(path.read_text())
    if cfg["top_fractions"] != [.1, .2, .5]:
        raise ValueError("This experiment expects top_fractions: [0.1, 0.2, 0.5]")
    if not isinstance(cfg["bootstrap_samples"], int) or cfg["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    run_dir = (args.run_dir or Path(cfg["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config_used.yaml"
    if snapshot.exists():
        if yaml.safe_load(snapshot.read_text()) != cfg:
            raise ValueError("Configuration changed; use a new output folder")
    else:
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Run folder: {run_dir}", flush=True)
    prepare(run_dir, cfg)
    counts = read_json(run_dir / "prepared.json")["transition_counts"]
    print(f"Distinct transitions: {counts}; selected endpoints: {cfg['lower_asr_order']} vs {cfg['higher_asr_order']}", flush=True)
    if args.phase == "prepare":
        return
    if args.phase == "all":
        for phase in ("geometry", "analyze"):
            subprocess.run([sys.executable, __file__, "--run-dir", str(run_dir), "--phase", phase], check=True)
        return
    if args.phase == "geometry":
        import importlib.metadata
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Geometry runs on your NVIDIA server; --phase prepare is available without CUDA")
        torch.set_num_threads(1)
        environment = {"time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")}}
        write_json(run_dir / "environment.json", environment)
        from mechanistic_analysis.geometry import measure
        measure(run_dir, cfg)
    else:
        from mechanistic_analysis.analyze import analyze
        analyze(run_dir, cfg)


if __name__ == "__main__":
    main()
