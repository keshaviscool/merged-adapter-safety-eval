"""Experiment 3: ASR-only cumulative checkpoints on an A100-80GB."""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent
REMOTE_DIR = "/root/safety_bench"
GPU = "A100-80GB"   # holds each of the 8B / 12B / 13B models comfortably, one at a time

app = modal.App("safety-bench-experiment3")

# Volumes persist between runs: model weights are downloaded only once.
hf_cache_volume = modal.Volume.from_name("safety-bench-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("safety-bench-results", create_if_missing=True)
adapters_volume = modal.Volume.from_name("safety-bench-adapters", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # required for the git+https transformers pin in requirements.txt
    .pip_install_from_requirements(str(PROJECT_DIR / "requirements.txt"))
    .env({"HF_HOME": "/hf-cache"})
    # Local files are added last so editing code does not rebuild the pip layer.
    .add_local_dir(PROJECT_DIR / "bench", remote_path=f"{REMOTE_DIR}/bench")
    .add_local_file(PROJECT_DIR / "run_benchmark.py", remote_path=f"{REMOTE_DIR}/run_benchmark.py")
    .add_local_file(EXPERIMENT_DIR / "weighted_merging.py", remote_path=f"{REMOTE_DIR}/weighted_merging.py")
    .add_local_file(EXPERIMENT_DIR / "benchmark_entrypoint.py", remote_path=f"{REMOTE_DIR}/benchmark_entrypoint.py")
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/hf-cache": hf_cache_volume,
        "/results": results_volume,
        "/adapters": adapters_volume,
    },
)
def run_benchmark(config_yaml: str, run_label: str, reuse_generations: str | None = None) -> str:
    """Runs run_benchmark.py remotely and returns the name of the new results folder."""
    # Imported here because PyYAML is installed in the Modal image, not necessarily on your machine.
    import yaml

    config = yaml.safe_load(config_yaml)

    # Point outputs and the dataset cache at the persistent volumes.
    output_root = Path("/results") / "experiment3" / run_label
    config.setdefault("output", {})["root"] = str(output_root)
    config.setdefault("datasets", {})["cache_dir"] = "/hf-cache/bench_data"

    config_path = Path(REMOTE_DIR) / "config_modal.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    command = ["python", "benchmark_entrypoint.py", "--config", str(config_path)]
    if reuse_generations:
        command += ["--reuse-generations", reuse_generations]

    try:
        subprocess.run(command, cwd=REMOTE_DIR, check=True)
    finally:
        # Persist whatever was produced, even if the run failed part-way.
        results_volume.commit()
        hf_cache_volume.commit()

    return str((output_root / sorted(p.name for p in output_root.iterdir() if p.is_dir())[-1]).relative_to("/results"))


@app.local_entrypoint()
def main(config: str, reuse_generations: str = "") -> None:
    config_yaml = (PROJECT_DIR / config).read_text()
    run_name = run_benchmark.remote(config_yaml, Path(config).stem, reuse_generations or None)

    print(f"\nDone. Results folder on the volume: {run_name}")
    print(f"Download with:\n  modal volume get safety-bench-results {run_name} ./results/")