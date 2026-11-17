"""Runs the whole ordering experiment on Modal A100-80GB GPUs, in parallel.

    modal run --detach ordering_experiment/modal_app.py                  # full run, ~4-5h of GPU time in total
    modal run --detach ordering_experiment/modal_app.py --run-dir /results/ordering/<timestamp>   # resume

The orchestration itself runs on Modal (CPU), so closing your laptop is fine with --detach.
Steps: N generation containers (each takes a share of the 46 models) -> 3 judging containers
(Llama Guard / HarmCLS / the two small detectors) -> a final CPU step that writes summaries and the analysis.
Every step skips work that is already on the volume, so a resume only does what is missing.

Download:  modal volume get ordering-results ordering/<timestamp> ./ordering_experiment/results/
Needs the `huggingface-secret` Modal secret (token with Llama Guard 3 access).
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/bench_repo"
N_GENERATION_CONTAINERS = 5

app = modal.App("ordering-experiment")
hf_cache = modal.Volume.from_name("safety-bench-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("ordering-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .env({"HF_HOME": "/hf-cache"})
    .add_local_dir(REPO / "bench", remote_path=f"{REMOTE}/bench")
    .add_local_dir(REPO / "ordering_experiment", remote_path=f"{REMOTE}/ordering_experiment", ignore=["results", "__pycache__"])
    .add_local_file(REPO / "sequential_merging" / "delta_norms.py", remote_path=f"{REMOTE}/sequential_merging/delta_norms.py")
)
common = dict(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/hf-cache": hf_cache, "/results": results},
    timeout=12 * 60 * 60,
)
gpu = dict(gpu="A100-80GB", **common)


def write_config(config_yaml: str, tag: str) -> Path:
    import yaml

    config = yaml.safe_load(config_yaml)
    config["output"] = {"root": "/results/ordering"}
    config["datasets"]["cache_dir"] = "/hf-cache/bench_data"
    path = Path(REMOTE) / f"config_modal_{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_script(config_yaml: str, run_dir: str, *extra: str) -> None:
    """Runs run.py with the config, committing the results volume every 5 min so nothing is lost on a crash."""
    config_path = write_config(config_yaml, "".join(c for c in "_".join(extra) if c.isalnum() or c == "_"))

    results.reload()
    proc = subprocess.Popen(
        ["python", "ordering_experiment/run.py", "--config", str(config_path), "--run-dir", run_dir, *extra], cwd=REMOTE
    )
    while True:
        try:
            code = proc.wait(timeout=300)
            break
        except subprocess.TimeoutExpired:
            results.commit()
    results.commit()
    hf_cache.commit()
    if code != 0:
        raise RuntimeError(f"run.py {' '.join(extra)} failed with exit code {code}")


@app.function(**common)
def prefetch_datasets(config_yaml: str) -> None:
    """Downloads the prompt datasets once, so parallel containers do not race to write the same cache files."""
    config_path = write_config(config_yaml, "prefetch")
    code = ("import sys; sys.path.insert(0, '.'); from bench.config import load_config; "
            "from bench.datasets import load_all_datasets; "
            f"print({{k: len(v) for k, v in load_all_datasets(load_config('{config_path}').datasets).items()}})")
    subprocess.run(["python", "-c", code], cwd=REMOTE, check=True)
    hf_cache.commit()


@app.function(**gpu)
def generate(config_yaml: str, run_dir: str, shard: int, n_shards: int) -> None:
    run_script(config_yaml, run_dir, "--shard", f"{shard}/{n_shards}")


@app.function(**gpu)
def judge(config_yaml: str, run_dir: str, detectors: str) -> None:
    run_script(config_yaml, run_dir, "--judge-detectors", detectors)


@app.function(**common)
def finalize(config_yaml: str, run_dir: str) -> None:
    run_script(config_yaml, run_dir, "--phase", "judge")  # all verdicts cached: writes summaries + analysis


@app.function(**common)
def orchestrate(config_yaml: str, run_dir: str) -> None:
    n = N_GENERATION_CONTAINERS
    prefetch_datasets.remote(config_yaml)
    list(generate.starmap([(config_yaml, run_dir, i, n) for i in range(n)]))
    list(judge.starmap([(config_yaml, run_dir, d) for d in ("llama_guard_3", "harm_cls", "rejection_classifier,keyword_refusal")]))
    finalize.remote(config_yaml, run_dir)


@app.local_entrypoint()
def main(run_dir: str = "") -> None:
    config_yaml = (REPO / "ordering_experiment" / "config.yaml").read_text()
    run_dir = run_dir or f"/results/ordering/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Run folder on the volume: {run_dir}")
    orchestrate.remote(config_yaml, run_dir)
    name = run_dir.removeprefix("/results/")
    print(f"\nDone. Download with:\n  modal volume get ordering-results {name} ./ordering_experiment/results/")
