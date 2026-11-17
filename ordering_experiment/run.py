"""Adapter-order experiment.  python ordering_experiment/run.py

For all 24 orders of the 4 adapters, two merge methods are followed checkpoint by checkpoint:
  LINEAR: after k adapters the weights are W + sum of their deltas (order cannot matter for the endpoint).
  TIES:   S1 = delta_1, S2 = TIES(S1, delta_2), S3 = TIES(S2, delta_3), ...   (running pair merges)
Each distinct model is generated and judged ONCE (paths that reach the same model share its results),
then every path is reported checkpoint by checkpoint.

Phases (all resumable; re-run with --run-dir to continue an unfinished run):
  generate  one worker process per GPU, each takes a share of the distinct models
  judge     every detector is loaded once and judges all models
  analyze   see analyze.py (ordering effects + local vs cumulative gate), CPU only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from itertools import permutations
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import analyze  # noqa: E402
import compose  # noqa: E402
from bench.config import load_config  # noqa: E402
from bench.datasets import DATASET_METRICS, load_all_datasets  # noqa: E402
from bench.detectors import build_detectors  # noqa: E402
from bench.detectors.base import Verdict  # noqa: E402
from bench.generation import ResponseGenerator  # noqa: E402
from bench.model_builder import build_model  # noqa: E402
from bench.reporting import load_generations  # noqa: E402
from bench.scoring import attach_verdicts, compute_summary  # noqa: E402
from bench.utils import free_gpu_memory, set_global_seed, setup_logging  # noqa: E402

logger = logging.getLogger("ordering")
BASE = "base"


# ---------------------------------------------------------------------------
# Plan: which model is reached after k adapters, for every order and method
# ---------------------------------------------------------------------------
def build_plan(names: list[str]):
    """Returns (paths, exprs). paths[method][order] = [model key at checkpoint 0..4]; exprs[key] = merge expression."""
    paths = {"linear": {}, "ties": {}}
    exprs = {BASE: None}
    for order in permutations(names):
        label = ">".join(order)
        linear, ties = [BASE], [BASE]
        running = None
        for k in range(1, len(order) + 1):
            lin_expr = compose.make_sum(list(order[:k]), names)
            running = ("leaf", order[0]) if k == 1 else compose.make_ties(running, ("leaf", order[k - 1]))
            for method, expr, keys in (("linear", lin_expr, linear), ("ties", running, ties)):
                exprs[compose.key(expr)] = expr
                keys.append(compose.key(expr))
        paths["linear"][label], paths["ties"][label] = linear, ties
    return paths, exprs


def state_dir(run_dir: Path, key: str) -> Path:
    return run_dir / "states" / key.replace("(", "_").replace(")", "").replace(",", "-")


# ---------------------------------------------------------------------------
# Sanity checks on the merge math (CPU, a few modules)
# ---------------------------------------------------------------------------
def sanity_checks(config, density: float) -> None:
    adapters = compose.Adapters(config.model.adapters, "cpu")
    names = [a.name for a in config.model.adapters]
    modules = adapters.modules()[:: max(1, len(adapters.modules()) // 3)][:3]
    for module in modules:
        d = {n: adapters.delta(n, module) for n in names}
        x, y = compose.ties(d[names[0]], d[names[1]], density), compose.ties(d[names[1]], d[names[0]], density)
        assert torch.equal(x, y), f"TIES is not symmetric on {module}"
        forward, backward = sum_in_order(d, names), sum_in_order(d, names[::-1])
        diff = (forward.bfloat16() != backward.bfloat16()).float().mean().item()
        logger.info("sanity %s: TIES(a,b)==TIES(b,a) exactly; bf16 weights differing between two linear "
                    "summation orders: %.2e of entries (rounding noise only)", module, diff)


def sum_in_order(deltas: dict, order: list[str]):
    total = deltas[order[0]]
    for n in order[1:]:
        total = total + deltas[n]
    return total


# ---------------------------------------------------------------------------
# Phase: generate (one process per GPU)
# ---------------------------------------------------------------------------
def generate_states(config, density, run_dir: Path, exprs: dict, shard: tuple[int, int]) -> None:
    keys = [k for i, k in enumerate(sorted(exprs)) if i % shard[1] == shard[0]]
    todo = [k for k in keys if not (state_dir(run_dir, k) / "generated.ok").exists()]
    logger.info("shard %d/%d: %d models, %d still to generate", shard[0], shard[1], len(keys), len(todo))
    if not todo:
        return

    prompts_by_dataset = load_all_datasets(config.datasets)
    model, tokenizer = build_model(replace(config.model, adapters=[]), config.merge)
    adapters = compose.Adapters(config.model.adapters, "cuda")
    params = dict(model.named_parameters())
    modules = adapters.modules()
    original = {m: params[m + ".weight"].detach().cpu().clone() for m in modules}  # base weights W0
    generator = ResponseGenerator(model, tokenizer, config.generation)

    for n, key in enumerate(todo, 1):
        logger.info("[%d/%d] building model %s", n, len(todo), key)
        with torch.no_grad():
            for module in modules:
                weight = original[module].cuda().float()
                if exprs[key] is not None:
                    weight = weight + compose.evaluate(exprs[key], module, adapters, density)
                params[module + ".weight"].copy_(weight.to(params[module + ".weight"].dtype))
        model.eval()

        out = state_dir(run_dir, key)
        (out / "generations").mkdir(parents=True, exist_ok=True)
        set_global_seed(config.generation.seed)
        for dataset, prompts in prompts_by_dataset.items():
            generations = generator.generate(prompts)
            with open(out / "generations" / f"{dataset}.json", "w", encoding="utf-8") as f:
                json.dump([g.to_dict() for g in generations], f, indent=2, ensure_ascii=False)
        (out / "generated.ok").write_text(key)


def run_generation_phase(args, config, density, run_dir: Path, exprs: dict, n_workers: int) -> None:
    if n_workers <= 1:
        generate_states(config, density, run_dir, exprs, (0, 1))
        return
    procs = []
    for i in range(n_workers):
        cmd = [sys.executable, __file__, "--config", args.config, "--run-dir", str(run_dir),
               "--phase", "generate", "--shard", f"{i}/{n_workers}"]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}))
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError("a generation worker failed; fix the error and re-run with --run-dir to resume")


# ---------------------------------------------------------------------------
# Phase: judge (every detector loaded once, verdicts cached per model)
# ---------------------------------------------------------------------------
def run_judging_phase(config, run_dir: Path, exprs: dict, only: list[str] | None = None) -> None:
    datasets = [d for d in ("harmbench", "advbench", "alpaca") if getattr(config.datasets, d).enabled]
    for detector in build_detectors(config.detectors):
        if only and detector.name not in only:
            continue
        wanted = [d for d in datasets if DATASET_METRICS[d] == detector.metric]
        todo = [(k, d) for k in sorted(exprs) for d in wanted
                if not (state_dir(run_dir, k) / "verdicts" / f"{d}__{detector.name}.json").exists()]
        if not todo:
            continue
        with detector:
            for n, (key, dataset) in enumerate(todo, 1):
                logger.info("[%s %d/%d] judging %s / %s", detector.name, n, len(todo), key, dataset)
                out = state_dir(run_dir, key)
                verdicts = detector.judge(load_generations(out, dataset))
                (out / "verdicts").mkdir(exist_ok=True)
                with open(out / "verdicts" / f"{dataset}__{detector.name}.json", "w", encoding="utf-8") as f:
                    json.dump([asdict(v) for v in verdicts], f, ensure_ascii=False)

    if only:  # partial judging (parallel jobs): summaries are written by the final full pass
        return
    for key in exprs:  # per-model summary with CIs, same format as the other experiments
        out = state_dir(run_dir, key)
        judged, names = {}, {}
        for dataset in datasets:
            files = sorted((out / "verdicts").glob(f"{dataset}__*.json"))
            names[dataset] = [f.stem.split("__")[1] for f in files]
            verdicts = {n: [Verdict(**v) for v in json.load(open(f))] for n, f in zip(names[dataset], files)}
            judged[dataset] = attach_verdicts(load_generations(out, dataset), verdicts)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(compute_summary(judged, names), f, indent=2)


# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Adapter-order experiment (LINEAR and TIES chains)")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--run-dir", default=None, help="Continue an earlier run folder")
    parser.add_argument("--phase", default="all", choices=["all", "generate", "judge", "analyze"])
    parser.add_argument("--judge-detectors", default=None, help="comma-separated detector names: only judge with these, then stop")
    parser.add_argument("--shard", default=None, help="internal: i/n, used by worker processes")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    ordering = yaml.safe_load(open(args.config))["ordering"]
    density = ordering["ties_density"]
    names = [a.name for a in config.model.adapters]
    paths, exprs = build_plan(names)

    run_dir = Path(args.run_dir) if args.run_dir else Path(config.output.root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.shard:  # worker process: only generate its share
        i, n = map(int, args.shard.split("/"))
        generate_states(config, density, run_dir, exprs, (i, n))
        return

    if args.judge_detectors:  # worker process: only judge with some detectors
        run_judging_phase(config, run_dir, exprs, only=args.judge_detectors.split(","))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "config_used.yaml").exists():
        shutil.copy(args.config, run_dir / "config_used.yaml")
    with open(run_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump({"adapters": names, "ties_density": density, "paths": paths, "models": sorted(exprs)}, f, indent=2)
    logger.info("Run folder: %s | %d orders x 2 methods, %d distinct models to evaluate", run_dir, len(paths["linear"]), len(exprs))

    if args.phase in ("all", "generate"):
        sanity_checks(config, density)
        n_workers = ordering.get("workers") or max(1, torch.cuda.device_count())
        run_generation_phase(args, config, density, run_dir, exprs, n_workers)
    if args.phase in ("all", "judge"):
        run_judging_phase(config, run_dir, exprs)
    if args.phase in ("all", "analyze", "judge"):
        analyze.main(run_dir, ordering)
    print(f"\nAll files written to: {run_dir}")


if __name__ == "__main__":
    main()
