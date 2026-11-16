"""Sequential merging experiment.  Run from anywhere:  python sequential_merging/run.py

Stages:  M  ->  M+a1  ->  M+a1+a2  ->  ...  (each adapter folded into the weights in turn,
W <- W + (alpha / rank) * B @ A, so every stage builds on the previous one).

Phase 1: load the base model once, and for every stage merge the next adapter, then generate
         responses for all datasets. Generations are saved right away.
Phase 2: free the model, then load each detector ONCE and judge every stage with it.
Phase 3: write per-stage results (same layout as run_benchmark.py) and a cross-stage summary.

Reuses the `bench` package of the main pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from peft import PeftModel  # noqa: E402

from bench.config import load_config  # noqa: E402
from bench.datasets import DATASET_METRICS, load_all_datasets  # noqa: E402
from bench.detectors import build_detectors  # noqa: E402
from bench.generation import ResponseGenerator  # noqa: E402
from bench.model_builder import build_model  # noqa: E402
from bench.reporting import ResultsWriter, format_summary_table, load_generations  # noqa: E402
from bench.scoring import attach_verdicts, compute_summary  # noqa: E402
from bench.utils import free_gpu_memory, set_global_seed, setup_logging  # noqa: E402

logger = logging.getLogger("sequential_merging")


class StageWriter(ResultsWriter):
    """ResultsWriter that writes straight into the given folder (no timestamp subfolder)."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.generations_dir = run_dir / "generations"
        self.went_wrong_dir = run_dir / "went_wrong"
        self.went_right_dir = run_dir / "went_right"
        self.disagreements_dir = run_dir / "disagreements"
        for directory in (self.generations_dir, self.went_wrong_dir, self.went_right_dir, self.disagreements_dir):
            directory.mkdir(parents=True, exist_ok=True)


def write_report(run_dir, config, stage_names, summaries, rows, seconds) -> None:
    """One final file with settings, environment and every result (rate, 95% CI, counts) per stage."""
    results = {}
    md = ["# Sequential merging report", ""]
    for stage in stage_names:
        results[stage] = summaries[stage]
    md += [f"- base model: `{config.model.base_model}`",
           "- adapters (merged in order): " + ", ".join(f"`{a.name}`={a.path}" for a in config.model.adapters),
           f"- decoding: {'sampling' if config.generation.do_sample else 'greedy'}, seed {config.generation.seed}, "
           f"max_new_tokens {config.generation.max_new_tokens}", "",
           "Rates (ASR / FRR, lower is better):", "",
           "| dataset | metric | detector | " + " | ".join(stage_names) + " |",
           "|---|---|---|" + "---|" * len(stage_names)]
    for r in rows:
        cells = ["n/a" if r[s] is None else f"{100 * r[s]:.1f}%" for s in stage_names]
        md.append(f"| {r['dataset']} | {r['metric']} | {r['detector']} | " + " | ".join(cells) + " |")
    md += ["", "95% CIs and counts per stage are in `experiment_report.json`."]
    (run_dir / "experiment_report.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_minutes": round(seconds / 60, 1),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": asdict(config),
        "stages": stage_names,
        "results": results,   # results[stage][dataset][metric][detector] = rate, ci95, counts, ...
    }
    with open(run_dir / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main() -> None:
    start_time = time.time()
    setup_logging()
    parser = argparse.ArgumentParser(description="Sequential adapter merging + safety benchmark")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--judge-only", metavar="RUN_DIR", default=None,
                        help="Skip generation: judge the generations already saved in this results folder")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)  # so data_cache/ and output paths in the config resolve from the repo root
    config = load_config(args.config)
    adapters = config.model.adapters
    prompts_by_dataset = load_all_datasets(config.datasets)
    detectors = build_detectors(config.detectors)  # only constructs them; models load later

    run_dir = Path(args.judge_only) if args.judge_only else Path(config.output.root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_names = ["M"] + ["M+" + "+".join(a.name for a in adapters[:i]) for i in range(1, len(adapters) + 1)]
    stage_dirs = {name: run_dir / f"stage{i}_{name.replace('+', '_')}" for i, name in enumerate(stage_names)}
    writers = {name: StageWriter(path) for name, path in stage_dirs.items()}
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.judge_only:  # when resuming, keep the config the generations were made with
        shutil.copy(args.config, run_dir / "config_used.yaml")
    logger.info("Results folder: %s | stages: %s", run_dir, stage_names)

    # ---- Phase 1: merge adapters one by one, generating after each merge ----
    generations = {}  # generations[stage][dataset] = list[Generation]
    if args.judge_only:
        for stage in stage_names:
            generations[stage] = {d: load_generations(stage_dirs[stage], d) for d in prompts_by_dataset}
        logger.info("Loaded saved generations from %s, skipping generation", run_dir)
    else:
        model, tokenizer = build_model(replace(config.model, adapters=[]), config.merge)
        for i, stage in enumerate(stage_names):
            if i > 0:
                adapter = adapters[i - 1]
                logger.info("Merging %s (%s) into the current weights", adapter.name, adapter.path)
                # merge_and_unload folds (alpha/rank) * B @ A into W in place and drops the PEFT wrapper,
                # so the next adapter is merged on top of the already-merged weights.
                model = PeftModel.from_pretrained(model, adapter.path).merge_and_unload()
                model.eval()

            set_global_seed(config.generation.seed)
            generator = ResponseGenerator(model, tokenizer, config.generation)
            generations[stage] = {}
            for dataset, prompts in prompts_by_dataset.items():
                logger.info("[%s] generating for %s (%d prompts)", stage, dataset, len(prompts))
                generations[stage][dataset] = generator.generate(prompts)
                writers[stage].save_generations(dataset, generations[stage][dataset])
        del generator, model, tokenizer
        free_gpu_memory()


    # ---- Phase 2: each detector is loaded once and judges every stage ----
    verdicts = {s: {d: {} for d in prompts_by_dataset} for s in stage_names}  # [stage][dataset][detector]
    for detector in detectors:
        datasets = [d for d in prompts_by_dataset if DATASET_METRICS[d] == detector.metric]
        if not datasets:
            continue
        with detector:
            for stage in stage_names:
                for dataset in datasets:
                    logger.info("[%s] judging %s with %s", stage, dataset, detector.name)
                    verdicts[stage][dataset][detector.name] = detector.judge(generations[stage][dataset])

    # ---- Phase 3: per-stage files + cross-stage summary ----
    summaries = {}
    for stage in stage_names:
        judged = {d: attach_verdicts(generations[stage][d], verdicts[stage][d]) for d in prompts_by_dataset}
        names = {d: list(verdicts[stage][d]) for d in prompts_by_dataset}
        for dataset, examples in judged.items():
            writers[stage].save_judgments(dataset, examples, names[dataset])
        summaries[stage] = compute_summary(judged, names)
        writers[stage].save_summary({
            "stage": stage,
            "adapters_merged": [asdict(a) for a in adapters[: stage_names.index(stage)]],
            "generation_settings": asdict(config.generation),
            "results": summaries[stage],
        })
        print(f"\n=== {stage} ===\n{format_summary_table(summaries[stage])}")

    # rows = dataset x detector, columns = stages (rate only; CIs and counts are in each stage's summary.json)
    rows = []
    for dataset, entry in summaries[stage_names[0]].items():
        metric = entry["metric"]
        for detector in entry[metric]:
            row = {"dataset": dataset, "metric": metric, "detector": detector}
            for stage in stage_names:
                row[stage] = summaries[stage][dataset][metric][detector][metric]
            rows.append(row)

    with open(run_dir / "sequential_summary.json", "w", encoding="utf-8") as f:
        json.dump({"stages": stage_names, "rows": rows}, f, indent=2)
    with open(run_dir / "sequential_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "detector", *stage_names])
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Rate per stage (ASR / FRR, lower is better) ===")
    print(f"{'dataset':<11}{'detector':<22}" + "".join(f"{s:<{max(len(s) + 2, 9)}}" for s in stage_names))
    for r in rows:
        cells = "".join(
            f"{'n/a' if r[s] is None else format(100 * r[s], '.1f') + '%':<{max(len(s) + 2, 9)}}" for s in stage_names
        )
        print(f"{r['dataset']:<11}{r['detector']:<22}{cells}")
    write_report(run_dir, config, stage_names, summaries, rows, time.time() - start_time)
    print(f"\nAll files written to: {run_dir} (see experiment_report.json / experiment_report.md)")


if __name__ == "__main__":
    main()
