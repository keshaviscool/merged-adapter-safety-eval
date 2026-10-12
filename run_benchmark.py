"""Entry point: python run_benchmark.py --config config.yaml

Pipeline:
  1. Load the prompt datasets: HarmBench + AdvBench (harmful prompts -> ASR)
     and Alpaca (benign prompts -> FRR).
  2. Build the model (base only, or base + merged adapters) and generate responses.
     The model is then freed from the GPU.
  3. Load each detector in turn, judge the datasets of its metric, free the detector.
  4. Compute ASR / FRR per dataset x detector and write all results to results/<timestamp>/.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

from bench.config import BenchConfig, load_config
from bench.datasets import DATASET_METRICS, Prompt, load_all_datasets
from bench.detectors import Detector, Verdict, build_detectors
from bench.generation import Generation, ResponseGenerator
from bench.model_builder import build_model, describe_model
from bench.reporting import ResultsWriter, format_summary_table, load_generations
from bench.scoring import JudgedExample, attach_verdicts, compute_summary
from bench.utils import free_gpu_memory, set_global_seed, setup_logging

logger = logging.getLogger("run_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety benchmark: ASR (HarmBench, AdvBench) + FRR (Alpaca)")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config")
    parser.add_argument(
        "--reuse-generations",
        default=None,
        metavar="RUN_DIR",
        help="Skip generation and re-judge the saved generations of an earlier run folder",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Which detector judges which dataset
# ---------------------------------------------------------------------------
def detector_names_for(dataset: str, detectors: list[Detector]) -> list[str]:
    """Detectors whose metric matches the dataset's metric (asr or frr)."""
    metric = DATASET_METRICS[dataset]
    return [d.name for d in detectors if d.metric == metric]


def check_every_dataset_has_a_detector(
    prompts_by_dataset: dict[str, list[Prompt]], detectors: list[Detector]
) -> None:
    """Fail before the expensive generation step, not after it."""
    for dataset in prompts_by_dataset:
        if not detector_names_for(dataset, detectors):
            raise ValueError(
                f"Dataset '{dataset}' is scored with the '{DATASET_METRICS[dataset]}' metric, "
                "but no enabled detector provides it. Enable a matching detector or disable the dataset."
            )


# ---------------------------------------------------------------------------
# Phase 1: generation
# ---------------------------------------------------------------------------
def run_generation_phase(
    config: BenchConfig, prompts_by_dataset: dict[str, list[Prompt]], writer: ResultsWriter
) -> dict[str, list[Generation]]:
    model, tokenizer = build_model(config.model, config.merge)
    generator = ResponseGenerator(model, tokenizer, config.generation)

    generations_by_dataset: dict[str, list[Generation]] = {}
    for dataset, prompts in prompts_by_dataset.items():
        logger.info("Generating for %s (%d prompts)", dataset, len(prompts))
        generations = generator.generate(prompts)
        writer.save_generations(dataset, generations)  # saved right away in case later steps fail
        generations_by_dataset[dataset] = generations

    # Release the target model before loading the (large) detectors.
    del generator, model, tokenizer
    free_gpu_memory()
    return generations_by_dataset


def reuse_saved_generations(
    run_dir: str, prompts_by_dataset: dict[str, list[Prompt]], writer: ResultsWriter
) -> dict[str, list[Generation]]:
    generations_by_dataset = {}
    for dataset in prompts_by_dataset:
        generations = load_generations(run_dir, dataset)
        writer.save_generations(dataset, generations)
        generations_by_dataset[dataset] = generations
    return generations_by_dataset


# ---------------------------------------------------------------------------
# Phase 2: judging
# ---------------------------------------------------------------------------
def run_judging_phase(
    detectors: list[Detector], generations_by_dataset: dict[str, list[Generation]]
) -> dict[str, list[JudgedExample]]:
    # verdicts[dataset][detector_name] = list of Verdicts
    verdicts: dict[str, dict[str, list[Verdict]]] = {d: {} for d in generations_by_dataset}

    for detector in detectors:  # one detector in memory at a time
        datasets_to_judge = [
            dataset for dataset in generations_by_dataset if DATASET_METRICS[dataset] == detector.metric
        ]
        if not datasets_to_judge:
            logger.info("Skipping %s: no enabled dataset uses the '%s' metric", detector.name, detector.metric)
            continue

        with detector:
            for dataset in datasets_to_judge:
                logger.info("Judging %s with %s", dataset, detector.name)
                verdicts[dataset][detector.name] = detector.judge(generations_by_dataset[dataset])

    return {
        dataset: attach_verdicts(generations_by_dataset[dataset], verdicts[dataset])
        for dataset in generations_by_dataset
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config.generation.seed)

    prompts_by_dataset = load_all_datasets(config.datasets)
    detectors = build_detectors(config.detectors)  # only constructs them, models load later
    check_every_dataset_has_a_detector(prompts_by_dataset, detectors)

    writer = ResultsWriter(config.output.root)
    model_info = describe_model(config.model, config.merge)
    if args.reuse_generations:
        model_info["generations_reused_from"] = args.reuse_generations
        logger.warning(
            "Reusing generations from %s: the model section of the config is NOT used.",
            args.reuse_generations,
        )
    writer.save_config_snapshot(config, model_info)

    if args.reuse_generations:
        generations_by_dataset = reuse_saved_generations(
            args.reuse_generations, prompts_by_dataset, writer
        )
    else:
        generations_by_dataset = run_generation_phase(config, prompts_by_dataset, writer)

    judged_by_dataset = run_judging_phase(detectors, generations_by_dataset)
    names_by_dataset = {
        dataset: detector_names_for(dataset, detectors) for dataset in generations_by_dataset
    }

    for dataset, examples in judged_by_dataset.items():
        writer.save_judgments(dataset, examples, names_by_dataset[dataset])

    summary = {
        "model": model_info,
        "generation_settings": asdict(config.generation),
        "results": compute_summary(judged_by_dataset, names_by_dataset),
    }
    writer.save_summary(summary)

    print()
    print(format_summary_table(summary["results"]))
    print(f"\nAll files written to: {writer.run_dir}")


if __name__ == "__main__":
    main()