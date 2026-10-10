"""Loaders that turn HarmBench and AdvBench into one common `Prompt` record."""

from __future__ import annotations

import csv
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bench.config import AdvBenchConfig, DatasetsConfig, HarmBenchConfig

logger = logging.getLogger(__name__)

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

# The copyright category is excluded because HarmCLS does not cover it.
SUPPORTED_HARMBENCH_CATEGORIES = {"standard", "contextual"}


@dataclass
class Prompt:
    id: str
    dataset: str                              # "harmbench" or "advbench"
    text: str                                 # exactly what the target model receives
    behavior: str                             # behavior description (HarmCLS needs this)
    context: str | None = None                # only for HarmBench contextual behaviors
    category: str | None = None               # HarmBench semantic category
    functional_category: str | None = None    # HarmBench: standard / contextual


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def _read_csv_with_cache(url: str, cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        logger.info("Downloading %s", url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)

    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# HarmBench
# ---------------------------------------------------------------------------
def _build_harmbench_prompt_text(behavior: str, context: str | None) -> str:
    """Contextual behaviors are given as '<context> --- <behavior>', as in HarmBench."""
    if context:
        return f"{context}\n\n---\n\n{behavior}"
    return behavior


def load_harmbench(config: HarmBenchConfig, cache_dir: Path) -> list[Prompt]:
    unsupported = set(config.functional_categories) - SUPPORTED_HARMBENCH_CATEGORIES
    if unsupported:
        raise ValueError(
            f"Unsupported HarmBench categories {sorted(unsupported)}. "
            f"Supported: {sorted(SUPPORTED_HARMBENCH_CATEGORIES)}"
        )

    rows = _read_csv_with_cache(HARMBENCH_URL, cache_dir / "harmbench_behaviors_text_all.csv")

    prompts = []
    for row in rows:
        if row["FunctionalCategory"] not in config.functional_categories:
            continue

        context = row["ContextString"].strip() or None
        prompts.append(
            Prompt(
                id=row["BehaviorID"],
                dataset="harmbench",
                text=_build_harmbench_prompt_text(row["Behavior"], context),
                behavior=row["Behavior"],
                context=context,
                category=row["SemanticCategory"],
                functional_category=row["FunctionalCategory"],
            )
        )

    if config.max_examples is not None:
        prompts = prompts[: config.max_examples]

    logger.info("Loaded %d HarmBench prompts", len(prompts))
    return prompts


# ---------------------------------------------------------------------------
# AdvBench
# ---------------------------------------------------------------------------
def load_advbench(config: AdvBenchConfig, cache_dir: Path) -> list[Prompt]:
    rows = _read_csv_with_cache(ADVBENCH_URL, cache_dir / "advbench_harmful_behaviors.csv")

    prompts = [
        Prompt(
            id=f"advbench_{index:04d}",
            dataset="advbench",
            text=row["goal"],
            behavior=row["goal"],
        )
        for index, row in enumerate(rows)
    ]

    if config.max_examples is not None:
        prompts = prompts[: config.max_examples]

    logger.info("Loaded %d AdvBench prompts", len(prompts))
    return prompts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_all_datasets(config: DatasetsConfig) -> dict[str, list[Prompt]]:
    """Returns {dataset_name: prompts} for every enabled dataset."""
    cache_dir = Path(config.cache_dir)
    datasets: dict[str, list[Prompt]] = {}

    if config.harmbench.enabled:
        datasets["harmbench"] = load_harmbench(config.harmbench, cache_dir)
    if config.advbench.enabled:
        datasets["advbench"] = load_advbench(config.advbench, cache_dir)

    if not datasets:
        raise ValueError("No datasets enabled. Enable at least one in the config.")
    return datasets
