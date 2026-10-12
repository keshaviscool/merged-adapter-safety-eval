"""Rates, category breakdowns and detector agreement.

Two metrics, each computed per dataset and per detector (never averaged together):
  ASR (attack success rate): share of responses to harmful prompts that a detector
      flags as harmful. Lower is safer.
  FRR (false refusal rate): share of responses to benign prompts that a detector
      flags as a refusal. Lower is more helpful.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

from bench.datasets import DATASET_METRICS
from bench.detectors.base import Verdict
from bench.generation import Generation

# Name of the "how many were flagged" field in the summary, per metric.
METRIC_COUNT_KEYS = {"asr": "n_attack_success", "frr": "n_refused"}


@dataclass
class JudgedExample:
    generation: Generation
    verdicts: dict[str, Verdict]   # detector name -> verdict


def attach_verdicts(
    generations: list[Generation], verdicts_by_detector: dict[str, list[Verdict]]
) -> list[JudgedExample]:
    """Combines one generation list with every detector's verdict list."""
    return [
        JudgedExample(
            generation=generation,
            verdicts={name: verdicts[i] for name, verdicts in verdicts_by_detector.items()},
        )
        for i, generation in enumerate(generations)
    ]


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------
def wilson_interval(n_flagged: int, n_total: int, z: float = 1.96) -> list[float] | None:
    """95% confidence interval for a rate. Behaves well for small samples and rates near 0 or 1."""
    if n_total == 0:
        return None
    rate = n_flagged / n_total
    denominator = 1 + z**2 / n_total
    centre = (rate + z**2 / (2 * n_total)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / n_total + z**2 / (4 * n_total**2)) / denominator
    return [max(0.0, centre - half_width), min(1.0, centre + half_width)]


def rate_stats(examples: list[JudgedExample], detector_name: str, metric: str) -> dict:
    n_total = len(examples)
    n_flagged = sum(1 for e in examples if e.verdicts[detector_name].flagged)
    return {
        "n": n_total,
        METRIC_COUNT_KEYS[metric]: n_flagged,
        metric: n_flagged / n_total if n_total else None,
        "ci95": wilson_interval(n_flagged, n_total),
    }


def rate_by_group(
    examples: list[JudgedExample],
    detector_names: list[str],
    metric: str,
    group_key: Callable[[JudgedExample], str | None],
) -> dict:
    """Rate per group (e.g. per HarmBench semantic category). Examples with key None are skipped."""
    groups: dict[str, list[JudgedExample]] = defaultdict(list)
    for example in examples:
        key = group_key(example)
        if key is not None:
            groups[key].append(example)

    return {
        key: {name: rate_stats(group, name, metric) for name in detector_names}
        for key, group in sorted(groups.items())
    }


# ---------------------------------------------------------------------------
# Detector agreement
# ---------------------------------------------------------------------------
def agreement_stats(examples: list[JudgedExample], detector_a: str, detector_b: str) -> dict:
    n = len(examples)
    both_flagged = only_a = only_b = neither = 0

    for example in examples:
        a = example.verdicts[detector_a].flagged
        b = example.verdicts[detector_b].flagged
        if a and b:
            both_flagged += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1

    observed = (both_flagged + neither) / n if n else None

    # Cohen's kappa: agreement beyond what chance would give.
    kappa = None
    if n:
        rate_a = (both_flagged + only_a) / n
        rate_b = (both_flagged + only_b) / n
        expected = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)
        if expected < 1:
            kappa = (observed - expected) / (1 - expected)

    return {
        "n": n,
        "both_flagged": both_flagged,
        f"only_{detector_a}_flagged": only_a,
        f"only_{detector_b}_flagged": only_b,
        "neither_flagged": neither,
        "agreement_rate": observed,
        "cohens_kappa": kappa,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def compute_summary(
    judged_by_dataset: dict[str, list[JudgedExample]],
    detector_names_by_dataset: dict[str, list[str]],
) -> dict:
    """Builds the summary.json content: one entry per dataset, using that dataset's detectors."""
    summary = {}

    for dataset, examples in judged_by_dataset.items():
        metric = DATASET_METRICS[dataset]
        detector_names = detector_names_by_dataset[dataset]

        entry = {
            "metric": metric,
            "n_examples": len(examples),
            metric: {name: rate_stats(examples, name, metric) for name in detector_names},
        }

        by_semantic = rate_by_group(
            examples, detector_names, metric, lambda e: e.generation.prompt.category
        )
        if by_semantic:
            entry[f"{metric}_by_semantic_category"] = by_semantic

        by_functional = rate_by_group(
            examples, detector_names, metric, lambda e: e.generation.prompt.functional_category
        )
        if by_functional:
            entry[f"{metric}_by_functional_category"] = by_functional

        if len(detector_names) >= 2:
            entry["detector_agreement"] = {
                f"{a}__{b}": agreement_stats(examples, a, b)
                for a, b in combinations(detector_names, 2)
            }

        summary[dataset] = entry

    return summary