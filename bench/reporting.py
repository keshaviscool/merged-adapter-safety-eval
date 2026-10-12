"""Writes every result into a new folder named after the current timestamp.

results/<YYYYMMDD_HHMMSS>/
    config_snapshot.yaml                       exact settings and model description
    summary.json                               ASR / FRR per dataset x detector (+ breakdowns, agreement)
    generations/<dataset>.json                 raw prompts and responses
    went_wrong/<dataset>__<detector>.json      responses the detector flagged
                                               (harmful response for ASR, refusal of a benign prompt for FRR)
    went_right/<dataset>__<detector>.json      responses the detector did not flag
    disagreements/<dataset>.json               responses where the detectors disagree
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from bench.config import BenchConfig
from bench.generation import Generation
from bench.scoring import METRIC_COUNT_KEYS, JudgedExample

logger = logging.getLogger(__name__)


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _judged_to_record(example: JudgedExample) -> dict:
    return {
        "prompt": asdict(example.generation.prompt),
        "response": example.generation.response,
        "verdicts": {name: asdict(verdict) for name, verdict in example.verdicts.items()},
    }


class ResultsWriter:
    def __init__(self, output_root: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_root) / timestamp

        self.generations_dir = self.run_dir / "generations"
        self.went_wrong_dir = self.run_dir / "went_wrong"
        self.went_right_dir = self.run_dir / "went_right"
        self.disagreements_dir = self.run_dir / "disagreements"
        for directory in (
            self.generations_dir,
            self.went_wrong_dir,
            self.went_right_dir,
            self.disagreements_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Results folder: %s", self.run_dir)

    # ------------------------------------------------------------------
    def save_config_snapshot(self, config: BenchConfig, model_info: dict) -> None:
        snapshot = {"model_description": model_info, "config": asdict(config)}
        with open(self.run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)

    def save_generations(self, dataset: str, generations: list[Generation]) -> None:
        _write_json(self.generations_dir / f"{dataset}.json", [g.to_dict() for g in generations])

    def save_judgments(
        self, dataset: str, examples: list[JudgedExample], detector_names: list[str]
    ) -> None:
        """Splits the judged examples into went_wrong / went_right per detector, plus disagreements.

        `detector_names` are the detectors that judged this dataset."""
        for detector in detector_names:
            wrong = [_judged_to_record(e) for e in examples if e.verdicts[detector].flagged]
            right = [_judged_to_record(e) for e in examples if not e.verdicts[detector].flagged]
            _write_json(self.went_wrong_dir / f"{dataset}__{detector}.json", wrong)
            _write_json(self.went_right_dir / f"{dataset}__{detector}.json", right)

        if len(detector_names) >= 2:
            disagreements = [
                _judged_to_record(e)
                for e in examples
                if len({v.flagged for v in e.verdicts.values()}) > 1
            ]
            _write_json(self.disagreements_dir / f"{dataset}.json", disagreements)

    def save_summary(self, summary: dict) -> None:
        _write_json(self.run_dir / "summary.json", summary)


# ---------------------------------------------------------------------------
# Reading saved generations back (used by --reuse-generations)
# ---------------------------------------------------------------------------
def load_generations(run_dir: str | Path, dataset: str) -> list[Generation]:
    path = Path(run_dir) / "generations" / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved generations for '{dataset}' at {path}. "
            "That run did not include this dataset: disable it in the config or run without --reuse-generations."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [Generation.from_dict(item) for item in json.load(f)]


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def format_summary_table(summary: dict) -> str:
    """One row per dataset x detector. `summary` is the "results" part of summary.json."""
    header = f"{'dataset':<11}{'metric':<7}{'detector':<22}{'rate':>7}  {'95% CI':<15}count"
    lines = [header, "-" * len(header)]

    for dataset, entry in summary.items():
        metric = entry["metric"]
        count_key = METRIC_COUNT_KEYS[metric]

        for detector, stats in entry[metric].items():
            rate = "n/a" if stats[metric] is None else f"{100 * stats[metric]:.1f}%"
            low, high = stats["ci95"] if stats["ci95"] else (None, None)
            interval = "n/a" if low is None else f"[{100 * low:.1f}-{100 * high:.1f}]"
            lines.append(
                f"{dataset:<11}{metric.upper():<7}{detector:<22}{rate:>7}  {interval:<15}"
                f"{stats[count_key]}/{stats['n']}"
            )
    return "\n".join(lines)