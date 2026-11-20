"""Small CPU-only helpers. Importing this module never loads a model or torch."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

METHODS = ("linear_sum", "linear_average", "coefficient_normalized", "frobenius_normalized", "ties")
DETECTORS = {
    "harmbench": ("llama_guard_3", "harm_cls"),
    "advbench": ("llama_guard_3", "harm_cls"),
    "alpaca": ("rejection_classifier", "keyword_refusal"),
}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    """Atomic replacement: a killed process cannot leave a half-written cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_states(names):
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("Expected four uniquely named adapters")
    states = {"base": {"method": "base", "adapters": []}}
    states.update({n: {"method": "single", "adapters": [n]} for n in names})
    for a, b in combinations(names, 2):
        for method in METHODS:
            states[f"{method}__{a}__{b}"] = {"method": method, "adapters": [a, b]}
    return states


def multiplier(method, scales, norms):
    if method == "linear_sum":
        return 1.0
    if method == "linear_average":
        return 0.5
    values = scales if method == "coefficient_normalized" else norms
    if method not in ("coefficient_normalized", "frobenius_normalized"):
        raise ValueError(f"No linear multiplier for {method}")
    denominator = math.hypot(*values)
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError(f"Cannot normalize {method}: denominator is {denominator}")
    return 1.0 / denominator


def validate_generations(records, expected_prompts=None):
    if not records:
        raise ValueError("Empty generation file")
    prompts = [r["prompt"] for r in records]
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prompt IDs")
    if any(not isinstance(r["response"], str) for r in records):
        raise ValueError("Missing/non-string response")
    if expected_prompts is not None and prompts != expected_prompts:
        raise ValueError("Prompt identity, content, or order differs from the source run")
    return prompts


def validate_verdicts(records, generations):
    if len(records) != len(generations):
        raise ValueError("Verdict count differs from generation count")
    for v, g in zip(records, generations):
        if type(v.get("flagged")) is not bool:
            raise ValueError("Verdict must contain a Boolean flagged value")
        if "prompt_id" in v and v["prompt_id"] != g["prompt"]["id"]:
            raise ValueError("Verdict prompt IDs are misaligned")


def save_verdicts(path, generations_path, verdicts):
    generations = read_json(generations_path)
    validate_verdicts(verdicts, generations)
    records = [dict(v, prompt_id=g["prompt"]["id"]) for v, g in zip(verdicts, generations)]
    write_json(path, {"generations_sha256": digest(generations_path), "records": records})


def load_verdicts(path, generations_path):
    data = read_json(path)
    if data["generations_sha256"] != digest(generations_path):
        raise ValueError(f"Stale verdict cache: {path}")
    validate_verdicts(data["records"], read_json(generations_path))
    return data["records"]


def retry_batches(items, batch_size, call, oom_type, cleanup, log):
    """Retry only the failed batch, outside the exception frame; preserve order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    def run(chunk):
        failed = False
        try:
            result = call(chunk)
        except oom_type:
            if len(chunk) == 1:
                raise
            failed = True
        if failed:
            cleanup()
            middle = len(chunk) // 2
            log(f"OOM: splitting batch {len(chunk)} into {middle} + {len(chunk) - middle}")
            return run(chunk[:middle]) + run(chunk[middle:])
        if len(result) != len(chunk):
            raise ValueError("Batch result count mismatch")
        log(f"Completed batch of {len(chunk)}")
        return result

    result = []
    for start in range(0, len(items), batch_size):
        result.extend(run(items[start:start + batch_size]))
    return result
