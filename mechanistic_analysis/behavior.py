"""Read and validate saved behavioural results; no torch/model imports."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from pairwise_experiment.common import (DETECTORS, digest, read_json, validate_generations,
                                       validate_verdicts, write_csv, write_json)


def state_directory(source, key):
    return source / "states" / key.replace("(", "_").replace(")", "").replace(",", "-")


def parse_expression(key):
    """Parse the saved, canonical merge keys without eval or importing torch."""
    if "(" not in key:
        return ("leaf", key)
    kind, rest = key.split("(", 1)
    if not rest.endswith(")") or kind not in ("sum", "ties"):
        raise ValueError(f"Invalid expression: {key}")
    parts, depth, start = [], 0, 0
    inside = rest[:-1]
    for i, char in enumerate(inside):
        depth += (char == "(") - (char == ")")
        if depth < 0:
            raise ValueError(f"Unbalanced expression: {key}")
        if char == "," and depth == 0:
            parts.append(inside[start:i])
            start = i + 1
    parts.append(inside[start:])
    if depth or any(not part for part in parts):
        raise ValueError(f"Invalid expression: {key}")
    if kind == "sum":
        if any("(" in part for part in parts):
            raise ValueError("LINEAR sums must contain leaves")
        return ("sum", tuple(parts))
    if len(parts) != 2:
        raise ValueError("Binary TIES requires two children")
    return ("ties", parse_expression(parts[0]), parse_expression(parts[1]))


def transition_plan(plan):
    """Deduplicate transitions rather than treating 24 shared paths as replicates."""
    transitions, lookup, occurrences = {}, {}, []
    for method in ("linear", "ties"):
        paths = plan["paths"][method]
        if len(paths) != math.factorial(len(plan["adapters"])):
            raise ValueError("Expected all 24 orders")
        for order, checkpoints in sorted(paths.items()):
            names = order.split(">")
            if sorted(names) != sorted(plan["adapters"]) or len(checkpoints) != len(names) + 1:
                raise ValueError("Invalid saved trajectory")
            for step, incoming in enumerate(names, 1):
                before, after = checkpoints[step - 1:step + 1]
                key = (method, before, incoming, after)
                if key not in lookup:
                    identifier = f"{method}_{sum(t['method'] == method for t in transitions.values()) + 1:03d}"
                    lookup[key] = identifier
                    transitions[identifier] = {"transition_id": identifier, "method": method,
                        "before": before, "incoming": incoming, "after": after, "step": step}
                occurrences.append({"method": method, "order": order, "step": step, "transition_id": lookup[key]})
    return transitions, occurrences


def paired_change(before, after, samples, seed):
    """Paired percentile bootstrap, using the four joint binary outcomes."""
    before, after = np.asarray(before, dtype=int), np.asarray(after, dtype=int)
    n = len(before)
    if n == 0 or len(after) != n:
        raise ValueError("Paired outcomes must have equal nonzero length")
    counts = np.bincount(2 * before + after, minlength=4)
    draws = np.random.default_rng(seed).multinomial(n, counts / n, size=samples)
    differences = 100 * (draws[:, 1] - draws[:, 2]) / n
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"n": n, "before_flagged": int(before.sum()), "after_flagged": int(after.sum()),
            "before_pct": 100 * int(before.sum()) / n, "after_pct": 100 * int(after.sum()) / n,
            "change_pp": 100 * int(after.sum() - before.sum()) / n,
            "ci95_low_pp": float(low), "ci95_high_pp": float(high)}


def prepare(run_dir, cfg):
    marker = run_dir / "prepared.json"
    if marker.exists():
        for path, expected in read_json(marker)["input_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Source input changed: {path}")
        return
    source = Path(cfg["source_run"])
    raw = yaml.safe_load((source / "config_used.yaml").read_text())
    plan = read_json(source / "plan.json")
    if len(plan["adapters"]) != 4 or plan["ties_density"] != raw["ordering"]["ties_density"]:
        raise ValueError("Expected a consistent four-adapter ordering run")
    if raw["generation"]["do_sample"] is not False or raw["model"]["dtype"] != "bfloat16":
        raise ValueError("This reconstruction expects greedy generation and BF16 base weights")
    expected_pairs = read_json(Path(cfg["pairwise_run"]) / "adapter_norms.json")
    earlier_cfg = yaml.safe_load((Path(cfg["pairwise_run"]) / "config_used.yaml").read_text())
    if earlier_cfg["model"]["base_model"] != raw["model"]["base_model"]:
        raise ValueError("Earlier analysis used a different base model")
    for adapter in raw["model"]["adapters"]:
        if expected_pairs[adapter["name"]]["path"] != adapter["path"]:
            raise ValueError("Earlier analysis used different adapters")
    transitions, occurrences = transition_plan(plan)
    hashes = {}
    def load(path):
        hashes[str(path)] = digest(path)
        return read_json(path)
    canonical = {d: validate_generations(load(state_directory(source, "base") / "generations" / f"{d}.json")) for d in DETECTORS}
    flags = {}
    rates = []
    for key in plan["models"]:
        directory = state_directory(source, key)
        for dataset, detectors in DETECTORS.items():
            records = load(directory / "generations" / f"{dataset}.json")
            validate_generations(records, canonical[dataset])
            for detector in detectors:
                verdicts = load(directory / "verdicts" / f"{dataset}__{detector}.json")
                validate_verdicts(verdicts, records)
                values = [v["flagged"] for v in verdicts]
                flags[key, dataset, detector] = values
                rates.append({"state": key, "dataset": dataset, "detector": detector,
                              "n": len(values), "flagged": sum(values), "rate_pct": 100 * sum(values) / len(values),
                              "parse_errors": sum(bool(v.get("details", {}).get("parse_error")) for v in verdicts)})
    linear_ends = {p[-1] for p in plan["paths"]["linear"].values()}
    canonical_sum = "sum(" + ",".join(plan["adapters"]) + ")"
    if linear_ends != {canonical_sum}:
        raise ValueError("LINEAR endpoints do not share the expected canonical sum")
    for name, extreme in (("lower_asr_order", min), ("higher_asr_order", max)):
        endpoint = plan["paths"]["ties"][cfg[name]][-1]
        values = flags[endpoint, "harmbench", "llama_guard_3"]
        all_counts = [sum(flags[path[-1], "harmbench", "llama_guard_3"]) for path in plan["paths"]["ties"].values()]
        if sum(values) != extreme(all_counts):
            raise ValueError(f"Configured {name} is not the Llama Guard HarmBench extreme")
    samples, seed = cfg["bootstrap_samples"], cfg["bootstrap_seed"]
    results = []
    for identifier, transition in transitions.items():
        for dataset, detectors in DETECTORS.items():
            for detector in detectors:
                before = flags[transition["before"], dataset, detector]
                after = flags[transition["after"], dataset, detector]
                base = flags["base", dataset, detector]
                results.append({**transition, "dataset": dataset, "detector": detector,
                    "metric": "frr" if dataset == "alpaca" else "asr",
                    **paired_change(before, after, samples, seed),
                    "cumulative_change_pp": 100 * (sum(after) - sum(base)) / len(base)})
    low = plan["paths"]["ties"][cfg["lower_asr_order"]][-1]
    high = plan["paths"]["ties"][cfg["higher_asr_order"]][-1]
    contrasts = [{"dataset": dataset, "detector": detector,
                  "lower_asr_order": cfg["lower_asr_order"], "higher_asr_order": cfg["higher_asr_order"],
                  **paired_change(flags[low, dataset, detector], flags[high, dataset, detector], samples, seed)}
                 for dataset in DETECTORS for detector in DETECTORS[dataset]]
    write_csv(run_dir / "transition_asr.csv", results)
    write_csv(run_dir / "state_rates.csv", rates)
    write_csv(run_dir / "path_steps.csv", occurrences)
    write_csv(run_dir / "ties_endpoint_asr.csv", contrasts)
    write_json(run_dir / "transitions.json", transitions)
    write_json(run_dir / "source_plan.json", plan)
    for path in (source / "config_used.yaml", source / "plan.json",
                 Path(cfg["pairwise_run"]) / "geometry.csv", Path(cfg["pairwise_run"]) / "adapter_norms.json",
                 Path(cfg["pairwise_run"]) / "config_used.yaml"):
        hashes[str(path)] = digest(path)
    write_json(marker, {"source_model": raw["model"], "ties_density": plan["ties_density"],
        "lower_endpoint": low, "higher_endpoint": high, "linear_endpoint": canonical_sum,
        "input_sha256": hashes, "expected_adapters": expected_pairs,
        "transition_counts": {method: sum(t["method"] == method for t in transitions.values()) for method in ("linear", "ties")}})
