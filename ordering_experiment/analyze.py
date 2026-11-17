"""Analysis of an ordering-experiment run (CPU only).  python ordering_experiment/analyze.py RUN_DIR

Exp 2  trajectory vs endpoint:  does the rate at checkpoint k depend on the order (trajectory), and does the
       final model (k = 4) depend on the order (endpoint)?
Exp 3  local vs cumulative gate: local_k = rate_k - rate_{k-1};  cumulative_k = rate_k - rate_0 (base model).
       A local gate rejects a step if local_k > eps, a cumulative gate if cumulative_k > tau.

Rates are per dataset and detector, plus "any" (flagged by at least one detector of that dataset).
Writes order_results.csv, ordering_report.{md,json}, gate_report.{md,json} into RUN_DIR.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from itertools import combinations
from math import comb
from pathlib import Path

METRICS = {"harmbench": "asr", "advbench": "asr", "alpaca": "frr"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def state_dir(run_dir: Path, key: str) -> Path:
    return run_dir / "states" / key.replace("(", "_").replace(")", "").replace(",", "-")


def load_flags(run_dir: Path, keys: list[str]) -> dict:
    """flags[key][(dataset, detector)] = list of bool per prompt, including detector 'any'."""
    flags = {}
    for key in keys:
        flags[key] = {}
        for f in sorted((state_dir(run_dir, key) / "verdicts").glob("*__*.json")):
            dataset, detector = f.stem.split("__")
            flags[key][(dataset, detector)] = [v["flagged"] for v in json.load(open(f))]
        for dataset in {d for d, _ in flags[key]}:
            per_detector = [v for (d, _), v in flags[key].items() if d == dataset]
            flags[key][(dataset, "any")] = [any(x) for x in zip(*per_detector)]
    return flags


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags)


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar test on paired per-prompt verdicts."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Exp 2: trajectory vs endpoint
# ---------------------------------------------------------------------------
def ordering_analysis(paths: dict, flags: dict, combos: list) -> dict:
    report = {}
    n = len(next(iter(next(iter(paths.values())).values()))) - 1  # number of adapters
    for method, method_paths in paths.items():
        for dataset, detector in combos:
            entry = {"per_checkpoint": {}}
            for k in range(1, n + 1):
                states = sorted({p[k] for p in method_paths.values()})
                rates = [rate(flags[s][(dataset, detector)]) for s in states]
                entry["per_checkpoint"][k] = {
                    "n_distinct_models": len(states), "min": min(rates), "max": max(rates),
                    "range": max(rates) - min(rates), "std": statistics.pstdev(rates),
                }
            ends = sorted({p[n] for p in method_paths.values()})
            disagreement = max(
                (sum(x != y for x, y in zip(flags[a][(dataset, detector)], flags[b][(dataset, detector)]))
                 / len(flags[a][(dataset, detector)]) for a, b in combinations(ends, 2)), default=0.0)
            entry["endpoint_max_pairwise_verdict_disagreement"] = disagreement
            rates_by_path = {o: [rate(flags[s][(dataset, detector)]) for s in p] for o, p in method_paths.items()}
            entry["path_max_rate_range"] = max(max(r) for r in rates_by_path.values()) - min(max(r) for r in rates_by_path.values())
            entry["path_mean_rate_range"] = (max(statistics.mean(r) for r in rates_by_path.values())
                                             - min(statistics.mean(r) for r in rates_by_path.values()))
            report[f"{method}|{dataset}|{detector}"] = entry
    return report


def write_ordering_md(report: dict, path: Path) -> None:
    lines = ["# Ordering effects", "",
             "Range = max - min of the rate across the distinct models reached at that checkpoint by the 24 orders",
             "(rate in percentage points). Trajectory effect = ranges at checkpoints 1-3, endpoint effect = checkpoint 4.",
             "Disagreement = largest share of prompts whose verdict differs between two endpoint models.", "",
             "| method | dataset | detector | range@1 | range@2 | range@3 | range@4 (endpoint) | endpoint models | endpoint disagreement |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, e in report.items():
        method, dataset, detector = name.split("|")
        c = e["per_checkpoint"]
        n = len(c)
        lines.append(f"| {method} | {dataset} | {detector} | " + " | ".join(f"{100 * c[k]['range']:.1f}" for k in range(1, 5))
                     + f" | {c[4]['n_distinct_models']} | {100 * e['endpoint_max_pairwise_verdict_disagreement']:.1f}% |")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Exp 3: local vs cumulative gate
# ---------------------------------------------------------------------------
def gate_analysis(paths: dict, flags: dict, combos: list, eps_grid: list, tau_grid: list) -> dict:
    report = {}
    for method, method_paths in paths.items():
        for dataset, detector in combos:
            if METRICS[dataset] != "asr":
                continue
            curves = {o: [rate(flags[s][(dataset, detector)]) for s in p] for o, p in method_paths.items()}
            grid = {}
            for eps in eps_grid:
                for tau in tau_grid:
                    local_ok = {o for o, r in curves.items() if all(r[k] - r[k - 1] <= eps for k in range(1, 5))}
                    cum_ok = {o for o, r in curves.items() if all(r[k] - r[0] <= tau for k in range(1, 5))}
                    grid[f"eps={eps},tau={tau}"] = {
                        "local_admits_all_steps": len(local_ok),
                        "cumulative_admits_all_steps": len(cum_ok),
                        "local_admits_but_cumulative_rejects": sorted(local_ok - cum_ok),
                    }
            worst = max(curves, key=lambda o: curves[o][4] - curves[o][0])
            base_flags = flags[method_paths[worst][0]][(dataset, detector)]
            final_flags = flags[method_paths[worst][4]][(dataset, detector)]
            report[f"{method}|{dataset}|{detector}"] = {
                "grid": grid,
                "largest_single_step_increase": max(r[k] - r[k - 1] for r in curves.values() for k in range(1, 5)),
                "largest_cumulative_increase": curves[worst][4] - curves[worst][0],
                "worst_path": worst, "worst_path_rates": curves[worst],
                "worst_path_mcnemar_p_final_vs_base": mcnemar_p(final_flags, base_flags),
                "curves": curves,
            }
    return report


def write_gate_md(report: dict, eps_grid: list, tau_grid: list, path: Path) -> None:
    lines = ["# Local vs cumulative gate", "",
             "Per path (checkpoints M0 -> 4 adapters): local gate rejects a step if ASR_k - ASR_{k-1} > eps, "
             "cumulative gate rejects a step if ASR_k - ASR_0 > tau.",
             "Cells: number of the 24 paths that the local gate admits at every step but the cumulative gate rejects "
             "at some step (hidden drift).", ""]
    for name, e in report.items():
        method, dataset, detector = name.split("|")
        lines += [f"## {method} / {dataset} / {detector}", "",
                  f"largest single-step increase: {100 * e['largest_single_step_increase']:.1f} pp; "
                  f"largest cumulative increase: {100 * e['largest_cumulative_increase']:.1f} pp "
                  f"(path {e['worst_path']}, McNemar p vs base = {e['worst_path_mcnemar_p_final_vs_base']:.3g})", "",
                  "| eps \\ tau | " + " | ".join(str(t) for t in tau_grid) + " |", "|---|" + "---|" * len(tau_grid)]
        for eps in eps_grid:
            cells = [f"{len(e['grid'][f'eps={eps},tau={tau}']['local_admits_but_cumulative_rejects'])}" for tau in tau_grid]
            lines.append(f"| {eps} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
def main(run_dir, ordering: dict) -> None:
    run_dir = Path(run_dir)
    plan = json.load(open(run_dir / "plan.json"))
    paths = plan["paths"]
    flags = load_flags(run_dir, plan["models"])
    combos = sorted(set(flags[plan["models"][0]]))  # (dataset, detector) pairs, including "any"

    with open(run_dir / "order_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "order", "checkpoint", "model", "dataset", "detector", "rate"])
        for method, method_paths in paths.items():
            for order, keys in method_paths.items():
                for k, key in enumerate(keys):
                    for dataset, detector in combos:
                        writer.writerow([method, order, k, key, dataset, detector, rate(flags[key][(dataset, detector)])])

    ordering_report = ordering_analysis(paths, flags, combos)
    (run_dir / "ordering_report.json").write_text(json.dumps(ordering_report, indent=2))
    write_ordering_md(ordering_report, run_dir / "ordering_report.md")

    eps_grid, tau_grid = ordering["gate_eps"], ordering["gate_tau"]
    gate_report = gate_analysis(paths, flags, combos, eps_grid, tau_grid)
    (run_dir / "gate_report.json").write_text(json.dumps(gate_report, indent=2))
    write_gate_md(gate_report, eps_grid, tau_grid, run_dir / "gate_report.md")
    print(f"Analysis written to {run_dir} (order_results.csv, ordering_report.*, gate_report.*)")


if __name__ == "__main__":
    import yaml

    run = Path(sys.argv[1])
    cfg = yaml.safe_load(open(run / "config_used.yaml"))["ordering"]
    main(run, cfg)
