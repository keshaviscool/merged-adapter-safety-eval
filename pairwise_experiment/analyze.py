"""CPU-only rates, paired composition-effect intervals and six-pair correlations."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from pairwise_experiment.common import (DETECTORS, METHODS, load_verdicts, read_json,
    validate_generations, write_csv, write_json)

FEATURES = ("cosine", "norm_balance", "sign_conflict", "top_jaccard")


def composition_effect(pair, single_i, single_j):
    return float(100 * (np.mean(pair) - max(np.mean(single_i), np.mean(single_j))))


def bootstrap_effect(pair, single_i, single_j, samples=20000, seed=42):
    """Resample joint prompt outcomes, recomputing the max on EACH replicate.

    Three binary verdicts have only eight possible joint outcomes. A multinomial
    draw over their observed frequencies is exactly the ordinary paired prompt
    bootstrap, without allocating samples x prompts index arrays.
    """
    arrays = [np.asarray(x, dtype=np.int64) for x in (pair, single_i, single_j)]
    if not len(arrays[0]) or len({len(x) for x in arrays}) != 1:
        raise ValueError("Paired bootstrap requires equal nonempty prompt arrays")
    if any(np.any((x != 0) & (x != 1)) for x in arrays):
        raise ValueError("Expected binary verdicts")
    codes = arrays[0] * 4 + arrays[1] * 2 + arrays[2]
    n = len(codes)
    probabilities = np.bincount(codes, minlength=8) / n
    counts = np.random.default_rng(seed).multinomial(n, probabilities, size=samples)
    outcomes = np.array([[(code >> bit) & 1 for bit in (2, 1, 0)] for code in range(8)])
    means = counts @ outcomes / n
    effects = 100 * (means[:, 0] - np.maximum(means[:, 1], means[:, 2]))
    return np.percentile(effects, [2.5, 97.5]).tolist()


def ranks(values):
    """Average ranks for ties, without a scipy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(run_dir, raw, states):
    canonical = read_json(run_dir / "prompts.json")
    imported = read_json(run_dir / "imports.json")["states"]
    flags, rates, summaries = {}, [], {}
    for state_id, state in states.items():
        summary = {}
        for dataset, detectors in DETECTORS.items():
            gen_path = run_dir / "states" / state_id / "generations" / f"{dataset}.json"
            generations = read_json(gen_path)
            validate_generations(generations, canonical[dataset])
            summary[dataset] = {}
            for detector in detectors:
                path = gen_path.parent.parent / "verdicts" / f"{dataset}__{detector}.json"
                records = load_verdicts(path, gen_path)
                verdicts = np.array([v["flagged"] for v in records], dtype=bool)
                flags[state_id, dataset, detector] = verdicts
                metric = "frr" if dataset == "alpaca" else "asr"
                row = {"state": state_id, "method": state["method"],
                       "adapters": "+".join(state["adapters"]), "dataset": dataset,
                       "detector": detector, "metric": metric, "n": len(verdicts),
                       "flagged": int(verdicts.sum()), "rate_pct": float(verdicts.mean() * 100),
                       "parse_errors": sum(bool(v.get("details", {}).get("parse_error")) for v in records),
                       "empty_responses": sum(not g["response"].strip() for g in generations),
                       "provenance": "imported" if state_id in imported else "new"}
                rates.append(row)
                summary[dataset][detector] = row
        summaries[state_id] = summary
    effects = []
    for state_id, state in states.items():
        if state["method"] not in METHODS:
            continue
        a, b = state["adapters"]
        for dataset, detectors in DETECTORS.items():
            for detector in detectors:
                pair, x, y = [flags[s, dataset, detector] for s in (state_id, a, b)]
                low, high = bootstrap_effect(pair, x, y,
                    raw["pairwise"]["bootstrap_samples"], raw["pairwise"]["bootstrap_seed"])
                effect = composition_effect(pair, x, y)
                effects.append({"adapter_i": a, "adapter_j": b, "method": state["method"],
                    "dataset": dataset, "detector": detector, "metric": "frr" if dataset == "alpaca" else "asr",
                    "n": len(pair), "pair_rate_pct": float(pair.mean() * 100),
                    "single_i_rate_pct": float(x.mean() * 100), "single_j_rate_pct": float(y.mean() * 100),
                    "reference_max_pct": float(max(x.mean(), y.mean()) * 100),
                    "effect_pp": effect, "ci95_low_pp": low, "ci95_high_pp": high,
                    "direction": (("increased_refusal" if effect > 0 else "decreased_refusal" if effect < 0 else "unchanged")
                                  if dataset == "alpaca" else
                                  ("amplification" if effect > 0 else "attenuation" if effect < 0 else "unchanged")),
                    "comparison_source": "within_imported_run" if state_id in imported else "new_pair_vs_imported_singles"})
    with open(run_dir / "geometry.csv") as f:
        geometry_rows = list(csv.DictReader(f))
    global_geometry = {(r["adapter_i"], r["adapter_j"]): r for r in geometry_rows if r["level"] == "global"}
    if len(global_geometry) != 6:
        raise ValueError("Expected six global pair geometry rows")
    correlations, grouped = [], defaultdict(list)
    for row in effects:
        if row["metric"] == "asr":
            grouped[row["method"], row["dataset"], row["detector"]].append(row)
    for (method, dataset, detector), rows in grouped.items():
        for feature in FEATURES:
            valid = [(float(global_geometry[r["adapter_i"], r["adapter_j"]][feature]), r["effect_pp"])
                     for r in rows if global_geometry[r["adapter_i"], r["adapter_j"]][feature] != ""]
            x, y = zip(*valid) if valid else ([], [])
            correlations.append({"method": method, "dataset": dataset, "detector": detector,
                "feature": feature, "n_pairs": len(valid), "spearman_rho": spearman(x, y),
                "interpretation": "exploratory; six dependent adapter pairs; no confirmatory inference"})
    write_csv(run_dir / "rates.csv", rates)
    write_csv(run_dir / "composition_effects.csv", effects)
    write_csv(run_dir / "correlations.csv", correlations)
    write_json(run_dir / "summary.json", {
        "states": summaries, "pair_results": len(effects) // 6,
        "effect_definition": "pair ASR minus max(original full-strength singleton ASRs), in percentage points",
        "bootstrap": {"method": "paired percentile; max recomputed per resample", "samples": raw["pairwise"]["bootstrap_samples"],
                      "seed": raw["pairwise"]["bootstrap_seed"], "limitations": "Prompt uncertainty only; not cross-run/hardware uncertainty."},
        "notes": ["FRR differences are supplementary and are not evidence of attack amplification.",
                  "Normalized variants change update strength; the singleton reference remains full strength.",
                  "Adapter revisions of the imported run were not pinned; current files are hashed.",
                  "Overlap aggregates per-matrix top-50% masks, not a new global top-50% selection.",
                  "Correlations use six pair-level observations; layers are not independent behavioural replicates."]})
    plot(run_dir, grouped, global_geometry)
    print(f"Saved 30 pair results, rates, paired intervals, geometry correlations and figures in {run_dir}")


def plot(run_dir, grouped, geometry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    labels = {"cosine": "Update cosine", "norm_balance": "min(norms) / max(norms)",
              "sign_conflict": "Opposite signs / jointly nonzero", "top_jaccard": "Top-magnitude Jaccard"}
    for (method, dataset, detector), rows in grouped.items():
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), layout="constrained")
        for axis, feature in zip(axes, FEATURES):
            for row in rows:
                value = geometry[row["adapter_i"], row["adapter_j"]][feature]
                if value == "":
                    continue
                x, y = float(value), row["effect_pp"]
                low, high = row["ci95_low_pp"], row["ci95_high_pp"]
                axis.vlines(x, low, high, color="0.7", linewidth=1)
                axis.scatter(x, y, s=25)
                axis.annotate(f"{row['adapter_i'][1:]}+{row['adapter_j'][1:]}", (x, y), xytext=(3, 4), textcoords="offset points", fontsize=8)
            axis.axhline(0, color="0.5", linewidth=.8)
            axis.set_xlabel(labels[feature])
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Composition effect (pp)")
        fig.suptitle(f"{method} | {dataset} | {detector} | six pairs (exploratory)", fontsize=11)
        fig.savefig(out / f"scatter__{method}__{dataset}__{detector}.pdf")
        plt.close(fig)
    pairs = sorted(geometry)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    maximum = max(abs(r["effect_pp"]) for rows in grouped.values() for r in rows) or 1
    for axis, (dataset, detector) in zip(axes.flat,
            [(d, c) for d in ("harmbench", "advbench") for c in DETECTORS[d]]):
        values = np.array([[next(r["effect_pp"] for r in grouped[m, dataset, detector]
                                if (r["adapter_i"], r["adapter_j"]) == pair)
                            for m in METHODS] for pair in pairs])
        image = axis.imshow(values, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            axis.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8,
                      color="white" if abs(value) > .6 * maximum else "black")
        axis.set_xticks(range(5), ["sum", "average", "coeff norm", "Frob norm", "TIES"], rotation=20)
        axis.set_yticks(range(6), [f"{a}+{b}" for a, b in pairs])
        axis.set_title(f"{dataset} / {detector}", fontsize=10)
    fig.colorbar(image, ax=axes, label="Composition effect (percentage points)", shrink=.7)
    fig.savefig(out / "composition_effects.pdf")
    plt.close(fig)
