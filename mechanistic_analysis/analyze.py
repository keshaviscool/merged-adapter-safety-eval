"""Aggregate geometry, join local ASR changes, and produce exploratory summaries."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from pairwise_experiment.common import read_json, write_csv, write_json
from pairwise_experiment.analyze import spearman
from mechanistic_analysis.geometry import locations, ratios


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def add_stats(accumulator, key, stats):
    target = accumulator.setdefault(key, {})
    for field, value in stats.items():
        target[field] = max(target.get(field, 0), value) if field == "max_abs_difference" else target.get(field, 0) + value


def correlation_rows(geometry, labels, features):
    groups = defaultdict(list)
    for row in geometry:
        if row["level"] not in ("global", "block", "module_type") or row["step"] == 1:
            continue
        for label in labels[row["transition_id"]]:
            if label["metric"] != "asr":
                continue
            for stage in ("all_nonbase", str(row["step"])):
                groups[row["method"], row["level"], row["location"], stage, label["dataset"], label["detector"]].append((row, label))
    results = []
    for (method, level, location, stage, dataset, detector), pairs in groups.items():
        for feature in features:
            valid = [(row[feature], int(label["after_flagged"]) - int(label["before_flagged"]))
                     for row, label in pairs if row.get(feature) is not None and math.isfinite(row[feature])]
            x, y = zip(*valid) if valid else ([], [])
            results.append({"method": method, "level": level, "location": location, "step": stage,
                "dataset": dataset, "detector": detector, "feature": feature, "n_transitions": len(valid),
                "spearman_rho": spearman(x, y), "interpretation": "exploratory; shared adapters/states; no independent-layer inference"})
    return results


def earlier_comparison(endpoints, cfg):
    earlier = defaultdict(list)
    for row in read_csv(Path(cfg["pairwise_run"]) / "geometry.csv"):
        if row["level"] == "module":
            earlier[row["location"]].append(row)
    result = []
    for row in endpoints:
        if row["level"] != "module":
            continue
        previous = earlier.get(row["location"], [])
        if not previous:
            raise ValueError(f"Earlier module geometry missing: {row['location']}")
        result.append({"representation": row["representation"], "module": row["location"],
            "endpoint_difference_energy_fraction": row["difference_energy_fraction"],
            "endpoint_difference_norm": row["difference_norm"],
            "earlier_max_abs_pair_cosine": max(abs(float(p["cosine"])) for p in previous if p["cosine"]),
            "earlier_max_top50_jaccard": max(float(p["top_jaccard"]) for p in previous if p["top_jaccard"]),
            "previously_highlighted": row["location"] in cfg["earlier_interesting_modules"]})
    for representation in {r["representation"] for r in result}:
        rows = [r for r in result if r["representation"] == representation]
        for field, rank_field in (("endpoint_difference_energy_fraction", "difference_rank"),
                                  ("earlier_max_abs_pair_cosine", "earlier_cosine_rank"),
                                  ("earlier_max_top50_jaccard", "earlier_overlap_rank")):
            for rank, row in enumerate(sorted(rows, key=lambda r: -r[field]), 1):
                row[rank_field] = rank
    return result


def analyze(run_dir, cfg):
    transitions = read_json(run_dir / "transitions.json")
    module_names = read_json(run_dir / "modules.json")
    accumulated, endpoint_accumulated, controls = {}, {}, []
    for module in module_names:
        cached = read_json(run_dir / "module_cache" / f"{module}.json")
        if (cached["module"] != module or len(cached["transitions"]) != len(transitions)
                or {r["transition_id"] for r in cached["transitions"]} != set(transitions)
                or len(cached["endpoints"]) != 2
                or {r["representation"] for r in cached["endpoints"]} != {"merge_update_fp32", "checkpoint_effective_bfloat16"}):
            raise ValueError(f"Incomplete module cache: {module}")
        if not cached["linear_control"]["equal"]:
            raise ValueError("LINEAR control failed")
        controls.append(cached["linear_control"])
        for level, location in locations(module):
            for row in cached["transitions"]:
                add_stats(accumulated, (row["transition_id"], level, location), row["stats"])
            for row in cached["endpoints"]:
                add_stats(endpoint_accumulated, (row["representation"], level, location), row["stats"])
    fractions = cfg["top_fractions"]
    geometry = [{**transitions[identifier], "level": level, "location": location,
                 **stats, **ratios(stats, fractions)} for (identifier, level, location), stats in accumulated.items()]
    endpoints = [{"representation": representation, "level": level, "location": location,
                  **stats, **ratios(stats, fractions)}
                 for (representation, level, location), stats in endpoint_accumulated.items()]
    total_difference = {r["representation"]: r["difference_sq"] for r in endpoints if r["level"] == "global"}
    if total_difference.get("merge_update_fp32", 0) <= 0:
        raise ValueError("Selected TIES endpoints have identical updates; verify reconstruction/source revisions")
    concentration = []
    for representation in total_difference:
        for level in ("module", "block", "module_type", "global"):
            rows = sorted([r for r in endpoints if r["representation"] == representation and r["level"] == level],
                          key=lambda r: -r["difference_sq"])
            cumulative = 0.
            for rank, row in enumerate(rows, 1):
                row["difference_energy_fraction"] = row["difference_sq"] / total_difference[representation] if total_difference[representation] else 0.
                cumulative += row["difference_energy_fraction"]
                row["difference_rank"] = rank
                row["cumulative_difference_energy_fraction"] = cumulative
                row["difference_energy_per_coordinate"] = row["difference_sq"] / row["n_coordinates"]
                if rank in (1, 3, 5, 10, 20, len(rows)):
                    concentration.append({"representation": representation, "level": level,
                        "top_n": rank, "total_locations": len(rows), "difference_energy_fraction": cumulative})
    comparisons = earlier_comparison(endpoints, cfg)
    labels = defaultdict(list)
    for label in read_csv(run_dir / "transition_asr.csv"):
        labels[label["transition_id"]].append(label)
    features = ["cosine", "incoming_over_composite_norm", "incoming_relative_to_base", "composite_relative_to_base"]
    for fraction in fractions:
        features.extend(f"top{round(100 * fraction)}_{name}" for name in ("sign_conflict", "energy_weighted_conflict", "jaccard"))
    correlations = correlation_rows(geometry, labels, features)
    joined, layer_joined = [], []
    for row in geometry:
        if row["level"] == "global":
            for label in labels[row["transition_id"]]:
                joined.append({**label, **{f: row[f] for f in features},
                               "integration_step_norm": row["integration_step_norm"],
                               "deviation_from_sum_norm": row["deviation_from_sum_norm"]})
        if row["level"] == "block":
            for label in labels[row["transition_id"]]:
                if label["metric"] == "asr":
                    layer_joined.append({**label, "layer": row["location"], **{f: row[f] for f in features}})
    write_csv(run_dir / "transition_geometry.csv", geometry)
    write_csv(run_dir / "transition_effects_global.csv", joined)
    write_csv(run_dir / "transition_effects_by_layer.csv", layer_joined)
    global_lookup = defaultdict(list)
    for row in joined:
        global_lookup[row["transition_id"]].append(row)
    selected = []
    for occurrence in read_csv(run_dir / "path_steps.csv"):
        if occurrence["order"] in (cfg["lower_asr_order"], cfg["higher_asr_order"]):
            for row in global_lookup[occurrence["transition_id"]]:
                selected.append({"order": occurrence["order"],
                                 "endpoint_group": "lower_asr" if occurrence["order"] == cfg["lower_asr_order"] else "higher_asr",
                                 **row})
    write_csv(run_dir / "selected_path_interactions.csv", selected)
    method_pairs = {}
    for occurrence in read_csv(run_dir / "path_steps.csv"):
        method_pairs.setdefault((occurrence["order"], occurrence["step"]), {})[occurrence["method"]] = occurrence["transition_id"]
    matched = []
    for linear_id, ties_id in sorted({(p["linear"], p["ties"]) for p in method_pairs.values()}):
        for linear in global_lookup[linear_id]:
            tied = next(r for r in global_lookup[ties_id] if r["dataset"] == linear["dataset"] and r["detector"] == linear["detector"])
            matched.append({"linear_transition_id": linear_id, "ties_transition_id": ties_id,
                "step": linear["step"], "incoming": linear["incoming"], "dataset": linear["dataset"], "detector": linear["detector"],
                "linear_local_change_pp": linear["change_pp"], "ties_local_change_pp": tied["change_pp"],
                "ties_minus_linear_local_change_pp": float(tied["change_pp"]) - float(linear["change_pp"]),
                "ties_deviation_from_sum_norm": tied["deviation_from_sum_norm"],
                **{f"linear_{f}": linear[f] for f in features}, **{f"ties_{f}": tied[f] for f in features}})
    write_csv(run_dir / "matched_method_transitions.csv", matched)
    write_csv(run_dir / "exploratory_correlations.csv", correlations)
    write_csv(run_dir / "ties_endpoint_differences.csv", sorted(endpoints, key=lambda r: (r["representation"], r["level"], r["difference_rank"])))
    write_csv(run_dir / "ties_difference_concentration.csv", concentration)
    write_csv(run_dir / "earlier_analysis_overlap.csv", comparisons)
    control = {"orders": 24, "distinct_final_expressions": 1, "modules_checked": len(controls),
        "canonical_fp32_endpoint_equal": all(c["equal"] for c in controls),
        "max_absolute_difference": max(c["max_absolute_difference"] for c in controls),
        "bf16_endpoint_equal": True,
        "interpretation": "Same canonical FP32 sum and one BF16 cast, by source construction. Not a test of arbitrary-order in-place BF16 addition."}
    write_json(run_dir / "linear_control.json", control)
    write_json(run_dir / "analysis_summary.json", {
        "transition_counts": read_json(run_dir / "prepared.json")["transition_counts"],
        "endpoint_global": [r for r in endpoints if r["level"] == "global"],
        "earlier_highlighted_modules": [r for r in comparisons if r["previously_highlighted"]],
        "notes": ["Explanatory associations only; localization of weight differences is not proof of causal safety effects.",
            "Extreme paths were selected using existing ASR results; CIs are descriptive and unadjusted for selection/multiplicity.",
            "Repeated paths are deduplicated; transitions still share adapters and states.",
            "Correlations exclude the first integration and use exact flagged-count changes to preserve ties.",
            "No prediction model or new generation/judging is run.",
            "Original ordering-run adapter/base revisions were not pinned. Current adapter hashes match the earlier pairwise analysis.",
            "Transition geometry is measured in the FP32 merge space. Endpoint comparisons also include reconstructed BF16 effective updates."]})
    plots(run_dir, cfg, geometry, endpoints, labels)
    print(f"Analysis complete: {run_dir}", flush=True)


def plots(run_dir, cfg, geometry, endpoints, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    representation = "checkpoint_effective_bfloat16"
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    for axis, level in zip(axes, ("module", "block", "module_type")):
        rows = sorted([r for r in endpoints if r["representation"] == representation and r["level"] == level],
                      key=lambda r: -r["difference_sq"])[:12]
        axis.barh([r["location"].replace("model.layers.", "L") for r in rows][::-1],
                  [100 * r["difference_energy_fraction"] for r in rows][::-1])
        axis.set_xlabel("Share of squared endpoint difference (%)")
        axis.set_title(level.replace("_", " "))
    fig.suptitle("TIES lower-ASR vs higher-ASR endpoint: BF16 effective weight differences")
    fig.savefig(out / "ties_endpoint_localization.pdf")
    plt.close(fig)
    features = ["cosine", "incoming_over_composite_norm", "incoming_relative_to_base",
                "top10_sign_conflict", "top20_sign_conflict", "top50_sign_conflict"]
    for method in ("linear", "ties"):
        for detector in ("llama_guard_3", "harm_cls"):
            fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), layout="constrained")
            for axis, feature in zip(axes.flat, features):
                points = 0
                for row in geometry:
                    if row["level"] != "global" or row["method"] != method or row["step"] == 1 or row[feature] is None:
                        continue
                    label = next(l for l in labels[row["transition_id"]] if l["dataset"] == "harmbench" and l["detector"] == detector)
                    color = {"a1": "C0", "a2": "C1", "a3": "C2", "a4": "C3"}[row["incoming"]]
                    axis.scatter(row[feature], float(label["change_pp"]), color=color, s=18, alpha=.8)
                    points += 1
                axis.axhline(0, color="0.6", linewidth=.7)
                if not points:
                    axis.text(.5, .5, "No defined measurements", ha="center", va="center", transform=axis.transAxes)
                axis.set_xlabel(feature.replace("_", " "), fontsize=9)
                axis.set_ylabel("Local ASR change (pp)")
            handles = [plt.Line2D([], [], marker="o", linestyle="", color=f"C{i}", label=f"incoming A{i+1}") for i in range(4)]
            fig.legend(handles=handles, loc="outside lower center", ncol=4)
            fig.suptitle(f"{method.upper()} / HarmBench / {detector}: distinct non-base transitions")
            fig.savefig(out / f"transition_geometry__{method}__{detector}.pdf")
            plt.close(fig)
    plan = read_json(run_dir / "source_plan.json")
    rates = {(r["state"], r["dataset"], r["detector"]): float(r["rate_pct"]) for r in read_csv(run_dir / "state_rates.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for axis, detector in zip(axes, ("llama_guard_3", "harm_cls")):
        for method in ("linear", "ties"):
            for order in (cfg["lower_asr_order"], cfg["higher_asr_order"]):
                axis.plot(range(5), [rates[s, "harmbench", detector] for s in plan["paths"][method][order]], marker="o",
                          label=f"{method}: {order.replace('a', '').replace('>', '-')}")
        axis.set_title(detector)
        axis.set_xlabel("Adapters integrated")
        axis.set_ylabel("HarmBench ASR (%)")
        axis.set_xticks(range(5))
        axis.legend(fontsize=8)
    fig.savefig(out / "selected_trajectories.pdf")
    plt.close(fig)
