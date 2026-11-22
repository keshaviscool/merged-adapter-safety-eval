"""Cosine of each scaled adapter update with the four-adapter LINEAR SUM.

Mi = base; Mf = base + D1 + D2 + D3 + D4; D = Mf - Mi = sum(Di).
Use saved dot products: <Di, D> = sum_j <Di, Dj>.
No model loading, downloads, torch, or dense weight calculations are needed.
This measures the mathematical update before casting merged weights to BF16.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pairwise_experiment/results/20261120_162057_002837"


def similarities(source):
    with open(source / "adapter_norms.json") as f:
        adapters = json.load(f)
    names = list(adapters)
    if len(names) != 4:
        raise ValueError("Expected exactly four adapters")
    norms_sq = {name: adapters[name]["frobenius_norm"] ** 2 for name in names}
    if any(not math.isfinite(v) or v <= 0 for v in norms_sq.values()):
        raise ValueError("Cosine requires finite, nonzero adapter norms")
    dots = {(name, name): value for name, value in norms_sq.items()}
    seen = set()
    with open(source / "geometry.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["level"] != "global":
                continue
            a, b = row["adapter_i"], row["adapter_j"]
            if a not in adapters or b not in adapters or a == b:
                raise ValueError("Unexpected adapter pair in geometry")
            pair = frozenset((a, b))
            if pair in seen:
                raise ValueError("Duplicate global pair")
            seen.add(pair)
            for name, column in ((a, "x_sq"), (b, "y_sq")):
                if not math.isclose(float(row[column]), norms_sq[name], rel_tol=1e-10):
                    raise ValueError(f"Inconsistent saved norm for {name}")
            dot = float(row["dot"])
            if not math.isfinite(dot):
                raise ValueError("Nonfinite pairwise dot product")
            if abs(dot) > math.sqrt(norms_sq[a] * norms_sq[b]) * (1 + 1e-10):
                raise ValueError("Pairwise dot product exceeds the Cauchy-Schwarz bound")
            dots[a, b] = dots[b, a] = dot
    if seen != {frozenset(pair) for pair in itertools.combinations(names, 2)}:
        raise ValueError("All six global pair measurements are required")

    total_sq = math.fsum(dots[a, b] for a in names for b in names)
    if total_sq <= 0:
        raise ValueError("Total update has zero/invalid norm; cosine is undefined")
    results = []
    for name in names:
        dot_with_total = math.fsum(dots[name, other] for other in names)
        cosine = dot_with_total / math.sqrt(norms_sq[name] * total_sq)
        if not math.isfinite(cosine) or abs(cosine) > 1 + 1e-10:
            raise ValueError("Inconsistent saved dot products produce an invalid cosine")
        results.append({"adapter": name, "cosine_similarity": max(-1., min(1., cosine))})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Completed pairwise result folder")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results.csv"))
    args = parser.parse_args()
    rows = similarities(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["adapter", "cosine_similarity"])
        writer.writeheader()
        writer.writerows(rows)
    print(args.out.resolve())
    for row in rows:
        print(f"{row['adapter']}: {row['cosine_similarity']:.8f}")


if __name__ == "__main__":
    main()
