# Adapter similarity to the complete LINEAR SUM update

Run from the repository root with standard Python (no additional dependencies):

```bash
python linear_sum_similarity/run.py
```

Writes one `results.csv`, with four rows and two columns: `adapter` and
`cosine_similarity`.

Here `Mi = base`, `Mf = base + D1 + D2 + D3 + D4`, and
`Di = (alpha_i / rank_i) * Bi @ Ai`. Each row reports `cos(Di, Mf - Mi)` over
all flattened adapted weights, using weight 1 for every adapter.

The completed pairwise run already contains the norms and all six pairwise dot
products. The calculation uses these sufficient statistics:

```text
<Di, sum(Dj)> = ||Di||² + sum_{j != i} <Di, Dj>
||sum(Dj)||² = sum_j ||Dj||² + 2 * sum_{j < k} <Dj, Dk>
```

This calculates similarity to the mathematical sum of the saved FP32 adapter
updates; it does not include rounding from constructing and subtracting BF16
model checkpoints. No models are loaded and no generation is run.

Default input: `pairwise_experiment/results/20261120_162057_002837/`.
Use `--source PATH` for a different completed pairwise run or `--out FILE.csv`
to change the output path. Original pairwise results are only read.
