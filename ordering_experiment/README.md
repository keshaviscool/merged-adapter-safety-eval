# Adapter-order experiment

For all 24 orders of the 4 adapters, follows two merge methods checkpoint by checkpoint (M0, after 1, 2, 3, 4 adapters):

- **LINEAR**: `W + sum of the deltas` of the adapters merged so far, delta = `(alpha/rank) * B @ A`. Order cannot change the endpoint.
- **TIES** (density in `config.yaml`): `S1 = delta_1`, `S2 = TIES(S1, delta_2)`, `S3 = TIES(S2, delta_3)`, ... (trim, elect sign, average agreeing entries).

Every distinct model is generated and judged once (46 incl. the base: 15 linear, 30 TIES; the first TIES merge is symmetric, so orders
that differ only in the first two adapters share models). All 24 paths per method are still reported.

```bash
python ordering_experiment/run.py                                   # generate (one worker per GPU) -> judge -> analyze
python ordering_experiment/run.py --run-dir ordering_experiment/results/<ts>     # resume an unfinished run
python ordering_experiment/run.py --run-dir <dir> --phase judge     # only judging (+ analysis)
python ordering_experiment/analyze.py <dir>                         # only the analysis
```

Outputs in `ordering_experiment/results/<timestamp>/`:
- `order_results.csv`: rate for every method x order x checkpoint x dataset x detector (+ `any` = flagged by any detector).
- `ordering_report.md/json`: trajectory effect (spread of the rate across orders at checkpoints 1-3) vs endpoint effect (checkpoint 4, incl. prompt-level verdict disagreement).
- `gate_report.md/json`: local step `ASR_k - ASR_{k-1}` vs cumulative `ASR_k - ASR_0`; counts of paths a local gate (eps) admits but a cumulative gate (tau) rejects, McNemar test of the worst path vs base.
- `states/<model>/`: generations, per-prompt verdicts, summary with 95% CIs. `plan.json` maps every path to its models.

Run `sanity_checks` output (in the log) shows that TIES(a,b) == TIES(b,a) exactly and how much bf16 rounding noise a different linear summation order would add.
