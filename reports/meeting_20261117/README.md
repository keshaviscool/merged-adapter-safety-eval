# LoRA security composition: meeting report

`meeting_report.pdf` contains the two-page, two-column results brief followed
by two landscape appendix pages: one for LINEAR and one for TIES.
`meeting_report.tex` and `figures/compact_*.pdf` are its editable LaTeX source.
The brief focuses on what was run and the observed results, with three figures,
two tables, and key paired confidence intervals. Each appendix includes all 24
orders, all four integration checkpoints and the shared baseline, covering both
ASR classifiers on HarmBench/AdvBench and both FRR detectors on Alpaca.
Appendix rates have two decimals; `data/complete_results.csv` preserves full precision.
Full trajectories and additional comparisons remain available in `data/`.

The report covers the completed fixed-chain run `20261117_001902` and the
LINEAR/TIES ordering run `20261117_045604`. The terminology in the report is
**adapter ordering**, not adaptive selection: all permutations were enumerated.

## Rebuild the existing LaTeX report

From the repository root:

```bash
tectonic reports/meeting_20261117/meeting_report.tex
```

Or upload `meeting_report.tex`, `figures/`, and `tables/raw_*.tex` to Overleaf and set
`meeting_report.tex` as the main file. Standard pdfLaTeX packages suffice.

## Recompute the analysis and figures

Requirements: Python 3.11+, numpy, matplotlib, PyYAML. Optional PDF inspection
uses pypdf and Poppler. No model loading, torch, CUDA, or remote calls are needed.
The two original result folders must be present at their repository locations.

```bash
env/bin/python reports/meeting_20261117/analyze_results.py
env/bin/python reports/meeting_20261117/compact_figures.py
env/bin/python reports/meeting_20261117/raw_tables.py
tectonic reports/meeting_20261117/meeting_report.tex
```

Figures are regenerated from the checked analysis CSVs. The appendix tables are generated from the same CSVs. Summary tables and narrative
claims are written for the two named runs and should be reviewed if inputs change.
`make_tables.py` can also generate supplementary LaTeX tables; those tables are
not required to compile the compact paper.

## Analysis choices and checks

- Match full prompt records across all states and both runs. For sequential
  data, join `went_right` and `went_wrong` records by prompt ID, verifying the
  saved response matches the generation file.
- Check verdict lengths and all detector rates against saved summaries; verify
  the original 2,160-row ordering CSV.
- Paired prompt bootstrap: 20,000 resamples, seed 20261117, percentile 95% CI.
  For binary paired differences, multinomial resampling of {-1,0,+1} counts
  exactly reproduces ordinary paired resampling for the mean-difference statistic.
  Intervals are pointwise, unadjusted, conditional on fixed saved generations
  and judge labels. Extremum comparisons and gate examples are exploratory.
- Recompute gates with 1e-10 pp comparison tolerance to accept exact-boundary
  steps. This corrects floating-point artifacts in the supplied gate summary.
- Compare each experiment against its own baseline. Base response texts differ
  between runs even with greedy decoding; batch sizes are 16 versus 256.
- Do not treat reused paths as independent trials. LINEAR endpoint invariance
  follows from the experiment's canonical summation and result sharing.

## Companion data

- `data/complete_results.csv`: all 240 method/order/checkpoint rows with six
  named-detector rates side by side, without rounding.
- `data/audit.json`: completeness checks, counts, gate witness, and headline metrics.
- `data/input_sha256.json`: hashes of the input files consumed by the analysis.
- `data/state_rates.csv`: all individual state rates, parsing errors, skipped empties.
- `data/all_path_regressions.csv`: every path/checkpoint/dataset/detector,
  plus local and cumulative changes; includes the exploratory `any` union.
- `data/paired_bootstrap.csv`: 78 paired HarmBench comparisons and confidence intervals.
- `data/gates.csv`: complete-path admissions and first reachable cumulative violations.
- `data/ordering_spreads.csv`, `endpoints.csv`, `trajectory_shapes.csv`: ordering results.
- `data/single_adapters.csv`, `ties_last_adapter.csv`, `category_breakdown.csv`:
  singleton and descriptive diagnostic analyses.
- `data/between_run_differences.csv`, `detector_agreement.csv`, `empty_responses.csv`:
  comparability, judge disagreement, and response-quality audit.

The report does not modify the original experiment outputs or claim to run
additional experiments. Two defects in the original automatic gate report
(floating boundaries and endpoint-only labeling of largest cumulative change)
are documented and handled independently here.
