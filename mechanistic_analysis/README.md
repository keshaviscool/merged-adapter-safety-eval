# Sequential composition: explanatory weight-space analysis

Uses the completed ordering experiment's **24 LINEAR and 24 sequential TIES
paths**. Reconstructs their weight updates and joins geometric measurements with
their already measured ASR changes. No new generation, judging, predictor
training or checkpoint intervention is performed.

## Run

From the repo root, in the server's existing environment:

```bash
pip install -r mechanistic_analysis/requirements.txt
python mechanistic_analysis/run.py
```

The same command works on an RTX 5000 Ada 32 GB or H100 80 GB. There is **no
inference batch size**: tensor calculations process one adapted matrix at a
time. One BF16 Qwen base model is loaded for base-relative magnitudes and
checkpoint-rounding measurements. LoRA factors stay on CPU; only the current
matrix's updates and masks are reconstructed on GPU. No dense adapter/model
checkpoints are saved. The default device is CUDA; CPU-only preparation is
available separately.

Keep these existing results on the server (paths are in `config.yaml`):

```text
ordering_experiment/results/20261117_045604
pairwise_experiment/results/20261120_162057_002837
```

The second folder supplies adapter hashes and the earlier layerwise geometry.
Hugging Face needs access only to the base and adapter repositories, not the
judges. Original results are only read.

The command prints a timestamped run folder under `mechanistic_analysis/results/`.
Resume with:

```bash
python mechanistic_analysis/run.py --run-dir mechanistic_analysis/results/RUN_FOLDER
```

Completed matrices are cached atomically under `module_cache/`; interruption
recomputes only the unfinished matrix. Configuration changes require a new run.
Source-file hashes are checked on resume. The current adapter hashes must match
the earlier pairwise run. A resolved base revision is saved and reused on
resume, when available.

Optional phases:

```bash
python mechanistic_analysis/run.py --phase prepare
python mechanistic_analysis/run.py --run-dir PATH --phase geometry
python mechanistic_analysis/run.py --run-dir PATH --phase analyze
```

`prepare` validates the saved verdicts/prompts and computes ASR comparisons on
CPU. `geometry` runs on the server GPU. `analyze` aggregates cached geometry and
creates CSVs/plots on CPU. Default `all` runs these in order, with geometry in a
separate process so its model is released before analysis.

## 1. Current composite versus incoming adapter

Reconstruct every adapter as `D_i = (alpha_i/r_i) B_i A_i` in FP32 (or its
recorded rsLoRA scaling). LINEAR uses the original canonical sum; sequential
TIES uses the original trim/elect-sign/disjoint-mean operator and saved density
0.5. No replacement merge library is used.

At each integration, compare current accumulated update `C` with incoming
update `D`, then attach:

```text
local ASR change = ASR(after) - ASR(before)
cumulative change = ASR(after) - ASR(original base)
```

There are **32 distinct LINEAR transitions and 40 distinct TIES transitions**.
Repeated occurrences on different paths share one measurement. All 192 path
steps retain references to these measurements in `path_steps.csv`.

Measurements include cosine, both update norms relative to the corresponding
base weights, incoming/composite norm ratio, high-magnitude sign conflict,
overlap, and magnitude-weighted conflict. The actual integration-step norm
`||after-C||` and deviation from additive merging `||after-(C+D)||` help describe
TIES' nonlinear changes. LINEAR's latter deviation can include tiny FP32
summation-order rounding differences, since its sum uses canonical ordering.

Save per-matrix, transformer-block, module-type (`q_proj`, `down_proj`, etc.)
and global results. Aggregates combine squared norms, dot products and support
counts; they do not average module cosines or percentages.

The first integration starts from a zero accumulated update. Its cosine,
incoming/composite ratio and conflict are undefined and saved as blanks. It
remains in the ASR tables, but is excluded from geometry–ASR correlations.

This analysis consistently uses the ordering run's sequential chains and ASRs.
The separate PEFT sequential run has different casting behaviour and its own
baseline; its results are not pooled into these relationships.

## 2. Large-update conflict: top 10%, 20%, 50%

For each fraction, independently select the largest absolute entries of `C`
and `D` **per weight matrix**. Fractions refer to all matrix coordinates, as in
the original TIES implementation. Zeros are excluded and threshold ties are
retained, so actual retained fractions can differ from the requested fraction,
especially for sparse TIES composites. Both retained counts and energy
fractions are saved.

Primary conflict is the opposite-sign fraction on the **intersection** of the
two high-magnitude masks: both updates must be large at that coordinate.
An empty intersection gives an undefined conflict, not zero conflict.

Also save:

- Jaccard overlap: intersection count / union count.
- Magnitude-weighted conflict: sum of `|C*D|` on conflicting intersection
  coordinates / sum of `|C*D|` on all intersection coordinates.
- A sensitivity measure on the union masks, restricted to jointly nonzero
  entries, so a large entry in either update can enter the comparison.
- Raw all-nonzero sign conflict as a reference, rather than the main signal.

Block/type/global masks aggregate these per-matrix selections; no new global
top-fraction threshold is applied after concatenation.

## 3. Compare the selected TIES endpoints

Default paths: lower HarmBench/Llama Guard endpoint `1-4-2-3`, higher endpoint
`3-4-2-1`. Preparation verifies they are extrema in the saved results. Their
observed HarmBench ASRs are 20.33% versus 25.33% for Llama Guard, and 19.67%
versus 24.67% for HarmBench classifier. All datasets/detectors are retained.

Compare two representations:

- `merge_update_fp32`: the effective TIES expression before the model cast.
- `checkpoint_effective_bfloat16`: `BF16(W0 + update) - W0`, matching the
  ordering runner's once-cast construction from BF16 base weights.

Rank matrices, blocks and module types by squared endpoint difference. Save
their share of total squared difference, cumulative share, mean squared
difference per coordinate, base-relative difference norm, endpoint cosine,
changed-coordinate count, and high-magnitude conflict/overlap. This separates
absolute concentration from size-normalized differences.

Compare difference ranks with earlier pairwise cosine/overlap ranks. The three
previously highlighted matrices are listed explicitly in `config.yaml`; they
are not selected after seeing the new endpoint comparison.

## 4. LINEAR control and interpretation

Verify that every LINEAR path reaches the same canonical expression and that
its reconstructed endpoint matrix is identical across the 24 paths. This also
implies identical once-cast BF16 endpoint matrices. The control is **shared by
source construction**, not an independent claim that arbitrary-order in-place
BF16 addition is associative. Different intermediate LINEAR states remain in
the trajectory tables.

Geometry–ASR correlations are exploratory Spearman associations, separately by
method, harmful dataset, classifier and location. Tables include pooled
non-base transitions and step-specific results. Equal ASR changes are ranked
using integer flagged-count differences to avoid floating-point tie errors.
No p-values or automatic significance claims are produced. Shared adapters,
shared states and selection of extreme paths prevent treating these rows or
layers as independent replications. Endpoint/local ASR intervals are paired
percentile bootstrap intervals with 20,000 resamples; they are unadjusted for
selection and multiple comparisons.

Localizing a weight difference does **not** establish that those layers cause
the safety difference. Source model revisions were not pinned in the ordering
run; matching current adapter hashes to the earlier pairwise run and saving
the current base revision improves reproducibility but cannot prove exact
identity to the original evaluated checkpoints.

## Main outputs

| File | Contents |
|---|---|
| `transition_effects_global.csv` | Each distinct transition's geometry joined with local/cumulative ASR or FRR change and local paired CI |
| `transition_effects_by_layer.csv` | Block geometry joined with local ASR for both harmful datasets and judges |
| `transition_geometry.csv` | Full matrix/block/type/global geometry and support counts |
| `selected_path_interactions.csv` | Integration geometry and outcomes along both selected orders, for LINEAR and TIES |
| `matched_method_transitions.csv` | LINEAR/TIES comparisons at the same integration order and step, deduplicated; descriptive differences, not causal operator effects |
| `ties_endpoint_differences.csv` | Ranked endpoint differences, norms, cosine and high-magnitude relationships in both precision representations |
| `ties_difference_concentration.csv` | Fraction of squared endpoint difference in the top 1/3/5/10/20 locations |
| `earlier_analysis_overlap.csv` | Endpoint difference ranks alongside previous layerwise geometry and preselected-module flags |
| `exploratory_correlations.csv` | Distinct-transition correlations, globally and per block/type, pooled and step-specific |
| `linear_control.json` | LINEAR endpoint-invariance check and its construction-specific interpretation |
| `ties_endpoint_asr.csv` | Selected endpoints' paired behavioural comparison |
| `figures/` | Endpoint localization, transition geometry versus local ASR, selected LINEAR/TIES ASR trajectories |

Supporting files contain raw ASR labels, path mappings, configuration snapshots,
source hashes, adapter/base metadata, environment details and resumable caches.
There is no generated prediction model or additional model benchmark.

## Lightweight validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest mechanistic_analysis.test_analysis -v
```

Tests use tiny tensors and existing saved verdicts. They verify the 32/40
transition counts, reconstruction against the original operators, LINEAR
invariance, large-update conflict/support handling, BF16 rounding, paired ASR
changes, aggregation/concentration, source checks and the output pipeline.
They do not download models or adapters.
