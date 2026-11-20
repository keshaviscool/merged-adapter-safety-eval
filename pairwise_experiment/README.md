# Pairwise composition experiment

All six pairs of the four existing Qwen2.5-3B adapters, under five merge rules.
The primary composition effect is always:

```text
effect_pp = ASR(merged pair) - max(ASR(original singleton i), ASR(original singleton j))
```

Rates are percentages and effects are percentage points. Positive = amplification;
negative = attenuation. Classifiers and datasets are reported separately.
The singleton reference stays at its original full strength for **every** variant.

## Run on the server

From the repository root, using the existing Python environment:

```bash
pip install -r pairwise_experiment/requirements.txt
python pairwise_experiment/run.py --profile rtx5000
```

For an H100 80 GB:

```bash
python pairwise_experiment/run.py --profile h100
```

Use one visible GPU, with enough free memory for a BF16 13B judge. If the machine
has several GPUs, prefix the command with `CUDA_VISIBLE_DEVICES=0`. Hugging Face
access to Llama Guard 3 is required, as in the existing experiments. Model and
adapter downloads use the usual Hugging Face cache. No Modal account is used.

The completed source run must exist on the server at:

```text
ordering_experiment/results/20261117_045604
```

Change `pairwise.source_run` in `config.yaml` if its location differs. The script
copies the base, four singleton and twelve pair results after validating their
configuration, prompts and verdict counts. It never modifies the source run.
It then generates **18 new pairs**, judges them, and produces the analysis.
Prompts come directly from the completed run, so there is no dataset resampling
or download. All HarmBench/AdvBench examples and the same 500 Alpaca prompts
are retained; generation remains greedy with a 512-token output limit.

## Batch sizes

These are starting values, not measured throughput optima. H100 judge sizes
match the previously successful settings in this project.

| Setting | RTX 5000 Ada 32 GB | H100 80 GB |
|---|---:|---:|
| Generation | 32 | 128 |
| Llama Guard 3 | 8 | 32 |
| HarmBench classifier | 2 | 16 |
| Alpaca rejection classifier | 32 | 64 |

Edit the **profiles at the bottom** of `config.yaml`:

```yaml
profiles:
  rtx5000:  # or h100
    generation:
      batch_size: 32
    detectors:
      llama_guard3:
        batch_size: 8
      harm_cls:
        batch_size: 2
      rejection:
        batch_size: 32
```

The selected profile overrides the corresponding top-level batch settings.
Generation and judging halve failed batches on CUDA OOM, preserve prompt order,
and log completed batch sizes in `run.log`. An OOM at batch size 1 stops with an
error. Only one model phase runs at a time; separate processes release GPU
memory between geometry, generation and judging.

## Merge definitions

First reconstruct each full update in FP32:

```text
s_i = alpha_i / rank_i
Delta_i = s_i * (B_i @ A_i)
n_i = sqrt(sum of Delta_i[element]**2 over all adapted matrices)
```

The Frobenius norm is calculated **after applying alpha/rank**. The method names
in the results mean:

| Method | Update added to the base |
|---|---|
| `linear_sum` | `Delta_i + Delta_j` |
| `linear_average` | `0.5 * (Delta_i + Delta_j)` |
| `coefficient_normalized` | `(Delta_i + Delta_j) / sqrt(s_i**2 + s_j**2)` |
| `frobenius_normalized` | `(Delta_i + Delta_j) / sqrt(n_i**2 + n_j**2)` |
| `ties` | Existing binary TIES, density 0.5, imported from the ordering run |

Normalization applies one common coefficient to both updates, retaining their
relative scaling. This joint Frobenius rule does not force the *sum's* norm to
one, because its denominator excludes the cross inner product. There is no
additional rescaling to match the unnormalized pairs' strength.

`linear_sum` and `ties` reuse existing results. The other three variants are new.
Every newly constructed model starts from the original base weights; the FP32
sum is cast once to the base model dtype. This is dense-update addition, rather
than addition of LoRA A/B factors through PEFT's `linear` adapter operator.

Plain LoRA and rsLoRA are supported (`alpha/sqrt(rank)` for rsLoRA). DoRA,
rank/alpha patterns, transposed weights, bias updates and extra saved modules
are rejected explicitly, rather than silently measured incorrectly. For
rsLoRA, coefficient normalization uses its actual `alpha/sqrt(rank)` scale.

## Weight-space measurements

Compare the **original scaled updates** for each pair, independent of the merge
variant. Missing modules are zero updates. Process only one module at a time;
do not build four full dense model deltas in memory.

- Cosine similarity of flattened updates.
- Each update's Frobenius norm, `norm_i/norm_j`, the symmetric balance
  `min(norms)/max(norms)`, and each norm relative to base weights.
- Sign conflict: opposite-sign entries divided by jointly nonzero entries.
- High-magnitude overlap: Jaccard of top-50%-magnitude masks per matrix.
  Threshold ties can retain more than 50%; actual retained fractions are saved.
  Zero entries are excluded from masks.

Save module, transformer-block and global measurements. Block/global values
aggregate dot products, squared norms and coordinate counts; they are not
unweighted averages of module cosines or percentages. Global overlap aggregates
the per-matrix masks rather than selecting a new global top half. Undefined
cosines, ratios or conflict fractions are blank in CSV, not fabricated zeros.

Correlations use the six **global pair-level** observations for cosine, norm
balance, sign conflict and overlap, separately for every method, harmful dataset
and judge. Spearman correlations are descriptive and exploratory. The six pairs
share adapters; neither these pairs nor individual layers provide independent
replications for confirmatory inference.

## Results and resuming

The command prints a timestamped run folder under `pairwise_experiment/results/`.
Key outputs:

- `composition_effects.csv`: all 30 pair configurations, both ASR judges on both
  harmful datasets, supplementary Alpaca FRR differences, and paired 95% CIs.
- `rates.csv`: base, four singletons and all pairs; counts, rates, parse errors,
  empty-response counts and whether results were imported or newly generated.
- `geometry.csv`: all pair measurements at module, block and global levels.
- `adapter_norms.json`, `coefficients.csv`: scaled norms, actual merge weights,
  adapter metadata and current file hashes.
- `correlations.csv`, `figures/`: six-pair Spearman results, labelled scatter
  plots with effect intervals and a composition-effect heatmap.
- `states/`: saved generations and prompt-aligned verdicts. Verdict files
  include a generation-file SHA256 to detect stale caches.
- `imports.json`, `environment/`, `config_used.yaml`, `run.log`: source hashes,
  original settings, runtime packages/GPU details, and execution history.

Resume by passing the printed folder:

```bash
python pairwise_experiment/run.py --run-dir pairwise_experiment/results/RUN_FOLDER
```

Completed geometry, generation files (per state/dataset) and verdict files
(per state/dataset/detector) are reused. An interrupted file is recomputed;
progress within an unfinished dataset is not persisted. You can change batch
sizes with an explicit `--config pairwise_experiment/config.yaml --profile h100`
on resume; scientific settings must match the saved run.

Optional phases are `--phase import`, `geometry`, `generate`, `judge`, `analyze`.
Run them in that order; `all` is the default. `import` and `analyze` need no GPU.
`--dry-run` validates source configuration and prints the planned work without
writing files or loading models. `analyze` expects every pair and geometry to
be complete, and fails on missing/stale records rather than dropping them.

Paired percentile intervals use 20,000 prompt resamples, recomputing the maximum
singleton ASR in each resample. A multinomial over the eight possible joint
binary verdict patterns implements the exact same paired bootstrap distribution
without a large resample-index matrix. These CIs capture prompt uncertainty,
not differences between execution environments. New pairs are explicitly
labelled as comparisons against imported singletons. Greedy generation does
not guarantee bitwise identical outputs across hardware or batch settings.
Original adapter revisions were not pinned in the source run; current adapter
files are hashed, but identity across the two runs cannot be proven from the
saved repository IDs alone. Low Alpaca refusal rates do not establish utility.

## Lightweight checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest pairwise_experiment.test_experiment -v
```

Tests use tiny CPU tensors and temporary records, plus validation against the
saved source results if present. They never download models or run inference.
