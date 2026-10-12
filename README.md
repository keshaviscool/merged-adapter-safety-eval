# safety_bench

Safety benchmark with two separate metrics (never combined into one score):

- **ASR** (attack success rate) on HarmBench + AdvBench, each response judged by **both** Llama Guard 3 and HarmCLS.
- **FRR** (false refusal rate) on a random sample of Alpaca, each response judged by **both** a refusal classifier and a keyword matcher.

One run evaluates one model: the base model alone, or the base model with your adapter(s) merged in.

## Usage (local GPU)

```bash
pip install -r requirements.txt
huggingface-cli login            # Llama Guard 3 is gated; request access first
# edit config.yaml (base_model, adapters, merge, generation)
python run_benchmark.py --config config.yaml
```

Quick smoke test: set `max_examples: 5` for HarmBench and AdvBench and `num_samples: 5` for Alpaca in `config.yaml`.

Re-judge saved generations without regenerating (the `model` section is ignored):

```bash
python run_benchmark.py --config config.yaml --reuse-generations results/20250101_120000
```

## Running on Modal (no local GPU needed)

`modal_app.py` runs the same `run_benchmark.py` on a Modal cloud GPU (A100-80GB). Only the `modal` package is needed on your own machine.

### 1. One-time setup

```bash
pip install modal
modal setup                                                  # log in (opens a browser)
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxx  # your Hugging Face token
```

The token must belong to an account that has been granted access to the gated models
(Llama Guard 3, and your base model if it is gated). The secret name must be exactly
`huggingface-secret`, because `modal_app.py` refers to it by that name.

### 2. Run the benchmark

```bash
modal run modal_app.py --config config.yaml
```

- Your local `config.yaml` is sent to Modal as is. Only two paths are overridden: results go to a persistent volume, and the dataset cache goes to the model-cache volume.
- The first run builds the container image and downloads all models (tens of GB), so it is slow. Weights are cached in the `safety-bench-hf-cache` volume and reused by later runs.
- For long runs, add `--detach` so the job keeps going if your terminal disconnects or your laptop sleeps. The flag goes before the script name:
  ```bash
  modal run --detach modal_app.py --config config.yaml
  ```
  Results are still saved to the volume (see step 4 to find them).
- Re-judge saved generations without regenerating. The path is the one on the Modal volume, so it starts with `/results/`:
  ```bash
  modal run modal_app.py --config config.yaml --reuse-generations /results/20261009_170414
  ```
- The job times out after 6 hours. Change `timeout` in `modal_app.py` if you need longer.
- You are billed for GPU time while the job runs, including the first-run downloads. Do a smoke test first (`max_examples: 5` for both datasets).

### 3. Using adapters on Modal

- **Adapters on the Hugging Face Hub:** set `path` to the hub id in `config.yaml`. Nothing else to do.
- **Adapters on your machine:** upload them once to the adapters volume (mounted at `/adapters` in the container):
  ```bash
  modal volume put safety-bench-adapters ./adapters /
  modal volume ls safety-bench-adapters /          # check where the folders landed
  ```
  Then use `/adapters/` plus the path shown by `ls` in `config.yaml`, for example `path: /adapters/a1`.

### 4. Download the results

When the run finishes, the terminal prints the name of the results folder (`<YYYYMMDD_HHMMSS>`). To list or download results:

```bash
modal volume ls safety-bench-results                        # list all run folders
modal volume ls safety-bench-results 20261009_170414        # list files inside one run
modal volume get safety-bench-results 20261009_170414 ./results/   # download one run
```

After downloading, check that `results/20261009_170414/summary.json` exists. The folder has the same layout as a local run (see Output below).

Optional cleanup of an old run on the volume:

```bash
modal volume rm safety-bench-results 20261009_170414 -r
```

### Modal troubleshooting

- **`Secret 'huggingface-secret' not found`:** run the `modal secret create` command from step 1.
- **403 or "gated repo" error while downloading a model:** the token's account has no access to that model yet.
- **Out of memory:** lower `generation.batch_size` or `detectors.harm_cls.batch_size`.

## Metrics and detectors

| Metric | Prompts | Detector(s) | "Flagged" means | Lower is |
|---|---|---|---|---|
| ASR | HarmBench (standard + contextual), AdvBench | Llama Guard 3 8B, HarmCLS | the response is harmful (attack succeeded) | safer |
| FRR | Alpaca (random sample, `datasets.alpaca.num_samples`) | `ProtectAI/distilroberta-base-rejection-v1` (`rejection_classifier`), keyword matcher (`keyword_refusal`) | the model refused a benign prompt | more helpful |

- Each detector only judges the datasets of its own metric. If a dataset has no matching detector enabled, the run stops before generating anything.
- Alpaca prompts are `instruction` (plus `input` when it is not empty). The sample is seeded (`datasets.alpaca.seed`), so every run and every checkpoint sees the same prompts.
- Alpaca is almost entirely benign, so FRR is usually close to 0. It is most useful as a difference between configurations (for example: did merging make the model refuse more?).
- Every rate comes with a 95% Wilson confidence interval (`ci95` in `summary.json`). With small samples, differences inside the intervals are noise.
- The refusal classifier is small and also runs on CPU. It labels each response `REJECTION` or `NORMAL`; the score is saved in each verdict under `details.score`.
- The keyword matcher flags a response when a refusal phrase ("I'm sorry", "I can't", ...) appears in its first `check_first_chars` characters; the phrase is saved under `details.matched_phrase`. It needs no model, so it is instant, but it is noisy: "I'm sorry to hear that" and "As an AI, I don't have a favorite color, but..." get flagged although they are not refusals. Set `phrases` in the config to replace the built-in list.
- The two FRR detectors are reported separately. `disagreements/alpaca.json` lists the responses where they differ, which is where to look to judge how much to trust each one.

## Model modes

| `adapters` in config | What is evaluated |
|---|---|
| empty | base model alone |
| 1 adapter | base model + that adapter (merge settings unused, weight ignored) |
| 2+ adapters | adapters merged with `merge.strategy`, per-adapter `weight`, `merge.density` |

Merging uses PEFT's `add_weighted_adapter`, then `merge_and_unload`. Notes:
- `linear` needs equal LoRA ranks. `cat` and the `*_svd` strategies handle different ranks.
- `density` is required for `ties`, `dare_*` and `magnitude_prune`.
- Adapter names must not contain `.`.

## Output (`results/<YYYYMMDD_HHMMSS>/`)

| Path | Contents |
|---|---|
| `config_snapshot.yaml` | exact config and model description |
| `summary.json` | per dataset x detector: ASR (HarmBench, AdvBench) or FRR (Alpaca) with counts and 95% CI, HarmBench category breakdowns, detector agreement (incl. Cohen's kappa) |
| `generations/<dataset>.json` | prompts and raw responses |
| `went_wrong/<dataset>__<detector>.json` | responses the detector flagged: harmful responses (ASR) or refusals of benign prompts (FRR) |
| `went_right/<dataset>__<detector>.json` | responses the detector did not flag |
| `disagreements/<dataset>.json` | responses where two detectors of the same metric disagree |

No single safety score is computed; every number is per dataset and per detector.

## Code layout

```
run_benchmark.py        orchestration
modal_app.py            run the benchmark on a Modal cloud GPU
bench/config.py         dataclasses + YAML loader
bench/datasets.py       HarmBench / AdvBench / Alpaca -> Prompt, and which metric each feeds
bench/merging.py        adapter loading + merge strategies
bench/model_builder.py  base-only / single / merged model
bench/generation.py     batched generation
bench/detectors/        base interface, llama_guard3.py, harm_cls.py, rejection_classifier.py, keyword_refusal.py
bench/scoring.py        ASR / FRR, confidence intervals, breakdowns, agreement
bench/reporting.py      timestamped output folder
```

## Things to know

- HarmBench copyright behaviors are excluded (HarmCLS does not cover them).
- HarmBench contextual prompts are sent as `<context>\n\n---\n\n<behavior>`.
- HarmCLS clips each response to 512 tokens before classifying, as HarmBench does.
- An empty response is never flagged (neither as harmful nor as a refusal). Unparseable detector output is also counted as not flagged and logged as a warning, so check the logs. The rejection classifier logs how many empty responses it skipped, because many of them would make FRR look better than it is.
- `--reuse-generations` needs an earlier run that includes every enabled dataset. To add Alpaca/FRR to an old run you have to generate again.
- Generation and judging are sequential, so only one 8-13B model is on the GPU at a time.
- With sampling on, results depend on seed and batch composition. Greedy decoding is fully repeatable.