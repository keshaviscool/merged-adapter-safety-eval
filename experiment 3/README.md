# Experiment 3: cumulative adapter checkpoints, ASR only

Measure how harmful-response rates change as adapters are added in this fixed
order: A1 → A1+A2 → A1+A2+A3 → A1+A2+A3+A4. Every checkpoint starts from the
original Qwen/Qwen2.5-3B-Instruct base and merges its adapter list together.
These are cumulative subsets, not sequential modifications of a saved model.

| Adapter | Hugging Face ID | Weight |
| --- | --- | --- |
| A1 | Akarshika/qwen25-3b-text2sql-lora | 0.25 |
| A2 | madokalif/qwen2.5-3b-translation-warmup-adapter | 0.25 |
| A3 | balaguhanesh/tool-call-ft | 0.25 |
| A4 | Pilestin/qwen2.5-3b-debug-v1-lora | 0.25 |

## Config matrix and instances

| Checkpoint | Linear | TIES | DARE-linear | DARE-TIES | Total |
| --- | --- | --- | --- | --- | --- |
| A1 only, shared control | 1 | — | — | — | 1 |
| A1+A2 | 1 | 2 | 2 | 2 | 7 |
| A1+A2+A3 | 1 | 2 | 2 | 2 | 7 |
| A1+A2+A3+A4 | 1 | 2 | 2 | 2 | 7 |
| **Total** | **4** | **6** | **6** | **6** | **22** |

TIES and both DARE variants use densities 0.5 and 0.75. Linear uses no density.
No SVD variants are included. Majority sign method is total. Weights stay at
0.25 per adapter at every checkpoint; they are not renormalized by adapter count.
The A1 control is evaluated once and shared across all seven method/density variants.
See manifest.csv for all 22 configurations.

Running the launcher requests **22 concurrent Modal instances, each with one
A100-80GB GPU**. Actual scheduling depends on workspace GPU limits and availability.

## Evaluation settings

- Generation batch size: **32** (previously 16).
- HarmBench classifier batch size: **16** (previously 8).
- HarmBench and AdvBench enabled, with all examples from the template.
- Llama Guard 3 and HarmBench classifier enabled for ASR. Llama Guard remains unbatched.
- Alpaca, rejection classifier, and keyword refusal detector disabled: **no FRR evaluation**.
- Other inference settings follow the root config.yaml, including 512 maximum new tokens and seed 42.

The larger batch sizes have not been tested on a GPU. Model downloads, actual
adapter compatibility, runtime, and cost have not been verified for this matrix.

## Run manually

From the project root, launch all 22 configs:

```bash
env/bin/python "experiment 3/run_all.py"
```

Or run a single checkpoint:

```bash
modal run "experiment 3/modal_app.py" --config "experiment 3/stage2_a1_a2_ties_d050.yaml"
```

Use this experiment's modal_app.py so the A1-only weight is applied correctly.
The experiment-local weighted_merging.py uses PEFT's weighted linear merge for
the single-adapter control; the shared runner ignores single-adapter weights.
benchmark_entrypoint.py installs that local implementation before invoking the
shared benchmark. Shared project source and earlier experiments are unchanged.

Each launcher invocation writes separate stdout/stderr logs, PIDs, exit codes,
and summary.csv under a new timestamped experiment 3/logs directory. Keep the
launcher running until all clients finish; Ctrl+C stops remaining local clients.
The launcher exits nonzero if any job fails. Optional flags:
--modal /path/to/modal and --logs-root /path/to/logs.

Remote results use the safety-bench-results volume under
experiment3/<config-stem>/<timestamp>/ to separate concurrent checkpoint outputs.
Each completed log prints its exact result path and download command.

## Local validation

All 22 configs parse through the benchmark's config loader. The cumulative
matrix, weights, batch sizes, disabled FRR settings, and unchanged template
settings were checked. A tiny CPU-only LoRA test verified that the single-adapter
control applies weight 0.25. A mock launcher test verified 22 concurrent starts,
separate logs and exit codes, and failure reporting. These checks used no model
downloads or Modal GPU runs.

**No Modal benchmarks have been launched as part of creating this experiment.**
