# Experiment 2: merge all four adapters

13 configs copied from the current root config.yaml, with only model.adapters
and merge settings changed. No runs have been launched for this experiment.

| Strategy | Densities | Configs |
| --- | --- | --- |
| linear | unused (null) | 1 |
| ties | 0.25, 0.5, 0.75, 1.0 | 4 |
| dare_linear | 0.25, 0.5, 0.75, 1.0 | 4 |
| dare_ties | 0.25, 0.5, 0.75, 1.0 | 4 |

All configs merge these adapters together, each with weight 0.5:

- a1: Akarshika/qwen25-3b-text2sql-lora
- a2: madokalif/qwen2.5-3b-translation-warmup-adapter
- a3: balaguhanesh/tool-call-ft
- a4: Pilestin/qwen2.5-3b-debug-v1-lora

Majority sign method is total; svd_rank is null. No SVD strategies are included.
The filename suffix d025/d050/d075/d100 identifies density.
See manifest.csv for the complete matrix.

To run an individual config manually from the project root:

```bash
modal run modal_app.py --config "experiment 2/all4_ties_d050.yaml"
```

Validation: all 13 YAML files parse with bench.config.load_config, and their
settings match the template outside the adapters and merge sections.
Adapter compatibility has not been tested by executing merges.

## Run all 13 configs concurrently

From the project root:

```bash
env/bin/python "experiment 2/run_batches.py"
```

The launcher starts all 13 configs in manifest.csv order without waiting between
launches. It waits for all jobs to finish and exits nonzero if any run fails.

Each invocation creates a new timestamped directory inside experiment 2/logs,
containing a separate .log (stdout and stderr), .pid, and .exit_code for each
config, plus summary.csv. Existing logs are preserved. Keep the launcher running
until all runs finish. Ctrl+C stops the remaining local Modal clients.

Optional arguments: --modal /path/to/modal and --logs-root /path/to/logs.
The script has been tested with a local mock executable; no benchmarks were launched.
