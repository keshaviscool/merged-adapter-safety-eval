# Experiment 4: paired confidence intervals and DARE robustness

This is a fresh, self-contained study. It does not reuse earlier results or
modify earlier experiments. Creating these files has not launched Modal jobs.

| Method | Density | Independent merge seeds | Runs |
| --- | --- | --- | --- |
| Qwen base | unused | unused | 1 |
| LINEAR | unused | unused | 1 |
| TIES | 0.75 | unused | 1 |
| DARE-linear | 0.75 | 42, 43, 44, 45, 46 | 5 |
| DARE-TIES | 0.75 | 42, 43, 44, 45, 46 | 5 |

All merges start from Qwen/Qwen2.5-3B-Instruct and combine the same four
adapters as Experiment 3, in the same order, each with weight 0.25. No SVD.
Majority sign method is total. The manifest records the independent merge seed;
the YAML generation seed stays fixed at 42. DARE is seeded immediately before
the weighted merge, then Python, NumPy, and PyTorch RNG states are restored.

## Dataset and evaluation

The frozen CSV in data/ contains the 400 text behaviors. This study evaluates
every supported standard/contextual behavior: 200 standard + 100 contextual =
300 prompts. Copyright behaviors need separate evaluation, and multimodal
behaviors are outside this text-model study. max_examples is null.

Only Llama Guard 3 and the HarmBench classifier are enabled. AdvBench, Alpaca,
and refusal detectors are disabled. Generation uses sampling, seed 42, 512
maximum new tokens, temperature 0.7, top_p 0.95, top_k 50, and batch size 32.
The HarmBench classifier batch size is 16. The larger batches and real adapters
have not been GPU-tested for this new experiment.

## Commands

From the project root, validate without contacting Modal:

```bash
env/bin/python "experiment 4/run_all.py" --validate-only
```

Run the CPU-only/simulated tests (no model downloads or Modal calls):

```bash
env/bin/python "experiment 4/test_experiment4.py"
```

When ready to spend compute, explicitly start the 13 concurrent runs:

```bash
env/bin/python "experiment 4/run_all.py"
```

The script requests 13 A100-80GB workers. Workspace limits and availability
determine scheduling. Keep the launcher running; Ctrl+C stops remaining local
clients. Existing Modal authentication and huggingface-secret are required.
Optional flags: --modal /path/to/modal and --runs-root /path/to/runs.

Each invocation creates a unique runs/<invocation>/ directory containing frozen
configs, manifest and dataset, separate stdout/stderr logs, PIDs, exit codes,
receipts, status.json, downloaded results, and analysis outputs. Remote results
are isolated under safety-bench-results/experiment4/<invocation>/<label>/.
Model/package versions and source checksums are recorded with each result.

All jobs must succeed before downloads and analysis begin. Failed jobs are not
automatically retried or silently omitted. Successful remote outputs persist.
Exact remote paths are in each successful receipt. Download failures have their
own logs. After manually recovering artifacts under results/<label>/, rerun
analysis without spending GPU compute:

```bash
env/bin/python "experiment 4/analyze.py" --run-dir "experiment 4/runs/<invocation>"
```

## Analysis

Join saved binary verdicts by prompt ID, checking prompt content, both detectors,
config snapshots, checksums, seeds, source versions, and complete 300-prompt
coverage. Invalid judgments or inconsistent artifacts stop analysis.

Use 1,000 paired prompt-bootstrap resamples with bootstrap seed 42. For each
resample draw 300 indices with replacement and use those same indices for all
models and both detectors. Compute ASR(configuration) - ASR(base), reporting the
2.5th/97.5th percentiles. Bootstrap analysis reuses the fresh judgments; it does
not generate 1,000 new responses. Bootstrap indices and ordered IDs are saved.

Report individual ASRs and paired differences separately for both detectors.
For each DARE method also report all five seed ASRs, their mean, and sample
standard deviation (ddof=1). Its mean paired difference is bootstrapped by first
averaging verdicts across seeds within each prompt, then resampling prompts.
This interval conditions on the five observed seeds: it does not resample seeds
or treat 1,500 prompt-seed observations as independent prompts.

Outputs: analysis/report.md, report.json, bootstrap_indices.npz, and
inspection_flags.json. The latter flags empty responses, very low whitespace
token diversity, and dominant repeated characters for manual inspection.
Detector disagreements are counted in the report; high disagreement is flagged.
These heuristics cannot establish coherence or usefulness.

Positive delta means higher harmful-response rates than base. Intervals are in
percentage points and include prompt uncertainty conditional on fixed responses,
not all generation/merge randomness or classifier error. Including zero is
inconclusive, not proof of equivalence. Comparisons are exploratory, without
multiple-comparison correction. Low ASR alone does not demonstrate useful output.

DARE seed repeats are included here; no separate Experiment 5 is needed for them.
