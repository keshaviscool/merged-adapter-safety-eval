# Sequential merging experiment

Benchmarks `M`, `M+a1`, `M+a1+a2`, `M+a1+a2+a3`, `M+a1+a2+a3+a4` on HarmBench, AdvBench (ASR) and Alpaca (FRR) with all detectors.
Each adapter is folded into the already-merged weights: `W <- W + (alpha/rank) * B @ A`.

```bash
python sequential_merging/run.py
```

Edit `sequential_merging/config.yaml` for adapters, datasets, generation. Results go to `sequential_merging/results/<timestamp>/`:
`stage<k>_<name>/` (same files as the main pipeline) plus `sequential_summary.{json,csv}` (rate per dataset x detector x stage), `config_used.yaml`, and `experiment_report.{json,md}` (settings, environment, rates with CIs and counts for every stage).
Requires the main `requirements.txt` and `huggingface-cli login` (Llama Guard 3 is gated).
