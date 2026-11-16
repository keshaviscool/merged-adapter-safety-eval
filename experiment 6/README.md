# Experiment 6: adapter weight-space diagnostics

Analyze the four existing LoRA adapters, module by module, on CPU. This script
does not load the base model, generate responses, call Modal, or change adapters.
The first real run downloads adapter configurations and weights from Hugging Face
if they are not already cached. PyTorch, NumPy, Pillow, safetensors, and
huggingface_hub are required; these are available in the project's current env.

From the project root:

```bash
env/bin/python "experiment 6/analyze_weights.py"
```

The defaults are:

| Label | Checkpoint |
| --- | --- |
| a1 | Akarshika/qwen25-3b-text2sql-lora |
| a2 | madokalif/qwen2.5-3b-translation-warmup-adapter |
| a3 | balaguhanesh/tool-call-ft |
| a4 | Pilestin/qwen2.5-3b-debug-v1-lora |

Each invocation creates a unique directory under `experiment 6/outputs`:

- `update_magnitudes.csv`: effective-update Frobenius norm, scaled norm,
  internal LoRA scale, shape, and zero counts for every module and adapter.
- `pairwise_metrics.csv`: cosine similarity and sign-conflict rate for every
  module and all six adapter pairs, plus a nonzero-only conflict rate.
- `magnitude_heatmap.png`: log10 magnitude; zero and missing updates are gray.
- `cosine_heatmap.png`: directional similarity from -1 to +1.
- `sign_conflict_heatmap.png`: sign disagreement from 0 to 1.
- `metadata.json`: adapter configurations, checkpoint hashes, resolved Hub
  revisions, numerical precision, and analysis options.
- `report.md`: pair summaries and interpretation limits.

## Definitions and safeguards

For each ordinary LoRA module, reconstruct `delta_W = scale * B @ A`. Use
`alpha/r`, or `alpha/sqrt(r)` for rsLoRA, including module-specific overrides.
Different adapter ranks are allowed if their effective update shapes match.
Metrics are computed in row chunks: FP32 matrix products and FP64 reductions.
No complete collection of dense model updates is kept in memory.

The raw metrics use each adapter's internal scale, without external merge
coefficients. `weighted_frobenius_norm` additionally applies the absolute value
of `--merge-weight` (default 0.25); it does not simulate a merged adapter.

Sign conflict is the fraction of all corresponding entries with different signs,
including zero versus nonzero, using sign(0)=0. The additional nonzero-only rate
counts opposing signs only where both entries are nonzero. Zero-norm cosine is
undefined. Missing modules and incompatible shapes get explicit status rows,
not fabricated zero metrics. Unsupported tensors/variants fail with an error.

Metrics use the BA matrix orientation. A common transpose for fan_in_fan_out
does not change these measurements. Adapter base-path differences are reported;
verify common model ancestry before making scientific comparisons.
Mixed fan_in_fan_out orientations are rejected rather than compared unaligned.

These are descriptive diagnostics. They do not establish safety, causality,
or predict ASR. Non-SVD PEFT TIES works on LoRA factors rather than directly on
these effective updates. No behavior results are inferred or imported here.

## Options

Use `--chunk-rows 64` to reduce temporary memory, `--output-root PATH` to change
the output directory, or `--merge-weight 0.5` for another scaled magnitude column.

Repeat `--adapter LABEL=LOCAL_PATH_OR_HUB_ID` exactly four times to replace all
defaults. Local paths must contain adapter_config.json and either
adapter_model.safetensors or adapter_model.bin. Sharded, embedding, DoRA, and
non-LoRA checkpoints are not supported by this first version.

Pin Hub checkpoints with repeated `--revision LABEL=COMMIT_SHA` flags. Without
pins, the first download resolves the current default revision; the resulting
commit and file hashes are recorded for later reproduction.

Run the CPU-only synthetic tests:

```bash
env/bin/python "experiment 6/test_analyze_weights.py"
```

Creating this experiment did not download or analyze the real adapters.
