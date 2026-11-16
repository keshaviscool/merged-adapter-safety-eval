# Experiment 6: weight-space diagnostics

Analyzed 252 distinct modules across four adapters and six pairs.

Updates are reconstructed as sBA, using configured alpha/r or alpha/sqrt(r), including per-module overrides.
Raw diagnostics exclude merge coefficients. A separate magnitude column uses coefficient 0.25.

## Pair summaries

| Pair | Comparable modules | Mean cosine | Mean conflict rate |
| --- | ---: | ---: | ---: |
| a1-a2 | 252 | 0.0024 | 49.93% |
| a1-a3 | 252 | 0.0067 | 49.78% |
| a1-a4 | 252 | 0.0070 | 49.80% |
| a2-a3 | 252 | 0.0012 | 49.96% |
| a2-a4 | 252 | 0.0009 | 49.98% |
| a3-a4 | 252 | 0.0038 | 49.89% |

**Warning:** adapters declare different base model paths; verify common model ancestry before interpreting alignment.

## Interpretation and limitations

- Pair summary means weight modules equally; inspect the layer-by-layer CSVs and heatmaps.
- Larger absolute magnitudes across differently sized modules can reflect matrix size.
- Missing modules and shape mismatches are explicit; they are not replaced by zero updates.
- Sign conflict uses sign(0)=0, so zero versus nonzero counts as a disagreement. The additional nonzero-only rate excludes these cases.
- Cosine is undefined for zero-norm updates. No epsilon threshold or magnitude pruning is applied.
- PEFT non-SVD TIES operates on LoRA factors; these effective-update metrics do not directly measure its internal factor conflicts.
- These are descriptive diagnostics, not evidence of safety, causation, or predictive performance.
- No behavioral measurements are inferred here. Compare with matching merge experiments separately.
- Unsupported tensors or adapter variants cause an error instead of being silently omitted.

See metadata.json for checkpoint hashes, configuration, and arithmetic details.
