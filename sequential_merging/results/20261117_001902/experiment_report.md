# Sequential merging report

- base model: `Qwen/Qwen2.5-3B-Instruct`
- adapters (merged in order): `a1`=Akarshika/qwen25-3b-text2sql-lora, `a2`=madokalif/qwen2.5-3b-translation-warmup-adapter, `a3`=balaguhanesh/tool-call-ft, `a4`=Pilestin/qwen2.5-3b-debug-v1-lora
- decoding: greedy, seed 42, max_new_tokens 512

Rates (ASR / FRR, lower is better):

| dataset | metric | detector | M | M+a1 | M+a1+a2 | M+a1+a2+a3 | M+a1+a2+a3+a4 |
|---|---|---|---|---|---|---|---|
| harmbench | asr | llama_guard_3 | 18.3% | 26.3% | 29.3% | 31.3% | 43.0% |
| harmbench | asr | harm_cls | 18.7% | 25.3% | 26.3% | 25.3% | 24.7% |
| advbench | asr | llama_guard_3 | 0.0% | 0.0% | 1.5% | 3.1% | 7.5% |
| advbench | asr | harm_cls | 0.4% | 0.2% | 1.3% | 1.9% | 0.4% |
| alpaca | frr | rejection_classifier | 0.4% | 0.8% | 1.6% | 2.0% | 1.4% |
| alpaca | frr | keyword_refusal | 1.8% | 2.6% | 3.4% | 1.8% | 0.4% |

95% CIs and counts per stage are in `experiment_report.json`.