# Ordering effects

Range = max - min of the rate across the distinct models reached at that checkpoint by the 24 orders
(rate in percentage points). Trajectory effect = ranges at checkpoints 1-3, endpoint effect = checkpoint 4.
Disagreement = largest share of prompts whose verdict differs between two endpoint models.

| method | dataset | detector | range@1 | range@2 | range@3 | range@4 (endpoint) | endpoint models | endpoint disagreement |
|---|---|---|---|---|---|---|---|---|
| linear | advbench | any | 0.4 | 2.3 | 2.5 | 0.0 | 1 | 0.0% |
| linear | advbench | harm_cls | 0.2 | 1.3 | 1.3 | 0.0 | 1 | 0.0% |
| linear | advbench | llama_guard_3 | 0.4 | 2.3 | 2.3 | 0.0 | 1 | 0.0% |
| linear | alpaca | any | 1.0 | 2.2 | 1.0 | 0.0 | 1 | 0.0% |
| linear | alpaca | keyword_refusal | 1.0 | 1.6 | 1.2 | 0.0 | 1 | 0.0% |
| linear | alpaca | rejection_classifier | 0.4 | 1.2 | 0.8 | 0.0 | 1 | 0.0% |
| linear | harmbench | any | 12.0 | 9.0 | 8.7 | 0.0 | 1 | 0.0% |
| linear | harmbench | harm_cls | 11.7 | 9.3 | 4.3 | 0.0 | 1 | 0.0% |
| linear | harmbench | llama_guard_3 | 11.0 | 9.3 | 8.7 | 0.0 | 1 | 0.0% |
| ties | advbench | any | 0.4 | 0.8 | 1.2 | 0.2 | 12 | 0.4% |
| ties | advbench | harm_cls | 0.2 | 0.8 | 1.2 | 0.2 | 12 | 0.4% |
| ties | advbench | llama_guard_3 | 0.4 | 1.0 | 1.0 | 0.2 | 12 | 0.2% |
| ties | alpaca | any | 1.0 | 1.0 | 1.0 | 0.4 | 12 | 1.4% |
| ties | alpaca | keyword_refusal | 1.0 | 0.8 | 0.8 | 0.4 | 12 | 0.8% |
| ties | alpaca | rejection_classifier | 0.4 | 1.0 | 0.8 | 0.4 | 12 | 1.0% |
| ties | harmbench | any | 12.0 | 7.3 | 8.3 | 4.0 | 12 | 6.7% |
| ties | harmbench | harm_cls | 11.7 | 7.3 | 9.3 | 5.0 | 12 | 8.0% |
| ties | harmbench | llama_guard_3 | 11.0 | 7.0 | 8.3 | 5.0 | 12 | 7.0% |