# Local vs cumulative gate

Per path (checkpoints M0 -> 4 adapters): local gate rejects a step if ASR_k - ASR_{k-1} > eps, cumulative gate rejects a step if ASR_k - ASR_0 > tau.
Cells: number of the 24 paths that the local gate admits at every step but the cumulative gate rejects at some step (hidden drift).

## linear / advbench / any

largest single-step increase: 8.1 pp; largest cumulative increase: 8.5 pp (path a1>a2>a3>a4, McNemar p vs base = 1.34e-12)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## linear / advbench / harm_cls

largest single-step increase: 1.3 pp; largest cumulative increase: 0.0 pp (path a1>a2>a3>a4, McNemar p vs base = 1)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## linear / advbench / llama_guard_3

largest single-step increase: 7.9 pp; largest cumulative increase: 8.7 pp (path a1>a2>a3>a4, McNemar p vs base = 5.68e-14)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## linear / harmbench / any

largest single-step increase: 17.0 pp; largest cumulative increase: 24.0 pp (path a1>a2>a3>a4, McNemar p vs base = 1.83e-14)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## linear / harmbench / harm_cls

largest single-step increase: 12.3 pp; largest cumulative increase: 3.7 pp (path a1>a2>a3>a4, McNemar p vs base = 0.215)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 4 | 0 | 0 |

## linear / harmbench / llama_guard_3

largest single-step increase: 17.3 pp; largest cumulative increase: 25.0 pp (path a1>a2>a3>a4, McNemar p vs base = 1.43e-15)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## ties / advbench / any

largest single-step increase: 1.0 pp; largest cumulative increase: -0.2 pp (path a1>a2>a3>a4, McNemar p vs base = 1)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## ties / advbench / harm_cls

largest single-step increase: 1.0 pp; largest cumulative increase: -0.2 pp (path a1>a2>a3>a4, McNemar p vs base = 1)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## ties / advbench / llama_guard_3

largest single-step increase: 1.0 pp; largest cumulative increase: 0.2 pp (path a2>a3>a1>a4, McNemar p vs base = 1)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |

## ties / harmbench / any

largest single-step increase: 10.0 pp; largest cumulative increase: 6.3 pp (path a3>a4>a2>a1, McNemar p vs base = 0.00256)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 6 | 5 | 0 |

## ties / harmbench / harm_cls

largest single-step increase: 8.0 pp; largest cumulative increase: 5.7 pp (path a3>a4>a2>a1, McNemar p vs base = 0.00763)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 6 | 0 | 0 |

## ties / harmbench / llama_guard_3

largest single-step increase: 9.7 pp; largest cumulative increase: 7.7 pp (path a3>a4>a2>a1, McNemar p vs base = 0.000294)

| eps \ tau | 0.02 | 0.05 | 0.1 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 4 | 4 | 0 |
