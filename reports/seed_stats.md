# Real vs shuffled, at the level of trained models

Generated 2026-09-16 14:54 by `scripts/20_seed_stats.py`.
The unit of observation is one trained network, not one test image:
runs are paired by model seed and each shuffled seed is an independent
random graph.  Intervals are Student t on n-1 degrees of freedom.

Runs considered: 35.

## replication: core + 20k, 60 epochs

3 paired seeds: [0, 1, 2]

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7689 ± 0.0189 | 0.6613 ± 0.0087 | **+0.1076** ± 0.0152 | [+0.0924, +0.1228] | 17.63 | 12.82 | 0.0011 | 0.2500 | ±0.0176 |
| B | 0.6021 ± 0.0553 | 0.4048 ± 0.0031 | **+0.1973** ± 0.0536 | [+0.1437, +0.2510] | 9.14 | 6.65 | 0.0040 | 0.2500 | ±0.0192 |
| C | 0.8977 ± 0.1361 | 0.4167 ± 0.0276 | **+0.4811** ± 0.1635 | [+0.3176, +0.6446] | 7.31 | 5.32 | 0.0062 | 0.2500 | ±0.0160 |
| D | 0.6009 ± 0.0806 | 0.3653 ± 0.0152 | **+0.2356** ± 0.0758 | [+0.1598, +0.3114] | 7.72 | 5.62 | 0.0055 | 0.2500 | ±0.0190 |

- **A** per-seed deltas: +0.1104, +0.1006, +0.1118
- **B** per-seed deltas: +0.2040, +0.1732, +0.2148
- **C** per-seed deltas: +0.4896, +0.4114, +0.5422
- **D** per-seed deltas: +0.2462, +0.2012, +0.2594

