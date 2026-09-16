# Real vs shuffled, at the level of trained models

Generated 2026-09-16 16:23 by `scripts/20_seed_stats.py`.
The unit of observation is one trained network, not one test image:
runs are paired by model seed and each shuffled seed is an independent
random graph.  Intervals are Student t on n-1 degrees of freedom.

Runs considered: 38.

## replication: core + 20k, 60 epochs

4 paired seeds: [0, 1, 2, 3]

Control strength: each seed's own shuffled graph retains 0.0219-0.0222 of the real edges (mean 0.0220), so the randomisation is comparable across seeds and the pairing is not confounded by varying control strength.

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7672 ± 0.0112 | 0.6636 ± 0.0087 | **+0.1036** ± 0.0150 | [+0.0886, +0.1186] | 10.99 | 8.79 | 0.0002 | 0.1250 | ±0.0176 |
| B | 0.6014 ± 0.0290 | 0.4073 ± 0.0081 | **+0.1941** ± 0.0299 | [+0.1642, +0.2240] | 10.34 | 8.27 | 0.0002 | 0.1250 | ±0.0192 |
| C | 0.8973 ± 0.0712 | 0.4158 ± 0.0147 | **+0.4816** ± 0.0855 | [+0.3960, +0.5671] | 8.96 | 7.17 | 0.0004 | 0.1250 | ±0.0160 |
| D | 0.5970 ± 0.0439 | 0.3682 ± 0.0122 | **+0.2288** ± 0.0452 | [+0.1836, +0.2740] | 8.06 | 6.45 | 0.0005 | 0.1250 | ±0.0191 |

- **A** per-seed deltas: +0.1104, +0.1006, +0.1118, +0.0916
- **B** per-seed deltas: +0.2040, +0.1732, +0.2148, +0.1844
- **C** per-seed deltas: +0.4896, +0.4114, +0.5422, +0.4830
- **D** per-seed deltas: +0.2462, +0.2012, +0.2594, +0.2084

