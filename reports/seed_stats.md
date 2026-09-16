# Real vs shuffled, at the level of trained models

Generated 2026-09-16 16:40 by `scripts/20_seed_stats.py`.
The unit of observation is one trained network, not one test image:
runs are paired by model seed and each shuffled seed is an independent
random graph.  Intervals are Student t on n-1 degrees of freedom.

Runs considered: 39.

## replication: core + 20k, 60 epochs

5 paired seeds: [0, 1, 2, 3, 4]

Control strength: each seed's own shuffled graph retains 0.0219-0.0225 of the real edges (mean 0.0221), so the randomisation is comparable across seeds and the pairing is not confounded by varying control strength.

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7689 ± 0.0089 | 0.6650 ± 0.0070 | **+0.1039** ± 0.0102 | [+0.0937, +0.1140] | 12.69 | 10.69 | 0.0000 | 0.0625 | ±0.0175 |
| B | 0.6040 ± 0.0209 | 0.4076 ± 0.0055 | **+0.1965** ± 0.0212 | [+0.1752, +0.2177] | 11.48 | 9.67 | 0.0000 | 0.0625 | ±0.0192 |
| C | 0.9016 ± 0.0496 | 0.4152 ± 0.0101 | **+0.4864** ± 0.0593 | [+0.4270, +0.5457] | 10.18 | 8.57 | 0.0000 | 0.0625 | ±0.0159 |
| D | 0.6009 ± 0.0316 | 0.3701 ± 0.0098 | **+0.2308** ± 0.0310 | [+0.1998, +0.2618] | 9.24 | 7.78 | 0.0000 | 0.0625 | ±0.0191 |

- **A** per-seed deltas: +0.1104, +0.1006, +0.1118, +0.0916, +0.1050
- **B** per-seed deltas: +0.2040, +0.1732, +0.2148, +0.1844, +0.2060
- **C** per-seed deltas: +0.4896, +0.4114, +0.5422, +0.4830, +0.5056
- **D** per-seed deltas: +0.2462, +0.2012, +0.2594, +0.2084, +0.2386

