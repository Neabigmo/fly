# Real vs shuffled, at the level of trained models

Generated 2026-09-16 21:41 by `scripts/20_seed_stats.py`.
The unit of observation is one trained network, not one test image:
runs are paired by model seed and each shuffled seed is an independent
random graph.  Intervals are Student t on n-1 degrees of freedom.

Runs considered: 49.

## replication: core + 20k, 60 epochs

10 paired seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Control strength: each seed's own shuffled graph retains 0.0215-0.0227 of the real edges (mean 0.0222), so the randomisation is comparable across seeds and the pairing is not confounded by varying control strength.

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7716 ± 0.0057 | 0.6640 ± 0.0033 | **+0.1075** ± 0.0073 | [+0.1003, +0.1148] | 10.60 | 9.79 | 0.0000 | 0.0020 | ±0.0175 |
| B | 0.6131 ± 0.0134 | 0.4049 ± 0.0041 | **+0.2082** ± 0.0144 | [+0.1938, +0.2226] | 10.34 | 9.54 | 0.0000 | 0.0020 | ±0.0192 |
| C | 0.9169 ± 0.0276 | 0.4216 ± 0.0101 | **+0.4952** ± 0.0256 | [+0.4696, +0.5208] | 13.84 | 12.77 | 0.0000 | 0.0020 | ±0.0156 |
| D | 0.6123 ± 0.0194 | 0.3686 ± 0.0056 | **+0.2437** ± 0.0189 | [+0.2247, +0.2626] | 9.20 | 8.49 | 0.0000 | 0.0020 | ±0.0190 |

- **A** per-seed deltas: +0.1104, +0.1006, +0.1118, +0.0916, +0.1050, +0.1028, +0.1258, +0.1162, +0.0968, +0.1142
- **B** per-seed deltas: +0.2040, +0.1732, +0.2148, +0.1844, +0.2060, +0.2126, +0.2332, +0.2384, +0.1968, +0.2186
- **C** per-seed deltas: +0.4896, +0.4114, +0.5422, +0.4830, +0.5056, +0.4880, +0.5284, +0.5218, +0.4836, +0.4986
- **D** per-seed deltas: +0.2462, +0.2012, +0.2594, +0.2084, +0.2386, +0.2406, +0.2820, +0.2810, +0.2334, +0.2458

