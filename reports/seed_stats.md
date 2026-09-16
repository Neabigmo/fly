# Real vs shuffled, at the level of trained models

Generated 2026-09-16 21:06 by `scripts/20_seed_stats.py`.
The unit of observation is one trained network, not one test image:
runs are paired by model seed and each shuffled seed is an independent
random graph.  Intervals are Student t on n-1 degrees of freedom.

Runs considered: 47.

## replication: core + 20k, 60 epochs

9 paired seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8]

Control strength: each seed's own shuffled graph retains 0.0215-0.0227 of the real edges (mean 0.0222), so the randomisation is comparable across seeds and the pairing is not confounded by varying control strength.

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7712 ± 0.0064 | 0.6644 ± 0.0036 | **+0.1068** ± 0.0080 | [+0.0987, +0.1148] | 10.20 | 9.33 | 0.0000 | 0.0039 | ±0.0175 |
| B | 0.6134 ± 0.0153 | 0.4064 ± 0.0028 | **+0.2070** ± 0.0162 | [+0.1909, +0.2232] | 9.85 | 9.01 | 0.0000 | 0.0039 | ±0.0192 |
| C | 0.9181 ± 0.0313 | 0.4233 ± 0.0107 | **+0.4948** ± 0.0292 | [+0.4657, +0.5240] | 13.04 | 11.93 | 0.0000 | 0.0039 | ±0.0156 |
| D | 0.6136 ± 0.0219 | 0.3702 ± 0.0049 | **+0.2434** ± 0.0216 | [+0.2218, +0.2650] | 8.67 | 7.93 | 0.0000 | 0.0039 | ±0.0190 |

- **A** per-seed deltas: +0.1104, +0.1006, +0.1118, +0.0916, +0.1050, +0.1028, +0.1258, +0.1162, +0.0968
- **B** per-seed deltas: +0.2040, +0.1732, +0.2148, +0.1844, +0.2060, +0.2126, +0.2332, +0.2384, +0.1968
- **C** per-seed deltas: +0.4896, +0.4114, +0.5422, +0.4830, +0.5056, +0.4880, +0.5284, +0.5218, +0.4836
- **D** per-seed deltas: +0.2462, +0.2012, +0.2594, +0.2084, +0.2386, +0.2406, +0.2820, +0.2810, +0.2334

