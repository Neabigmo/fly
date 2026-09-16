# Real vs shuffled, at the level of trained models

Generated 2026-09-17 07:45 by `scripts/20_seed_stats.py`.
The unit of observation is one trained network, not one test image:
runs are paired by model seed and each shuffled seed is an independent
random graph.  Intervals are Student t on n-1 degrees of freedom.

Runs considered: 76.

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

## fixed updates: core + 5000, 18780 steps

3 paired seeds: [0, 1, 2]

Control strength: each seed's own shuffled graph retains 0.0219-0.0221 of the real edges (mean 0.0220), so the randomisation is comparable across seeds and the pairing is not confounded by varying control strength.

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7229 ± 0.0085 | 0.5903 ± 0.0166 | **+0.1327** ± 0.0134 | [+0.1192, +0.1461] | 24.56 | 17.86 | 0.0006 | 0.2500 | ±0.0184 |
| B | 0.5479 ± 0.0161 | 0.3368 ± 0.0305 | **+0.2111** ± 0.0245 | [+0.1866, +0.2355] | 21.43 | 15.59 | 0.0007 | 0.2500 | ±0.0190 |
| C | 0.6773 ± 0.0762 | 0.3994 ± 0.0246 | **+0.2779** ± 0.0934 | [+0.1845, +0.3712] | 7.39 | 5.38 | 0.0060 | 0.2500 | ±0.0188 |
| D | 0.5273 ± 0.0178 | 0.3144 ± 0.0302 | **+0.2129** ± 0.0138 | [+0.1991, +0.2266] | 38.45 | 27.96 | 0.0002 | 0.2500 | ±0.0189 |

- **A** per-seed deltas: +0.1328, +0.1380, +0.1272
- **B** per-seed deltas: +0.2224, +0.2062, +0.2046
- **C** per-seed deltas: +0.2934, +0.2350, +0.3052
- **D** per-seed deltas: +0.2180, +0.2136, +0.2070

## fixed updates: core + 10000, 18780 steps

3 paired seeds: [0, 1, 2]

Control strength: each seed's own shuffled graph retains 0.0219-0.0221 of the real edges (mean 0.0220), so the randomisation is comparable across seeds and the pairing is not confounded by varying control strength.

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7533 ± 0.0094 | 0.6437 ± 0.0136 | **+0.1096** ± 0.0060 | [+0.1036, +0.1156] | 45.20 | 32.87 | 0.0002 | 0.2500 | ±0.0179 |
| B | 0.5835 ± 0.0403 | 0.3833 ± 0.0105 | **+0.2002** ± 0.0444 | [+0.1558, +0.2446] | 11.20 | 8.15 | 0.0026 | 0.2500 | ±0.0192 |
| C | 0.8349 ± 0.1732 | 0.4760 ± 0.0671 | **+0.3589** ± 0.2269 | [+0.1320, +0.5858] | 3.93 | 2.86 | 0.0209 | 0.2500 | ±0.0172 |
| D | 0.5882 ± 0.0557 | 0.3709 ± 0.0176 | **+0.2173** ± 0.0540 | [+0.1634, +0.2713] | 10.00 | 7.27 | 0.0033 | 0.2500 | ±0.0191 |

- **A** per-seed deltas: +0.1122, +0.1092, +0.1074
- **B** per-seed deltas: +0.2146, +0.1802, +0.2058
- **C** per-seed deltas: +0.4116, +0.2534, +0.4116
- **D** per-seed deltas: +0.2384, +0.1950, +0.2186

## Fly-v2 signed synapses: core + 20k

3 paired seeds: [0, 1, 2]

Control strength: each seed's own shuffled graph retains 0.0219-0.0221 of the real edges (mean 0.0220), so the randomisation is comparable across seeds and the pairing is not confounded by varying control strength.

| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.6759 ± 0.0170 | 0.6323 ± 0.0211 | **+0.0435** ± 0.0372 | [+0.0063, +0.0808] | 2.90 | 2.11 | 0.0373 | 0.2500 | ±0.0186 |
| B | 0.4215 ± 0.0263 | 0.3555 ± 0.0388 | **+0.0660** ± 0.0388 | [+0.0272, +0.1048] | 4.23 | 3.08 | 0.0181 | 0.2500 | ±0.0191 |
| C | 0.4577 ± 0.0953 | 0.3845 ± 0.0242 | **+0.0731** ± 0.0945 | [-0.0213, +0.1676] | 1.92 | 1.40 | 0.0795 | 0.2500 | ±0.0193 |
| D | 0.3778 ± 0.0352 | 0.3304 ± 0.0138 | **+0.0474** ± 0.0461 | [+0.0013, +0.0935] | 2.56 | 1.86 | 0.0474 | 0.2500 | ±0.0187 |

- **A** per-seed deltas: +0.0338, +0.0608, +0.0360
- **B** per-seed deltas: +0.0564, +0.0576, +0.0840
- **C** per-seed deltas: +0.0580, +0.0450, +0.1164
- **D** per-seed deltas: +0.0296, +0.0460, +0.0666

