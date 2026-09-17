# Phase I — 一个果蝇 connectome 能被教育到多复杂的数学

- spec fingerprint: `fd0c71ca2f0b9046`
- 教学条件: ('A', 'B') 混合（每个线索只在半个数据里有效），评估条件 ('A', 'B', 'C')
- 动力: core / signed=True / w_scale=0.5 / alpha=0.1
- optimizer: AdamW, constant lr 0.003, batch 64, grad clip 1, gain penalty 0.001
- 比较单位: **optimiser updates**（不是 epoch）；探针固定在同一批 update 数
- 单 seed；无 early stopping；不用留出集选 checkpoint

## Line A — 教学能到达的复杂度

| cell | task | brain | budget | train | **seen** | area B | env C | AUC(seen) | →0.50 | →0.80 | →0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | count | scratch | 100000 | 0.776 | **0.757** | 0.811 | 0.797 | 0.645 | 20000 | -- | -- |
| A1-S | add | scratch | 50000 | 0.221 | **0.204** | 0.163 | 0.148 | 0.179 | -- | -- | -- |
| A1-C | add | count | 50000 | 0.172 | **0.174** | 0.153 | 0.138 | 0.147 | -- | -- | -- |

`seen` 是教学内容上的准确率（chance: count 0.143 / add 0.077 / addsub 0.111 / two_step 0.143）；`area B`、`env C` 是同一批 item 在另外两个条件下的读数，C 从未参与教学。

## Line B — 长期训练会不会从记忆跃迁到规则

_尚未有完成的 Line B cell。_

## 内部记录（行为变化是否伴随内部重组）

| cell | δ mean | activity dim | 探针 |
|---|---|---|---|
| A1-C | 0.6273 | 19.2 | a=0.375 b=0.153 sum=0.171 |
| A1-S | 0.2444 | 34.4 | a=0.383 b=0.273 sum=0.204 |
| C0 | 0.6371 | 25.3 | n=0.428 |
