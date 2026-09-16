"""Degree- and weight-preserving edge randomisation (the shuffled-fly control).

The control must differ from the real connectome *only* in who connects to whom.
We therefore use the weighted paired edge swap:

    pick two existing edges  a --w1--> b   and   c --w2--> d
    replace them with        a --w1--> d   and   c --w2--> b

This preserves, exactly and by construction:

* the out-degree of every neuron (``a`` and ``c`` each keep one outgoing edge),
* the in-degree of every neuron (``b`` and ``d`` each keep one incoming edge),
* the **multiset of edge weights** (``w1`` and ``w2`` are carried along unharmed),

so the shuffled network has an identical parameter count and an identical
distribution of synaptic strengths.  Only the wiring changes.

Proposals are rejected when they would create a self-loop, re-create an existing
edge, or leave the edge set unchanged (the ``a == c`` / ``b == d`` cases).
Proposals are generated in large batches and validated against a sorted key
array with ``np.searchsorted``, which keeps throughput high even for the
12.5 M-edge ``full`` circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ShuffleResult:
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    stats: dict = field(default_factory=dict)


def _keys(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    return pre.astype(np.int64) * n + post.astype(np.int64)


def shuffle_edges(
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    n: int,
    *,
    rounds_per_edge: float = 20.0,
    seed: int = 0,
    batch: int = 1_000_000,
    record_overlap: bool = True,
    logger=None,
) -> ShuffleResult:
    """Randomise an edge list while preserving in/out degrees and weights."""
    rng = np.random.default_rng(seed)
    pre = np.ascontiguousarray(pre, dtype=np.int32).copy()
    post = np.ascontiguousarray(post, dtype=np.int32).copy()
    weight = np.ascontiguousarray(weight, dtype=np.float32).copy()
    weight_ref = weight.copy()

    e = len(pre)
    if e == 0:
        return ShuffleResult(pre, post, weight, {"attempts": 0, "swaps": 0})

    in0 = np.bincount(post, minlength=n)
    out0 = np.bincount(pre, minlength=n)
    keys = np.sort(_keys(pre, post, n))
    if keys.size != e or np.unique(keys).size != e:
        raise ValueError(
            "shuffle_edges requires an aggregated edge list without duplicate "
            "(pre, post) pairs; the MaleCNS connectome file satisfies this"
        )
    original = keys.copy()

    n_target = int(round(rounds_per_edge * e))
    attempts = swaps = 0
    overlap_curve: list[tuple[int, float]] = []
    snapshot_every = max(n_target // 40, 1)
    next_snapshot = snapshot_every
    k = max(2, min(batch, e))

    while attempts < n_target:
        # Draw an index pool that is distinct by construction.  (Requiring
        # uniqueness is what keeps two accepted proposals from writing to the
        # same edge slot, which would silently corrupt the degree sequence.)
        n_pairs = max(1, min(k, e // 2))
        pool = rng.integers(0, e, size=int(n_pairs * 2 * 1.35) + 16)
        pool = np.unique(pool)
        n_pairs = min(n_pairs, pool.size // 2)
        if n_pairs < 1:
            break
        i1 = pool[:n_pairs]
        i2 = pool[n_pairs : 2 * n_pairs]
        attempts += n_pairs

        a, b = pre[i1], post[i1]
        c, d = pre[i2], post[i2]

        # no self-loops, and the swap must actually change the edge set
        ok = (a != d) & (c != b) & (a != c) & (b != d)
        if not ok.any():
            continue
        i1, i2, a, b, c, d = (x[ok] for x in (i1, i2, a, b, c, d))

        new1 = a.astype(np.int64) * n + d.astype(np.int64)
        new2 = c.astype(np.int64) * n + b.astype(np.int64)
        ok = new1 != new2
        if not ok.any():
            continue
        i1, i2, a, b, c, d, new1, new2 = (
            x[ok] for x in (i1, i2, a, b, c, d, new1, new2)
        )

        # neither replacement edge may already exist in the current graph
        p1 = np.clip(np.searchsorted(keys, new1), 0, e - 1)
        p2 = np.clip(np.searchsorted(keys, new2), 0, e - 1)
        ok = (keys[p1] != new1) & (keys[p2] != new2)
        if not ok.any():
            continue
        i1, i2, a, b, c, d, new1, new2 = (
            x[ok] for x in (i1, i2, a, b, c, d, new1, new2)
        )

        # ...and no two proposals inside this batch may create the same edge
        uniq, cnt = np.unique(np.concatenate([new1, new2]), return_counts=True)
        clashing = uniq[cnt > 1]
        if clashing.size:
            keep = ~np.isin(new1, clashing) & ~np.isin(new2, clashing)
            if not keep.any():
                continue
            i1, i2, a, b, c, d, new1, new2 = (
                x[keep] for x in (i1, i2, a, b, c, d, new1, new2)
            )

        # apply (all edge indices in this batch are distinct)
        pre[i1], post[i1] = a, d
        pre[i2], post[i2] = c, b
        keys = np.sort(_keys(pre, post, n))

        swaps += len(i1)

        if record_overlap and attempts >= next_snapshot:
            overlap_curve.append(
                (
                    int(attempts),
                    float(np.intersect1d(keys, original, assume_unique=True).size / e),
                )
            )
            next_snapshot += snapshot_every

    in1 = np.bincount(post, minlength=n)
    out1 = np.bincount(pre, minlength=n)
    if not np.array_equal(in0, in1):
        raise AssertionError("shuffle changed the in-degree sequence")
    if not np.array_equal(out0, out1):
        raise AssertionError("shuffle changed the out-degree sequence")
    if not np.array_equal(np.sort(weight), np.sort(weight_ref)):
        raise AssertionError("shuffle changed the weight multiset")
    if np.unique(keys).size != e:
        raise AssertionError(
            f"shuffle produced duplicate edges ({e - np.unique(keys).size} collisions)"
        )

    final_overlap = float(np.intersect1d(keys, original, assume_unique=True).size / e)
    stats = {
        "edges": int(e),
        "attempts": int(attempts),
        "swaps": int(swaps),
        "swap_rate": round(swaps / max(attempts, 1), 4),
        "rounds_per_edge": rounds_per_edge,
        "final_edge_overlap": round(final_overlap, 6),
        "degrees_preserved": True,
        "weights_preserved": True,
        "overlap_curve": overlap_curve,
        "seed": int(seed),
    }
    if logger:
        logger.info(
            "shuffle(seed=%d): %d swaps / %d attempts (%.1f%% accepted) | "
            "edge overlap with real graph %.2f%%",
            seed,
            swaps,
            attempts,
            100 * swaps / max(attempts, 1),
            100 * final_overlap,
        )
    return ShuffleResult(pre, post, weight, stats)


def edge_overlap(pre_a, post_a, pre_b, post_b, n: int) -> float:
    """Fraction of edges in A that also exist in B."""
    ka = np.unique(_keys(pre_a, post_a, n))
    kb = np.unique(_keys(pre_b, post_b, n))
    return float(np.intersect1d(ka, kb, assume_unique=True).size / max(len(ka), 1))
