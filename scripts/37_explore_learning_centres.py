"""Step 3: identify the learning centres by cell, and test whether vision can teach them.

The annotation table is optic-lobe-deep and central-brain-blind; the neurotransmitter file
carries the rich vocabulary (Kenyon cells, MBONs, dopamine, central complex).  This joins
the two and answers the question the whole plan depends on: can a visual signal reach the
mushroom body and the central complex, and can their outputs come back to the read-out?

Breadth-first search runs on CSR directly -- a Python list of per-neuron neighbour arrays
costs more memory than the graph itself and died on a 32 GB machine.

    python scripts/37_explore_learning_centres.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.data import connectome as cx  # noqa: E402
from flynum.data.annotations import load_annotations  # noqa: E402
from flynum.retina.encoder import select_input_neurons  # noqa: E402
from flynum.retina.hexmap import build_hex_field, column_lookup  # noqa: E402

GROUPS = {
    "Kenyon cell (MB)": ("kc",),
    "MBON": ("mbon",),
    "dopamine DAN/PPL": ("ppl", "dan"),
    "octopamine": ("oa-", "vum", "vpm"),
    "central complex": ("fb", "eb", "pb", "no", "epg", "pfn", "pfl", "hdelta", "cl1",
                        "cl2", "ib", "lno"),
    "lateral horn": ("lh",),
    "AOTU/tubercle": ("aotu", "ibu", "tu-"),
    "clock (LN)": ("ln", "sln"),
}


def csr(pre: np.ndarray, post: np.ndarray, n: int):
    """CSR (indptr, indices) for pre -> post, and its reverse."""
    order = np.argsort(pre, kind="stable")
    counts = np.bincount(pre, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    ind = post[order].astype(np.int32)
    order_r = np.argsort(post, kind="stable")
    counts_r = np.bincount(post, minlength=n)
    indptr_r = np.concatenate([[0], np.cumsum(counts_r)]).astype(np.int64)
    ind_r = pre[order_r].astype(np.int32)
    return (indptr, ind), (indptr_r, ind_r)


def csr_bfs(graph, sources: np.ndarray, n: int) -> np.ndarray:
    """Multi-source breadth-first search over CSR, returning hop distance (-1 = none)."""
    indptr, ind = graph
    dist = np.full(n, -1, dtype=np.int32)
    dist[sources] = 0
    frontier = sources.astype(np.int64)
    depth = 0
    while frontier.size:
        starts, ends = indptr[frontier], indptr[frontier + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            break
        offs = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts)
        nbr = ind[np.repeat(starts, counts) + offs]
        nbr = np.unique(nbr)
        nbr = nbr[dist[nbr] < 0]
        if nbr.size == 0:
            break
        depth += 1
        dist[nbr] = depth
        frontier = nbr
    return dist


def main() -> int:
    ann = load_annotations()
    nt = pd.read_feather(paths.raw_path("neurotransmitters"))[["body", "cell_type"]]
    nt = nt.dropna(subset=["cell_type"])
    per_body = nt.groupby("body")["cell_type"].first()
    pos = {int(b): i for i, b in enumerate(ann.body_ids)}
    keep = [(int(b), str(c)) for b, c in per_body.items() if int(b) in pos]
    idx = np.array([pos[b] for b, _ in keep], dtype=np.int64)
    types = np.array([c for _, c in keep])
    print(f"{len(idx):,} of {ann.n:,} annotated neurons carry a cell type "
          f"({len(set(types)):,} distinct types)")

    print("\n=== learning / navigation populations (by cell) ===")
    found: dict[str, np.ndarray] = {}
    for label, keys in GROUPS.items():
        m = np.array([any(t.lower().startswith(k) for k in keys) for t in types])
        found[label] = idx[m]
        print(f"  {label:20s} {int(m.sum()):6d}   {dict(Counter(types[m]).most_common(4))}")

    print("\n=== graph ===")
    edges = cx.load_edge_list()
    pre, post = edges["pre"].astype(np.int64), edges["post"].astype(np.int64)
    n = ann.n
    alive = np.bincount(pre, minlength=n) + np.bincount(post, minlength=n) > 0
    print(f"  neurons {n:,} | edges {len(pre):,} | with an edge {int(alive.sum()):,}")
    fwd, bwd = csr(pre, post, n)
    del edges, pre, post

    hex_ok = ~np.isnan(ann.hex1)
    field = build_hex_field(ann.hex1[hex_ok], ann.hex2[hex_ok], ann.soma_side[hex_ok])
    col = np.full(n, -1, dtype=np.int32)
    col[hex_ok] = column_lookup(field, ann.hex1[hex_ok], ann.hex2[hex_ok],
                                ann.soma_side[hex_ok])
    retina = np.nonzero(select_input_neurons(ann.cell_type, ann.soma_side, col,
                                             input_types="lamina", input_eye="both"))[0]
    vpn = np.nonzero(ann.superclass == "visual_projection")[0]
    print(f"  lamina input {len(retina):,} | visual projection (read-out pool) {len(vpn):,}")

    d_in = csr_bfs(fwd, retina, n)
    d_out = csr_bfs(bwd, vpn, n)
    print("\n=== vision -> learning centre -> read-out ===")
    print(f"  {'population':20s} {'cells':>6s} {'<=4 hops from retina':>21s} "
          f"{'<=3 hops to read-out':>22s} {'median in/out':>14s}")
    for label, cells in found.items():
        if not len(cells):
            continue
        di, do = d_in[cells], d_out[cells]
        near_in = int(((di >= 0) & (di <= 4)).sum())
        near_out = int(((do >= 0) & (do <= 3)).sum())
        mi = int(np.median(di[di >= 0])) if (di >= 0).any() else -1
        mo = int(np.median(do[do >= 0])) if (do >= 0).any() else -1
        print(f"  {label:20s} {len(cells):6d} {near_in:21d} {near_out:22d} "
              f"{mi:7d}/{mo:<6d}")

    kc = found["Kenyon cell (MB)"]
    if len(kc):
        indptr, ind = bwd          # reverse graph: in-edges of a node
        seg = np.concatenate([ind[indptr[c]:indptr[c + 1]] for c in kc[:2000]])
        print(f"\n=== presynaptic partners of {min(len(kc), 2000)} Kenyon cells ===")
        for k, v in Counter(ann.superclass[seg]).most_common(8):
            print(f"    {str(k):26s} {v:,} synapses")
        print(f"    from visual projection neurons: {int(np.isin(seg, vpn).sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
