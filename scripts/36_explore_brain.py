"""Explore the MaleCNS connectome as a whole brain: is it usable, and can vision reach
the central brain within a useful number of hops?

Step 2 of the data exploration.

    python scripts/36_explore_brain.py
"""

from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.data import connectome as cx  # noqa: E402
from flynum.data.annotations import load_annotations  # noqa: E402
from flynum.retina.encoder import select_input_neurons  # noqa: E402
from flynum.retina.hexmap import build_hex_field, column_lookup  # noqa: E402


def bfs_dist(n: int, adj, sources: np.ndarray) -> np.ndarray:
    dist = np.full(n, -1, dtype=np.int32)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(int(s))
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def build_adj(pre, post, n):
    order = np.argsort(pre, kind="stable")
    src, dst = pre[order], post[order]
    starts = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
    return [dst[starts[i]:starts[i + 1]] for i in range(n)]


def main() -> int:
    # ---- is there a richer cell-type vocabulary anywhere else? ----------------- #
    print("=== neurotransmitter file cell_type vocabulary ===")
    nt = pd.read_feather(paths.raw_path("neurotransmitters"))
    types = Counter(nt["cell_type"].astype(str))
    print(f"  {len(types)} distinct cell types over {len(nt)} rows; top 10: "
          f"{dict(types.most_common(10))}")
    for label, keys in (("Kenyon", ("kc", "kenyon")), ("MBON", ("mbon",)),
                        ("dopamine", ("dan", "ppl")), ("octopamine", ("oa", "vum")),
                        ("central complex", ("pb", "eb", "fb"))):
        hits = {c: v for c, v in types.items()
                if any(str(k) == str(c).lower() or str(c).lower().startswith(str(k))
                       for k in keys)}
        print(f"  {label:16s} {sum(hits.values()):7d}  "
              f"{dict(list(hits.items())[:5]) if hits else '(none)'}")

    # ---- the whole-brain graph ------------------------------------------------- #
    print("\n=== whole-brain graph ===")
    ann = load_annotations()
    edges = cx.load_edge_list()
    pre, post, w = edges["pre"], edges["post"], edges["weight"]
    print(f"  usable edges {len(pre):,}  over {ann.n:,} annotated neurons")
    deg_out = np.bincount(pre, minlength=ann.n)
    deg_in = np.bincount(post, minlength=ann.n)
    alive = (deg_out + deg_in) > 0
    print(f"  neurons with at least one edge: {int(alive.sum()):,} "
          f"({alive.mean():.1%} of annotated)")
    print(f"  edges among alive neurons: {int(((deg_out + deg_in) > 0)[pre].sum()):,}")
    print(f"  degree: out mean {deg_out.mean():.1f} max {deg_out.max():,} | "
          f"in mean {deg_in.mean():.1f} max {deg_in.max():,}")

    # ---- can vision reach the central brain, and in how many hops? ------------- #
    print("\n=== hops from the retina ===")
    hex_ok = ~np.isnan(ann.hex1)
    field = build_hex_field(ann.hex1[hex_ok], ann.hex2[hex_ok], ann.soma_side[hex_ok])
    col = np.full(ann.n, -1, dtype=np.int32)
    col[hex_ok] = column_lookup(field, ann.hex1[hex_ok], ann.hex2[hex_ok],
                               ann.soma_side[hex_ok])
    retina = np.nonzero(select_input_neurons(ann.cell_type, ann.soma_side, col,
                                             input_types="lamina", input_eye="both"))[0]
    print(f"  lamina input neurons: {len(retina):,}")
    adj = build_adj(pre.astype(np.int64), post.astype(np.int64), ann.n)
    dist = bfs_dist(ann.n, adj, retina)
    reach = dist >= 0
    print(f"  reachable from retina at all: {int(reach.sum()):,} ({reach.mean():.1%})")
    for sup in ("ol_intrinsic", "visual_projection", "cb_intrinsic", "vnc_intrinsic",
                "__unassigned__", "descending_neuron", "ascending_neuron"):
        m = (ann.superclass == sup) & reach
        if m.sum():
            d = dist[m]
            print(f"  {sup:22s} reachable {int(m.sum()):7d}  hops: "
                  f"median {int(np.median(d)):3d}  p90 {int(np.percentile(d, 90)):3d}  "
                  f"max {int(d.max()):3d}")
    # visual projection neurons are the bridge: where do they send?
    vpn = np.nonzero(ann.superclass == "visual_projection")[0]
    d2 = bfs_dist(ann.n, adj, vpn)
    print(f"\n  from visual projection neurons: reachable {int((d2 >= 0).sum()):,}")
    for sup in ("cb_intrinsic", "vnc_intrinsic", "__unassigned__", "descending_neuron"):
        m = (ann.superclass == sup) & (d2 >= 0)
        if m.sum():
            print(f"    {sup:22s} {int(m.sum()):7d}  median hops {int(np.median(d2[m]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
