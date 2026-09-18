"""Which neurons can a stimulus actually reach -- and which can reach the readout?

Answers one question with data instead of intuition: is the extra mass of the ``full``
circuit causally *on the path* from the retina to the read-out, or is it mostly tissue the
visual computation never touches?  Three sets are computed per circuit:

``reachable``
    neurons a signal can get to from the injected input neurons, at any depth
``ancestors``
    neurons that can influence the read-out population, at any depth
``on path``
    the intersection -- the only neurons that can matter for the task at all

Depth matters as much as connectivity: one recurrent step moves a signal one hop, so with a
sequence of T steps only neurons within about T hops of the read-out are usable.  The
distance distribution is therefore reported next to the raw counts, restricted to the
retina-reachable set.

    python scripts/34_reachability.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.data.annotations import load_annotations  # noqa: E402
from flynum.data.subgraph import Subgraph  # noqa: E402
from flynum.retina.encoder import select_input_neurons  # noqa: E402
from flynum.retina.hexmap import build_hex_field, column_lookup  # noqa: E402

LEARNING_HINTS = ("mushroom", "kenyon", "kc", "mb", "ellipsoid", "fan-shaped",
                  "gall", "protocerebral bridge", "central complex", "nodulus",
                  "asymmetrical")


def bfs(n: int, adj, sources: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    """Shortest-path distance (in hops) from ``sources``; -1 where unreachable."""
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


def build_adj(pre: np.ndarray, post: np.ndarray, n: int, *, reverse: bool = False):
    src, dst = (post, pre) if reverse else (pre, post)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    counts = np.bincount(src, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    return [dst[starts[i]:starts[i + 1]] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=20,
                    help="sequence length whose hop budget to use for 'usable'")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    ann = load_annotations()
    out: dict = {}
    report = {}

    for circuit in ("core", "full"):
        sg = Subgraph.load(paths.DATA_PROCESSED / "circuits" / f"{circuit}.npz")
        n = sg.n_neurons
        sub_ann = ann.index_of(sg.body_ids)
        h1, h2 = ann.hex1[sub_ann], ann.hex2[sub_ann]
        side = ann.soma_side[sub_ann]
        has_hex = ~np.isnan(h1)
        field = build_hex_field(h1[has_hex], h2[has_hex], side[has_hex])
        col = np.full(n, -1, dtype=np.int32)
        col[has_hex] = column_lookup(field, h1[has_hex], h2[has_hex], side[has_hex])
        mask = select_input_neurons(ann.cell_type[sub_ann], side, col,
                                    input_types=cfg.retina.input_types,
                                    input_eye=cfg.retina.input_eye)
        inputs = np.nonzero(mask)[0]
        readout = np.nonzero(ann.superclass[sub_ann] == "visual_projection")[0]

        fwd = build_adj(sg.pre.astype(np.int64), sg.post.astype(np.int64), n)
        bwd = build_adj(sg.pre.astype(np.int64), sg.post.astype(np.int64), n,
                        reverse=True)
        d_from_in = bfs(n, fwd, inputs)
        d_to_out = bfs(n, bwd, readout)
        reachable = d_from_in >= 0
        ancestors = d_to_out >= 0
        on_path = reachable & ancestors
        usable = on_path & (d_to_out <= args.hops)

        sup = ann.superclass[sub_ann]
        out[circuit] = {
            "neurons": int(n), "edges": int(len(sg.pre)),
            "input_neurons": int(len(inputs)), "readout_neurons": int(len(readout)),
            "reachable": int(reachable.sum()),
            "ancestors": int(ancestors.sum()),
            "on_path": int(on_path.sum()),
            "on_path_fraction": round(float(on_path.mean()), 4),
            "usable_within_hops": int(usable.sum()),
            "hops": args.hops,
            "median_hops_to_readout": float(np.median(d_to_out[reachable & (d_to_out >= 0)])),
            "on_path_by_superclass": Counter(sup[on_path]).most_common(18),
            "extra_vs_core_top": None,
        }
        report[circuit] = (sup, on_path, usable, d_to_out)

        print(f"\n=== {circuit} ===  {n} neurons / {len(sg.pre)} edges")
        print(f"  inputs (retina)            {len(inputs):7d}")
        print(f"  readout (VPN)              {len(readout):7d}")
        print(f"  reachable from retina      {reachable.sum():7d}  "
              f"({reachable.mean():.1%})")
        print(f"  ancestors of readout       {ancestors.sum():7d}  "
              f"({ancestors.mean():.1%})")
        print(f"  ON PATH (both)             {on_path.sum():7d}  "
              f"({on_path.mean():.1%})")
        print(f"  on path within {args.hops} hops    {usable.sum():7d}")
        print(f"  median hops retina->readout (where reachable): "
              f"{out[circuit]['median_hops_to_readout']:.0f}")
        print("  top superclasses on the path: " + ", ".join(
            f"{k} {v}" for k, v in out[circuit]["on_path_by_superclass"][:10]))

    # what does full add, and is it on the path?
    sup_f, path_f, usable_f, _ = report["full"]
    sup_c, path_c, _, _ = report["core"]
    core_supers = set(sup_c)
    extra = ~np.isin(sup_f, list(core_supers)) | np.array(
        [s not in set(report["core"][0]) for s in sup_f])
    in_core = np.isin(sup_f, list(core_supers))
    extra_on_path = (~in_core) & path_f
    print("\n=== full vs core ===")
    print(f"  full neurons not belonging to a core superclass: {int(extra_on_path.sum())} "
          f"of them on the retina->readout path")
    print("  biggest on-path superclasses unique-ish to full:")
    for k, v in Counter(sup_f[extra_on_path]).most_common(12):
        print(f"    {k:34s} {v}")

    print("\n=== learning-centre check (on the path?) ===")
    for circuit, (sup, path, usable, _) in report.items():
        hits = Counter()
        for s in set(sup[path]):
            if any(h in str(s).lower() for h in LEARNING_HINTS):
                hits[s] = int(((sup == s) & path).sum())
        print(f"  {circuit}: " + (", ".join(f"{k} {v}" for k, v in hits.most_common(8))
                                 or "no learning-centre superclass on the path"))

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
