"""Step 4: does the mushroom body close the loop -- can it be taught, and can it speak?

The canonical MB microcircuit is KC -> MBON with dopamine (DAN) and octopamine (OA)
injecting the learning signal onto KC axons and MBON dendrites.  For a *visually educated*
brain the loop must also close in the other direction: MBON output has to reach the
read-out population, or the learning centre can never change behaviour.

    python scripts/38_explore_mb_loop.py
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


def main() -> int:
    ann = load_annotations()
    nt = pd.read_feather(paths.raw_path("neurotransmitters"))[["body", "cell_type"]]
    nt = nt.dropna(subset=["cell_type"])
    per_body = nt.groupby("body")["cell_type"].first()
    pos = {int(b): i for i, b in enumerate(ann.body_ids)}
    idx = np.array([pos[int(b)] for b in per_body.index if int(b) in pos])
    types = np.array([str(per_body[b]) for b in per_body.index if int(b) in pos])

    def pick(*prefixes):
        return idx[np.array([any(t.lower().startswith(p) for p in prefixes)
                             for t in types])]

    kc = pick("kc")
    mbon = pick("mbon")
    dan = pick("ppl", "dan")
    oa = pick("oa-", "vum", "vpm")
    vpn = np.nonzero(ann.superclass == "visual_projection")[0]
    print(f"KC {len(kc)} | MBON {len(mbon)} | DAN {len(dan)} | OA {len(oa)} | VPN {len(vpn)}")

    e = cx.load_edge_list()
    pre, post = e["pre"].astype(np.int64), e["post"].astype(np.int64)
    del e

    def count(src, dst):
        return int(np.isin(pre, src).__and__(np.isin(post, dst)).sum())

    print("\n=== MB microcircuit (synapse counts) ===")
    print(f"  KC   -> MBON   {count(kc, mbon):>9,}")
    print(f"  DAN  -> KC     {count(dan, kc):>9,}    DAN -> MBON  {count(dan, mbon):>9,}")
    print(f"  OA   -> KC     {count(oa, kc):>9,}    OA  -> MBON  {count(oa, mbon):>9,}")
    print(f"  VPN  -> KC     {count(vpn, kc):>9,}    (visual input to the learning centre)")
    print(f"  VPN  -> MBON   {count(vpn, mbon):>9,}")
    print(f"  KC   -> VPN    {count(kc, vpn):>9,}    MBON -> VPN  {count(mbon, vpn):>9,}")

    # one and two hop routes MBON -> VPN
    print("\n=== can the learning centre reach the read-out? ===")
    n = ann.n
    mbon_to = np.isin(pre, mbon)
    nxt = post[mbon_to]
    print(f"  MBON targets (1 hop): {len(np.unique(nxt)):,} neurons, "
          f"{len(nxt):,} synapses")
    print(f"    top superclasses: {Counter(ann.superclass[nxt]).most_common(6)}")
    hop2 = np.isin(pre, np.unique(nxt))
    print(f"  2-hop targets: {len(np.unique(post[hop2])):,} neurons "
          f"({int(np.isin(post[hop2], vpn).sum()):,} synapses onto the read-out pool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
