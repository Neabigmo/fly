"""Explore the MaleCNS annotation table: what is actually in the brain we never used.

Step 1 of the data exploration: the cell-type vocabulary, where the learning centres are,
how many neurons carry each neurotransmitter, and the id-range problem in the existing
sign loader.

    python scripts/35_explore_annotations.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.data.annotations import load_annotations  # noqa: E402

#: Substrings that identify the circuits a study about *learning* cannot do without.
HINTS = {
    "Kenyon cell (MB)": ("kc", "kenyon"),
    "MB output (MBON)": ("mbon",),
    "MB dopamine (DAN/PPL)": ("dan", "ppl", "pal"),
    "octopamine (OA/VUM)": ("oa", "vum", "octopamine"),
    "central complex (PB/EB/FB/NO)": ("pb", "eb", "fb", "no", "cx"),
    "lateral horn": ("lh", "lateral"),
    "AOTU / tubercle (visual->MB)": ("aotu", "tu", "atl"),
    "olfactory (ORN/PN/LN)": ("orn", "pn", "ln", "antennal"),
    "clock": ("clock", "ln_", "sln", "lnv", "lnd"),
}


def main() -> int:
    ann = load_annotations()
    print(f"annotated neurons: {ann.n}")
    sup = ann.superclass
    ct = ann.cell_type

    print("\n--- superclass counts (whole table) ---")
    for k, v in Counter(sup).most_common(12):
        print(f"  {str(k):32s} {v}")

    print("\n--- top 30 cell_type values overall ---")
    for k, v in Counter(ct).most_common(30):
        print(f"  {str(k):34s} {v}")

    print("\n--- cell types matching learning / navigation hints ---")
    for label, keys in HINTS.items():
        hits = Counter()
        for c, s in zip(ct, sup):
            cl = str(c).lower()
            if any(k == cl or cl.startswith(k + "_") or f"_{k}" in cl or cl == k
                   for k in keys):
                hits[(str(c), str(s))] += 1
        total = sum(hits.values())
        print(f"  {label:32s} {total:6d}  " +
              (", ".join(f"{c}[{s}]={n}" for (c, s), n in hits.most_common(4))
               or "(none)"))

    print("\n--- how much of each superclass carries hex coordinates ---")
    hex_ok = ~np.isnan(ann.hex1)
    for s in ("ol_intrinsic", "visual_projection", "cb_intrinsic", "vnc_intrinsic",
              "__unassigned__"):
        m = sup == s
        if m.sum():
            print(f"  {s:24s} n={int(m.sum()):7d}  with hex={int((m & hex_ok).sum()):7d}")

    # --- the sign loader's id-range problem ------------------------------------- #
    print("\n--- neurotransmitter file ---")
    path = paths.raw_path("neurotransmitters")
    import pandas as pd

    df = pd.read_feather(path)
    print(f"  rows={len(df)}  columns={list(df.columns)[:8]}")
    idc = [c for c in df.columns if "body" in c.lower() or "id" in c.lower()]
    if idc:
        ids = df[idc[0]].to_numpy()
        print(f"  id column {idc[0]!r}: min={ids.min()} max={ids.max()} "
              f"-> an id-sized lookup array would need {ids.max() / 1e9:.2f} GB")
    for col in df.columns:
        if col.lower() in ("nt_type", "neurotransmitter", "top_nt", "sign", "nt"):
            counts = Counter(df[col].astype(str))
            print(f"  {col}: {dict(counts.most_common(10))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
