"""Print one Phase I cell's probe history as a table.

    python scripts/33_phase1_curves.py --cell B1-S --tag 10

The compact view the reports are built from: accuracy on the taught items, on the
withheld pairs, and on the unsupported items, next to the continuous metrics that decide
whether a change in accuracy is a change in behaviour.  Nothing here is computed -- it
prints what the run recorded, so it can be used to watch a cell while it trains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--tag", default="10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keys", default="")
    args = ap.parse_args()

    run = paths.run_dir(f"{args.tag}_{args.cell}_s{args.seed}")
    path = run / "metrics.jsonl"
    if not path.exists():
        print(f"no metrics at {path}")
        return 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        return 1
    keys = ([k.strip() for k in args.keys.split(",")] if args.keys else
            ["train_acc", "taught_area_acc", "hold_acc", "unsup_acc",
             "hold_ce", "hold_p_correct", "hold_margin"])
    keys = [k for k in keys if any(k in r for r in rows)]

    width = max(9, *(len(k) for k in keys))
    header = "".join(f"{k:>{width}s}" for k in keys)
    print(f"{'updates':>9s}{header}")
    for r in rows:
        cells = ""
        for k in keys:
            v = r.get(k)
            cells += (f"{'--':>{width}s}" if not isinstance(v, (int, float)) or v != v
                      else f"{v:>{width}.3f}")
        print(f"{r['updates']:>9d}{cells}")
    print(f"\n{len(rows)} probes in {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
