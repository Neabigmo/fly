"""Where is the Phase I queue now, and when does each cell finish?

    python scripts/29_phase1_status.py            # every Phase I run in runs/
    python scripts/29_phase1_status.py --tag 10

Reads only ``metrics.jsonl`` and ``config.json``, so it is safe to run against a live
queue -- it never touches a checkpoint or the GPU.  The estimate is deliberately naive:
it extrapolates the observed updates-per-second of the current cell, which is what the
elapsed record already contains, rather than assuming a throughput measured elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.phase1 import spec  # noqa: E402

#: Cells in the order the queue runs them, so progress reads top to bottom.
ORDER = [t.run for t in (spec.FOUNDATION + spec.LINE_A + spec.LINE_B)]


def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarise(run_dir: Path, tag: str) -> dict | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    p1 = cfg.get("phase1") or {}
    if not p1 or (tag and not run_dir.name.startswith(f"{tag}_")):
        return None
    rows = load_rows(run_dir)
    if not rows:
        return None
    budget = int(p1.get("budget") or 0)
    last = rows[-1]
    done = int(last.get("updates") or 0)
    elapsed = float(last.get("elapsed") or 0.0)
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    eta = (budget - done) / rate if rate > 0 else float("nan")
    finished = (run_dir / "full_summary.json").exists()
    return {
        "cell": p1.get("cell", run_dir.name), "dir": run_dir.name,
        "task": p1.get("task", "?"), "brain": p1.get("brain", "?"),
        "budget": budget, "done": done, "probes": len(rows), "finished": finished,
        "rate": rate, "eta_h": eta / 3600 if eta == eta else float("nan"),
        "elapsed_h": elapsed / 3600, "last": last,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="10")
    ap.add_argument("--runs", default="")
    ap.add_argument("--all", action="store_true", help="include tiny/dry runs")
    args = ap.parse_args()

    root = Path(args.runs) if args.runs else paths.RUNS
    found = [s for s in (summarise(d, args.tag) for d in sorted(root.iterdir())
                         if d.is_dir()) if s]
    if not args.all:
        found = [s for s in found if s["budget"] >= 1000]
    if not found:
        print(f"no Phase I runs for tag {args.tag!r} under {root}")
        return 1

    order = {name: i for i, name in enumerate(ORDER)}
    found.sort(key=lambda s: order.get(s["cell"], 99))
    now = datetime.now(timezone.utc)

    print(f"{'cell':7s} {'task':9s} {'brain':11s} {'progress':>17s} {'rate':>8s} "
          f"{'elapsed':>8s} {'eta':>8s}  {'train':>7s} {'seen':>7s} {'hold':>7s} "
          f"{'unsup':>7s} {'probe':>7s}")
    total_left = 0.0
    for s in found:
        r = s["last"]
        prog = f"{s['done']}/{s['budget']}"
        if s["finished"]:
            state, eta = "done", "   --"
        else:
            state = "running"
            eta = f"{s['eta_h']:.1f}h"
            total_left += s["eta_h"]
        probes = {k: v for k, v in r.items() if k.startswith("probe_")}
        pv = max(probes.values()) if probes else float("nan")
        print(f"{s['cell']:7s} {s['task']:9s} {s['brain']:11s} {prog:>17s} "
              f"{s['rate']:8.2f} {s['elapsed_h']:7.2f}h {eta:>8s}  "
              f"{r.get('train_acc', float('nan')):7.3f} "
              f"{r.get('seen_acc', float('nan')):7.3f} "
              f"{r.get('hold_acc', float('nan')):7.3f} "
              f"{r.get('unsup_acc', float('nan')):7.3f} {pv:7.3f}"
              + ("  [finished]" if s["finished"] else f"  [{state}]"))

    planned = [c for c in ORDER if any(s["cell"] == c for s in found)]
    print(f"\n{len(found)} of {len(planned)} planned cells have started; "
          f"remaining work in started cells: {total_left:.1f} h")
    if total_left:
        print(f"if nothing else starts, the queue finishes around "
              f"{(now + timedelta(hours=total_left)).strftime('%Y-%m-%d %H:%M')} UTC "
              f"(cells still queued are not counted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
