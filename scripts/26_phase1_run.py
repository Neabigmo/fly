"""Run one or more Phase I cells.

    python scripts/26_phase1_run.py --list
    python scripts/26_phase1_run.py --cells C0                 # the foundation brain
    python scripts/26_phase1_run.py --line A                   # Line A, in dependency order
    python scripts/26_phase1_run.py --cells B2-S,B2-C --device cuda
    python scripts/26_phase1_run.py --cells A1-S --dry-run      # CPU, seconds, checks wiring

Cells must respect the warm-start chain C0 -> A1-C -> A2-C -> A3-C; the script orders a
requested set topologically and refuses to start a dependent cell whose source run has no
checkpoint, rather than silently training it from scratch and mislabelling the row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.devices import pick_device  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from flynum.phase1 import spec  # noqa: E402
from flynum.phase1.run import (Cell, cell_from_spec, find_checkpoint,  # noqa: E402
                               run_cell, run_id_for)

ALL: dict[str, spec.TaskSpec] = {
    t.run: t for t in (spec.FOUNDATION + spec.LINE_A + spec.LINE_B + spec.LINE_B_CONTROLS)
}

#: Dry-run shape: enough updates to exercise every probe and every metric path, with
#: pools small enough to render on the CPU in seconds.
DRY = {"budget": 40, "probes": (0, 10, 20, 40), "reps": 3, "eval_items": 48,
       "probe_steps": 25, "ckpt_every": 20, "permanent_every": 40}


def order(cells: list[str]) -> list[str]:
    """Topological order of the warm-start chain, keeping the caller's order otherwise."""
    out: list[str] = []
    pending = list(cells)
    while pending:
        progressed = False
        for name in list(pending):
            src = spec.WARM_START.get(ALL[name].brain, "")
            if not src or src in out or src not in pending:
                out.append(name)
                pending.remove(name)
                progressed = True
        if not progressed:
            raise SystemExit(f"warm-start cycle among {pending}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma-separated run ids")
    ap.add_argument("--line", default="", choices=["", "F", "A", "B"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget", type=int, default=0,
                    help="override the update budget (a short real run on the GPU; the "
                         "probe schedule is re-derived from it)")
    ap.add_argument("--device", default="", help="cuda | cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=600, help="layouts per item")
    ap.add_argument("--eval-items", type=int, default=1500)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--tag", default="10")
    ap.add_argument("--render-workers", type=int, default=1,
                    help="processes for the one-off stimulus render")
    ap.add_argument("--out", default="", help="write the per-cell summaries here")
    args = ap.parse_args()

    log = get_logger("phase1.run")
    if args.list:
        log.info("%-8s %-9s %-9s %9s  %-24s %s", "cell", "task", "brain", "budget",
                 "probes", "holdout")
        for name, t in ALL.items():
            log.info("%-8s %-9s %-9s %9d  %-24s %d pairs", name, t.task, t.brain,
                     t.budget, ",".join(str(p) for p in t.probes), len(t.holdout))
        return 0

    names = [c.strip() for c in args.cells.split(",") if c.strip()]
    if args.line:
        names = [n for n, t in ALL.items() if t.line == args.line]
    if not names:
        raise SystemExit("nothing to run: pass --cells or --line")
    unknown = [n for n in names if n not in ALL]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; known: {list(ALL)}")
    names = order(names)

    device = args.device or ("" if args.dry_run else str(pick_device()))
    if args.dry_run:
        device = "cpu"
    log.info("=" * 88)
    log.info("PHASE I RUN | %s | device=%s | spec %s", ",".join(names),
             device or "auto", spec.fingerprint())
    for name, t in ALL.items():
        if name in names:
            log.info("  %-6s %-9s brain=%-10s budget=%-7d holdout=%s", name, t.task,
                     t.brain, t.budget, list(t.holdout) or "none")

    summaries = {}
    skipped: dict[str, str] = {}
    for name in names:
        t = ALL[name]
        kw = dict(seed=args.seed, device=device, tag=args.tag,
                  reps_per_item=args.reps, eval_items=args.eval_items,
                  probe_steps=args.probe_steps, render_workers=args.render_workers)
        if args.dry_run:
            t = replace(t, budget=DRY["budget"], probes=DRY["probes"])
            kw.update(reps_per_item=DRY["reps"], eval_items=DRY["eval_items"],
                      probe_steps=DRY["probe_steps"], ckpt_every=DRY["ckpt_every"],
                      permanent_every=DRY["permanent_every"], min_train_rows=0)
        elif args.budget:
            t = replace(t, budget=args.budget, probes=spec.probes_for(args.budget))
        src = spec.WARM_START.get(t.brain, "")
        if src and find_checkpoint(src, tag=args.tag, seed=args.seed) is None:
            resolved = paths.run_dir(run_id_for(src, args.tag, args.seed))
            if args.dry_run:
                log.warning("  %s: dry run, no checkpoint at %s/ckpt/final.pt, so it "
                            "trains from scratch.  A dry run cannot validate a warm "
                            "start: it writes its own tagged directories.  The resolution "
                            "itself is covered by tests/test_phase1.py.", name, resolved)
                kw["tag"] = "dry"
                summaries[name] = run_cell(replace(cell_from_spec(t, **kw), brain="scratch"))
                continue
            # One missing prerequisite must not cost the whole queue: this aborted at cell
            # two of twelve and left ten unrun, which is exactly the failure mode a long
            # unattended queue cannot afford.  Dependent cells are recorded as skipped and
            # the process still exits non-zero, so a partial run is loud, not silent.
            reason = (f"needs the checkpoint of {src}, absent at {resolved}/ckpt/")
            log.error("  SKIPPING %s: %s", name, reason)
            skipped[name] = reason
            continue
        t0 = time.time()
        summaries[name] = run_cell(cell_from_spec(t, **kw))
        log.info("  %s finished in %.1f min", name, (time.time() - t0) / 60)

    out = Path(args.out) if args.out else paths.DATA_PROCESSED / (
        "phase1_dry_run.json" if args.dry_run else "phase1_runs.json")
    payload = {k: {kk: vv for kk, vv in v.items() if kk != "history"}
               for k, v in summaries.items()}
    if skipped:
        payload["_skipped"] = skipped
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)
    if skipped:
        log.error("%d cell(s) skipped for a missing prerequisite: %s",
                  len(skipped), ", ".join(skipped))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
