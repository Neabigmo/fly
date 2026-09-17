"""Run the 2x2 addition-teaching pilot: scratch vs count-pretrained, full vs holdout.

Four runs, one seed, everything else locked -- in particular the **number of optimiser
updates**, because this project already found that an epoch-matched comparison gave one
condition four times the compute of another and produced a knee that was an artefact.

    A1  scratch    + full      can a fly brain be taught addition at all?
    A2  pretrained + full      does having learned to count make that easier?
    B1  scratch    + holdout   can it infer the one pair it was never taught?
    B2  pretrained + holdout   does counting help it infer that pair?

Usage::

    python scripts/24_addition_pilot.py --pretrain <count_run_id>      # all four
    python scripts/24_addition_pilot.py --pretrain <id> --cells A2 B2  # a subset
    python scripts/24_addition_pilot.py --pretrain <id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.experiments.addition_pilot import (  # noqa: E402
    HOLDOUT_PAIR,
    PROBE_STEPS,
    PilotConfig,
    run_addition_pilot,
)
from flynum.logging_utils import get_logger  # noqa: E402

#: The four cells.  ``pretrain`` marks which ones warm-start from the counting run.
CELLS: dict[str, dict] = {
    "A1": {"brain": "scratch", "teach_holdout": True,
           "question": "can addition be taught from scratch at all?"},
    "A2": {"brain": "pretrained", "teach_holdout": True,
           "question": "does counting make the taught pairs easier to learn?"},
    "B1": {"brain": "scratch", "teach_holdout": False,
           "question": "can the withheld pair be inferred from scratch?"},
    "B2": {"brain": "pretrained", "teach_holdout": False,
           "question": "does counting help infer the withheld pair?"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrain", default="",
                    help="run id of the counting run used as the warm start (required "
                         "for the A2/B2 cells)")
    ap.add_argument("--cells", nargs="*", default=sorted(CELLS), choices=sorted(CELLS))
    ap.add_argument("--budget-steps", type=int, default=10_000, dest="budget_steps")
    ap.add_argument("--reps", type=int, default=600,
                    help="items per ordered pair in the addition dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-items", type=int, default=2000, dest="eval_items")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="integration rate; the addition horizon diverges at 0.2")
    ap.add_argument("--w-scale", type=float, default=None, dest="w_scale")
    ap.add_argument("--steps-a", type=int, default=5, dest="steps_a")
    ap.add_argument("--steps-gap", type=int, default=4, dest="steps_gap")
    ap.add_argument("--steps-b", type=int, default=5, dest="steps_b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = get_logger("pilot_cli")
    w_scale = args.w_scale
    if w_scale is None:
        from flynum.pipeline import load_w_scale

        w_scale = load_w_scale("core")

    needs_pretrain = [c for c in args.cells if CELLS[c]["brain"] == "pretrained"]
    if needs_pretrain and not args.pretrain:
        log.error("cells %s need --pretrain <count_run_id>", needs_pretrain)
        return 2

    log.info("=" * 92)
    log.info("2x2 ADDITION PILOT | cells=%s | seed=%d | budget=%d updates | probes=%s",
             args.cells, args.seed, args.budget_steps, list(PROBE_STEPS))
    log.info("teach %d ordered pairs; hold out exactly %s (its sum stays reachable "
             "through 1+4, 3+2, 4+1)", 16 if True else 15, HOLDOUT_PAIR)
    log.info("locked: seed, dataset, batch-order seed, optimiser, lr, updates, "
             "alpha=%.2f, w_scale=%.3g, T=%d+%d+%d", args.alpha, w_scale,
             args.steps_a, args.steps_gap, args.steps_b)

    results = []
    for cell in args.cells:
        spec = CELLS[cell]
        cfg = ExperimentConfig()
        cfg.stage = "9"
        cfg.task = "add"
        cfg.model = "M1"
        cfg.data.circuit = "core"
        cfg.data.graph = "real"
        # the split boundary is always 2+3, so all four cells share one probe set
        cfg.stimulus.holdout_pairs = (HOLDOUT_PAIR,)
        cfg.stimulus.add_reps_per_pair = args.reps
        cfg.model_cfg.alpha = args.alpha
        cfg.model_cfg.w_scale = w_scale
        cfg.model_cfg.signed_synapses = False
        cfg.model_cfg.readout_standardize = False
        cfg.time.steps_a = args.steps_a
        cfg.time.steps_gap = args.steps_gap
        cfg.time.steps_b = args.steps_b
        cfg.train.batch_size = 64
        cfg.seeds["model_seed"] = args.seed
        pc = PilotConfig(
            label=cell,
            pretrain_from=args.pretrain if spec["brain"] == "pretrained" else "",
            teach_holdout=spec["teach_holdout"],
            budget_steps=args.budget_steps,
            eval_items=args.eval_items,
        )
        log.info("-" * 92)
        log.info("%s (%s/%s): %s", cell, spec["brain"],
                 "full" if spec["teach_holdout"] else "holdout", spec["question"])
        if args.dry_run:
            log.info("    would train %s for %d updates",
                     "16 pairs" if spec["teach_holdout"] else "15 pairs",
                     args.budget_steps)
            continue
        t0 = time.time()
        try:
            s = run_addition_pilot(cfg, pc, logger=log)
            results.append(s)
            log.info("%s done in %.1f min", cell, (time.time() - t0) / 60)
        except Exception as exc:
            log.exception("%s FAILED: %s", cell, exc)
            results.append({"console": cell, "status": "error", "error": str(exc)})

    if results:
        out = paths.DATA_PROCESSED / "addition_pilot.json"
        out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        log.info("wrote %s", out)
    bad = [r for r in results if r.get("status") == "error"]
    if bad:
        log.error("%d/%d cell(s) failed", len(bad), len(results))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
