"""Stage 8 -- run the guided addition curriculum and the leave-pair-out test.

Four conditions, forming the 2x2 table that isolates *what* transfers::

    real   + count-pretrained      real   + scratch
    shuffled + count-pretrained    shuffled + scratch

"Count-pretrained" warm-starts the recurrent parameters (the learned gains and
biases) from a counting run and leaves the readout fresh; "scratch" starts from
g=1 everywhere.  Because the readout is task specific and is reset in both cases,
any difference is attributable to the learned connectome.

The three-lesson schedule (operands first, sum later) lives in
:mod:`flynum.experiments.curriculum`.  The leave-pair-out split always keeps every
sum reachable from the remaining pairs, so an unseen pair tests composition
rather than an unseen output class.

Usage::

    python scripts/16_run_curriculum.py --graph real     --pretrain 7_rep_real_s0
    python scripts/16_run_curriculum.py --graph shuffled --pretrain 7_rep_shuffled_s0
    python scripts/16_run_curriculum.py --graph real     --scratch
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
from flynum.experiments.curriculum import (  # noqa: E402
    CurriculumConfig,
    run_curriculum,
)
from flynum.logging_utils import get_logger  # noqa: E402

#: Leave-pair-out splits.  In every case each sum stays reachable through at
#: least one other ordered pair, which is what makes the test a composition test.
LPO_SPLITS: dict[str, tuple[tuple[int, int], ...]] = {
    "23": ((2, 3), (3, 2)),
    "13": ((1, 3), (3, 1)),
    "14": ((1, 4), (4, 1)),
    "24": ((2, 4), (4, 2)),
}


def _check_split(holdout: tuple[tuple[int, int], ...], a_min: int, a_max: int) -> None:
    """Every held-out sum must still be producible by a training pair."""
    hold = set(holdout)
    train_sums = {
        a + b
        for a in range(a_min, a_max + 1)
        for b in range(a_min, a_max + 1)
        if (a, b) not in hold
    }
    for a, b in hold:
        if a + b not in train_sums:
            raise ValueError(
                f"holding out {a}+{b} removes sum {a + b} from training entirely; "
                "the test would measure class generalisation, not composition"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="real", choices=["real", "shuffled"])
    ap.add_argument("--holdout", default="23", choices=sorted(LPO_SPLITS))
    ap.add_argument("--pretrain", default="", help="run id to warm-start from")
    ap.add_argument("--scratch", action="store_true",
                    help="ignore --pretrain and train from g=1")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--operand-epochs", type=int, default=15, dest="operand_epochs")
    ap.add_argument("--w-sum", type=float, default=1.0, dest="w_sum")
    ap.add_argument("--max-epochs", type=int, default=70)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--steps-a", type=int, default=5)
    ap.add_argument("--steps-gap", type=int, default=4)
    ap.add_argument("--steps-b", type=int, default=5)
    ap.add_argument("--w-scale", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = get_logger("curriculum_cli")
    holdout = LPO_SPLITS[args.holdout]
    _check_split(holdout, 1, 4)
    w_scale = args.w_scale
    if w_scale is None:
        from flynum.pipeline import load_w_scale

        w_scale = load_w_scale("core")

    results = []
    for seed in args.seeds:
        cfg = ExperimentConfig()
        cfg.stage = "8"
        cfg.task = "add"
        cfg.model = "M1"
        cfg.data.circuit = "core"
        cfg.data.graph = args.graph
        cfg.stimulus.holdout_pairs = holdout
        cfg.stimulus.add_reps_per_pair = args.reps
        cfg.time.steps_a = args.steps_a
        cfg.time.steps_gap = args.steps_gap
        cfg.time.steps_b = args.steps_b
        cfg.model_cfg.alpha = args.alpha
        cfg.model_cfg.w_scale = w_scale
        cfg.model_cfg.signed_synapses = args.signed
        cfg.model_cfg.readout_standardize = False
        cfg.train.batch_size = 64
        cfg.train.max_epochs = args.max_epochs
        cfg.train.patience = args.patience
        cfg.seeds["model_seed"] = seed
        cfg.seeds["shuffle_seed"] = seed if args.graph == "shuffled" else 0

        cc = CurriculumConfig(
            operand_epochs=args.operand_epochs,
            w_sum=args.w_sum,
            pretrain_from="" if args.scratch else args.pretrain,
            scratch=args.scratch,
        )
        label = f"{args.graph}/{'scratch' if args.scratch else 'pretrained'}/lpo{args.holdout}/s{seed}"
        log.info("=" * 78)
        log.info("CURRICULUM %s | holdout=%s | stages %d+%d+%d | w_scale=%.3g | signed=%s",
                 label, holdout, args.steps_a, args.steps_gap, args.steps_b, w_scale,
                 args.signed)
        if args.dry_run:
            continue
        t0 = time.time()
        try:
            s = run_curriculum(cfg, cc, logger=log)
            s["label"] = label
            s["holdout"] = [list(p) for p in holdout]
            results.append(s)
            log.info(
                "  -> a %.4f  b %.4f  sum %.4f  (chance a 0.25 / sum 0.143)  %.1f min",
                s["acc_a_test"], s["acc_b_test"], s["sum_test"], (time.time() - t0) / 60,
            )
        except Exception as exc:
            log.exception("curriculum failed: %s", exc)
            results.append({"label": label, "status": "error", "error": str(exc)})

    if results:
        out = paths.DATA_PROCESSED / f"curriculum_{args.graph}_{args.holdout}.json"
        out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
