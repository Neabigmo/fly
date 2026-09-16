"""Stage 4 / 5 -- train and test two-epoch addition.

``a`` dots are shown for a few steps, then a blank delay, then ``b`` dots; the
network must output ``a + b``.  Specific ordered pairs are held out of training
entirely (default ``2+3`` and ``3+2``), so accuracy on them measures
compositional generalisation rather than lookup.

Examples
--------
    python scripts/08_run_addition.py --graph real
    python scripts/08_run_addition.py --graph shuffled --shuffle-seed 1
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
from flynum.experiments.addition import run_addition_experiment  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="real", choices=["real", "shuffled"])
    p.add_argument("--model", default="M1", choices=["M0", "M1"])
    p.add_argument("--circuit", default="core", choices=["core", "full"])
    p.add_argument("--steps-a", type=int, default=5)
    p.add_argument("--steps-gap", type=int, default=4)
    p.add_argument("--steps-b", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--reps", type=int, default=600, help="images per ordered pair")
    p.add_argument("--a-min", type=int, default=1)
    p.add_argument("--a-max", type=int, default=4)
    p.add_argument("--holdout", nargs="*", default=["2,3", "3,2"])
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--w-scale", type=float, default=None,
                   help="override the calibrated w_scale. The addition sequence is "
                        "14 steps long, and the connectome recurrence with unsigned "
                        "weights is only marginally stable: at the counting value "
                        "(1.0) the state grows from 0.32 at t=8 to 3.3 at t=14 and "
                        "training diverges. 0.5 is the largest value that stays "
                        "bounded over 14 steps.")
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--all-seeds", action="store_true",
                   help="run every model seed sequentially (3 replicates)")
    p.add_argument("--standardize", action="store_true",
                   help="enable readout feature standardisation (off by default: it "
                        "destabilises training at the default learning rate)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    log = get_logger("run_addition")
    w_scale = float(args.w_scale) if args.w_scale is not None else float(
        json.loads((paths.DATA_PROCESSED / "calibration.json").read_text(encoding="utf-8"))[
            "chosen_w_scale"
        ]
    )
    holdout = tuple(tuple(int(v) for v in h.split(",")) for h in args.holdout)

    seeds = [0, 1, 2] if args.all_seeds else [args.model_seed]
    configs = []
    for seed in seeds:
        cfg = ExperimentConfig()
        cfg.stage = "4"
        cfg.task = "add"
        cfg.model = args.model
        cfg.data.circuit = args.circuit
        cfg.data.graph = args.graph
        cfg.stimulus.a_min = args.a_min
        cfg.stimulus.a_max = args.a_max
        cfg.stimulus.holdout_pairs = holdout
        cfg.stimulus.add_reps_per_pair = args.reps
        cfg.time.steps_a = args.steps_a
        cfg.time.steps_gap = args.steps_gap
        cfg.time.steps_b = args.steps_b
        cfg.model_cfg.alpha = args.alpha
        cfg.model_cfg.w_scale = w_scale
        cfg.model_cfg.readout_standardize = args.standardize
        cfg.train.batch_size = args.batch_size
        cfg.train.max_epochs = args.max_epochs
        cfg.train.patience = args.patience
        cfg.seeds["model_seed"] = seed
        cfg.seeds["shuffle_seed"] = seed if args.graph == "shuffled" else 0
        # label the run by its actual dataset size (ordered pairs minus holdouts)
        n_pairs = (args.a_max - args.a_min + 1) ** 2 - len(holdout)
        cfg.train.n_train = n_pairs * args.reps
        configs.append(cfg)

    log.info(
        "addition: %d run(s) | graph=%s model=%s circuit=%s | holdout=%s | "
        "steps %d+%d+%d | reps=%d | w_scale=%.3g",
        len(configs), args.graph, args.model, args.circuit, holdout,
        args.steps_a, args.steps_gap, args.steps_b, args.reps, w_scale,
    )
    if args.dry_run:
        for c in configs:
            log.info("  %s", c.run_id())
        return 0

    t0 = time.time()
    failures = 0
    for i, cfg in enumerate(configs, 1):
        log.info("[%d/%d] %s", i, len(configs), cfg.run_id())
        try:
            run_addition_experiment(cfg, logger=log)
        except Exception as exc:
            log.exception("addition run failed: %s", exc)
            failures += 1
    log.info("addition finished in %.1f min (%d failures)", (time.time() - t0) / 60, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
