"""Architecture ablation: find settings that let the fly actually learn.

Runs a configurable grid of short experiments at a fixed (small) sample budget
and prints one compact table, so design choices can be compared without
committing to the full 30-run learning curve.

Knobs, and why each is worth testing:

``--readout``      the readout population.  ``vpn`` is the biologically clean
                   choice (optic-lobe output, contains LC11), but it may discard
                   numerosity information that lives in the medulla, in which
                   case ``all_except_input`` should do better.
``--alpha``        the leak rate.  It sets the memory time constant, which the
                   addition task depends on, and also how fast the state settles.
``--lambda-gain``  how strongly the gains are tethered to the measured wiring.
``--steps``        how many recurrent steps the computation gets.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.experiments.count import run_count_experiment  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--readout", nargs="*", default=["vpn"])
    ap.add_argument("--alpha", nargs="*", type=float, default=[0.2])
    ap.add_argument("--steps", nargs="*", type=int, default=[8])
    ap.add_argument("--act", nargs="*", default=["relu"])
    ap.add_argument("--w-scale", nargs="*", type=float, default=[1.0])
    ap.add_argument("--lambda-gain", nargs="*", type=float, default=[1e-3],
                    dest="lambda_gain")
    ap.add_argument("--lr-gain", type=float, default=1e-2)
    ap.add_argument("--lr-readout", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--circuit", default="core")
    ap.add_argument("--graph", default="real", choices=["real", "shuffled"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    log = get_logger("ablation")
    grids = list(itertools.product(
        args.act, args.w_scale, args.steps, args.readout, args.alpha, args.lambda_gain
    ))
    if args.limit:
        grids = grids[: args.limit]
    log.info("ablation grid: %d configurations", len(grids))

    rows = []
    t0 = time.time()
    for i, (act, ws, steps, readout, alpha, lam) in enumerate(grids, 1):
        cfg = ExperimentConfig()
        cfg.stage = "abl"
        cfg.task = "count"
        cfg.model = "M1"
        cfg.data.circuit = args.circuit
        cfg.data.graph = args.graph
        cfg.train.n_train = args.n_train
        cfg.train.max_epochs = args.max_epochs
        cfg.train.patience = args.patience
        cfg.train.batch_size = args.batch_size
        cfg.train.lr_gain = args.lr_gain
        cfg.train.lr_readout = args.lr_readout
        cfg.model_cfg.alpha = alpha
        cfg.model_cfg.nonlinearity = act
        cfg.model_cfg.w_scale = ws
        cfg.model_cfg.readout = readout
        cfg.model_cfg.lambda_gain = lam
        cfg.time.steps = steps

        label = (f"{act}|w{ws:g}|T{steps}|{readout}|a{alpha:g}|l{lam:g}")
        log.info("[%d/%d] %s", i, len(grids), label)
        try:
            s = run_count_experiment(cfg, logger=log)
            rows.append(
                {
                    "label": label, "act": act, "w_scale": ws, "steps": steps,
                    "readout": readout, "alpha": alpha, "lambda_gain": lam,
                    "val": s.get("best_val_acc"), "A": s.get("test_acc_a"),
                    "B": s.get("test_acc_b"), "C": s.get("test_acc_c"),
                    "D": s.get("test_acc_d"), "epochs": s.get("train_info", {}).get("epochs_run"),
                    "params": s.get("n_params"), "run_id": s.get("run_id"),
                }
            )
        except Exception as exc:
            log.exception("configuration failed: %s", exc)
            rows.append({"label": label, "A": float("nan"), "error": str(exc)})

    out = paths.DATA_PROCESSED / "ablation.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    log.info("=" * 96)
    log.info("%-46s %7s %7s %7s %7s %6s", "config", "val", "A", "B", "C", "ep")
    log.info("-" * 96)
    for r in sorted(rows, key=lambda r: -(r.get("A") or 0)):
        log.info(
            "%-46s %7.4f %7.4f %7.4f %7.4f %6s",
            r["label"], r.get("val") or float("nan"), r.get("A") or float("nan"),
            r.get("B") or float("nan"), r.get("C") or float("nan"), r.get("epochs"),
        )
    best = max(rows, key=lambda r: (r.get("A") or 0))
    log.info("-" * 96)
    log.info("best: %s -> test A %.4f (%.1f min total)",
             best["label"], best.get("A") or float("nan"), (time.time() - t0) / 60)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
