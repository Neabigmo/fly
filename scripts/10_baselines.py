"""Baselines -- how hard is the task, and which shortcuts exist?

Trains three non-connectome references on exactly the same images:

* ``area``   -- only the total ink is visible,
* ``pixels`` -- logistic regression on the raw 32x32 image,
* ``cnn``    -- a small convolutional network (task-difficulty ceiling).

The scientific point is the *contrast*: ``area`` must be near chance on the
controlled conditions, and the CNN shows how much of the task is solvable
without any connectome structure at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.logging_utils import RunContext  # noqa: E402
from flynum.train.baselines import (  # noqa: E402
    run_area_baseline,
    run_cnn_baseline,
    run_pixel_baseline,
)
from flynum.train.data import build_count_data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()

    ctx = RunContext("baselines")
    log = ctx.log
    log.info("=" * 78)
    log.info("BASELINES (non-connectome references)")
    log.info("=" * 78)

    cfg = ExperimentConfig()
    cfg.train.n_train = args.n_train
    data = build_count_data(cfg, logger=log)
    n_classes = cfg.stimulus.n_max - cfg.stimulus.n_min + 1

    train = {
        "images": data.pool_images,
        "labels": data.pool_labels,
        "ink": data.pool_ink,
    }
    if len(train["images"]) > args.n_train:
        idx = np.random.default_rng(0).permutation(len(train["images"]))[: args.n_train]
        train = {k: v[idx] for k, v in train.items()}

    results: dict = {}
    for name, fn in (
        ("area", lambda: run_area_baseline(train, data.tests, n_classes)),
        ("pixels", lambda: run_pixel_baseline(train, data.tests, n_classes)),
        ("cnn", None if args.skip_cnn else lambda: run_cnn_baseline(
            train, data.tests, n_classes, epochs=args.epochs,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu")),
    ):
        if fn is None:
            continue
        log.info("-" * 78)
        log.info("baseline: %s", name)
        out = fn()
        results[name] = {mode: r.as_dict() for mode, r in out.items()}
        for mode, r in out.items():
            log.info(
                "  %s -> acc %.4f | macro-F1 %.4f", mode, r.accuracy, r.macro_f1
            )
            ctx.metric(baseline=name, split=mode, accuracy=r.accuracy)

    out_path = paths.DATA_PROCESSED / "baselines.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("-" * 78)
    log.info("summary (accuracy by condition):")
    header = "%-8s " + " ".join(f"{m:>8s}" for m in ("A", "B", "C", "D"))
    log.info(header, "model")
    for name, res in results.items():
        log.info(
            "%-8s " + " ".join(f"{res[m]['accuracy']:8.4f}" for m in ("A", "B", "C", "D")),
            name,
        )
    ctx.finish("ok", baselines=list(results))
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
