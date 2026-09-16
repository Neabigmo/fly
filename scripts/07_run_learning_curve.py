"""Stage 2 -- the real-vs-shuffled sample-efficiency learning curve.

Grid: ``N_train in {100, 500, 1000, 5000, 20000}`` x ``{real, shuffled}`` x
``3 model seeds`` = 30 runs.

Design notes that matter for the statistics
-------------------------------------------
* Each model seed pairs a real run with a *distinct* shuffled control graph
  (``shuffle_seed = model_seed``), so the comparison is not a single control
  realisation.
* ``data_seed`` and ``split_seed`` are shared by every run, so a given
  ``(N_train, model_seed)`` pair trains on byte-identical images whether the
  graph is real or shuffled.
* Subsets are nested, so a curve mixes sample size, not stimulus luck.
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
from flynum.experiments.count import run_count_grid  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from flynum.train.data import STIMULUS_VERSION  # noqa: E402

N_GRID = (100, 500, 1000, 5000, 20000)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--circuit", default="core", choices=["core", "full"])
    p.add_argument("--models", nargs="*", default=["M1"], choices=["M0", "M1"])
    p.add_argument("--n-grid", nargs="*", type=int, default=list(N_GRID))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--graphs", nargs="*", default=["real", "shuffled"],
                   choices=["real", "shuffled"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="run at most N runs (pilot)")
    p.add_argument("--shuffle-rounds", type=float, default=20.0)
    p.add_argument("--standardize", action="store_true",
                   help="enable readout feature standardisation (off by default: it "
                        "made no measurable difference at N=5000 and destabilises "
                        "training at the default learning rate)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    log = get_logger("learning_curve")
    from flynum.pipeline import load_w_scale

    w_scale = load_w_scale(args.circuit)

    configs: list[ExperimentConfig] = []
    for model in args.models:
        for n in args.n_grid:
            for seed in args.seeds:
                for graph in args.graphs:
                    cfg = ExperimentConfig()
                    cfg.stage = "2"
                    cfg.task = "count"
                    cfg.model = model
                    cfg.data.circuit = args.circuit
                    cfg.data.graph = graph
                    cfg.data.shuffle_rounds_per_edge = args.shuffle_rounds
                    cfg.train.n_train = n
                    cfg.train.batch_size = args.batch_size
                    cfg.train.max_epochs = args.max_epochs
                    cfg.train.patience = args.patience
                    cfg.time.steps = args.steps
                    cfg.model_cfg.w_scale = w_scale
                    cfg.model_cfg.readout_standardize = args.standardize
                    cfg.seeds["model_seed"] = seed
                    # a distinct control graph per replicate; irrelevant for real
                    cfg.seeds["shuffle_seed"] = seed if graph == "shuffled" else 0
                    configs.append(cfg)

    log.info(
        "learning-curve grid: %d runs | circuit=%s models=%s N=%s seeds=%s graphs=%s "
        "| w_scale=%.3g | stimulus version %s",
        len(configs), args.circuit, args.models, args.n_grid, args.seeds,
        args.graphs, w_scale, STIMULUS_VERSION,
    )
    if args.dry_run:
        for c in configs:
            log.info("  %s", c.run_id())
        return 0
    if args.limit:
        configs = configs[: args.limit]
        log.info("limited to %d runs", len(configs))

    t0 = time.time()
    results = run_count_grid(configs, logger=log)
    ok = [r for r in results if r.get("status") != "error"]
    log.info(
        "learning curve finished: %d/%d ok in %.1f min",
        len(ok), len(results), (time.time() - t0) / 60,
    )
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
