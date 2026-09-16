"""Run many training jobs in parallel processes on one GPU.

The core model peaks at ~1.2 GB, so three workers share the 8 GB card
comfortably and cut a 20-run replication from ~10 h to ~3.5 h.  Each worker gets
its own run directory (the run-id suffixing in ``RunContext`` guarantees that),
so there is no shared mutable state between them.

Grids are named so the exact configuration of every study block is recorded in
one place rather than scattered across shell history.

Usage::

    python scripts/15_run_parallel.py --grid replication --workers 3
    python scripts/15_run_parallel.py --grid fixed_updates --workers 3
    python scripts/15_run_parallel.py --grid signed_count --workers 3 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402

# --------------------------------------------------------------------------- #
# Study blocks
# --------------------------------------------------------------------------- #
#: Every grid returns a list of ``(run_label, config_overrides)`` where the
#: overrides are applied on top of an ExperimentConfig with the shared settings
#: below.  ``seed`` is expanded per replicate.
SHARED = {
    "stage": "7",
    "task": "count",
    "model": "M1",
    "circuit": "core",
    "steps": 8,
    "alpha": 0.2,
    "batch_size": 64,
    "max_epochs": 60,
    "patience": 15,
    "readout_standardize": False,
}


def _grid_replication(n_seeds: int, seed_start: int = 0) -> list[dict]:
    """Stage 7a -- the headline replication.

    ``core`` + 20k, ``n_seeds`` real runs against ``n_seeds`` independent shuffled
    graphs, so the real-vs-shuffled gap can be reported with a seed-level CI
    instead of treating thousands of correlated test images as independent.
    """
    out = []
    for seed in range(seed_start, seed_start + n_seeds):
        for graph in ("real", "shuffled"):
            out.append(
                {
                    **SHARED,
                    "graph": graph,
                    "n_train": 20000,
                    "model_seed": seed,
                    "shuffle_seed": seed if graph == "shuffled" else 0,
                    "tag": f"rep_{graph}_s{seed}",
                }
            )
    return out


#: The compute the original learning curve handed to its largest condition:
#: 60 epochs x ceil(20000/64) = 60 x 313 updates.  Every fixed-update run spends
#: exactly this, so the only thing left varying between conditions is how many
#: *distinct* stimuli the model sees.
BASELINE_STEPS = 60 * 313  # 18_780

#: Validation points per run.  The N=20000 baseline selected its checkpoint from
#: 60 validation points, so every fixed-update run gets 60 as well: equal updates
#: *and* equal model-selection granularity.
BASELINE_EVALS = 60


def _grid_fixed_updates() -> list[dict]:
    """Stage 7b -- disentangle sample size from optimiser updates.

    The original curve gave every N the same 60 *epochs*, so N=20000 received
    18,780 updates while N=5000 (which early-stopped between epochs 41 and 60)
    received at most 4,740.  A gap that appears only at N=20000 therefore
    confounds data diversity with compute, and the user's reading of it as
    "20k samples unlock a topological advantage" is not licensed by that design.

    Here every run spends the same 18,780 updates.  If the real-vs-shuffled gap
    is still ~0 at N=5000 and ~0.11 at N=20000, the knee is a data-diversity
    effect; if the gap opens at N=5000 too, the earlier knee was compute.

    ``max_epochs`` is raised per condition so the budget is actually reachable,
    and ``eval_every`` keeps the *number* of validation points at 60 so a
    small-N run is not silently advantaged by selecting over 238 checkpoints.
    """
    out = []
    for n in (5000, 10000, 20000):
        spe = -(-n // SHARED["batch_size"])              # steps per epoch
        epochs = -(-BASELINE_STEPS // spe)               # epochs to spend it
        eval_every = max(1, round(epochs / BASELINE_EVALS))
        for seed in range(3):
            for graph in ("real", "shuffled"):
                if n == 20000 and graph == "shuffled":
                    # control for the machinery itself: N=20000 real must land on
                    # the 0.7684/0.7616/0.7768 the epoch-budget runs already gave
                    continue
                out.append(
                    {
                        **SHARED,
                        "graph": graph,
                        "n_train": n,
                        "model_seed": seed,
                        "shuffle_seed": seed if graph == "shuffled" else 0,
                        "max_steps": BASELINE_STEPS,
                        "max_epochs": epochs,
                        "eval_every": eval_every,
                        "tag": f"fxu_{graph}_N{n}_s{seed}",
                    }
                )
    return out


def _grid_signed_count(n_seeds: int, w_scale: float) -> list[dict]:
    """Stage 7c -- Fly-v2: signed synapses, bounded dynamics, same counting task."""
    out = []
    for seed in range(n_seeds):
        for graph in ("real", "shuffled"):
            out.append(
                {
                    **SHARED,
                    "stage": "7c",
                    "graph": graph,
                    "n_train": 20000,
                    "model_seed": seed,
                    "shuffle_seed": seed if graph == "shuffled" else 0,
                    "w_scale": w_scale,
                    "signed_synapses": True,
                    "tag": f"v2_{graph}_s{seed}",
                }
            )
    return out


GRIDS = {
    "replication": lambda a: _grid_replication(a.seeds, a.seed_start),
    "fixed_updates": lambda a: _grid_fixed_updates(),
    "signed_count": lambda a: _grid_signed_count(a.seeds, a.w_scale),
}


# --------------------------------------------------------------------------- #
def _run_one(spec: dict) -> dict:
    """Worker: build the config, train one model, return its summary."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "2")

    import torch

    from flynum.config import ExperimentConfig
    from flynum.experiments.count import run_count_experiment
    from flynum.pipeline import load_w_scale

    cfg = ExperimentConfig()
    tag = spec.get("tag", "run")
    cfg.stage = spec.get("stage", "7")
    cfg.task = spec.get("task", "count")
    cfg.model = spec.get("model", "M1")
    cfg.data.circuit = spec.get("circuit", "core")
    cfg.data.graph = spec.get("graph", "real")
    cfg.time.steps = int(spec.get("steps", 8))
    cfg.model_cfg.alpha = float(spec.get("alpha", 0.2))
    cfg.model_cfg.signed_synapses = bool(spec.get("signed_synapses", False))
    cfg.model_cfg.readout_standardize = bool(spec.get("readout_standardize", False))
    cfg.model_cfg.weight_normalization = spec.get("normalization", "global")
    cfg.model_cfg.w_scale = float(
        spec.get("w_scale", load_w_scale(cfg.data.circuit))
    )
    cfg.train.n_train = int(spec.get("n_train", 20000))
    cfg.train.batch_size = int(spec.get("batch_size", 64))
    cfg.train.max_epochs = int(spec.get("max_epochs", 60))
    cfg.train.max_steps = int(spec.get("max_steps", 0))
    cfg.train.eval_every = int(spec.get("eval_every", 0))
    cfg.train.patience = int(spec.get("patience", 15))
    cfg.seeds["model_seed"] = int(spec.get("model_seed", 0))
    cfg.seeds["shuffle_seed"] = int(spec.get("shuffle_seed", 0))
    cfg.run_name = f"{cfg.stage}_{tag}"

    t0 = time.time()
    try:
        summary = run_count_experiment(cfg, logger=None)
        return {
            "tag": tag, "status": "ok",
            "run_id": summary.get("run_id"),
            "test_acc_a": summary.get("test_acc_a"),
            "test_acc_b": summary.get("test_acc_b"),
            "test_acc_c": summary.get("test_acc_c"),
            "test_acc_d": summary.get("test_acc_d"),
            "best_val_acc": summary.get("best_val_acc"),
            "optimizer_steps": summary.get("train_info", {}).get("optimizer_steps"),
            "wall_min": round((time.time() - t0) / 60, 1),
        }
    except Exception as exc:  # a failed worker must not kill the block
        import traceback

        return {
            "tag": tag, "status": "error", "error": str(exc),
            "traceback": traceback.format_exc()[-1500:],
            "wall_min": round((time.time() - t0) / 60, 1),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, choices=sorted(GRIDS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=10, help="replicates per condition")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                    help="first model seed (use to add replicates to an existing block)")
    ap.add_argument("--w-scale", type=float, default=0.5, dest="w_scale")
    ap.add_argument("--limit", type=int, default=0, help="run at most N jobs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = get_logger("parallel")
    specs = GRIDS[args.grid](args)
    if args.limit:
        specs = specs[: args.limit]
    log.info("grid=%s : %d jobs, %d parallel workers", args.grid, len(specs), args.workers)
    if args.dry_run:
        for s in specs:
            log.info("  %s", s["tag"])
        return 0

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, s): s["tag"] for s in specs}
        done = 0
        for fut in as_completed(futures):
            tag = futures[fut]
            done += 1
            try:
                r = fut.result()
            except Exception as exc:
                r = {"tag": tag, "status": "error", "error": str(exc)}
            results.append(r)
            if r["status"] == "ok":
                log.info(
                    "[%2d/%2d] %-28s A %.4f B %.4f C %.4f D %.4f  (%.1f min)",
                    done, len(specs), tag, r["test_acc_a"], r["test_acc_b"],
                    r["test_acc_c"], r["test_acc_d"], r["wall_min"],
                )
            else:
                log.error("[%2d/%2d] %-28s FAILED: %s", done, len(specs), tag, r["error"])

    ok = [r for r in results if r["status"] == "ok"]
    log.info("=" * 78)
    log.info("grid %s done: %d/%d ok in %.1f min wall",
             args.grid, len(ok), len(specs), (time.time() - t0) / 60)
    out = paths.DATA_PROCESSED / f"grid_{args.grid}.json"
    import json

    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)
    return 0 if len(ok) == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
