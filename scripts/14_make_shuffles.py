"""Generate several shuffled control graphs **in parallel**.

Why this exists
---------------
The degree/weight-preserving edge swap is CPU-bound and inherently sequential
within one graph: every accepted swap mutates the edge set that the next
proposal is checked against.  A single Python process therefore spends tens of
minutes on the 12.4M-edge ``full`` circuit (measured: ~40 min at 20 rounds per
edge), and the GPU sits idle behind it.

Different **seed values are independent graphs**, though, so they parallelise
perfectly.  This script runs one worker process per seed, each capped to a few
BLAS threads so the workers do not fight over the same cores.

Usage::

    python scripts/14_make_shuffles.py --circuit core --seeds 0 1 2 --workers 3
    python scripts/14_make_shuffles.py --circuit full --seeds 0 1 --workers 2

The resulting cache files are byte-compatible with what
:func:`flynum.pipeline.get_graph` expects, so training runs then load them
instantly instead of rebuilding them inline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402


def _build_one(payload: dict) -> dict:
    """Worker: build and cache one shuffled graph for one seed."""
    # keep each worker to a modest number of BLAS threads
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(payload["threads"]))

    import numpy as np

    from flynum.config import ExperimentConfig
    from flynum.data.annotations import load_annotations
    from flynum.pipeline import get_graph, get_subgraph

    circuit, seed, rounds = payload["circuit"], payload["seed"], payload["rounds"]
    t0 = time.time()
    ann = load_annotations()
    sg = get_subgraph(ann, circuit, force=False)
    cfg = ExperimentConfig()
    cfg.data.circuit = circuit
    cfg.data.graph = "shuffled"
    cfg.data.shuffle_rounds_per_edge = rounds
    cfg.seeds["shuffle_seed"] = seed
    res = get_graph(sg, cfg, logger=None)
    stats = dict(res.stats)
    stats.update(
        {
            "circuit": circuit,
            "seed": seed,
            "rounds_per_edge": rounds,
            "n_neurons": sg.n_neurons,
            "n_edges": sg.n_edges,
            "wall_seconds": round(time.time() - t0, 1),
        }
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="core", choices=["core", "full", "c3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rounds", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes (default: one per seed, capped "
                         "at 6).  Measured CPU utilisation during a single shuffle "
                         "is only ~20%%, so several seeds should always be built at "
                         "once rather than one after another.")
    ap.add_argument("--threads", type=int, default=2,
                    help="BLAS threads per worker, to avoid oversubscription")
    args = ap.parse_args()

    log = get_logger("make_shuffles")
    workers = args.workers or min(len(args.seeds), 6)
    log.info(
        "building %d shuffled graph(s) for circuit=%s, seeds=%s, rounds/edge=%g, "
        "workers=%d (each capped to %d BLAS threads)",
        len(args.seeds), args.circuit, args.seeds, args.rounds, workers, args.threads,
    )

    payloads = [
        {"circuit": args.circuit, "seed": s, "rounds": args.rounds, "threads": args.threads}
        for s in args.seeds
    ]
    results = []
    t0 = time.time()
    if workers <= 1:
        for p in payloads:
            log.info("seed %d ...", p["seed"])
            results.append(_build_one(p))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_build_one, p): p["seed"] for p in payloads}
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    stats = fut.result()
                    results.append(stats)
                    log.info(
                        "seed %d done: %d edges, overlap %.2f%%, %.1f min",
                        seed, stats["edges"],
                        100 * stats.get("final_edge_overlap", float("nan")),
                        stats["wall_seconds"] / 60,
                    )
                except Exception as exc:  # keep the other workers going
                    log.error("seed %d FAILED: %s", seed, exc)

    log.info("=" * 70)
    log.info("%-6s %12s %10s %12s %10s", "seed", "edges", "swaps", "overlap%", "minutes")
    for s in sorted(results, key=lambda r: r["seed"]):
        log.info(
            "%-6d %12d %10d %11.2f%% %10.1f",
            s["seed"], s["edges"], s.get("swaps", -1),
            100 * s.get("final_edge_overlap", float("nan")), s["wall_seconds"] / 60,
        )
    log.info(
        "done: %d/%d graphs in %.1f min wall",
        len(results), len(payloads), (time.time() - t0) / 60,
    )

    out = paths.DATA_PROCESSED / f"shuffle_report_{args.circuit}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)
    return 0 if len(results) == len(payloads) else 1


if __name__ == "__main__":
    raise SystemExit(main())
