"""Print a compact status table for every run in the study.

Useful while a long grid is running: shows which configurations have finished,
their headline metrics, and what is still pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="", help="filter by task (count|add)")
    ap.add_argument("--sort", default="run_id")
    ap.add_argument("--last", type=int, default=0, help="show only the last N runs")
    ap.add_argument("--rebuild-index", action="store_true", dest="rebuild_index",
                    help="rewrite runs/index.csv from the run directories, dropping "
                         "rows whose directory no longer exists")
    args = ap.parse_args()

    if args.rebuild_index:
        from flynum.logging_utils import rebuild_index

        info = rebuild_index()
        print(f"rebuilt runs/index.csv: {info['rows']} rows from "
              f"{info['live_dirs']} run directories, dropped {info['dropped']} "
              f"stale rows")

    rows = []
    # load_summaries backfills fields a summary may not carry (older runs recorded
    # graph=None), so the status table and the report agree about which runs exist
    # instead of one of them silently dropping rows.
    from flynum.analysis.report import load_summaries

    for s in load_summaries():
        if args.task and s.get("task") != args.task:
            continue
        rows.append(
            {
                "run_id": s.get("run_id"),
                "task": s.get("task"),
                "model": s.get("model"),
                "graph": s.get("graph"),
                "N": s.get("n_train"),
                "seed": s.get("model_seed"),
                "val": s.get("best_val_acc"),
                "A": s.get("test_acc_a"),
                "B": s.get("test_acc_b"),
                "C": s.get("test_acc_c"),
                "D": s.get("test_acc_d"),
                "add": s.get("addition_test_accuracy"),
                "unseen": s.get("unseen_pair_accuracy"),
                "status": s.get("status"),
                "min": round((s.get("wall_seconds_total") or 0) / 60, 1),
            }
        )

    if not rows:
        print("no completed runs found")
        return 1
    df = pd.DataFrame(rows)
    df = df.sort_values(args.sort)
    if args.last:
        df = df.tail(args.last)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"total runs: {len(df)}")
    if "task" in df:
        print(df.groupby(["task", "model", "graph"]).size().to_string())
    # how many runs are still in flight (directories without summary.json)
    with_metrics = {p.parent.name for p in paths.RUNS.glob("*/metrics.jsonl")}
    finished = {p.parent.name for p in paths.RUNS.glob("*/summary.json")}
    in_flight = sorted(with_metrics - finished)
    if in_flight:
        print(f"\nin flight ({len(in_flight)}): {', '.join(in_flight[:8])}"
              + (" ..." if len(in_flight) > 8 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
