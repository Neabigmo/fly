"""Phase I results: Line A capability, Line B emergence, and the internal record.

    python scripts/28_phase1_analysis.py                 # everything under runs/
    python scripts/28_phase1_analysis.py --tag 10

Reports are written to ``reports/phase1_report.md``, numbers to
``data/processed/phase1_summary.json`` and figures to ``figures/fig13..15``.  The script is
written to be honest about absence: a cell that has not run is listed as not run, a cell
whose curves never crossed a criterion says so, and a Line B verdict that the frozen
criteria reject is printed with the reason rather than quietly reported as accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.analysis import figures  # noqa: E402
from flynum.phase1 import emergence, spec  # noqa: E402

ORDER = [t.run for t in (spec.FOUNDATION + spec.LINE_A + spec.LINE_B)]


# --------------------------------------------------------------------------- #
def load_cells(root: Path, tag: str) -> dict[str, dict]:
    """Every completed Phase I cell, keyed by cell name."""
    out: dict[str, dict] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or (tag and not d.name.startswith(f"{tag}_")):
            continue
        cfg_path = d / "config.json"
        summary_path = d / "full_summary.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        p1 = cfg.get("phase1") or {}
        if not p1:
            continue
        history = []
        if summary_path.exists():
            history = json.loads(summary_path.read_text(encoding="utf-8")).get("history", [])
        elif (d / "metrics.jsonl").exists():
            for line in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    history.append(json.loads(line))
        if not history:
            continue
        cell = p1.get("cell", d.name)
        # a re-run with the same cell name wins if it got further
        if cell in out and out[cell]["history"][-1]["updates"] >= history[-1]["updates"]:
            continue
        out[cell] = {"dir": d.name, "phase1": p1, "history": history,
                     "finished": summary_path.exists(),
                     "spec_fingerprint": cfg.get("phase1", {}).get("spec", {})}
    return out


def _num(v, nd=4):
    return "  --  " if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def _final(history: list[dict], key: str):
    return history[-1].get(key)


def line_a_table(cells: dict[str, dict]) -> dict[str, dict]:
    """Final accuracy, area under the curve, and updates to each criterion."""
    out = {}
    for name in ORDER:
        c = cells.get(name)
        if not c or c["phase1"].get("line") not in ("A", "F"):
            continue
        row = emergence.summarise_cell(c["history"], "A")
        row["task"] = c["phase1"]["task"]
        row["brain"] = c["phase1"]["brain"]
        row["budget"] = c["phase1"]["budget"]
        row["finished"] = c["finished"]
        for key in ("seen_acc", "taught_area_acc", "taught_env_acc",
                    "train_acc", "probe_n", "probe_a", "probe_b", "probe_sum",
                    "probe_answer", "probe_partial", "delta_abs_mean", "activity_dim"):
            row[key] = _final(c["history"], key)
        out[name] = row
    return out


def line_b_verdicts(cells: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for name in ORDER:
        c = cells.get(name)
        if not c or c["phase1"].get("line") != "B":
            continue
        v = emergence.detect(c["history"]).as_dict()
        v["task"] = c["phase1"]["task"]
        v["brain"] = c["phase1"]["brain"]
        v["budget"] = c["phase1"]["budget"]
        v["finished"] = c["finished"]
        v["n_taught"] = len(c["phase1"].get("teach") or [])
        v["n_holdout"] = len(c["phase1"].get("holdout") or [])
        for key in ("hold_acc", "holdout_area_acc", "holdout_env_acc", "unsup_acc",
                    "train_acc", "hold_ce", "hold_p_correct", "hold_margin"):
            v[key] = _final(c["history"], key)
        out[name] = v
    return out


def write_report(cells, a_rows, b_rows, out_path: Path) -> str:
    L = []
    L.append("# Phase I — 一个果蝇 connectome 能被教育到多复杂的数学\n")
    L.append(f"- spec fingerprint: `{spec.fingerprint()}`")
    L.append(f"- 教学条件: {spec.TRAIN_MODES} 混合（每个线索只在半个数据里有效），"
             f"评估条件 {spec.EVAL_MODES}")
    L.append(f"- 动力: {spec.DYNAMICS['circuit']} / signed={spec.DYNAMICS['signed_synapses']}"
             f" / w_scale={spec.DYNAMICS['w_scale']} / alpha={spec.DYNAMICS['alpha']}")
    L.append(f"- optimizer: {spec.OPTIM['optimizer']}, {spec.OPTIM['schedule']} lr "
             f"{spec.OPTIM['lr_gain']:g}, batch {spec.OPTIM['batch_size']}, "
             f"grad clip {spec.OPTIM['grad_clip']:g}, gain penalty "
             f"{spec.OPTIM['lambda_gain']:g}")
    L.append("- 比较单位: **optimiser updates**（不是 epoch）；探针固定在同一批 update 数")
    L.append("- 单 seed；无 early stopping；不用留出集选 checkpoint\n")

    L.append("## Line A — 教学能到达的复杂度\n")
    if not a_rows:
        L.append("_尚未有完成的 Line A cell。_\n")
    else:
        L.append("| cell | task | brain | budget | train | **seen** | area B | env C | "
                 "AUC(seen) | →0.50 | →0.80 | →0.95 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, r in a_rows.items():
            L.append("| {} | {} | {} | {} | {} | **{}** | {} | {} | {} | {} | {} | {} |".format(
                name, r["task"], r["brain"], r["budget"], _num(r["final_train_acc"], 3),
                _num(r["final_seen_acc"], 3), _num(r.get("taught_area_acc"), 3),
                _num(r.get("taught_env_acc"), 3), _num(r["auc_seen_acc"], 3),
                r["updates_to_50"] or "--", r["updates_to_80"] or "--",
                r["updates_to_95"] or "--"))
        L.append("")
        L.append("`seen` 是教学内容上的准确率（chance: count 0.143 / add 0.077 / "
                 "addsub 0.111 / two_step 0.143）；`area B`、`env C` 是同一批 item 在"
                 "另外两个条件下的读数，C 从未参与教学。")
        L.append("")

    L.append("## Line B — 长期训练会不会从记忆跃迁到规则\n")
    if not b_rows:
        L.append("_尚未有完成的 Line B cell。_\n")
    else:
        L.append("| cell | task | budget | T_mem | T_grok | 延迟倍数 | train | hold | "
                 "unsupported | 连续指标同步 | 判定 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, v in b_rows.items():
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | **{}** |".format(
                name, v["task"], v["budget"], v["t_mem"] or "--", v["t_grok"] or "--",
                _num(v["delay_factor"], 2), _num(v["final_train_acc"], 3),
                _num(v["final_hold_acc"], 3), _num(v.get("unsup_acc"), 3),
                v["continuous_confirms"], v["verdict"]))
        L.append("")
        for name, v in b_rows.items():
            if v["notes"]:
                L.append(f"- **{name}**: " + "; ".join(v["notes"]))
        L.append("")
        L.append("判定规则在跑之前就冻结在 `flynum/phase1/spec.py`：T_mem = 训练准确率"
                 f"连续 {spec.CRITERIA.sustain} 个探针 >{spec.CRITERIA.mem_acc}，"
                 f"T_grok = 留出准确率连续 {spec.CRITERIA.sustain} 个探针 "
                 f">{spec.CRITERIA.grok_acc}，"
                 f"延迟要求 T_grok > {spec.CRITERIA.delay_factor:g}×T_mem，"
                 "并且连续指标必须同步移动（否则记为阈值假象）。")
        L.append("")

    L.append("## 内部记录（行为变化是否伴随内部重组）\n")
    L.append("| cell | δ mean | activity dim | 探针 |")
    L.append("|---|---|---|---|")
    for name, c in cells.items():
        last = c["history"][-1]
        probes = " ".join(f"{k[6:]}={last[k]:.3f}" for k in sorted(last)
                          if k.startswith("probe_") and isinstance(last[k], float)
                          and last[k] == last[k])
        L.append("| {} | {} | {} | {} |".format(
            name, _num(last.get("delta_abs_mean"), 4),
            _num(last.get("activity_dim"), 1), probes or "--"))
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="10")
    ap.add_argument("--runs", default="")
    ap.add_argument("--figures", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.runs) if args.runs else paths.RUNS
    cells = load_cells(root, args.tag)
    if not cells:
        print(f"no Phase I cells with a config under {root} (tag {args.tag!r})")
        return 1
    a_rows, b_rows = line_a_table(cells), line_b_verdicts(cells)

    report = write_report(cells, a_rows, b_rows, paths.REPORTS / "phase1_report.md")
    (paths.REPORTS / "phase1_report.md").write_text(report, encoding="utf-8")

    payload = {
        "spec_fingerprint": spec.fingerprint(),
        "cells": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                  for k, v in cells.items()},
        "line_a": a_rows, "line_b": b_rows,
    }
    (paths.DATA_PROCESSED / "phase1_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if args.figures:
        # the figure helpers take one flat record per cell: name, task, history, final
        font = {name: {"task": c["phase1"].get("task", "?"),
                       "brain": c["phase1"].get("brain", "?"),
                       "history": c["history"], "final": c["history"][-1]}
                for name, c in cells.items()}
        if any(v["phase1"].get("line") == "A" for v in cells.values()):
            figures.plot_phase1_line_a(font, paths.FIGURES / "fig13_phase1_line_a.png")
        if b_rows:
            figures.plot_phase1_emergence(font, paths.FIGURES / "fig14_phase1_emergence.png",
                                          b_rows)
        figures.plot_phase1_internal(font, paths.FIGURES / "fig15_phase1_internal.png")

    print(f"{len(cells)} cell(s) loaded: {', '.join(sorted(cells, key=ORDER.index))}")
    for name in sorted(a_rows, key=ORDER.index):
        r = a_rows[name]
        print("  A {:<7s} {:<9s} seen {:.3f} (B {:.3f} C {:.3f}) auc {:.3f}".format(
            name, r["task"], r["final_seen_acc"] or float("nan"),
            r.get("taught_area_acc") or float("nan"),
            r.get("taught_env_acc") or float("nan"), r["auc_seen_acc"] or float("nan")))
    for name in sorted(b_rows, key=ORDER.index):
        v = b_rows[name]
        print("  B {:<7s} train {:.3f} hold {:.3f} -> {}".format(
            name, v["final_train_acc"] or float("nan"),
            v["final_hold_acc"] or float("nan"), v["verdict"]))
    print(f"wrote {paths.REPORTS / 'phase1_report.md'} and "
          f"{paths.DATA_PROCESSED / 'phase1_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
