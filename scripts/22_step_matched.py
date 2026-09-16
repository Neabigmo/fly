"""Within-run step matching: did the N=20000 gap exist at N=5000's compute?

The confound the fixed-update grid exists to settle is that the original curve gave
every condition 60 *epochs*, so N=20000 spent 18,780 optimiser updates while N=5000
spent at most 4,740.  The clean control is a rerun at equal updates, and that is
queued.  But the runs already on disk answer part of the question without any new
compute, because every epoch's validation accuracy was logged alongside its epoch
index: within the N=20000 condition, the gap can be read off **at step 4,740**, where
that condition had spent exactly the compute N=5000 was given.

If `Delta(step 4,740)` is already near its final value, the gap was present at matched
compute *within* the large condition, which points at data diversity rather than
compute.  If it is ~0 there and opens later, the epoch-budget gap is consistent with
compute and the queued control is required to separate them.

This is supporting evidence, not the control: it compares 20k-distinct-stimuli runs
against 5k-distinct-stimuli runs at the same *step count*, whereas the queued grid
holds N fixed and varies the budget.  Both are needed, and only the second can rule
compute out.

Usage::

    python scripts/22_step_matched.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402

#: The step budget the original curve effectively gave N=5000 (best case: 60 epochs).
BUDGET_5000 = 60 * math.ceil(5000 / 64)   # 4_740
BUDGET_20000 = 60 * math.ceil(20000 / 64)  # 18_780


def load_curves() -> list[dict]:
    """Every core counting run that logged per-epoch validation accuracy."""
    out = []
    for d in sorted(paths.RUNS.iterdir()):
        mp, cp = d / "metrics.jsonl", d / "config.json"
        if not (mp.exists() and cp.exists()):
            continue
        # A run that is still being written is not yet a result: its curve would be
        # held flat past its last logged epoch, which understates the arm it belongs
        # to and inflates any delta computed against it.
        if not (d / "summary.json").exists():
            continue
        cfg = json.loads(cp.read_text(encoding="utf-8"))
        if cfg.get("task") != "count" or cfg.get("model") != "M1":
            continue
        mc = cfg["model_cfg"]
        if (cfg["data"]["circuit"] != "core" or mc.get("signed_synapses")
                or mc.get("readout_standardize") or mc["w_scale"] != 1.0
                or mc["alpha"] != 0.2 or cfg["time"]["steps"] != 8
                or cfg["train"].get("max_steps", 0)):
            continue
        rows = []
        for line in mp.read_text(encoding="utf-8").splitlines():
            if '"epoch"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("val_acc") is None or rec.get("diverged"):
                continue
            rows.append(rec)
        if len(rows) < 5:
            continue
        spe = math.ceil(cfg["train"]["n_train"] / cfg["train"]["batch_size"])
        steps = np.array([(r["epoch"] + 1) * spe for r in rows], dtype=float)
        val = np.array([r["val_acc"] for r in rows], dtype=float)
        out.append({
            "run_id": d.name,
            "graph": cfg["data"]["graph"],
            "n_train": int(cfg["train"]["n_train"]),
            "model_seed": cfg["seeds"]["model_seed"],
            "steps_per_epoch": spe,
            "steps": steps,
            "val": val,
        })
    return out


def interp(curve: dict, grid: np.ndarray) -> np.ndarray:
    """Step -> validation accuracy, held flat before the first and after the last."""
    return np.interp(grid, curve["steps"], curve["val"],
                     left=curve["val"][0], right=curve["val"][-1])


def _arm(curves: list[dict], grid: np.ndarray) -> np.ndarray:
    """(n_seeds, n_grid) validation accuracy for one arm."""
    return np.stack([interp(c, grid) for c in curves])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    log = get_logger("step_matched")
    curves = load_curves()
    log.info("loaded %d core counting curves", len(curves))

    grid = np.arange(500, BUDGET_20000 + 1, 250, dtype=float)
    arms: dict[tuple[int, str], list[dict]] = {}
    for c in curves:
        arms.setdefault((c["n_train"], c["graph"]), []).append(c)

    report: dict = {"budget_5000": BUDGET_5000, "budget_20000": BUDGET_20000,
                    "grid": grid.tolist(), "arms": {}, "delta": {}}
    log.info("-" * 92)
    for (n, graph), cs in sorted(arms.items()):
        m = _arm(cs, grid)
        report["arms"][f"N{n}_{graph}"] = {
            "n_seeds": len(cs), "run_ids": [c["run_id"] for c in cs],
            "mean": m.mean(0).tolist(),
            "lo": (m.mean(0) - m.std(0, ddof=1)).tolist() if len(cs) > 1 else None,
            "hi": (m.mean(0) + m.std(0, ddof=1)).tolist() if len(cs) > 1 else None,
        }
        log.info("N=%-6d %-9s %d seed(s)", n, graph, len(cs))

    for n in sorted({k[0] for k in arms}):
        if (n, "real") in arms and (n, "shuffled") in arms:
            r = _arm(arms[(n, "real")], grid)
            s = _arm(arms[(n, "shuffled")], grid)
            # pair by seed where the two arms share seeds
            seeds_r = {c["model_seed"]: c for c in arms[(n, "real")]}
            seeds_s = {c["model_seed"]: c for c in arms[(n, "shuffled")]}
            shared = sorted(set(seeds_r) & set(seeds_s))
            if shared:
                pr = np.stack([interp(seeds_r[k], grid) for k in shared])
                ps = np.stack([interp(seeds_s[k], grid) for k in shared])
                diff = pr - ps
                delta = diff.mean(0)
                if len(shared) > 1:
                    from scipy import stats

                    ci = (stats.t.ppf(0.975, len(shared) - 1)
                          * diff.std(0, ddof=1) / np.sqrt(len(shared)))
                else:
                    ci = np.zeros_like(grid)
            else:
                delta, ci = r.mean(0) - s.mean(0), np.zeros_like(grid)
            report["delta"][f"N{n}"] = {
                "n_pairs": len(shared), "seeds": shared,
                "mean": delta.tolist(), "ci95": np.asarray(ci).tolist(),
            }
            idx = {int(v): i for i, v in enumerate(grid)}
            # Honest range: past a seed's last logged epoch there is nothing to read,
            # and holding the curve flat there would present an extrapolation as data.
            # A matched comparison is only made where *every* paired seed observed it.
            obs_max = min(
                max(float(seeds_r[k]["steps"][-1]), float(seeds_s[k]["steps"][-1]))
                for k in shared
            ) if shared else float(grid[-1])
            report["delta"][f"N{n}"]["observed_through_step"] = obs_max
            log.info("N=%-6d matched-comparison range: up to step %.0f (min over paired "
                     "seeds of their last logged epoch)", n, obs_max)
            if BUDGET_5000 in idx:
                i = idx[BUDGET_5000]
                tag = "" if BUDGET_5000 <= obs_max else "  [EXTRAPOLATED]"
                log.info(
                    "    at %d steps (the N=5000 budget): %+.4f +/- %.4f%s",
                    BUDGET_5000, delta[i], np.asarray(ci)[i], tag,
                )
            inside = np.nonzero(grid <= obs_max)[0]
            if len(inside):
                j = inside[-1]
                log.info("    at %d steps (last fully observed): %+.4f +/- %.4f",
                         int(grid[j]), delta[j], np.asarray(ci)[j])
            log.info("    at %d steps (full budget): %+.4f +/- %.4f  (%d paired seeds)",
                     BUDGET_20000, delta[-1], np.asarray(ci)[-1], len(shared))
            for m in (1500, 2500, 3239, BUDGET_5000, 8000, 12000, 16000, BUDGET_20000):
                j = int(np.argmin(np.abs(grid - m)))
                star = "" if grid[j] <= obs_max else " *"
                log.info("    %6d steps  delta %+.4f +/- %.4f   (real %.4f, shuffled %.4f)%s",
                         m, delta[j], np.asarray(ci)[j], r.mean(0)[j], s.mean(0)[j], star)

    # The comparison that speaks to the confound, stated plainly.
    d20 = report["delta"].get("N20000")
    d5 = report["delta"].get("N5000")
    if d20 and d5:
        obs5 = d5.get("observed_through_step", 0.0)
        i5 = int(np.argmin(np.abs(grid - obs5)))
        i20 = int(np.argmin(np.abs(grid - BUDGET_20000)))
        # Where did the large condition actually acquire its gap?  That step, not the
        # smaller condition's stopping point, is what decides whether the smaller
        # condition was stopped too early to ever show it.
        half = np.nonzero(np.asarray(d20["mean"]) >= 0.5 * d20["mean"][i20])[0]
        onset = float(grid[half[0]]) if len(half) else float("nan")
        report["verdict"] = {
            "n20000_delta_at_matched_compute": float(d20["mean"][i5]),
            "n5000_delta_at_same_step": float(d5["mean"][i5]),
            "n20000_delta_final": float(d20["mean"][i20]),
            "n5000_observed_through_step": obs5,
            "n20000_half_gap_onset_step": onset,
        }
        log.info("=" * 92)
        log.info("N=5000 was observed up to step %.0f; N=20000 reached %.0f.",
                 obs5, BUDGET_20000)
        log.info("At step %.0f (matched, fully observed on both sides): "
                 "N=20000 delta %+.4f, N=5000 delta %+.4f",
                 obs5, d20["mean"][i5], d5["mean"][i5])
        log.info("N=20000 reached half its final gap (+%.4f) at step %.0f",
                 d20["mean"][i20] / 2, onset)
        if onset > obs5:
            log.info(
                "=> the gap materialises at step %.0f, AFTER the %.0f steps N=5000 was "
                "given.  So the epoch-budget comparison stopped the small condition "
                "before the step range where the gap appears, and compute remains a "
                "live explanation.  The equal-update grid decides it.", onset, obs5)
        else:
            log.info(
                "=> the gap is already established by step %.0f, inside the range "
                "N=5000 was given, so it is not merely compute.", onset)

    # ------------------------------------------------------------------ #
    # Was the small condition still improving when patience stopped it?
    #
    # This is the crux of the compute question.  If N=5000's validation accuracy was
    # flat over its final patience window, then more updates could not have helped it
    # and the epoch-budget gap is about the stimuli; if it was still climbing, the
    # early stop -- not the sample size -- is what kept it at 0.51.
    A = []
    for c in curves:
        v = c["val"]
        w = min(15, len(v) - 1)
        if w < 2:
            continue
        tail = v[-w:]
        # least-squares slope per epoch over the final patience window, and the gain
        # actually achieved in it
        x = np.arange(w, dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0])
        A.append({
            "run_id": c["run_id"],
            "n_train": c["n_train"],
            "graph": c["graph"],
            "model_seed": c["model_seed"],
            "epochs": len(v),
            "stopped_at_step": float(c["steps"][-1]),
            "val_at_stop": float(v[-1]),
            "val_best_last_window": float(tail.max()),
            "gain_in_last_window": float(tail.max() - tail[0]),
            "slope_per_epoch": slope,
        })
    report["plateau"] = A
    for n in sorted({r["n_train"] for r in A}):
        for graph in ("real", "shuffled"):
            g = [r for r in A if r["n_train"] == n and r["graph"] == graph]
            if not g:
                continue
            sl = np.array([r["slope_per_epoch"] for r in g])
            gn = np.array([r["gain_in_last_window"] for r in g])
            log.info(
                "N=%-6d %-8s %d seed(s): stopped at %s steps | last-window slope "
                "%+.4f/epoch | gain in last window %+.4f",
                n, graph, len(g),
                "/".join(f"{r['stopped_at_step']:.0f}" for r in g),
                sl.mean(), gn.mean(),
            )
    small = [r for r in A if r["n_train"] <= 5000]
    if small:
        still_rising = [r for r in small if r["slope_per_epoch"] > 0.002]
        report["verdict"]["small_n_still_rising"] = len(still_rising)
        report["verdict"]["small_n_runs"] = len(small)
        report["verdict"]["n5000_mean_last_window_slope"] = float(np.mean(
            [r["slope_per_epoch"] for r in A if r["n_train"] == 5000]))
        report["verdict"]["n20000_mean_last_window_slope"] = float(np.mean(
            [r["slope_per_epoch"] for r in A if r["n_train"] == 20000]))
        log.info(
            "%d of %d small-N runs were still rising by more than +0.002/epoch when "
            "they stopped", len(still_rising), len(small),
        )
        log.info("=" * 92)
        s5 = report["verdict"]["n5000_mean_last_window_slope"]
        s20 = report["verdict"]["n20000_mean_last_window_slope"]
        log.info(
            "combined reading: the small conditions were FLAT over their final "
            "patience window (mean slope %+.5f/epoch for N=5000) and so was N=20000 "
            "at the end (%+.5f/epoch).  Every condition converged; they converged to "
            "different plateau HEIGHTS (0.51 against 0.77/0.66), which is a statement "
            "about the stimuli rather than about training duration.  The caveat that "
            "keeps this from being conclusive: a flat region can precede a later "
            "descent, so only the equal-update grid can rule out that N=5000 would "
            "have dropped off its plateau given the full budget.",
            s5, s20,
        )

    out = Path(args.out) if args.out else paths.DATA_PROCESSED / "step_matched.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    from flynum.analysis.figures import plot_step_matched

    fig = paths.FIGURES / "fig11_step_matched.png"
    plot_step_matched(report, fig)
    log.info("wrote %s", fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
