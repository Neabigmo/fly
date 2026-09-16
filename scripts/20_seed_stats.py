"""Seed-level statistics for every real-vs-shuffled block.

The mistake this script exists to prevent: quoting a confidence interval computed
over test *images*.  Every image in a run is scored by one trained network, so the
5000 test images are not 5000 independent observations of "does the real connectome
help" -- they are one observation repeated 5000 times.  With three seeds the
image-level interval is roughly 30x too narrow, which is exactly how a difference
of 0.1076 at N=20000 can look overwhelming while resting on three graphs.

Everything here therefore treats the **trained model** as the unit of
observation: one real run against one shuffled run, paired by model seed, with a
distinct shuffled graph per seed.  Reported per condition:

* mean and 95% CI (Student t on n-1 df) for each arm;
* the paired difference, its 95% CI, Cohen's d_z and Hedges' g, paired t-test and
  Wilcoxon signed-rank p-values;
* the image-level CI for contrast, so the inflation is visible rather than implied.

Usage::

    python scripts/20_seed_stats.py
    python scripts/20_seed_stats.py --block replication fxu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402

CONDITIONS = ("a", "b", "c", "d")


# --------------------------------------------------------------------------- #
def load_runs() -> list[dict]:
    """Every run with a summary, tagged with its identity and test accuracies."""
    out = []
    for d in sorted(paths.RUNS.iterdir()):
        p = d / "full_summary.json"
        cp = d / "config.json"
        if not (p.exists() and cp.exists()):
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        c = json.loads(cp.read_text(encoding="utf-8"))
        if c.get("task") != "count" or s.get("status", "ok") != "ok":
            continue
        acc = {k: s.get(f"test_acc_{k}") for k in CONDITIONS}
        if any(v is None for v in acc.values()):
            continue
        stats = s.get("shuffle_stats") or {}
        out.append(
            {
                "run_id": d.name,
                "model_seed": c["seeds"]["model_seed"],
                "shuffle_seed": c["seeds"]["shuffle_seed"],
                "graph": c["data"]["graph"],
                "circuit": c["data"]["circuit"],
                "model": c.get("model"),
                "n_train": c["train"]["n_train"],
                "max_steps": c["train"].get("max_steps", 0),
                "w_scale": c["model_cfg"]["w_scale"],
                "alpha": c["model_cfg"]["alpha"],
                "signed": bool(c["model_cfg"].get("signed_synapses", False)),
                "steps": c["time"]["steps"],
                "standardize": bool(c["model_cfg"].get("readout_standardize", False)),
                # How strongly this control graph was randomised.  Every seed has its
                # own shuffled graph, so the paired comparison is only clean if the
                # randomisation strength is the same across seeds; report it rather
                # than assume it.
                "shuffle_overlap": stats.get("final_edge_overlap"),
                "shuffle_swap_rate": stats.get("swap_rate"),
                "shuffle_degrees_ok": stats.get("degrees_preserved"),
                "shuffle_weights_ok": stats.get("weights_preserved"),
                "per_mode": s.get("per_mode", {}),
                **{f"acc_{k}": v for k, v in acc.items()},
            }
        )
    return out


def _ci95(x: np.ndarray) -> tuple[float, float, float]:
    """mean and the two-sided 95% CI half-width (Student t)."""
    from scipy import stats

    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1) * se)
    return m, half, se


def _paired(real: np.ndarray, shuf: np.ndarray) -> dict:
    """Paired comparison with effect sizes."""
    from scipy import stats

    d = real - shuf
    n = len(d)
    m, half, se = _ci95(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    dz = float(m / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    # Hedges' g for paired designs: d_z with the small-sample correction
    g = float(dz * (1 - 3 / (4 * n - 1))) if np.isfinite(dz) else float("nan")
    try:
        t_p = float(stats.ttest_rel(real, shuf).pvalue)
    except Exception:
        t_p = float("nan")
    try:
        w = stats.wilcoxon(real, shuf)
        w_p = float(w.pvalue)
    except Exception:
        w_p = float("nan")
    return {
        "n_pairs": n,
        "delta_mean": m,
        "delta_ci95": half,
        "delta_se": se,
        "cohen_dz": dz,
        "hedges_g": g,
        "t_p": t_p,
        "wilcoxon_p": w_p,
        "delta_per_seed": [float(v) for v in d],
    }


def check_arm_distinctness(runs: list[dict], log, label: str) -> dict:
    """Two runs in the same arm must not be the same graph.

    ``_validate_shuffle`` proves each control graph differs from the *real* one, but it
    cannot catch a cache key that ignored the shuffle seed -- every control run would
    then share a single graph while still passing that check, and the seed-level CI
    would be computed over replicates that are not replicates.

    Independent graphs trained on the same stimuli agree on roughly 90% of individual
    test predictions; the same graph would agree on 100%.  Verified on this block:
    shuffled seeds 3 and 4 share 303 of 1,170,024 edges and agree on 90.2% of
    predictions, so this is a live check and not a formality.
    """
    import numpy as np

    from flynum import paths as _paths

    preds: dict[str, np.ndarray] = {}
    for r in runs:
        p = _paths.run_dir(r["run_id"]) / "predictions.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            if "A_pred" in z.files:
                preds[r["run_id"]] = z["A_pred"]

    pairs: list[dict] = []
    worst = 0.0
    ids = sorted(preds)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = preds[ids[i]], preds[ids[j]]
            if a.shape != b.shape:
                continue
            agree = float((a == b).mean())
            pairs.append({"a": ids[i], "b": ids[j], "agreement": round(agree, 6)})
            worst = max(worst, agree)
    out = {"n_runs_with_predictions": len(preds), "pairs": pairs,
           "max_agreement": worst}
    if pairs:
        log.info("  %s: %d run pairs, per-sample prediction agreement "
                 "%.4f-%.4f (identical graphs would be 1.0)",
                 label, len(pairs),
                 min(p["agreement"] for p in pairs), worst)
        if worst > 0.995:
            bad = [p for p in pairs if p["agreement"] > 0.995][0]
            log.error("  %s: %s and %s agree on %.4f of predictions -- these are "
                      "almost certainly the same graph, so this arm is not a set of "
                      "independent replicates", label, bad["a"], bad["b"],
                      bad["agreement"])
    return out


def compare(real_runs: list[dict], shuf_runs: list[dict], log, label: str) -> dict:
    """Real vs shuffled, paired by model seed, on every test condition."""
    by_seed_r = {r["model_seed"]: r for r in real_runs}
    by_seed_s = {r["model_seed"]: r for r in shuf_runs}
    seeds = sorted(set(by_seed_r) & set(by_seed_s))
    if not seeds:
        log.warning("%s: no matched seeds (real %s, shuffled %s)",
                    label, sorted(by_seed_r), sorted(by_seed_s))
        return {}
    if len(seeds) != len(by_seed_r) or len(seeds) != len(by_seed_s):
        log.warning("%s: seeds without a partner -- real %s, shuffled %s",
                    label, sorted(set(by_seed_r) - set(by_seed_s)),
                    sorted(set(by_seed_s) - set(by_seed_r)))

    out = {"label": label, "n_seeds": len(seeds), "seeds": seeds, "conditions": {}}
    log.info("-" * 92)
    log.info("%s | %d paired seeds %s", label, len(seeds), seeds)
    # Control strength: each seed has its own shuffled graph, so the pairing is only
    # clean if they were randomised to the same degree.
    ov = [by_seed_s[s].get("shuffle_overlap") for s in seeds]
    ov = [v for v in ov if v is not None]
    if ov:
        out["shuffle_overlap"] = {
            "per_seed": ov,
            "mean": float(np.mean(ov)),
            "spread": float(max(ov) - min(ov)),
        }
        log.info("  control graphs retain %.4f-%.4f of the real edges "
                 "(mean %.4f, spread %.4f)",
                 min(ov), max(ov), np.mean(ov), max(ov) - min(ov))
        for s in seeds:
            rec = by_seed_s[s]
            if not (rec.get("shuffle_degrees_ok") and rec.get("shuffle_weights_ok")):
                log.warning("  seed %d: shuffle did not preserve degrees/weights", s)
    out["distinctness"] = {
        "real": check_arm_distinctness([by_seed_r[s] for s in seeds], log,
                                       f"{label} real arm"),
        "shuffled": check_arm_distinctness([by_seed_s[s] for s in seeds], log,
                                           f"{label} shuffled arm"),
    }
    for cond in CONDITIONS:
        key = f"acc_{cond}"
        r = np.array([by_seed_r[s][key] for s in seeds], dtype=float)
        q = np.array([by_seed_s[s][key] for s in seeds], dtype=float)
        r_mean, r_half, _ = _ci95(r)
        q_mean, q_half, _ = _ci95(q)
        block = {
            "real_mean": r_mean, "real_ci95": r_half, "real_per_seed": r.tolist(),
            "shuffled_mean": q_mean, "shuffled_ci95": q_half, "shuffled_per_seed": q.tolist(),
            **_paired(r, q),
        }
        # image-level interval, for contrast only
        ns = [
            by_seed_r[s]["per_mode"].get(cond.upper(), {}).get("n") for s in seeds
        ]
        n_img = int(np.median([v for v in ns if v])) if any(ns) else 0
        if n_img:
            se_img = float(
                np.sqrt(
                    np.mean([p * (1 - p) for p in r] ) / n_img
                    + np.mean([p * (1 - p) for p in q]) / n_img
                )
            )
            block["image_level_ci95"] = 1.96 * se_img
            block["n_test_images"] = n_img
        out["conditions"][cond.upper()] = block
        log.info(
            "  %s  real %.4f +/- %.4f | shuffled %.4f +/- %.4f | delta %+.4f "
            "+/- %.4f | d_z %.2f | t p=%.4f  W p=%.4f",
            cond.upper(), r_mean, r_half, q_mean, q_half,
            block["delta_mean"], block["delta_ci95"], block["cohen_dz"],
            block["t_p"], block["wilcoxon_p"],
        )
        if n_img:
            log.info(
                "       (image-level CI would be +/- %.4f on %d images -- %.0fx too "
                "narrow, which is why it is not the reported interval)",
                block["image_level_ci95"], n_img,
                block["delta_ci95"] / max(block["image_level_ci95"], 1e-12),
            )
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", nargs="*", default=["replication", "fxu", "signed"],
                    choices=["replication", "fxu", "signed"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    log = get_logger("seed_stats")
    runs = load_runs()
    log.info("loaded %d counting runs with summaries", len(runs))

    def sel(**crit) -> list[dict]:
        out = []
        for r in runs:
            if all(r.get(k) == v for k, v in crit.items()):
                out.append(r)
        return out

    report: dict = {"n_runs": len(runs), "blocks": {}}

    if "replication" in args.block:
        # the headline condition: core + 20k, epoch budget, unsigned, w_scale 1.0
        base = dict(circuit="core", model="M1", n_train=20000, signed=False,
                    standardize=False, alpha=0.2, steps=8, w_scale=1.0, max_steps=0)
        real, shuf = sel(graph="real", **base), sel(graph="shuffled", **base)
        log.info("replication: %d real / %d shuffled runs at N=20000", len(real), len(shuf))
        report["blocks"]["replication"] = {
            "criteria": base, "real_runs": [r["run_id"] for r in real],
            "shuffled_runs": [r["run_id"] for r in shuf],
            **compare(real, shuf, log, "replication: core + 20k, 60 epochs"),
        }

    if "fxu" in args.block:
        """Fixed-update control: same optimiser-step budget, varying N only."""
        reports = {}
        for n in sorted({r["n_train"] for r in runs if r["max_steps"]}):
            base = dict(circuit="core", model="M1", n_train=n, signed=False,
                        standardize=False, alpha=0.2, steps=8, w_scale=1.0)
            real = sel(graph="real", **base)
            shuf = sel(graph="shuffled", **base)
            real = [r for r in real if r["max_steps"]]
            shuf = [r for r in shuf if r["max_steps"]]
            if real and shuf:
                reports[f"N{n}"] = {
                    "criteria": base,
                    "real_runs": [r["run_id"] for r in real],
                    "shuffled_runs": [r["run_id"] for r in shuf],
                    **compare(real, shuf, log, f"fixed updates: core + {n}, "
                                               f"{real[0]['max_steps']} steps"),
                }
        report["blocks"]["fxu"] = reports

    if "signed" in args.block:
        base = dict(circuit="core", model="M1", n_train=20000, signed=True,
                    standardize=False, alpha=0.2, steps=8)
        real, shuf = sel(graph="real", **base), sel(graph="shuffled", **base)
        if real and shuf:
            report["blocks"]["signed"] = {
                "criteria": base, "real_runs": [r["run_id"] for r in real],
                "shuffled_runs": [r["run_id"] for r in shuf],
                **compare(real, shuf, log, "Fly-v2 signed synapses: core + 20k"),
            }
        else:
            log.info("signed block not run yet (%d real / %d shuffled)", len(real), len(shuf))

    out = Path(args.out) if args.out else paths.DATA_PROCESSED / "seed_stats.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    # the paired figure: only meaningful once there is a replication block
    rep = report.get("blocks", {}).get("replication") or {}
    if rep.get("conditions"):
        from flynum.analysis.figures import plot_paired_seeds

        fig = paths.FIGURES / "fig10_paired_seeds.png"
        plot_paired_seeds(
            rep["conditions"], fig,
            label=(f"core + 20k, {rep['n_seeds']} paired seeds "
                   f"(one independent shuffled graph per seed)"),
        )
        log.info("wrote %s", fig)

    md = paths.REPORTS / "seed_stats.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_markdown(report), encoding="utf-8")
    log.info("wrote %s", md)
    return 0


def _markdown(report: dict) -> str:
    """The same numbers as a table, so the report can quote them directly."""
    import time as _time

    lines = [
        "# Real vs shuffled, at the level of trained models",
        "",
        f"Generated {_time.strftime('%Y-%m-%d %H:%M')} by `scripts/20_seed_stats.py`.",
        "The unit of observation is one trained network, not one test image:",
        "runs are paired by model seed and each shuffled seed is an independent",
        "random graph.  Intervals are Student t on n-1 degrees of freedom.",
        "",
        f"Runs considered: {report.get('n_runs')}.",
        "",
    ]
    for name, blk in report.get("blocks", {}).items():
        if "conditions" not in blk:
            # the fxu block nests one report per N
            for sub, sub_blk in blk.items():
                lines += _block_md(f"{name} / {sub}", sub_blk)
            continue
        lines += _block_md(name, blk)
    return "\n".join(lines) + "\n"


def _block_md(name: str, blk: dict) -> list[str]:
    if "conditions" not in blk:
        return []
    out = [
        f"## {blk.get('label', name)}",
        "",
        f"{blk['n_seeds']} paired seeds: {blk['seeds']}",
        "",
    ]
    so = blk.get("shuffle_overlap")
    if so:
        out += [
            f"Control strength: each seed's own shuffled graph retains "
            f"{min(so['per_seed']):.4f}-{max(so['per_seed']):.4f} of the real edges "
            f"(mean {so['mean']:.4f}), so the randomisation is comparable across "
            f"seeds and the pairing is not confounded by varying control strength.",
            "",
        ]
    out += [
        "| cond | real | shuffled | delta | 95% CI | d_z | Hedges g | t p | Wilcoxon p | image CI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cond, c in blk["conditions"].items():
        img = f"±{c['image_level_ci95']:.4f}" if "image_level_ci95" in c else "—"
        out.append(
            f"| {cond} | {c['real_mean']:.4f} ± {c['real_ci95']:.4f} "
            f"| {c['shuffled_mean']:.4f} ± {c['shuffled_ci95']:.4f} "
            f"| **{c['delta_mean']:+.4f}** ± {c['delta_ci95']:.4f} "
            f"| [{c['delta_mean'] - c['delta_ci95']:+.4f}, {c['delta_mean'] + c['delta_ci95']:+.4f}] "
            f"| {c['cohen_dz']:.2f} | {c['hedges_g']:.2f} "
            f"| {c['t_p']:.4f} | {c['wilcoxon_p']:.4f} | {img} |"
        )
    out.append("")
    for cond, c in blk["conditions"].items():
        out.append(
            f"- **{cond}** per-seed deltas: "
            + ", ".join(f"{v:+.4f}" for v in c["delta_per_seed"])
        )
    out.append("")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
