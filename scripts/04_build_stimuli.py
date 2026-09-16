"""Stage 0f -- generate the stimulus sets and audit the anti-shortcut controls.

Writes the training pool, validation set and the four test conditions, then
reports for each condition how much a *purely non-numerical* classifier could
achieve.  If the total-ink baseline is above chance on condition B or C, the
control has failed and the script fails loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig, StimulusConfig  # noqa: E402
from flynum.data.manifest import Manifest  # noqa: E402
from flynum.logging_utils import RunContext  # noqa: E402
from flynum.stimuli.dots import make_add_dataset, make_count_dataset, shortcut_report  # noqa: E402
from flynum.train.data import stratified_subset  # noqa: E402


def main() -> int:
    ctx = RunContext("stage0_stimuli")
    log = ctx.log
    log.info("=" * 78)
    log.info("STIMULI + CONTROL AUDIT")
    log.info("=" * 78)

    cfg = ExperimentConfig()
    sc = cfg.stimulus
    report: dict = {"stimulus_config": sc.__dict__, "conditions": {}, "learning_curve_subsets": {}}

    ok = True
    for mode in ("A", "B", "C", "D"):
        ds = make_count_dataset(cfg.train.n_test, sc, seed=cfg.seeds["data_seed"] + 100 + ord(mode),
                                mode=mode, balanced=True)
        rep = shortcut_report(ds)
        rep["blob_ok_fraction"] = float(ds["blob_ok"].mean())
        rep["mean_attempts"] = float(ds["attempts"].mean())
        rep["ink_cv"] = float(ds["ink"].std() / ds["ink"].mean())
        rep["spread_cv"] = float(ds["spread"].std() / ds["spread"].mean())
        report["conditions"][mode] = rep
        log.info(
            "condition %s | chance %.3f | ink-baseline %.3f | spread-baseline %.3f | "
            "ink CV %.3f | blobs ok %.4f",
            mode, rep["chance"], rep["ink_centroid_accuracy"],
            rep["spread_centroid_accuracy"], rep["ink_cv"], rep["blob_ok_fraction"],
        )
        if rep["blob_ok_fraction"] < 1.0:
            log.error("condition %s: %.2f%% of images have the wrong blob count",
                      mode, 100 * (1 - rep["blob_ok_fraction"]))
            ok = False
        # the controlled conditions must not leak the answer through total ink
        if mode in ("B", "C") and rep["ink_centroid_accuracy"] > 0.45:
            log.error("condition %s: total ink predicts the count (%.3f) - control FAILED",
                      mode, rep["ink_centroid_accuracy"])
            ok = False
        # ...nor through the spatial envelope
        if mode in ("C", "D") and rep["spread_centroid_accuracy"] > 0.60:
            log.error("condition %s: dot spacing predicts the count (%.3f) - control FAILED",
                      mode, rep["spread_centroid_accuracy"])
            ok = False

    # condition C envelope check: spread must be matched for n >= 2
    ds_c = make_count_dataset(800, sc, seed=7, mode="C")
    spreads = [ds_c["spread"][ds_c["ns"] == n].mean() for n in range(2, sc.n_max + 1)]
    report["condition_C_spread_mean"] = float(np.mean(spreads))
    report["condition_C_spread_cv"] = float(np.std(spreads) / np.mean(spreads))
    log.info("condition C spread CV (n>=2): %.4f", report["condition_C_spread_cv"])
    if report["condition_C_spread_cv"] > 0.10:
        log.error("condition C does not equalise the spatial envelope")
        ok = False

    # ---- area control must stay inside the trained radius range --------- #
    # Otherwise the "numerosity" test silently becomes a size-generalisation
    # test, which is a different (and much easier to fail) question.
    radii_by_n = {}
    for mode in ("B", "C", "D"):
        ds = make_count_dataset(500, sc, seed=11, mode=mode)
        r = {int(n): (float(ds["radius_mean"][ds["ns"] == n].min()),
                      float(ds["radius_mean"][ds["ns"] == n].max()))
             for n in range(sc.n_min, sc.n_max + 1)}
        radii_by_n[mode] = r
        lo = min(v[0] for v in r.values())
        hi = max(v[1] for v in r.values())
        report.setdefault("radius_ranges", {})[mode] = {"min": lo, "max": hi}
        log.info("condition %s radii span [%.2f, %.2f] (trained range [%.2f, %.2f])",
                 mode, lo, hi, sc.radius_min, sc.radius_max)
        if lo < sc.radius_min - 1e-6 or hi > sc.radius_max + 1e-6:
            log.error(
                "condition %s uses dot radii outside the trained range - the test "
                "would measure size generalisation, not numerosity", mode,
            )
            ok = False

    # ---- learning-curve subsetting ------------------------------------ #
    pool = make_count_dataset(cfg.train.n_train, sc, seed=cfg.seeds["data_seed"], mode="A")
    for n in (100, 500, 1000, 5000, 20000):
        idx = stratified_subset(pool["labels"], n)
        counts = np.bincount(pool["labels"][idx], minlength=sc.n_max - sc.n_min + 1)
        report["learning_curve_subsets"][str(n)] = {
            "n": int(len(idx)),
            "per_class": counts.tolist(),
        }
        log.info("  subset N=%6d -> %d items, per-class %s", n, len(idx), counts.tolist())
        if not np.all(counts == n // counts.size):
            log.error("subset N=%d is not class-balanced", n)
            ok = False
    # nestedness: subset(100) must be contained in subset(500)
    small = set(stratified_subset(pool["labels"], 100).tolist())
    large = set(stratified_subset(pool["labels"], 500).tolist())
    report["subsets_nested"] = small <= large
    if not small <= large:
        log.error("learning-curve subsets are not nested")
        ok = False

    # ---- addition task ------------------------------------------------ #
    tr = make_add_dataset(sc, seed=cfg.seeds["data_seed"] + 7, reps_per_pair=2, split="train")
    te = make_add_dataset(sc, seed=cfg.seeds["data_seed"] + 7, reps_per_pair=2, split="test")
    report["addition"] = {
        "train_pairs": sorted(set(zip(tr["a"].tolist(), tr["b"].tolist()))),
        "holdout_pairs": sorted(set(zip(te["a"].tolist(), te["b"].tolist()))),
        "train_sums": sorted(set((tr["a"] + tr["b"]).tolist())),
        "unseen_sums": sorted(set((te["a"] + te["b"]).tolist())),
        "n_classes": tr["n_classes"],
    }
    log.info("addition: %d train pairs, holdout %s, sums %s",
             len(report["addition"]["train_pairs"]),
             report["addition"]["holdout_pairs"], report["addition"]["train_sums"])
    if not set((te["a"] + te["b"]).tolist()) <= set((tr["a"] + tr["b"]).tolist()):
        log.error("a held-out pair creates a sum that was never trainable")
        ok = False

    out = paths.DATA_PROCESSED / "stimulus_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    man = Manifest()
    man.add("stimulus_report", out, conditions=list(report["conditions"]))
    man.save()
    log.info("wrote %s", out)

    ctx.finish("ok" if ok else "control_failed", passed=ok)
    log.info("STIMULI + CONTROL AUDIT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
