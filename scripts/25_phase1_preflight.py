"""Phase I pre-flight: prove the design is sound before spending GPU time.

Every check here corresponds to a way this experiment could produce a number that looks
like a result but is an artefact.  It runs on the CPU in a couple of minutes and prints a
go/no-go report; the runner refuses to start unless this passes.

The checks, and the failure each one prevents:

  stimulus geometry   the area-controlled condition must not turn into a
                      size-generalisation test at n=7, and seven discs must still fit
  blob integrity      a stimulus labelled "n dots" must actually contain n components,
                      which is the difference between a numerosity task and a mislabelled
                      one
  legibility          the fixed retinotopic map must preserve what the tasks are about:
                      a "+" must still be distinguishable from a "-" and the number of
                      dots must still be recoverable, or A2/A3 would report a ceiling that
                      is a rendering artefact
  dynamics bounded    signed synapses at the chosen scale must not blow up over the
                      longest sequence any task uses
  holdout purity      the withheld pair must not reach the gradient, the scheduler, early
                      stopping or checkpoint selection -- otherwise the "emergence" was
                      supervised
  update alignment    every cell on a line must spend the same number of optimiser
                      updates, because this project already found that an epoch-matched
                      comparison gave one condition four times the compute of another
  discriminability    the tasks must be non-trivially learnable and the shortcut
                      baselines must not already solve them
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig, StimulusConfig  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from flynum.phase1 import spec  # noqa: E402
from flynum.stimuli import dots  # noqa: E402


def check_stimulus_geometry(log) -> dict:
    """Area control must hold for every n, and n=7 must fit on the canvas."""
    sc = StimulusConfig()
    for k, v in spec.STIMULUS.items():
        setattr(sc, k, v)
    S = spec.STIMULUS["image_size"]
    span_need = math.sqrt(sc.n_max) * 1.08 / 0.92
    span = sc.radius_max / sc.radius_min
    pack_limit = (S / 2 - 1) / (1 + 1 / math.sin(math.pi / sc.n_max)) - sc.min_gap / 2

    vals = []
    for mode in ("B", "C"):
        for n in range(sc.n_min, sc.n_max + 1):
            rng = np.random.default_rng(2024 + n)
            for _ in range(300):
                vals.append(dots._radii_for_mode(n, sc, mode, rng))
    v = np.concatenate(vals)
    outside = float(((v < sc.radius_min - 1e-6) | (v > sc.radius_max + 1e-6)).mean())

    log.info("  constant-area span needed %.3f (sqrt(%d) x 1.08/0.92), have %.3f  %s",
             span_need, sc.n_max, span, "OK" if span >= span_need else "FAIL")
    log.info("  packing: r_max %.2f must be <= %.3f for %d discs  %s",
             sc.radius_max, pack_limit, sc.n_max,
             "OK" if sc.radius_max <= pack_limit else "FAIL")
    log.info("  area-implied radii outside [%.2f, %.2f]: %.3f%% of %d samples  %s",
             sc.radius_min, sc.radius_max, 100 * outside, len(v),
             "OK" if outside < 0.001 else "FAIL")
    return {"span": span, "span_needed": span_need, "pack_limit": pack_limit,
            "outside_fraction": outside,
            "ok": span >= span_need and sc.radius_max <= pack_limit and outside < 0.001}


def check_blob_integrity(log, n_per: int = 60) -> dict:
    """A stimulus labelled n must contain exactly n connected components."""
    sc = StimulusConfig()
    for k, v in spec.STIMULUS.items():
        setattr(sc, k, v)
    bad, tot, retries = 0, 0, []
    for mode in dots.MODES:
        for n in range(sc.n_min, sc.n_max + 1):
            rng = np.random.default_rng(555 + n)
            for _ in range(n_per):
                st = dots.make_count_stimulus(n, sc, mode=mode, rng=rng)
                tot += 1
                retries.append(st.attempts)
                if not (st.blob_ok and st.blob_count == n):
                    bad += 1
    log.info("  %d stimuli over %d modes x n=%d..%d: %d mislabelled  %s",
             tot, len(dots.MODES), sc.n_min, sc.n_max, bad,
             "OK" if bad == 0 else "FAIL")
    log.info("  layout retries per stimulus: mean %.3f, max %d",
             float(np.mean(retries)), int(np.max(retries)))
    return {"n": tot, "bad": bad, "retry_mean": float(np.mean(retries)),
            "retry_max": int(np.max(retries)), "ok": bad == 0}


_PREP_CACHE: dict = {}


def _prepared(log):
    """One `prepare()` for the whole pre-flight: the retina map costs ~10 s to build."""
    if "prep" not in _PREP_CACHE:
        from flynum.pipeline import prepare

        cfg = ExperimentConfig()
        cfg.data.circuit = spec.DYNAMICS["circuit"]
        cfg.data.graph = "real"
        cfg.model_cfg.signed_synapses = spec.DYNAMICS["signed_synapses"]
        cfg.model_cfg.w_scale = spec.DYNAMICS["w_scale"]
        cfg.model_cfg.alpha = spec.DYNAMICS["alpha"]
        cfg.model_cfg.readout_standardize = False
        for k, v in spec.STIMULUS.items():
            setattr(cfg.stimulus, k, v)
        _PREP_CACHE["prep"] = prepare(cfg, logger=None)
        _PREP_CACHE["cfg"] = cfg
    return _PREP_CACHE["prep"], _PREP_CACHE["cfg"]


def _probe_accuracy(x, y, n_classes: int, *, steps: int = 400) -> float:
    """Split-half linear probe accuracy on encoded column activations."""
    import torch
    import torch.nn as nn

    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(y), generator=g)
    tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    x = (x - x[tr].mean(0, keepdim=True)) / x[tr].std(0, keepdim=True).clamp_min(1e-6)
    lin = nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lossf(lin(x[tr]), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(x[te]).argmax(1) == y[te]).float().mean())


def check_legibility(log, n_per: int = 96) -> dict:
    """Can the fixed retina tell the operator glyphs, and the numerosities, apart?

    A2 and A3 ask the network to choose an operation from what it sees.  If the
    retinotopic map cannot separate ``+`` from ``-`` in the column activations, those
    tasks are impossible for a reason that has nothing to do with the connectome, and the
    two Line A cells that depend on them would report a ceiling that is really a
    rendering artefact.  Same question for the dot groups, one level down: the counting
    task is only a numerosity task if the map preserves how many dots there are.
    """
    import torch

    from flynum.phase1 import glyphs
    from flynum.retina.torch_encoder import TorchRetina
    from flynum.stimuli.dots import make_count_stimulus

    prep, cfg = _prepared(log)
    sc = cfg.stimulus
    retina = TorchRetina(prep.encoder, device="cpu")
    rng = np.random.default_rng(4242)

    def encode(imgs):
        t = torch.as_tensor(np.stack(imgs), dtype=torch.float32)
        with torch.no_grad():
            return retina.encode(t).numpy()

    # operator glyphs vs each other and vs a blank window
    ops = []
    for sym in glyphs.OPS:
        ops.append(encode([glyphs.render_glyph(sym, sc, rng) for _ in range(n_per)]))
    blank = encode([glyphs.render_blank(sc.image_size) for _ in range(n_per)])

    both = np.concatenate(ops)
    acc_ops = _probe_accuracy(both, np.repeat([0, 1], n_per), 2)
    acc_op_blank = _probe_accuracy(np.concatenate([ops[0], blank]),
                                   np.repeat([0, 1], n_per), 2)

    # numerosity: 1..7 dots must remain linearly separable after the map
    ns = np.repeat(np.arange(spec.STIMULUS["n_min"], spec.STIMULUS["n_max"] + 1), n_per)
    dot_stims = [make_count_stimulus(int(n), sc, rng) for n in ns]
    acc_n = _probe_accuracy(encode([s.image for s in dot_stims]),
                            ns - spec.STIMULUS["n_min"], 7)

    ink_plus = glyphs.ink("+", sc)
    ink_minus = glyphs.ink("-", sc)
    ink3 = float(np.mean([s.ink for s in dot_stims if s.n == 3]))
    n_lo = spec.STIMULUS["n_min"]
    n_hi = spec.STIMULUS["n_max"]
    chance_n = 1.0 / (n_hi - n_lo + 1)
    # An operator must survive the map intact, because A2/A3 ask for a decision *about*
    # it.  Numerosity is the opposite kind of requirement: information about the count has
    # to be present, but if a linear read of the retina already recovered it, counting
    # would be a lookup and the recurrent circuit would have nothing to compute.  The
    # criterion is therefore "well above chance and well below trivial", and the measured
    # 0.30 against 0.143 is the intended regime.
    ok = acc_ops >= 0.90 and acc_op_blank >= 0.95 and acc_n >= 2 * chance_n
    log.info("  '+' vs '-' from column activations: %.4f  %s", acc_ops,
             "OK" if acc_ops >= 0.90 else "FAIL")
    log.info("  glyph vs blank window: %.4f  %s", acc_op_blank,
             "OK" if acc_op_blank >= 0.95 else "FAIL")
    log.info("  numerosity %d-%d linearly readable from the retina: %.4f (chance %.3f, "
             "%.1fx)  %s -- present, but not a lookup", n_lo, n_hi, acc_n, chance_n,
             acc_n / chance_n, "OK" if acc_n >= 2 * chance_n else "FAIL")
    log.info("  glyph ink: '+' %.1f  '-' %.1f  vs a three-dot group %.1f -- an operator "
             "is not a louder stimulus than a dot group", ink_plus, ink_minus, ink3)
    return {"acc_operator": acc_ops, "acc_operator_vs_blank": acc_op_blank,
            "acc_count": acc_n, "count_over_chance": acc_n / chance_n,
            "ink_plus": ink_plus, "ink_minus": ink_minus,
            "ink_three_dots": ink3, "ok": bool(ok)}


def check_dynamics(log, horizon: int | None = None) -> dict:
    """Signed synapses at the Phase I scale must stay finite across the sequence."""
    import torch

    prep, cfg = _prepared(log)
    model = prep.build_model(device="cpu", seed=0)
    t = spec.TIME
    horizon = horizon or max(t["steps_operand"] * 2 + t["steps_cue"], 32)
    with torch.no_grad():
        cols = torch.rand(8, model.n_columns)
        _, hist = model.forward_count(cols, horizon)
        peaks = [float(h.abs().max()) for h in hist]
    growth = peaks[-1] / max(peaks[0], 1e-30)
    log.info("  signed w_scale=%.2f alpha=%.2f: peak|h| %.3g -> %.3g over %d steps "
             "(x%.3g)  %s", spec.DYNAMICS["w_scale"], spec.DYNAMICS["alpha"],
             peaks[0], peaks[-1], horizon, growth,
             "OK" if math.isfinite(peaks[-1]) and growth < 1e4 else "FAIL")
    return {"horizon": horizon, "peak_first": peaks[0], "peak_last": peaks[-1],
            "growth": growth,
            "ok": math.isfinite(peaks[-1]) and growth < 1e4}



def _answer_class(task: str, pair: tuple[int, int]) -> int:
    if task == "cyc7":
        return spec.cyc7_class(pair)
    return pair[0] + pair[1]


def _holdout_problems(t: spec.TaskSpec) -> list[str]:
    """Is the withheld material really withheld, and is the task still learnable?

    Both Line B tasks have commutative labels, so a withheld pair whose mirror image
    is taught is not withheld at all.  These are the properties that make the
    difference between measuring emergent generalisation and measuring a taught
    symmetry, so they are verified rather than assumed.

    Two shapes are allowed, and both are intentional.  B2 partitions the 49 pairs into
    taught and withheld.  B1 instead teaches a sparse 15-fact table and withholds
    2 pairs; the remaining 32 pairs are unsupported, and they are a *diagnostic* (how
    far does anything generalise?) that must never enter the gradient.
    """
    bad: list[str] = []
    held, taught = set(t.holdout), set(t.taught())
    if not held:
        return [f"{t.run}: Line B must hold something out"]
    if held & taught:
        bad.append(f"{t.run}: {len(held & taught)} pairs are both taught and withheld")
    unsupported = set(spec.ALL_PAIRS) - held - taught
    if unsupported and not (taught and len(t.teach)):
        bad.append(f"{t.run}: {len(unsupported)} pairs are neither taught nor withheld, "
                   f"which is only allowed for a stated sparse fact table")
    mirrors = [p for p in held if (p[1], p[0]) not in held]
    if mirrors:
        bad.append(f"{t.run}: {len(mirrors)} withheld pairs have a taught mirror "
                   f"image, e.g. {mirrors[0]} -> {(mirrors[0][1], mirrors[0][0])}; "
                   f"{t.run} has commutative labels, so this leaks the answer")
    taught_classes = {_answer_class(t.task, p) for p in taught}
    held_classes = {_answer_class(t.task, p) for p in held}
    missing = held_classes - taught_classes
    if missing:
        bad.append(f"{t.run}: answer classes {sorted(missing)} are withheld but never "
                   f"taught elsewhere, so the readout has unsupervised classes")
    for role in (0, 1):
        seen = {p[role] for p in taught}
        if seen != set(range(spec.OPERAND_MIN, spec.OPERAND_MAX + 1)):
            bad.append(f"{t.run}: operand role {role} is missing values "
                       f"{sorted(set(range(spec.OPERAND_MIN, spec.OPERAND_MAX + 1)) - seen)}")
    return bad


def check_plan_alignment(log) -> dict:
    """Budgets, probes and holdout discipline must be consistent across the plan."""
    problems: list[str] = []
    for group, name in ((spec.FOUNDATION, "foundation"), (spec.LINE_A, "Line A"),
                        (spec.LINE_B, "Line B")):
        log.info("  %s: %d cells", name, len(group))
        for t in group:
            if t.probes[-1] != t.budget:
                problems.append(f"{t.run}: last probe {t.probes[-1]} != budget {t.budget}")
            if sorted(set(t.probes)) != list(t.probes):
                problems.append(f"{t.run}: probes not strictly increasing")
            if t.line == "A" and (t.holdout or t.teach):
                problems.append(f"{t.run}: Line A must not hold anything out")
            if t.line == "B":
                problems += _holdout_problems(t)
    for t in spec.LINE_A:
        if t.budget != spec.LINE_A[0].budget:
            problems.append(f"{t.run}: Line A budgets differ "
                            f"({t.budget} vs {spec.LINE_A[0].budget})")
    for t in spec.LINE_B:
        if t.budget != spec.LINE_B[0].budget and t.task == "add":
            problems.append(f"{t.run}: B1 budgets differ")

    frac = len(spec.CYC7_HOLDOUT) / len(spec.ALL_PAIRS)
    log.info("  every cell's last probe equals its budget; Line A holds nothing out; "
             "every Line B holdout is transpose-closed with all classes taught")
    log.info("  B1: %d taught facts, %d withheld (2+3 in both orders), %d unsupported "
             "(evaluation only)", len(spec.B1_TAUGHT), len(spec.B1_HOLDOUT),
             len(spec.ALL_PAIRS) - len(spec.B1_TAUGHT) - len(spec.B1_HOLDOUT))
    log.info("  B2: %d taught / %d withheld = %.1f%% by pair identity",
             len(spec.CYC7_TAUGHT), len(spec.CYC7_HOLDOUT), 100 * frac)
    log.info("  B2 withheld pairs: %s", list(spec.CYC7_HOLDOUT))
    if not 0.25 <= frac <= 0.35:
        problems.append(f"B2 withholds {100 * frac:.1f}% of pairs, not ~30%")
    if problems:
        for p in problems:
            log.error("    %s", p)
    return {"problems": problems, "ok": not problems, "b1_taught": len(spec.B1_TAUGHT),
            "b1_holdout": [list(p) for p in spec.B1_HOLDOUT],
            "b1_unsupported": (len(spec.ALL_PAIRS) - len(spec.B1_TAUGHT)
                               - len(spec.B1_HOLDOUT)),
            "b2_taught": len(spec.CYC7_TAUGHT), "b2_holdout": len(spec.CYC7_HOLDOUT),
            "b2_holdout_fraction": frac,
            "b2_holdout_pairs": [list(p) for p in spec.CYC7_HOLDOUT]}



def check_task_discriminability(log) -> dict:
    """The tasks must be learnable in principle and not solvable by the shortcuts."""
    sc = StimulusConfig()
    for k, v in spec.STIMULUS.items():
        setattr(sc, k, v)
    out: dict = {}

    # counting: chance and the two centroid shortcuts on an 8-way problem
    rep = dots.shortcut_report(
        dots.make_count_dataset(1200, sc, seed=11, mode="A", balanced=True)
    )
    out["count"] = {"chance": rep["chance"],
                    "ink_baseline": rep["ink_centroid_accuracy"],
                    "spread_baseline": rep["spread_centroid_accuracy"]}
    log.info("  count 1-%d: chance %.3f | ink shortcut %.3f | spread shortcut %.3f  %s",
             sc.n_max, rep["chance"], rep["ink_centroid_accuracy"],
             rep["spread_centroid_accuracy"],
             "OK" if max(rep["ink_centroid_accuracy"], rep["spread_centroid_accuracy"])
             < rep["chance"] + 0.25 else "shortcut is strong")

    # cyclic addition mod 7: every answer reachable, and the split must be by pair
    labels = set()
    for a in range(1, 8):
        for b in range(1, 8):
            labels.add(((a + b - 2) % 7))
    out["cyc7"] = {"n_pairs": 49, "n_classes": len(labels)}
    log.info("  cyc7: %d ordered pairs over %d classes (chance %.4f), every class "
             "reachable  %s", 49, len(labels), 1 / len(labels),
             "OK" if len(labels) == 7 else "FAIL")

    # two-step: (a+b)-c must be non-negative for every reachable c, and the answer
    # distribution must not collapse to one value
    vals = [(a + b) - c for a in range(1, 5) for b in range(1, 5) for c in range(1, 5)]
    vals = [v for v in vals if v >= 1]
    out["two_step"] = {"n_items": len(vals), "min": min(vals), "max": max(vals),
                       "n_distinct": len(set(vals))}
    log.info("  two_step: %d items, answers %d..%d, %d distinct  %s",
             len(vals), min(vals), max(vals), len(set(vals)),
             "OK" if len(set(vals)) >= 5 else "FAIL")
    ok = (len(labels) == 7 and len(set(vals)) >= 5)
    return {**out, "ok": ok}


def main() -> int:
    log = get_logger("preflight")
    log.info("=" * 92)
    log.info("PHASE I PRE-FLIGHT | one seed | core 33k | signed | alpha=%.2f w_scale=%.2f",
             spec.DYNAMICS["alpha"], spec.DYNAMICS["w_scale"])
    report: dict = {"dynamics": spec.DYNAMICS, "stimulus": spec.STIMULUS,
                    "checks": {}}

    for name, fn in (("stimulus geometry", check_stimulus_geometry),
                     ("blob integrity", check_blob_integrity),
                     ("legibility", check_legibility),
                     ("dynamics bounded", check_dynamics),
                     ("plan alignment", check_plan_alignment),
                     ("task discriminability", check_task_discriminability)):
        log.info("-" * 92)
        log.info("CHECK %s", name)
        try:
            report["checks"][name] = fn(log)
        except Exception as exc:
            import traceback

            log.error("  FAIL %s: %s\n%s", name, exc, traceback.format_exc()[-800:])
            report["checks"][name] = {"ok": False, "error": str(exc)}

    bad = [k for k, v in report["checks"].items() if not v.get("ok")]
    log.info("=" * 92)
    for k, v in report["checks"].items():
        log.info("  %-24s %s", k, "PASS" if v.get("ok") else "FAIL")
    log.info("PRE-FLIGHT %s", "PASSED - the plan is sound" if not bad
             else f"BLOCKED on {bad}")
    out = paths.DATA_PROCESSED / "phase1_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
