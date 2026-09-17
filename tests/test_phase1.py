"""Phase I: task arithmetic, pool purity, and the emergence rules.

The emergence tests are the important ones.  They feed synthetic probe histories whose
answer is known by construction -- a genuine late transition, a late accuracy step with
flat continuous metrics, a memoriser that never generalises -- and check that the frozen
criteria classify each one correctly.  Without them, "delayed generalisation" would be a
label nobody could audit.
"""

from __future__ import annotations

import numpy as np
import pytest

from flynum.config import StimulusConfig
from flynum.phase1 import emergence, glyphs, spec, tasks
from flynum.phase1.data import build_pools


def _sc() -> StimulusConfig:
    sc = StimulusConfig()
    for k, v in spec.STIMULUS.items():
        setattr(sc, k, v)
    return sc


# --------------------------------------------------------------------------- #
def test_task_answers_are_in_range_and_correct():
    """Every item maps into its head, and the arithmetic is what it claims to be."""
    for name, t in tasks.TASKS.items():
        classes = {t.answer(it) for it in t.items}
        assert classes <= set(range(t.n_classes)), name
        assert len(classes) == t.n_classes or name == "two_step", (
            f"{name}: only {len(classes)} of {t.n_classes} classes are reachable")

    add = tasks.TASKS["add"]
    assert add.answer((2, 3)) == 3 and add.answer_value((2, 3)) == 5
    assert add.answer((7, 7)) == 12

    cyc = tasks.TASKS["cyc7"]
    assert cyc.answer_value((7, 7)) == (14 - 2) % 7 == 5
    assert {cyc.answer_value((1, n)) for n in range(1, 8)} == set(range(7))

    sub = tasks.TASKS["addsub"]
    assert sub.answer_value((3, 0, 2)) == 5          # 3 + 2
    assert sub.answer_value((3, 1, 2)) == 1          # 3 - 2
    assert all(it[0] >= it[2] for it in sub.items if glyphs.OPS[it[1]] == "-")

    two = tasks.TASKS["two_step"]
    assert two.answer_value((4, 0, 4, 1, 1)) == 7    # (4 + 4) - 1
    assert all(two.answer_value(it) >= 1 for it in two.items)


def test_sequence_lengths_and_phase_ends():
    """Cue and gap are the same length, so tasks differ only in what is drawn."""
    assert spec.TIME["steps_cue"] == spec.TIME["steps_gap"]
    # counting is the one task with a longer window: a single group, seven blobs to
    # integrate, and no arithmetic to do on it
    expected = {"count": 8, "add": 14, "cyc7": 14, "addsub": 14, "two_step": 23}
    for name, t in tasks.TASKS.items():
        plan = tasks.SequencePlan(t)
        assert plan.n_steps() == expected[name], name
    assert tasks.SequencePlan(tasks.TASKS["add"]).content_ends() == [5, 14]
    assert tasks.SequencePlan(tasks.TASKS["two_step"]).content_ends() == [5, 9, 14, 18, 23]


def test_the_taught_mixture_is_not_solved_by_a_single_scalar():
    """The design guard: no one statistic of the image answers the taught question.

    Each condition leaks something -- the natural one leaks brightness, any constant-area
    one unavoidably leaks per-dot size -- so the taught set is a mixture, and this asserts
    that the best single-scalar strategy stays well below solving it.  If a later change to
    the stimulus generator makes one cue decisive again, this test fails rather than letting
    the study report cue exploitation as capability.
    """
    from flynum.phase1 import cues

    table = cues.cue_table(_sc(), n_per_mode=400)
    ceiling = cues.mixture_ceiling(table, spec.TRAIN_MODES)
    assert ceiling["accuracy"] < 0.75, ceiling
    assert ceiling["chance"] < 0.2
    # and the conditions differ in what they leak, which is what makes the mixture bite
    assert table["A"]["ink"] > 0.3 and table["B"]["ink"] < 0.25
    assert table["B"]["radius_mean"] > 0.9 and table["A"]["radius_mean"] < 0.4


def test_glyphs_are_distinct_and_not_louder_than_dots():
    sc = _sc()
    rng = np.random.default_rng(0)
    plus = glyphs.render_glyph("+", sc, rng)
    minus = glyphs.render_glyph("-", sc, rng)
    blank = glyphs.render_blank(sc.image_size)
    assert plus.shape == minus.shape == (32, 32)
    assert np.array_equal(blank, np.zeros((32, 32), dtype=np.float32))
    assert not np.array_equal(plus, minus)
    # a "+" is a "-" plus a vertical arm, so it is inked in strictly more pixels, and a
    # three-dot group is in the same league -- the operator is not a beacon
    assert glyphs.ink("+", sc) > glyphs.ink("-", sc)
    from flynum.stimuli.dots import make_count_stimulus

    three = make_count_stimulus(3, sc, rng)
    assert 0.3 < glyphs.ink("-", sc) / three.ink < 3.0


def test_render_phase_matches_item_and_is_parallel_stable():
    """Parallel rendering must not change the pixels: same seed, same pool."""
    sc = _sc()
    t = tasks.TASKS["addsub"]
    a = tasks.build_dataset(t, sc, reps_per_item=2, seed=3, workers=1)
    b = tasks.build_dataset(t, sc, reps_per_item=2, seed=3, workers=4)
    assert np.array_equal(a["images"], b["images"])
    assert np.array_equal(a["item_index"], b["item_index"])
    assert np.array_equal(a["labels"], b["labels"])
    # the first phase of a row is the first operand's dot group, the last is the second's
    row = 0
    item = a["items"][a["item_index"][row]]
    assert a["images"][row, 0].sum() > 0 and a["images"][row, 1].sum() > 0
    assert a["labels"][row] == t.answer(tuple(item))


# --------------------------------------------------------------------------- #
def test_pools_never_train_on_a_withheld_item():
    """The purity guarantee, on the arrays training actually consumes."""
    sc = _sc()
    t = tasks.TASKS["add"]
    taught = set(spec.B1_TAUGHT)
    held = spec.B1_HOLDOUT
    pools = build_pools(t, sc, taught=taught, holdout=held, reps=2, seed=0,
                        eval_items=40)
    train_items = {tuple(it) for it in pools["splits"]["train"]["items"][
        pools["splits"]["train"]["item_index"]]}
    assert train_items == taught
    assert not (train_items & set(held))
    assert set(pools["unsupported_items"]) == set(t.items) - taught - set(held)
    # each withheld pair is present in the held-out split, and the diagnostic set is
    # disjoint from both
    hold_items = {tuple(it) for it in pools["splits"]["holdout"]["items"]}
    assert set(held) <= hold_items


def test_pools_reject_a_leaky_specification():
    sc = _sc()
    t = tasks.TASKS["cyc7"]
    with pytest.raises(AssertionError, match="taught and\\s+withheld"):
        build_pools(t, sc, taught=set(spec.CYC7_TAUGHT) | {spec.CYC7_HOLDOUT[0]},
                    holdout=spec.CYC7_HOLDOUT, reps=1, seed=0, eval_items=20)


def test_cyc7_split_properties_are_structural():
    held = set(spec.CYC7_HOLDOUT)
    assert len(held) == 15 and len(spec.CYC7_TAUGHT) == 34
    assert all((b, a) in held for a, b in held)                 # transpose-closed
    from collections import Counter

    per_class = Counter(spec.cyc7_class(p) for p in held)
    assert set(per_class) == set(range(7))
    assert max(per_class.values()) - min(per_class.values()) <= 1
    counts = Counter(p[0] for p in held)
    assert max(counts.values()) <= 3                            # operand balance


# --------------------------------------------------------------------------- #
def _history(steps, train, hold, ce, p_correct, margin, entropy=None):
    return [{"updates": s, "train_acc": a, "hold_acc": b, "hold_ce": c,
             "hold_p_correct": p, "hold_margin": m,
             "hold_entropy": e if e is not None else 1.0}
            for s, a, b, c, p, m, e in zip(steps, train, hold, ce, p_correct, margin,
                                           entropy or [1.0] * len(steps))]


def test_first_sustained_needs_a_run():
    steps = [0, 10, 20, 30, 40, 50]
    assert emergence.first_sustained(steps, [0, 1, 0, 1, 1, 1], 0.5, 3) == 30
    assert emergence.first_sustained(steps, [1, 1, 0, 0, 0, 0], 0.5, 3) is None
    # a NaN is not a crossing
    assert emergence.first_sustained(steps, [float("nan")] * 6, 0.5, 1) is None


def test_delayed_generalisation_is_detected_when_continuous_metrics_move():
    # memorised at 100, held-out criterion at 400: a 4x delay, which the frozen
    # criterion (>{:.0f}x) accepts as delayed rather than simultaneous
    steps = [0, 100, 200, 300, 400, 500, 600]
    train = [0.1, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99]
    hold = [0.13, 0.13, 0.14, 0.15, 0.85, 0.90, 0.92]
    ce = [2.5, 2.5, 2.4, 2.3, 0.5, 0.3, 0.2]
    p_correct = [0.14, 0.14, 0.15, 0.16, 0.75, 0.85, 0.90]
    margin = [-1.0, -1.0, -0.9, -0.8, 2.0, 3.0, 3.5]
    v = emergence.detect(_history(steps, train, hold, ce, p_correct, margin))
    assert v.t_mem == 100 and v.t_grok == 400
    assert v.delay_factor == pytest.approx(4.0)
    assert v.delayed and v.continuous_confirms is True
    assert v.verdict == "delayed_generalisation"


def test_a_two_fold_delay_is_not_called_delayed():
    """The criterion is a factor of three; 2x is 'simultaneous' and must say so."""
    steps = [0, 100, 200, 300, 400]
    train = [0.1, 0.97, 0.98, 0.99, 0.99]
    hold = [0.13, 0.13, 0.85, 0.90, 0.92]
    v = emergence.detect(_history(steps, train, hold, [2.5, 2.5, 0.5, 0.4, 0.35],
                                  [0.14, 0.14, 0.75, 0.85, 0.88],
                                  [-1.0, -1.0, 2.0, 3.0, 3.2]))
    assert v.t_mem == 100 and v.t_grok == 200 and v.delay_factor == pytest.approx(2.0)
    assert not v.delayed and v.verdict == "simultaneous"


def test_late_accuracy_step_with_flat_continuous_metrics_is_not_generalisation():
    """The failure this criterion exists to catch."""
    steps = [0, 100, 200, 300, 400, 500, 600]
    train = [0.1, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99]
    hold = [0.13, 0.13, 0.14, 0.15, 0.85, 0.90, 0.92]
    # the argmax flips while the probability mass barely moves: the exact artefact the
    # continuous test exists to catch, and it stays inside the CE/P(correct) consistency
    # bound (p_correct 0.2 permits CE >= 1.6)
    ce = [2.05, 2.05, 2.05, 2.04, 2.00, 1.98, 1.97]
    p_correct = [0.15, 0.15, 0.15, 0.155, 0.185, 0.195, 0.20]
    margin = [-1.0, -1.0, -1.0, -0.95, 0.02, 0.05, 0.06]
    v = emergence.detect(_history(steps, train, hold, ce, p_correct, margin))
    assert v.delayed and v.continuous_confirms is False
    assert v.verdict == "delayed_but_continuous_metrics_flat"
    assert "artefact" in " ".join(v.notes)


def test_memoriser_that_never_generalises():
    steps = [0, 100, 200, 300, 400]
    v = emergence.detect(_history(steps, [0.1, 0.5, 0.97, 0.98, 0.99],
                                  [0.13, 0.13, 0.15, 0.16, 0.16],
                                  [2.5, 2.4, 2.3, 2.2, 2.2],
                                  [0.14, 0.14, 0.15, 0.16, 0.16],
                                  [-1.0, -1.0, -0.9, -0.8, -0.8]))
    assert v.t_mem == 200 and v.t_grok is None
    assert v.verdict == "memorised_no_generalisation"


def test_simultaneous_learning_is_not_called_delayed():
    steps = [0, 100, 200, 300, 400]
    v = emergence.detect(_history(steps, [0.1, 0.5, 0.97, 0.99, 0.99],
                                  [0.13, 0.5, 0.85, 0.90, 0.92],
                                  [2.5, 1.5, 0.6, 0.4, 0.3],
                                  [0.14, 0.5, 0.80, 0.85, 0.88],
                                  [-1.0, 0.5, 2.0, 2.5, 2.8]))
    assert v.t_mem == 200 and v.t_grok == 200
    assert not v.delayed and v.verdict == "simultaneous"


def test_line_a_summary_reports_curve_metrics():
    steps = [0, 10, 20, 30]
    hist = [{"updates": s, "train_acc": t, "seen_acc": a, "unsup_acc": float("nan")}
            for s, t, a in zip(steps, [0.1, 0.4, 0.7, 0.9], [0.2, 0.6, 0.82, 0.9])]
    s = emergence.summarise_cell(hist, "A")
    assert s["line"] == "A" and s["final_seen_acc"] == 0.9
    # 0.50 is crossed strictly at the 0.6 probe; the 0.80 criterion needs two
    # consecutive probes, so it is dated to the first of the two
    assert s["updates_to_50"] == 10 and s["updates_to_80"] == 20
    assert 0.5 < s["auc_seen_acc"] < 0.9


# --------------------------------------------------------------------------- #
def test_warm_start_lookup_matches_the_directory_a_run_creates(tmp_path, monkeypatch):
    """The naming contract between a run's directory and the warm-start lookup.

    The first version of the runner looked for ``runs/C0/ckpt/final.pt`` while the run had
    written ``runs/10_C0_s0/ckpt/final.pt``.  Nothing failed until the second cell of a
    twelve-cell queue, which then aborted and left ten cells unrun.  A dry run did not
    catch it because its missing-prerequisite branch trained from scratch by design.
    """
    from flynum import paths
    from flynum.logging_utils import RunContext
    from flynum.phase1.run import find_checkpoint, run_id_for

    monkeypatch.setattr(paths, "RUNS", tmp_path)
    ctx = RunContext(run_id_for("C0", "10", 0), None)
    assert ctx.run_id == "10_C0_s0"
    assert find_checkpoint("C0", tag="10", seed=0) is None      # no checkpoint yet
    (ctx.dir / "ckpt" / "final.pt").write_bytes(b"weights")
    assert find_checkpoint("C0", tag="10", seed=0) == ctx.dir / "ckpt" / "final.pt"
    # a different tag or seed is a different run and must not be found
    assert find_checkpoint("C0", tag="11", seed=0) is None
    assert find_checkpoint("C0", tag="10", seed=1) is None
    # an untagged directory is still accepted, for checkpoints made outside the runner
    plain = tmp_path / "countA01_src" / "ckpt"
    plain.mkdir(parents=True)
    (plain / "final.pt").write_bytes(b"older")
    assert find_checkpoint("countA01_src", tag="10", seed=0) == plain / "final.pt"
