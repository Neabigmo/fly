"""Deciding whether a late held-out jump is generalisation or a threshold artefact.

The criteria are frozen in :mod:`flynum.phase1.spec` and this module only applies them.
Two things are worth stating precisely, because both are places where a plausible-looking
claim could be manufactured:

**Sustained, not first-crossing.**  ``T_mem`` and ``T_grok`` are the first probe of a run
of :attr:`EmergenceCriteria.sustain` consecutive probes above the threshold.  A single
probe above the line is noise at this sample size -- one held-out set of 15 pairs scored
at 1,500 rows moves in steps of 1/15 of a pair -- so a crossing that does not persist is
not a time.

**Accuracy alone is not evidence.**  Accuracy is an argmax crossing a cut: it steps even
when nothing about the representation changed, and it can step *late* purely because the
decision boundary drifted.  So a transition is only called generalisation if the
continuous quantities move with it, and "move" is defined as movement too large to be a
rounding error rather than as a sign:

    held-out cross-entropy must at least halve, mean P(correct) must at least double,
    and the correct-minus-runner-up margin must increase,

all measured from the probe before the crossing to the probe at the crossing.  A
direction-only test is not enough, and the difference is not academic: an argmax can flip
while the probability mass stays where it was, which is exactly the artefact this guard
exists to catch.  If the test fails, the verdict says so explicitly instead of quietly
reporting the accuracy number.

Everything here is a pure function of the probe history, which is what makes it testable
without training anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import CRITERIA, EmergenceCriteria

#: Continuous metrics, and the direction a genuine improvement moves them.
CONTINUOUS_DIRECTION = {"ce": -1, "p_correct": +1, "margin": +1, "entropy": -1}

#: How much each metric must move across the crossing to count as confirmation.  A halving
#: of cross-entropy and a doubling of P(correct) is the weakest movement that cannot be a
#: rounding artefact; the margin only has to rise, because its scale is not interpretable.
CONTINUOUS_REQUIREMENT = {"ce": ("ratio", 0.5), "p_correct": ("ratio", 2.0),
                          "margin": ("direction", None)}


@dataclass
class Verdict:
    """What the pre-registered criteria say about one cell's curves."""

    t_mem: int | None = None
    t_grok: int | None = None
    delay_factor: float | None = None
    delayed: bool = False
    continuous_confirms: bool | None = None
    continuous_moves: dict = field(default_factory=dict)
    max_hold_acc: float = float("nan")
    final_hold_acc: float = float("nan")
    final_train_acc: float = float("nan")
    largest_jump_at: int | None = None
    largest_jump: float = float("nan")
    verdict: str = "insufficient_data"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "t_mem": self.t_mem, "t_grok": self.t_grok,
            "delay_factor": self.delay_factor, "delayed": self.delayed,
            "continuous_confirms": self.continuous_confirms,
            "continuous_moves": self.continuous_moves,
            "max_hold_acc": self.max_hold_acc, "final_hold_acc": self.final_hold_acc,
            "final_train_acc": self.final_train_acc,
            "largest_jump_at": self.largest_jump_at, "largest_jump": self.largest_jump,
            "verdict": self.verdict, "notes": self.notes,
        }


def first_sustained(steps: list[int], values: list[float], threshold: float,
                    sustain: int) -> int | None:
    """First probe of ``sustain`` consecutive probes strictly above ``threshold``."""
    run = 0
    for i, v in enumerate(values):
        if v == v and v > threshold:          # NaN never counts as above threshold
            run += 1
            if run >= sustain:
                return steps[i - sustain + 1]
        else:
            run = 0
    return None


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for r in history:
        v = r.get(key)
        steps.append(int(r.get("updates", 0)))
        vals.append(float(v) if isinstance(v, (int, float)) and v == v else float("nan"))
    return steps, vals


def _index_at(steps: list[int], step: int | None) -> int | None:
    if step is None:
        return None
    return steps.index(step) if step in steps else None


def detect(history: list[dict], criteria: EmergenceCriteria = CRITERIA, *,
           train_key: str = "train_acc", hold_key: str = "hold_acc") -> Verdict:
    """Apply the frozen criteria to one cell's probe history."""
    v = Verdict()
    if len(history) < criteria.sustain + 1:
        v.notes.append(f"only {len(history)} probes; need at least "
                       f"{criteria.sustain + 1}")
        return v

    steps, train = _series(history, train_key)
    _, hold = _series(history, hold_key)
    v.final_train_acc, v.final_hold_acc = train[-1], hold[-1]
    v.max_hold_acc = max((x for x in hold if x == x), default=float("nan"))

    v.t_mem = first_sustained(steps, train, criteria.mem_acc, criteria.sustain)
    v.t_grok = first_sustained(steps, hold, criteria.grok_acc, criteria.sustain)
    if v.t_mem and v.t_grok:
        v.delay_factor = v.t_grok / v.t_mem
        v.delayed = v.delay_factor > criteria.delay_factor

    # where the biggest single-probe change in held-out accuracy happened, regardless of
    # any threshold: reported so the transition can be located even when it never reaches
    # the criterion
    jumps = [(steps[i], hold[i] - hold[i - 1]) for i in range(1, len(hold))
             if hold[i] == hold[i] and hold[i - 1] == hold[i - 1]]
    if jumps:
        v.largest_jump_at, v.largest_jump = max(jumps, key=lambda t: t[1])

    i0 = _index_at(steps, v.t_grok)
    if i0 is not None and i0 >= 1:
        window = history[max(i0 - 1, 0): i0 + 1]
        moves = {}
        for key in CONTINUOUS_DIRECTION:
            k = f"hold_{key}"
            a, b = window[0].get(k), window[-1].get(k)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            a, b = float(a), float(b)
            kind, need = CONTINUOUS_REQUIREMENT.get(key, ("direction", None))
            if kind == "ratio":
                # ratio > 1 means the metric moved the way an improvement moves it
                ratio = (a / b) if (key == "ce" and b > 0) else (
                    (b / a) if a > 0 else float("inf"))
                moved = ratio >= need
            else:
                moved = CONTINUOUS_DIRECTION[key] * (b - a) > 0
            moves[key] = {"before": round(a, 4), "at": round(b, 4),
                          "delta": round(b - a, 4), "moved_right_way": bool(moved)}
            if kind == "ratio":
                moves[key]["ratio"] = round(float(ratio), 3)
                moves[key]["required_ratio"] = need
        v.continuous_moves = moves
        required = tuple(CONTINUOUS_REQUIREMENT)
        have = [m for m in required if m in moves]
        v.continuous_confirms = (len(have) == len(required)
                                 and all(moves[m]["moved_right_way"] for m in required))

    if v.t_mem is None:
        v.verdict = "never_memorised"
        v.notes.append(f"training accuracy never held above {criteria.mem_acc:.2f}, so "
                       f"there is no memorisation phase to be late relative to")
    elif v.t_grok is None:
        v.verdict = "memorised_no_generalisation"
        v.notes.append(f"held-out accuracy never held above {criteria.grok_acc:.2f}"
                       + (f" (best {v.max_hold_acc:.3f})" if v.max_hold_acc == v.max_hold_acc
                          else ""))
    elif not v.delayed:
        v.verdict = "simultaneous"
        v.notes.append(f"held-out criterion reached at {v.t_grok} updates, only "
                       f"{v.delay_factor:.2f}x the memorisation time {v.t_mem}")
    elif criteria.require_continuous and v.continuous_confirms is False:
        v.verdict = "delayed_but_continuous_metrics_flat"
        v.notes.append("the held-out accuracy criterion was crossed late, but "
                       "cross-entropy / P(correct) / margin did not move with it: this "
                       "is an accuracy-threshold artefact, not a transition")
    else:
        v.verdict = "delayed_generalisation"
        v.notes.append(f"memorised at {v.t_mem}, held-out criterion at {v.t_grok} "
                       f"({v.delay_factor:.2f}x), continuous metrics moved with it")
    return v


def summarise_cell(history: list[dict], line: str) -> dict:
    """A verdict for either line: Line A has no held-out set, so it reports the curves."""
    if line == "A":
        steps, seen = _series(history, "seen_acc")
        _, train = _series(history, "train_acc")
        auc = _auc(steps, seen)
        return {
            "line": "A", "final_train_acc": train[-1], "final_seen_acc": seen[-1],
            "auc_seen_acc": auc,
            "updates_to_50": first_sustained(steps, seen, 0.50, 1),
            "updates_to_80": first_sustained(steps, seen, 0.80, 2),
            "updates_to_95": first_sustained(steps, seen, 0.95, 2),
        }
    return {"line": "B", **detect(history).as_dict()}


def _auc(steps: list[int], values: list[float]) -> float:
    """Area under the learning curve, normalised by the budget.

    Normalising is what makes cells with different budgets comparable: the number is the
    average accuracy over the run, in units of the run's own length.
    """
    xs = [s for s, v in zip(steps, values) if v == v]
    ys = [v for v in values if v == v]
    if len(xs) < 2 or xs[-1] <= xs[0]:
        return float("nan")
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * 0.5 * (ys[i] + ys[i - 1])
    return area / (xs[-1] - xs[0])
