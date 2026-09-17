"""Phase I: the frozen specification.

Everything that defines Phase I lives here, in one place, so that a run cannot drift
from the plan and so that the pre-flight checker and the runner read the same numbers.
The old study is kept intact and treated as pilot evidence; nothing here changes its
defaults.

Two questions, deliberately separated:

**Line A -- how complex a mathematics can this connectome be taught?**  No held-out
pairs anywhere in Line A.  A count model C0 and a scratch brain S0 are both carried
through a curriculum tree (count -> addition -> +/- rule selection -> two-step), and at
every level the same task is taught to both.  The comparison is a learning curve, not an
endpoint: final accuracy, area under the curve, and updates-to-criterion.

**Line B -- does long training jump from memorisation to a rule?**  Here part of the
answer is deliberately withheld and training runs far longer, because delayed
generalisation is by definition a gap between "the training set is memorised" and "the
held-out set suddenly works".  The withheld pairs never touch the gradient, the
scheduler, early stopping or checkpoint selection.

One seed throughout: this stage is for finding phenomena, not for error bars.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# The single digital fly used by every task in Phase I.
# --------------------------------------------------------------------------- #
DYNAMICS: dict = {
    "circuit": "core",          # 33k neurons; the full 99k connectome waits for Phase II
    "signed_synapses": True,    # real neurotransmitter signs: ~22% of edges inhibitory
    "w_scale": 0.5,             # the signed-bounded setting (48-step peak 13.8 vs 3.4e9)
    "alpha": 0.1,
    "learn": ("delta", "bias"),  # edge gains and per-neuron bias; topology is frozen
    "readout_standardize": False,
    "readout": "vpn",           # fixed retinotopic mapping, no CNN anywhere
}

#: The 1-7 dot stimulus.  These numbers are not free choices -- they are the solution of
#: three constraints that the pre-flight checker verifies by measurement:
#:   * constant-area control across n=1..7 needs r(1)/r(7) = sqrt(7) = 2.646, and the
#:     generator jitters radii by +-8% before renormalising, so the *range* must span
#:     2.646 * 1.08/0.92 = 3.107;
#:   * seven discs plus the minimum gap must fit: r_max <= 3.789 at S=32;
#:   * the area-implied radius for every n must land inside the range, or the
#:     area-controlled condition silently becomes a size-generalisation test.
#: Measured with [1.13, 3.70] and area_target 35.6: 0.000% of area-controlled radii fall
#: outside the range, and 2,240 generated stimuli across 4 modes x n=1..7 all recover
#: exactly n connected components.
STIMULUS: dict = {
    "image_size": 32,
    "n_min": 1,
    "n_max": 7,
    "radius_min": 1.13,
    "radius_max": 3.70,
    "area_target": 35.6,
    "min_gap": 1.5,
    "ring_radius": 9.5,
}

#: Sequence timing.  Every phase length is shared by every task so that the number of
#: recurrent steps per update is comparable across the curriculum.
TIME: dict = {
    "steps_operand": 5,   # each dot group is held for this many steps
    "steps_cue": 4,       # an operator glyph occupies the gap
    "steps_gap": 4,       # a blank delay when there is no glyph
}


# --------------------------------------------------------------------------- #
# What is taught and what is withheld.
#
# Both Line B tasks have *commutative* labels: 2+3 = 3+2, and (a+b) mod 7 =
# (b+a) mod 7.  So withholding one ordered pair withholds nothing: the network can
# answer it by evaluating the mirrored pair, which it was taught.  Every holdout
# below is therefore closed under transposition, and the pre-flight check verifies
# that property rather than trusting it.  Without this, "emergent generalisation"
# would be reported for what is really a taught symmetry.
# --------------------------------------------------------------------------- #
OPERAND_MIN = 1
OPERAND_MAX = 7

#: Every ordered operand pair the two Line B tasks can present.
ALL_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in range(OPERAND_MIN, OPERAND_MAX + 1)
    for b in range(OPERAND_MIN, OPERAND_MAX + 1)
)

#: B1 withholds the *fact* 2+3, in both orders, from a 15-fact table.
B1_HOLDOUT: tuple[tuple[int, int], ...] = ((2, 3), (3, 2))


def _b1_taught() -> tuple[tuple[int, int], ...]:
    """The 15 additions B1 is taught, built mechanically so nothing is cherry-picked.

    One canonical pair per sum class (the staircase, which is the only way to reach
    sums 9-14 at all), then the lexicographically first two pairs that are neither
    taught nor withheld.  This covers all 13 answer classes and every operand in
    both roles, so the readout has no unsupervised class and the model can read
    both operands -- while 2+3 stays outside the table in both orders.
    """
    pairs = [(max(OPERAND_MIN, s - OPERAND_MAX), min(OPERAND_MAX, s - OPERAND_MIN))
             for s in range(2 * OPERAND_MIN, 2 * OPERAND_MAX + 1)]
    held = set(B1_HOLDOUT)
    for p in ALL_PAIRS:
        if len(pairs) >= 15:
            break
        if p not in pairs and p not in held:
            pairs.append(p)
    return tuple(pairs)


B1_TAUGHT: tuple[tuple[int, int], ...] = _b1_taught()

#: B2 teaches 70% of the 49 residue pairs and withholds 30% by pair identity.
#: Answer class of an ordered pair of operands in 1..7: (a + b - 2) mod 7.
CYC7_HOLDOUT_FRACTION = 0.30
CYC7_SEED = 20_240_915


def cyc7_class(pair: tuple[int, int]) -> int:
    return (pair[0] + pair[1] - 2) % 7


def _cyc7_split() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """(taught, holdout): a 70/30 split fixed by an explicitly balanced design.

    Three properties are wanted at once, and all three are stated here before any
    result is seen:

      * closed under transposition -- the labels are commutative, so a withheld pair
        with a taught mirror image is not withheld;
      * every residue class withheld -- so held-out accuracy is not dominated by one
        answer, and chance stays exactly 1/7;
      * balanced over operands -- every value appears in exactly two of the seven
        withheld pairs (plus value 1 in the diagonal), so a low held-out score cannot
        be an artefact of the operands the holdout happens to favour.

    A 2-regular graph on the seven operands has exactly those degrees, so the split is
    a 7-edge 2-regular graph whose edges carry seven distinct classes.  There are
    finitely many; one is picked with a fixed seed.  The result is 14 ordered pairs
    plus the diagonal (1,1) = 15 of 49 (30.6%), 34 taught.
    """
    edges = [(a, b) for a in range(OPERAND_MIN, OPERAND_MAX + 1)
             for b in range(a + 1, OPERAND_MAX + 1)]
    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(edges, 7):
        deg = dict.fromkeys(range(OPERAND_MIN, OPERAND_MAX + 1), 0)
        for a, b in combo:
            deg[a] += 1
            deg[b] += 1
        if any(d != 2 for d in deg.values()):
            continue
        if len({cyc7_class(e) for e in combo}) != 7:
            continue
        valid.append(combo)
    if not valid:
        raise RuntimeError("no balanced 7-edge design exists for the cyc7 holdout")

    pick = random.Random(CYC7_SEED).choice(valid)
    hold = [p for e in pick for p in (e, (e[1], e[0]))]
    hold.append((OPERAND_MIN, OPERAND_MIN))          # the class-0 diagonal

    held = set(hold)
    taught = tuple(p for p in ALL_PAIRS if p not in held)
    return taught, tuple(sorted(held))


CYC7_TAUGHT, CYC7_HOLDOUT = _cyc7_split()


# --------------------------------------------------------------------------- #
@dataclass
class TaskSpec:
    """One cell of the plan: a task, a starting brain, and a measurement schedule."""

    run: str
    line: str                 # "A" | "B"
    task: str                 # count | add | addsub | two_step | cyc7
    brain: str                # scratch | count | add | curriculum
    #: optimiser updates; every cell on a line is compared at the same budget
    budget: int = 50_000
    #: update counts at which the full metric set is recorded
    probes: tuple[int, ...] = ()
    #: pairs deliberately withheld from teaching.  Empty for all of Line A.
    holdout: tuple[tuple[int, int], ...] = ()
    #: the pairs actually taught.  Empty means "every ordered pair except the
    #: holdout"; B1 states its 15-fact table explicitly instead.
    teach: tuple[tuple[int, int], ...] = ()
    question: str = ""

    def __post_init__(self) -> None:
        if not self.probes:
            self.probes = probes_for(self.budget)

    def taught(self) -> tuple[tuple[int, int], ...]:
        """The taught fact table, whether stated directly or as a complement."""
        if self.teach:
            return self.teach
        held = set(self.holdout)
        return tuple(p for p in ALL_PAIRS if p not in held)


def probes_for(budget: int) -> tuple[int, ...]:
    """A geometric-ish schedule whose last point is the budget itself."""
    base = [0, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000,
            350_000, 500_000]
    out = [p for p in base if p < budget]
    out.append(budget)
    return tuple(out)


#: The foundation both lines start from.
FOUNDATION: list[TaskSpec] = [
    TaskSpec(run="C0", line="F", task="count", brain="scratch", budget=50_000,
             question="a unified counting brain over 1-7 dots; also the pretrain source"),
]

#: Line A -- capability ceiling under systematic teaching.  No holdouts anywhere.
LINE_A: list[TaskSpec] = [
    TaskSpec(run="A1-S", line="A", task="add", brain="scratch", budget=50_000,
             question="can addition be taught from scratch, and how fast?"),
    TaskSpec(run="A1-C", line="A", task="add", brain="count", budget=50_000,
             question="does numerosity make addition easier to learn?"),
    TaskSpec(run="A2-S", line="A", task="addsub", brain="scratch", budget=50_000,
             question="can the +/- rule be selected from a visual cue, from scratch?"),
    TaskSpec(run="A2-C", line="A", task="addsub", brain="add", budget=50_000,
             question="does already holding one arithmetic rule help acquire a second?"),
    TaskSpec(run="A3-S", line="A", task="two_step", brain="scratch", budget=50_000,
             question="can a two-step computation be taught from scratch?"),
    TaskSpec(run="A3-C", line="A", task="two_step", brain="curriculum", budget=50_000,
             question="does the accumulated curriculum reach a two-step computation?"),
]

#: Line B -- emergence under withheld supervision and long training.
LINE_B: list[TaskSpec] = [
    TaskSpec(run="B1-S", line="B", task="add", brain="scratch", budget=200_000,
             holdout=B1_HOLDOUT, teach=B1_TAUGHT,
             question="does 2+3 ever appear if it is never taught and never selected on?"),
    TaskSpec(run="B1-C", line="B", task="add", brain="count", budget=200_000,
             holdout=B1_HOLDOUT, teach=B1_TAUGHT,
             question="does numerosity change whether 2+3 emerges?"),
    TaskSpec(run="B2-S", line="B", task="cyc7", brain="scratch", budget=500_000,
             holdout=CYC7_HOLDOUT, teach=CYC7_TAUGHT,
             question="cyclic addition mod 7: does held-out-pair accuracy jump late?"),
    TaskSpec(run="B2-C", line="B", task="cyc7", brain="count", budget=500_000,
             holdout=CYC7_HOLDOUT, teach=CYC7_TAUGHT,
             question="does a counting foundation make the jump earlier or larger?"),
]

#: Runs that are only justified once B2 shows a transition: is grokking a property of
#: this connectome's wiring, or of trainable RNNs in general?
LINE_B_CONTROLS: list[TaskSpec] = [
    TaskSpec(run="B2-SS", line="B", task="cyc7", brain="scratch", budget=500_000,
             holdout=CYC7_HOLDOUT, teach=CYC7_TAUGHT,
             question="shuffled control for B2-S"),
    TaskSpec(run="B2-SC", line="B", task="cyc7", brain="count", budget=500_000,
             holdout=CYC7_HOLDOUT, teach=CYC7_TAUGHT,
             question="shuffled control for B2-C"),
]


# --------------------------------------------------------------------------- #
# Emergence criteria, fixed before the runs so the claim cannot be fitted afterwards.
# --------------------------------------------------------------------------- #
@dataclass
class EmergenceCriteria:
    """When may we say 'delayed generalisation' rather than 'a threshold was crossed'?"""

    #: held-out accuracy that counts as generalisation (chance for 7 classes is 1/7)
    grok_acc: float = 0.80
    #: training accuracy that counts as memorisation
    mem_acc: float = 0.95
    #: how many consecutive probes a condition must hold before it is called a time
    sustain: int = 3
    #: T_grok must exceed T_mem by this factor to be called *delayed*
    delay_factor: float = 3.0
    #: and the continuous metrics must move too, or it is an accuracy artefact
    require_continuous: bool = True

    def notes(self) -> list[str]:
        return [
            f"T_mem  = first probe where train accuracy > {self.mem_acc:.2f} and holds "
            f"for {self.sustain} consecutive probes",
            f"T_grok = first probe where held-out accuracy > {self.grok_acc:.2f} and "
            f"holds for {self.sustain} consecutive probes",
            f"delayed generalisation requires T_grok > {self.delay_factor:g} x T_mem",
            "and requires the continuous metrics (held-out cross-entropy, P(correct), "
            "correct-minus-runner-up logit margin) to move at the same time; otherwise "
            "the jump is an accuracy-threshold artefact and is reported as such",
        ]


CRITERIA = EmergenceCriteria()

#: Continuous quantities recorded at every probe, on both the taught and held-out sets.
#: Accuracy is discrete and jumps when the argmax crosses; these do not.
CONTINUOUS_METRICS: tuple[str, ...] = (
    "ce",          # cross-entropy
    "p_correct",   # probability mass on the correct class
    "margin",      # correct logit minus the best wrong logit
    "entropy",     # predictive entropy, a collapse detector
)

#: Internal quantities recorded alongside, so a behavioural jump can be compared with an
#: internal reorganisation rather than asserted to be one.
INTERNAL_METRICS: tuple[str, ...] = (
    "delta_abs_mean", "delta_abs_max",       # edge-gain magnitude
    "bias_abs_mean",
    "h_peak", "h_mean",                      # activity scale
    "activity_dim",                          # effective dimensionality of the state
    "probe_acc_a", "probe_acc_b", "probe_acc_sum",   # decodability of intermediates
)
