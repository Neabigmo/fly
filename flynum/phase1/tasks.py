"""The five Phase I tasks: what is shown, in what order, and what the answer is.

Every task is a sequence of *phases*, and every phase has one of three kinds:

``dot``
    a group of 1-7 dots, held for ``TIME["steps_operand"]`` steps
``op``
    a ``+`` or ``-`` glyph, held for ``TIME["steps_cue"]`` steps
``blank``
    an empty window, held for ``TIME["steps_gap"]`` steps

The cue length and the blank length are equal, which is the point: addition, cyclic
addition and rule selection then differ only in *what is drawn in the middle window*,
never in how long anything lasts.  One consequence is that the number of recurrent steps
per update is a property of the task, not of the condition, so comparing cells by
optimiser updates compares them by a known amount of recurrent compute.

The answers are index spaces with an offset, exactly as in the rest of the project:

===========  ========  ======  ==========================================
task         classes   offset  answer
===========  ========  ======  ==========================================
``count``    7         1       n
``add``      13        2       a + b
``cyc7``     7         0       (a + b - 2) mod 7
``addsub``   9         0       a + b, or a - b when the glyph is ``-``
``two_step`` 7         1       (a + b) - c
===========  ========  ======  ==========================================

``addsub`` only ever subtracts a smaller-or-equal second operand, so the difference is
non-negative and every answer lands in the shared 0-8 space.  ``two_step`` only presents
triples whose result is at least 1, for the same reason.

Items are tuples of operand values in the order the content phases appear, so an item
encodes exactly the numbers on screen.  A blank phase contributes nothing to an item.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..stimuli.dots import make_count_stimulus
from . import glyphs
from .spec import OPERAND_MAX, OPERAND_MIN, TIME

#: Operand range for the multi-step tasks.  Deliberately smaller than the counting
#: range: the point of A2/A3 is whether an operation can be selected and composed, so
#: the operands stay inside the range counting already covers.
SMALL_MIN = 1
SMALL_MAX = 4


@dataclass(frozen=True)
class TaskDef:
    """One task: its item set, its phase template and its answer space."""

    name: str
    n_classes: int
    label_offset: int
    #: kinds of the content phases, in order: "dot" or "op"
    content: tuple[str, ...]
    #: full phase template including blanks, in order
    template: tuple[str, ...]
    items: tuple[tuple[int, ...], ...]
    #: overrides ``TIME["steps_operand"]`` for this task's dot windows.  Counting uses it:
    #: one group, no arithmetic on it, and seven blobs to integrate
    operand_steps: int | None = None

    @property
    def label(self) -> str:
        return f"{self.name}({self.n_classes} classes, offset {self.label_offset})"

    def answer(self, item: tuple[int, ...]) -> int:
        """The class index of an item."""
        return self.answer_value(item) - self.label_offset

    def answer_value(self, item: tuple[int, ...]) -> int:
        raise NotImplementedError

    def probes(self) -> tuple[tuple[str, int, int, object], ...]:
        """Internal decodability probes: ``(name, content index, n_classes, fn)``.

        Measured at the *end of a phase*, on the same pooled readout features the answer
        head sees, by a linear probe fitted at evaluation time.  These are what make it
        possible to say whether a behavioural jump was accompanied by the intermediate
        quantities becoming linearly available, instead of asserting that it was.
        """
        return ()

    def n_steps(self) -> int:
        return sum(TIME["steps_operand"] if p == "dot" else
                   TIME["steps_cue"] if p == "op" else TIME["steps_gap"]
                   for p in self.template)

    def n_dot_phases(self) -> int:
        return sum(1 for p in self.template if p == "dot")


class _Count(TaskDef):
    def answer_value(self, item):
        return item[0]

    def probes(self):
        return (("n", 0, 7, lambda it: it[0] - 1),)


class _Add(TaskDef):
    def answer_value(self, item):
        return item[0] + item[1]

    def probes(self):
        return (("a", 0, 7, lambda it: it[0] - 1),
                ("b", 1, 7, lambda it: it[1] - 1),
                ("sum", 1, 13, lambda it: it[0] + it[1] - 2))


class _Cyc7(TaskDef):
    def answer_value(self, item):
        return (item[0] + item[1] - 2) % 7

    def probes(self):
        return (("a", 0, 7, lambda it: it[0] - 1),
                ("b", 1, 7, lambda it: it[1] - 1),
                ("residue", 1, 7, lambda it: (it[0] + it[1] - 2) % 7))


class _AddSub(TaskDef):
    def answer_value(self, item):
        a, op, b = item
        return a + b if glyphs.OPS[op] == "+" else a - b

    def probes(self):
        return (("a", 0, 4, lambda it: it[0] - 1),
                ("b", 2, 4, lambda it: it[2] - 1),
                ("answer", 2, 9, lambda it: it[0] + it[2] if glyphs.OPS[it[1]] == "+"
                 else it[0] - it[2]))


class _TwoStep(TaskDef):
    def answer_value(self, item):
        a, plus, b, minus, c = item
        assert glyphs.OPS[plus] == "+" and glyphs.OPS[minus] == "-"
        return (a + b) - c

    def probes(self):
        return (("a", 0, 4, lambda it: it[0] - 1),
                ("b", 2, 4, lambda it: it[2] - 1),
                ("c", 4, 4, lambda it: it[4] - 1),
                ("partial", 2, 7, lambda it: it[0] + it[2] - 2),
                ("answer", 4, 7, lambda it: (it[0] + it[2]) - it[4] - 1))


def _build() -> dict[str, TaskDef]:
    out: dict[str, TaskDef] = {}

    # counting: a single dot group, read out at the end of its window
    out["count"] = _Count(
        name="count", n_classes=7, label_offset=OPERAND_MIN,
        content=("dot",), template=("dot",),
        items=tuple((n,) for n in range(OPERAND_MIN, OPERAND_MAX + 1)),
        operand_steps=TIME["steps_count"],
    )

    # addition and cyclic addition share a template exactly: operand, delay, operand
    add_items = tuple((a, b) for a in range(OPERAND_MIN, OPERAND_MAX + 1)
                      for b in range(OPERAND_MIN, OPERAND_MAX + 1))
    out["add"] = _Add(name="add", n_classes=13, label_offset=2 * OPERAND_MIN,
                      content=("dot", "dot"), template=("dot", "blank", "dot"),
                      items=add_items)
    out["cyc7"] = _Cyc7(name="cyc7", n_classes=7, label_offset=0,
                        content=("dot", "dot"), template=("dot", "blank", "dot"),
                        items=add_items)

    # rule selection: the middle window carries the operator instead of a delay
    add_items_small = tuple((a, b) for a in range(SMALL_MIN, SMALL_MAX + 1)
                            for b in range(SMALL_MIN, SMALL_MAX + 1))
    sub_items_small = tuple((a, b) for a in range(SMALL_MIN, SMALL_MAX + 1)
                            for b in range(SMALL_MIN, a + 1))
    items = ([(a, 0, b) for a, b in add_items_small]
             + [(a, 1, b) for a, b in sub_items_small])
    out["addsub"] = _AddSub(name="addsub", n_classes=9, label_offset=0,
                            content=("dot", "op", "dot"),
                            template=("dot", "op", "dot"), items=tuple(items))

    # two-step: (a + b) - c, with both operators shown and the answer kept >= 1
    ts = [(a, 0, b, 1, c)
          for a in range(SMALL_MIN, SMALL_MAX + 1)
          for b in range(SMALL_MIN, SMALL_MAX + 1)
          for c in range(SMALL_MIN, SMALL_MAX + 1)
          if (a + b) - c >= 1]
    out["two_step"] = _TwoStep(name="two_step", n_classes=7, label_offset=1,
                               content=("dot", "op", "dot", "op", "dot"),
                               template=("dot", "op", "dot", "op", "dot"),
                               items=tuple(ts))

    for t in out.values():
        assert t.label_offset >= 0
        for it in t.items:
            ans = t.answer(it)
            if not 0 <= ans < t.n_classes:
                raise AssertionError(f"{t.name}: item {it} gives class {ans} "
                                     f"outside 0..{t.n_classes - 1}")
    return out


TASKS: dict[str, TaskDef] = _build()


# --------------------------------------------------------------------------- #
def render_phase(task: TaskDef, index: int, item: tuple[int, ...], sc,
                 rng: np.random.Generator, mode: str = "A") -> np.ndarray:
    """Render the image for content phase ``index`` of ``item``.

    An item is aligned with :attr:`TaskDef.content` -- entry ``i`` *is* the number or
    the operator drawn in content phase ``i`` -- so there is no separate lookup table to
    keep in step with the template.
    """
    kind = task.content[index]
    value = int(item[index])
    if kind == "dot":
        return make_count_stimulus(value, sc, rng, mode=mode).image
    if kind == "op":
        return glyphs.render_glyph(glyphs.OPS[value], sc, rng)
    raise ValueError(f"unknown phase kind {kind!r}")


#: Rows rendered per work unit.  Fixed, and the RNG seed of a unit depends only on its
#: first row index, so a pool is byte-identical whether it was rendered by one worker or
#: thirty-two -- parallel rendering must not change the experiment.
_BLOCK = 256


def _render_block(payload) -> np.ndarray:
    task, sc, mode, items, rows, seed = payload
    rng = np.random.default_rng(seed)
    S = int(getattr(sc, "image_size", 32))
    out = np.zeros((len(rows), len(task.content), S, S), dtype=np.float32)
    for j, item_idx in enumerate(rows):
        item = items[int(item_idx)]
        for i in range(len(task.content)):
            out[j, i] = render_phase(task, i, item, sc, rng, mode=mode)
    return out


def build_dataset(task: TaskDef, sc, *, reps_per_item: int, seed: int,
                  mode: str = "A", items: tuple[tuple[int, ...], ...] | None = None,
                  workers: int = 1) -> dict:
    """Render a set of items ``reps_per_item`` times each, shuffled.

    Returns per-item arrays plus the item table itself, so a caller can always recover
    which operand tuple a row came from -- which is what makes it possible to hold whole
    operand pairs out of a *pre-rendered* pool without any leakage.

    ``workers`` splits the work across processes.  Rendering is the one CPU-bound step in
    Phase I (measured at ~250 phase-images/s per core, and the two-step pool is 162,000 of
    them), so on a many-core machine this shortens a one-off cost that would otherwise be
    twenty minutes of a GPU sitting idle.
    """
    items = task.items if items is None else tuple(items)
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(len(items)), reps_per_item)
    rng.shuffle(rows)

    blocks = [(start, rows[start:start + _BLOCK]) for start in range(0, len(rows), _BLOCK)]
    payloads = [(task, sc, mode, items, blk, seed * 1_000_003 + start)
                for start, blk in blocks]
    if workers > 1 and len(blocks) > 1:
        import multiprocessing as mp

        with mp.Pool(processes=min(workers, len(blocks))) as pool:
            chunks = pool.map(_render_block, payloads)
    else:
        chunks = [_render_block(p) for p in payloads]
    images = (np.concatenate(chunks, axis=0) if chunks
              else np.zeros((0, len(task.content),
                             int(getattr(sc, "image_size", 32)),
                             int(getattr(sc, "image_size", 32))), dtype=np.float32))

    return {
        "images": images,
        "item_index": rows.astype(np.int64),
        "items": np.array(items, dtype=np.int64),
        "labels": np.array([task.answer(it) for it in items], dtype=np.int64)[rows],
        "n_classes": task.n_classes,
        "label_offset": task.label_offset,
        "task": task.name,
        "mode": mode,
    }


# --------------------------------------------------------------------------- #
@dataclass
class SequencePlan:
    """How to turn one dataset row into a list of per-step column activations.

    Kept separate from the renderer so the same plan serves the trainer and every
    evaluation path, and so the number of recurrent steps is computed, not guessed.
    """

    task: TaskDef
    steps_operand: int = field(default_factory=lambda: TIME["steps_operand"])
    steps_cue: int = field(default_factory=lambda: TIME["steps_cue"])
    steps_gap: int = field(default_factory=lambda: TIME["steps_gap"])

    def __post_init__(self) -> None:
        if self.task.operand_steps is not None:
            self.steps_operand = int(self.task.operand_steps)

    def phase_images(self, row: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """The ``(kind, image)`` pairs of one row, in template order."""
        out, c = [], 0
        for kind in self.task.template:
            if kind == "blank":
                out.append((kind, None))
            else:
                out.append((kind, row[c]))
                c += 1
        return out

    def lengths(self) -> list[int]:
        return [self.steps_operand if k == "dot" else
                self.steps_cue if k == "op" else self.steps_gap
                for k in self.task.template]

    def n_steps(self) -> int:
        return int(sum(self.lengths()))

    def phase_ends(self) -> list[int]:
        """Step index (exclusive) at which each phase ends, counting from 1."""
        out, acc = [], 0
        for L in self.lengths():
            acc += L
            out.append(acc)
        return out

    def content_ends(self) -> list[int]:
        """Step index (exclusive) at which each *content* phase ends.

        This is where an internal probe reads the state, because it is the last moment
        at which the quantity of interest is the most recent thing shown: operand ``a``
        at the end of the first dot window, the running sum at the end of the second.
        """
        out, acc = [], 0
        for kind, L in zip(self.task.template, self.lengths()):
            acc += L
            if kind != "blank":
                out.append(acc)
        return out
