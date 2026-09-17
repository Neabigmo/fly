"""Phase I stimulus pools, and the guarantee that a withheld item never trains.

A pool is a pre-rendered, shuffled table of stimulus sequences.  Rendering happens once
per (task, mode, layout budget) and is cached, because 500,000 optimiser updates at batch
64 draw tens of millions of samples: generating them on the fly would stall the GPU on
the CPU renderer (measured at ~250 phase-images/s), while a fixed pool of ~20,000 items
is the classical memorisation-then-generalisation setting this line of work needs.

**Holdout purity is structural, not supervised.**  A pool stores the item index of every
row, so the training set is built by *selecting rows whose item is taught*, and the
withheld rows are never in it.  The assertion is checked before training starts and
recorded in the run.  Nothing else -- not the schedule, not checkpoint selection, not an
early-stopping rule -- ever reads the withheld rows; there is no early stopping in Phase I
at all, because stopping on a held-out metric is exactly how a delayed transition gets
manufactured.

Aside from the taught set, two evaluation sets are built: the withheld items (the
pre-registered test) and, when a task teaches only a sparse fact table, the items that
are neither taught nor withheld.  The last is a diagnostic -- how far does anything
generalise? -- and is never used for a threshold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .. import paths
from .spec import STIMULUS
from .tasks import TaskDef, build_dataset

CACHE = paths.DATA_PROCESSED / "stimuli_phase1"

#: Fields of the stimulus configuration that change a rendered image.  A pool rendered
#: under a different value of any of these is a different pool and must not be reused.
_RENDER_FIELDS = (
    "image_size", "n_min", "n_max", "radius_min", "radius_max", "area_target",
    "min_gap", "ring_radius", "noise_sigma", "supersample",
)


def stimulus_fingerprint(sc) -> str:
    payload = {k: getattr(sc, k) for k in _RENDER_FIELDS if hasattr(sc, k)}
    payload["spec"] = {k: v for k, v in sorted(STIMULUS.items())}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _pool_key(task: TaskDef, sc, items: tuple[tuple[int, ...], ...], *, reps: int,
              seed: int, mode: str) -> str:
    """Key on what actually determines the pixels, so add and cyc7 share one pool.

    ``add`` and ``cyc7`` present byte-identical sequences -- the same 49 operand pairs,
    delay in the middle -- and differ only in how the answer is read off.  Keying the
    cache on the item table and the template rather than the task name means they render
    once, which is both faster and one fewer place for the two tasks to drift apart.
    """
    blob = np.asarray(items, dtype=np.int64).tobytes()
    items_hash = hashlib.sha1(blob).hexdigest()[:10]
    return (f"{'-'.join(task.template)}_{items_hash}_rep{reps}_{mode}"
            f"_{stimulus_fingerprint(sc)}_seed{seed}")


def render_pool(task: TaskDef, sc, items: tuple[tuple[int, ...], ...], *, reps: int,
                seed: int, mode: str = "A", logger=None, workers: int = 1) -> dict:
    """Render (or load) the cached image pool for one item table.  Labels are not cached."""
    items = tuple(items)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{_pool_key(task, sc, items, reps=reps, seed=seed, mode=mode)}.npz"
    if path.exists():
        with np.load(path) as z:
            pool = {k: z[k] for k in z.files}
        if logger:
            logger.info("stimulus pool from cache: %s (%d rows x %d phases)",
                        path.name, len(pool["item_index"]), pool["images"].shape[1])
        return pool
    pool = build_dataset(task, sc, reps_per_item=reps, seed=seed, mode=mode, items=items,
                         workers=workers)
    np.savez(path, images=pool["images"], item_index=pool["item_index"],
             items=pool["items"])
    if logger:
        logger.info("stimulus pool rendered: %s (%d rows x %d phases, %.0f MB)",
                    path.name, pool["images"].shape[0], pool["images"].shape[1],
                    path.stat().st_size / 1e6)
    return pool


def split_of(pool: dict, task: TaskDef, *, n: int = 0, seed: int = 0) -> dict:
    """Turn a pool into an evaluation split, optionally capped at ``n`` rows.

    The item table travels with the split: an internal probe needs the operand values of
    the row it is decoding, and re-deriving them from the item index at every probe would
    be one more place to get the mapping wrong.
    """
    labels = np.array([task.answer(tuple(it)) for it in pool["items"][pool["item_index"]]],
                      dtype=np.int64)
    rows = np.arange(len(labels))
    if n and len(rows) > n:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=n, replace=False))
    return {
        "images": pool["images"][rows],
        "item_index": pool["item_index"][rows],
        "items": pool["items"],
        "labels": labels[rows],
        "n_classes": task.n_classes,
        "n": int(len(rows)),
    }


def build_pools(task: TaskDef, sc, *, taught: set[tuple[int, ...]],
                holdout: tuple[tuple[int, ...], ...], reps: int, seed: int,
                eval_items: int, logger=None, workers: int = 1) -> dict:
    """Taught / withheld / unsupported sets, with holdout purity asserted on the arrays.

    The training pool covers the whole task item table and training selects rows by
    membership, so the withheld rows are not merely unused -- they are not *reachable*
    from the training set.  Each evaluation set gets its own small pool sized so that one
    probe costs the same whether the task has 7 items or 54, which is what makes 12
    probes affordable inside a 500,000-update run.
    """
    held = set(holdout)
    taught = set(taught)
    if taught & held:
        raise AssertionError(f"{task.name}: {len(taught & held)} items are taught and "
                             f"withheld at the same time")
    unknown = taught - set(task.items)
    if unknown:
        raise AssertionError(f"{task.name}: taught items not in the task: {sorted(unknown)[:3]}")

    def eval_set(items: set[tuple[int, ...]], mode: str, salt: int) -> dict | None:
        if not items:
            return None
        ordered = tuple(it for it in task.items if it in items)
        per = max(1, int(np.ceil(eval_items / len(ordered))))
        pool = render_pool(task, sc, ordered, reps=per, seed=seed + salt, mode=mode,
                           logger=logger, workers=workers)
        return split_of(pool, task, n=eval_items, seed=seed + salt)

    unsupported = set(task.items) - taught - held
    pool = render_pool(task, sc, task.items, reps=reps, seed=seed, mode="A",
                       logger=logger, workers=workers)
    keep = np.array([tuple(it) in taught for it in pool["items"][pool["item_index"]]])
    rows = np.nonzero(keep)[0]
    train = {
        "images": pool["images"][rows],
        "item_index": pool["item_index"][rows],
        "items": pool["items"],
        "labels": np.array([task.answer(tuple(it))
                            for it in pool["items"][pool["item_index"][rows]]], dtype=np.int64),
        "n_classes": task.n_classes,
        "n": int(len(rows)),
    }

    splits = {
        "train": train,
        "taught": eval_set(taught, "A", 11),
        "taught_area": eval_set(taught, "B", 12),
        "holdout": eval_set(held, "A", 13),
        "holdout_area": eval_set(held, "B", 14),
        "unsupported": eval_set(unsupported, "A", 15),
        "unsupported_area": eval_set(unsupported, "B", 16),
    }

    # ---- purity, checked on the arrays that training will actually consume ------- #
    train_items = {tuple(it) for it in pool["items"][train["item_index"]]}
    leak = train_items & held
    if leak:
        raise AssertionError(f"{task.name}: {len(leak)} withheld items are in the "
                             f"training pool, e.g. {sorted(leak)[:3]}")
    if not train_items <= taught:
        raise AssertionError(f"{task.name}: training pool contains untaught items")
    if logger:
        logger.info(
            "pools: %d taught items -> %d training rows | %d withheld | %d unsupported "
            "| eval rows taught %d / holdout %d / unsupported %d",
            len(taught), train["n"], len(held), len(unsupported),
            splits["taught"]["n"],
            splits["holdout"]["n"] if splits["holdout"] else 0,
            splits["unsupported"]["n"] if splits["unsupported"] else 0,
        )
    return {
        "splits": splits,
        "taught_items": sorted(taught),
        "holdout_items": sorted(held),
        "unsupported_items": sorted(unsupported),
        "pool_key": _pool_key(task, sc, task.items, reps=reps, seed=seed, mode="A"),
        "stimulus_fingerprint": stimulus_fingerprint(sc),
        "n_train_rows": train["n"],
    }

