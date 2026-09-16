"""Dataset construction, caching and learning-curve subsetting.

Two properties matter for the science and are enforced here:

**Shared data.**  Stimuli are generated from ``data_seed`` alone.  Real and
shuffled networks therefore see byte-identical images, and any difference in
performance is attributable to the wiring.

**Nested subsets.**  The learning curve uses subsets where
``N=100 subset of N=500 subset of ... subset of N=20000`` and every subset is
exactly class-balanced.  A curve that mixes different data draws would
confound sample size with stimulus luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..config import ExperimentConfig
from ..stimuli.dots import make_add_dataset, make_count_dataset, shortcut_report

STIM_DIR = paths.DATA_PROCESSED / "stimuli"

#: Bump whenever the stimulus generator changes in a way that alters the images.
#: It is part of the cache key, so stale images are never silently reused.
STIMULUS_VERSION = "v3"

#: The training pool is generated once at a fixed size and every learning-curve
#: subset is drawn from it.  Coupling the pool to ``n_train`` would make each
#: sample size a different draw, confounding sample size with stimulus luck.
POOL_SIZE = 20000


@dataclass
class CountData:
    pool_images: np.ndarray
    pool_labels: np.ndarray
    pool_ns: np.ndarray
    pool_ink: np.ndarray
    pool_spread: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    tests: dict[str, dict]
    qc: dict

    def subset(self, n_train: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Nested, class-balanced subset of the training pool."""
        idx = stratified_subset(self.pool_labels, n_train)
        rng = np.random.default_rng(seed)
        idx = idx.copy()
        rng.shuffle(idx)
        return self.pool_images[idx], self.pool_labels[idx]


# --------------------------------------------------------------------------- #
def build_add_data(
    cfg: ExperimentConfig, *, force: bool = False, reps_per_pair: int = 250,
    logger=None,
) -> dict:
    """Build (or load) the two-epoch addition datasets.

    ``holdout`` holds only the compositional-generalisation pairs; ``train``
    holds every other ordered pair.  Each count appears in many random layouts,
    so the network has to *count* before it can add.
    """
    sc = cfg.stimulus
    key = (
        f"{STIMULUS_VERSION}_add_a{sc.a_min}-{sc.a_max}_rep{reps_per_pair}"
        f"_img{sc.image_size}"
        f"_hold{'-'.join(''.join(map(str, p)) for p in sc.holdout_pairs)}"
        f"_seed{cfg.seeds['data_seed']}"
    )
    train_path = _cache_path("addtrain", key)
    test_path = _cache_path("addtest", key)
    if train_path.exists() and test_path.exists() and not force:
        with np.load(train_path) as z:
            train = {k: z[k] for k in z.files if k != "holdout_pairs"}
        with np.load(test_path) as z:
            test = {k: z[k] for k in z.files if k != "holdout_pairs"}
        if logger:
            logger.info("addition data loaded from cache (%d train pairs items)", len(train["a"]))
    else:
        train = make_add_dataset(
            sc, seed=cfg.seeds["data_seed"] + 7, reps_per_pair=reps_per_pair, split="train"
        )
        test = make_add_dataset(
            sc, seed=cfg.seeds["data_seed"] + 7, reps_per_pair=reps_per_pair, split="test"
        )
        np.savez(train_path, **train)
        np.savez(test_path, **test)
        if logger:
            logger.info(
                "generated addition data: %d train items, %d holdout items",
                len(train["a"]), len(test["a"]),
            )
    train["holdout_pairs"] = np.array(
        ["+".join(map(str, p)) for p in sc.holdout_pairs], dtype=object
    )
    return {"train": train, "test": test}


def addition_matrix(
    a: np.ndarray, b: np.ndarray, labels: np.ndarray, pred: np.ndarray,
    a_min: int, a_max: int,
) -> dict:
    """Accuracy per (a, b) cell of the addition table."""
    out = {}
    for x in range(a_min, a_max + 1):
        for y in range(a_min, a_max + 1):
            sel = (a == x) & (b == y)
            out[f"{x}+{y}"] = {
                "n": int(sel.sum()),
                "accuracy": float((pred[sel] == labels[sel]).mean()) if sel.any() else float("nan"),
                "target_sum": x + y,
            }
    return out


# --------------------------------------------------------------------------- #
def _cache_path(tag: str, key: str) -> Path:
    STIM_DIR.mkdir(parents=True, exist_ok=True)
    return STIM_DIR / f"{tag}_{key}.npz"


def stratified_subset(labels: np.ndarray, n: int, *, shuffle_within_class: bool = False,
                      seed: int = 0) -> np.ndarray:
    """First ``n/len(classes)`` examples of every class, in pool order.

    Taking a *prefix* per class makes the subsets nested by construction.
    """
    classes = np.unique(labels)
    per_class = n // len(classes)
    if per_class * len(classes) != n:
        raise ValueError(
            f"n={n} is not divisible by the number of classes ({len(classes)})"
        )
    out = []
    rng = np.random.default_rng(seed)
    for c in classes:
        idx = np.nonzero(labels == c)[0]
        if len(idx) < per_class:
            raise ValueError(f"class {c} has only {len(idx)} samples, need {per_class}")
        if shuffle_within_class:
            idx = rng.permutation(idx)
        out.append(idx[:per_class])
    return np.sort(np.concatenate(out))


# --------------------------------------------------------------------------- #
def build_count_data(
    cfg: ExperimentConfig, *, force: bool = False, logger=None
) -> CountData:
    """Generate (or load) the count-task datasets.

    The full training pool is generated once and cached; learning-curve subsets
    are taken from it, so every run in the study sees the same pool.
    """
    sc = cfg.stimulus
    n_train = cfg.train.n_train
    pool_size = max(POOL_SIZE, n_train)
    base_key = (
        f"{STIMULUS_VERSION}_S{sc.image_size}_N{sc.n_min}-{sc.n_max}"
        f"_r{sc.radius_min}-{sc.radius_max}_g{sc.min_gap}_a{sc.area_target}"
        f"_ring{sc.ring_radius}_pool{pool_size}_seed{cfg.seeds['data_seed']}"
    )
    train_path = _cache_path("countpool", base_key)
    test_path = _cache_path("counttest", base_key)

    if train_path.exists() and test_path.exists() and not force:
        with np.load(train_path) as z:
            pool = {k: z[k] for k in z.files}
        with np.load(test_path) as z:
            flat = {k: z[k] for k in z.files}
        if logger:
            logger.info("count stimuli loaded from cache (%d pool images)", len(pool["labels"]))
    else:
        if logger:
            logger.info("generating count stimuli: pool=%d ...", pool_size)
        pool = make_count_dataset(
            pool_size, sc, seed=cfg.seeds["data_seed"], mode="A", balanced=True
        )
        np.savez(train_path, **pool)
        flat = {}
        for mode in ("A", "B", "C", "D"):
            ds = make_count_dataset(
                cfg.train.n_test,
                sc,
                seed=cfg.seeds["data_seed"] + 100 + ord(mode),
                mode=mode,
                balanced=True,
            )
            for k, v in ds.items():
                if k != "mode":
                    flat[f"{mode}_{k}"] = v
        flat["modes"] = np.array(["A", "B", "C", "D"])
        np.savez(test_path, **flat)

    val = make_count_dataset(
        cfg.train.n_val, sc, seed=cfg.seeds["data_seed"] + 999, mode="A", balanced=True
    )

    tests: dict[str, dict] = {}
    fields = ("images", "labels", "ns", "ink", "spread", "radius_mean")
    for mode in ("A", "B", "C", "D"):
        tests[mode] = {k: flat[f"{mode}_{k}"] for k in fields if f"{mode}_{k}" in flat}
        tests[mode]["mode"] = mode

    qc = {m: shortcut_report(t) for m, t in tests.items()}
    if logger:
        for m, rep in qc.items():
            logger.info(
                "stimuli %s: chance=%.2f  ink-baseline=%.3f  spread-baseline=%.3f",
                m,
                rep["chance"],
                rep["ink_centroid_accuracy"],
                rep["spread_centroid_accuracy"],
            )

    return CountData(
        pool_images=pool["images"],
        pool_labels=pool["labels"],
        pool_ns=pool["ns"],
        pool_ink=pool["ink"],
        pool_spread=pool["spread"],
        val_images=val["images"],
        val_labels=val["labels"],
        tests=tests,
        qc=qc,
    )
