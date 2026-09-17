"""How much of a counting condition is answerable without counting.

Every dot condition leaks something, and the leaks are measurable.  A nearest-centroid
classifier that sees exactly one scalar statistic of the image gives a lower bound on what
a network can achieve without ever resolving an individual dot:

``ink``
    total brightness.  Live in the natural condition, where ink grows with the number of
    dots -- this is the cue the first version of the plan accidentally taught.
``spread``
    mean distance of the dot centres from their centroid.  Live wherever dot positions are
    free, because more dots occupy more area.
``radius``
    mean per-dot radius.  Perfectly live in any constant-area condition, and unavoidably
    so: fixing the total area forces ``r proportional to 1/sqrt(n)``.  Matching the radius
    distribution instead re-opens the ink cue, so no condition is free of all three.

The number that matters for reading a result is :func:`mixture_ceiling`: the accuracy of
the best single scalar on the *taught mixture*.  A cell that scores at that level cannot
be said to have learned anything the cue did not already give away; a cell that scores
well above it in every evaluated condition cannot be explained by any one of these
statistics.
"""

from __future__ import annotations

import numpy as np

from ..stimuli import dots

#: Scalars that a classifier can read off an image, and the feature extractor for each.
CUE_FIELDS: tuple[str, ...] = ("ink", "spread", "radius_mean")


def centroid_accuracy(feature: np.ndarray, labels: np.ndarray) -> float:
    """Nearest-centroid accuracy of a single scalar (the cheapest possible baseline)."""
    classes = np.unique(labels)
    means = np.array([feature[labels == c].mean() for c in classes])
    pred = classes[np.abs(feature[:, None] - means[None, :]).argmin(1)]
    return float((pred == labels).mean())


def cue_table(sc, *, n_per_mode: int = 700, seed: int = 11,
              modes: tuple[str, ...] = ("A", "B", "C")) -> dict:
    """Per-condition accuracy of each single-scalar strategy."""
    out: dict[str, dict] = {}
    for i, mode in enumerate(modes):
        ds = dots.make_count_dataset(n_per_mode, sc, seed=seed + i, mode=mode,
                                     balanced=True)
        y = ds["labels"]
        out[mode] = {
            "chance": float(1.0 / len(np.unique(y))),
            "ink": centroid_accuracy(ds["ink"], y),
            "spread": centroid_accuracy(ds["spread"], y),
            "radius_mean": centroid_accuracy(ds["radius_mean"], y),
        }
    return out


def mixture_ceiling(table: dict, modes: tuple[str, ...]) -> dict:
    """Best single-scalar accuracy on the pooled taught mixture, by averaging.

    Averaging the per-condition accuracies is exact for a balanced mixture: each condition
    contributes half the rows, so a strategy that scores a in one and b in the other scores
    (a + b) / 2 on the pool.  A strategy could in principle do better by also detecting
    which condition it is looking at, and that is stated rather than hidden -- it is the
    residual hole in any design that must teach more than one cue condition.
    """
    best = {"cue": "", "accuracy": 0.0, "per_condition": {}}
    for cue in CUE_FIELDS:
        acc = float(np.mean([table[m][cue] for m in modes]))
        if acc > best["accuracy"]:
            best = {"cue": cue, "accuracy": acc,
                    "per_condition": {m: table[m][cue] for m in modes}}
    best["chance"] = float(np.mean([table[m]["chance"] for m in modes]))
    return best
