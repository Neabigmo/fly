"""Dot stimuli for the numerosity and addition tasks.

The single most important design constraint is that the network must not be able
to solve the task from a non-numerical visual cue.  Four test conditions are
generated:

``A`` natural
    Positions, radii and jitter all randomised.  Total ink grows with N, so a
    brightness detector can succeed here -- this is the "easy" condition.

``B`` area-controlled
    Total dot area is held constant across every N, so total ink carries no
    information about the number of items.

``C`` area- and envelope-controlled
    As B, and additionally the dots are placed on a ring of fixed radius, so the
    spatial spread of the stimulus is also matched.  Only ``N >= 2`` is
    meaningful (a single dot has no envelope).

``D`` unseen layout
    Dots are snapped to a regular 3x3 grid.  Used only at test time, to probe
    layout invariance.

Rendering uses 4x supersampling followed by box downsampling, so the total ink
of the final 32x32 image is exactly ``sum_i pi r_i^2`` and the anti-aliasing is
a true area average rather than an ad-hoc blur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("A", "B", "C", "D")


@dataclass
class CountStimulus:
    image: np.ndarray      # (S, S) float32
    n: int                 # true number of dots
    label: int             # n - n_min  (class index)
    ink: float             # total intensity = total dot area in px^2
    spread: float          # mean distance of dot centres from their centroid
    radius_mean: float
    mode: str
    attempts: int = 1      # how many draws were needed to satisfy the blob check
    blob_count: int = -1   # measured connected components

    @property
    def blob_ok(self) -> bool:
        return self.blob_count == self.n


@dataclass
class AddStimulus:
    image_a: np.ndarray
    image_b: np.ndarray
    a: int
    b: int
    label: int             # a + b - (a_min + a_min)
    is_holdout: bool


# --------------------------------------------------------------------------- #
def render_dots(
    size: int,
    centres: np.ndarray,
    radii: np.ndarray,
    *,
    supersample: int = 4,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render anti-aliased discs onto a ``size x size`` float image in [0, 1]."""
    ss = int(supersample)
    hi = size * ss
    yy, xx = np.mgrid[0:hi, 0:hi]
    canvas = np.zeros((hi, hi), dtype=np.float32)
    for (cx, cy), r in zip(centres, radii):
        rs = r * ss
        d2 = (xx + 0.5 - cx * ss) ** 2 + (yy + 0.5 - cy * ss) ** 2
        canvas = np.maximum(canvas, (d2 <= rs * rs).astype(np.float32))
    img = canvas.reshape(size, ss, size, ss).mean(axis=(1, 3)).astype(np.float32)
    if noise_sigma > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _count_blobs(image: np.ndarray, threshold: float = 0.35) -> int:
    """Number of connected components above ``threshold`` (4-connectivity)."""
    from scipy import ndimage

    _, k = ndimage.label(image > threshold)
    return int(k)


def _grid_cells(size: int) -> np.ndarray:
    """Nine evenly spaced candidate positions, well separated by construction."""
    mid = size / 2.0 - 0.5
    step = size / 4.0
    pts = [
        (mid - step, mid - step), (mid, mid - step), (mid + step, mid - step),
        (mid - step, mid), (mid, mid), (mid + step, mid),
        (mid - step, mid + step), (mid, mid + step), (mid + step, mid + step),
    ]
    return np.array(pts, dtype=np.float32)


def _ring_fallback(
    n: int, radii: np.ndarray, size: int, margin: float, rng: np.random.Generator
) -> np.ndarray:
    """Guaranteed-valid layout: ``n`` dots evenly spaced on a ring.

    For ``n`` points on a circle of radius ``rho`` the closest pair is separated
    by ``2 rho sin(pi/n)``, so choosing::

        rho >= max(radii) / sin(pi/n)        (separation satisfied)

    always yields a legal arrangement, provided the ring still fits, i.e.
    ``rho <= limit - max(radii)``.  With the configured radius range that window
    is wide (about 7.1 to 10.9 lattice units for five dots), so the fallback is
    always available.  A random rotation and a small per-dot radial jitter keep
    the layout from being degenerate.
    """
    limit = size / 2.0 - margin
    mid = size / 2.0 - 0.5
    if n == 1:
        return np.array([[mid, mid]], dtype=np.float32)

    r_max = float(radii.max())
    rho_min = r_max / np.sin(np.pi / n)
    rho_max = limit - r_max
    if rho_min > rho_max:
        raise RuntimeError(
            f"cannot place {n} dots of radius {r_max:.2f}: need ring radius "
            f">= {rho_min:.2f} but only {rho_max:.2f} is available"
        )
    rho = min(max(rho_min * 1.12, rho_min), rho_max)
    angles = rng.random() * 2 * np.pi + np.arange(n) * 2 * np.pi / n
    angles = angles + rng.normal(0.0, 0.06, n)
    radii_jitter = np.clip(rho + rng.normal(0.0, 0.05 * rho, n), 0.0, rho_max)
    return np.stack(
        [mid + radii_jitter * np.cos(angles), mid + radii_jitter * np.sin(angles)], axis=1
    ).astype(np.float32)


def _try_random_positions(
    n: int, radii: np.ndarray, size: int, margin: float, rng: np.random.Generator
) -> np.ndarray | None:
    """Rejection sampling; returns ``None`` if any dot cannot be placed."""
    limit = size / 2.0 - margin
    centres = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        r_i = radii[i]
        for _ in range(400):
            rr = np.sqrt(rng.random()) * max(limit - r_i, 0.5)
            th = rng.random() * 2 * np.pi
            cand = np.array(
                [size / 2.0 - 0.5 + rr * np.cos(th), size / 2.0 - 0.5 + rr * np.sin(th)],
                dtype=np.float32,
            )
            if i == 0 or np.all(
                np.linalg.norm(centres[:i] - cand[None, :], axis=1) >= radii[:i] + r_i
            ):
                centres[i] = cand
                break
        else:
            return None
    return centres


def _disc_positions(
    n: int, radii: np.ndarray, size: int, margin: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample non-overlapping centres inside a disc inscribed in the canvas.

    ``radii`` should already include half the required gap.  Random placement is
    attempted first; if it fails (five large dots leave almost no legal space
    for the last one) the whole layout is redrawn deterministically on a ring,
    which is guaranteed to satisfy the separation constraint.
    """
    for _ in range(3):
        centres = _try_random_positions(n, radii, size, margin, rng)
        if centres is not None:
            return centres
    return _ring_fallback(n, radii, size, margin, rng)


def _ring_positions(
    n: int, ring_radius: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Evenly spaced positions on a ring with small angular jitter."""
    angles = np.arange(n) / max(n, 1) * 2 * np.pi
    angles = angles + rng.normal(0.0, 0.18, n)
    mid = size / 2.0 - 0.5
    return np.stack(
        [mid + ring_radius * np.cos(angles), mid + ring_radius * np.sin(angles)], axis=1
    ).astype(np.float32)


def _grid_positions(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Condition D: dots on a regular grid, in an *unseen* orientation.

    A fixed subset of the 3x3 grid would make the mean dot spacing a perfect
    predictor of ``n`` (measured spread-baseline accuracy: 1.00), which would
    turn the layout-transfer test into a trivial spacing test.  The subset,
    orientation and scale are therefore randomised, so the layout stays
    grid-like while its geometry carries no information about the count.
    """
    cells = _grid_cells(size)
    mid = size / 2.0 - 0.5
    idx = rng.choice(len(cells), size=n, replace=False)
    pts = (cells[idx] - mid).astype(np.float64)
    angle = rng.choice([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]) + rng.uniform(-0.18, 0.18)
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    pts = (pts @ rot.T) * rng.uniform(0.80, 1.10) + mid
    return pts.astype(np.float32)


# --------------------------------------------------------------------------- #
def _radii_for_mode(
    n: int, cfg, mode: str, rng: np.random.Generator
) -> np.ndarray:
    if mode == "A":
        return rng.uniform(cfg.radius_min, cfg.radius_max, n).astype(np.float32)
    # B, C, D: constant total ink
    base = np.sqrt(cfg.area_target / (n * np.pi))
    radii = base * rng.uniform(0.92, 1.08, n)
    # renormalise so sum(pi r^2) == area_target exactly
    radii = radii * np.sqrt(cfg.area_target / (np.pi * np.sum(radii**2)))
    return radii.astype(np.float32)


def make_count_stimulus(
    n: int,
    cfg,
    rng: np.random.Generator,
    mode: str = "A",
    *,
    enforce: bool = True,
    max_attempts: int = 6,
) -> CountStimulus:
    """Sample one dot-count image, verified to contain exactly ``n`` blobs.

    The blob check matters: the label says "n objects", and if anti-aliasing
    merged two nearby dots the network would be trained on a mislabelled image.
    With ``min_gap`` above the 1-pixel downsampling footprint a violation is
    already rare; this loop makes it impossible.
    """
    last: CountStimulus | None = None
    for attempt in range(1, max_attempts + 1):
        stim = _render_once(n, cfg, rng, mode)
        stim.attempts = attempt
        stim.blob_count = _count_blobs(stim.image)
        if not enforce or stim.blob_ok:
            return stim
        last = stim
    assert last is not None
    return last


def _render_once(n: int, cfg, rng: np.random.Generator, mode: str) -> CountStimulus:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")
    size = cfg_image_size(cfg)
    radii = _radii_for_mode(n, cfg, mode, rng)

    if mode == "C":
        centres = _ring_positions(n, cfg.ring_radius, size, rng)
        centres = _separate(centres, radii, cfg.min_gap, size, rng)
    elif mode == "D":
        centres = _grid_positions(n, size, rng)
        centres = _separate(centres, radii, cfg.min_gap, size, rng)
    else:
        centres = _disc_positions(n, radii + cfg.min_gap / 2, size, cfg.margin, rng)

    img = render_dots(
        size,
        centres,
        radii,
        supersample=cfg.supersample,
        noise_sigma=cfg.noise_sigma,
        rng=rng,
    )
    centroid = centres.mean(0)
    spread = float(np.linalg.norm(centres - centroid[None, :], axis=1).mean())
    return CountStimulus(
        image=img,
        n=int(n),
        label=int(n - cfg.n_min),
        ink=float(img.sum()),
        spread=spread,
        radius_mean=float(radii.mean()),
        mode=mode,
    )


def cfg_image_size(cfg) -> int:
    """Stimulus canvas size (kept identical to the retina encoder's image size)."""
    return int(getattr(cfg, "image_size", 32))


def _separate(
    centres: np.ndarray, radii: np.ndarray, gap: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Push apart overlapping centres (used only for the controlled layouts)."""
    centres = centres.copy()
    n = len(centres)
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dv = centres[j] - centres[i]
                dist = float(np.linalg.norm(dv))
                need = radii[i] + radii[j] + gap
                if dist < need:
                    if dist < 1e-6:
                        dv = rng.normal(0, 1, 2).astype(np.float32)
                        dist = float(np.linalg.norm(dv)) + 1e-6
                    push = (need - dist) / 2.0 * dv / dist
                    centres[i] -= push
                    centres[j] += push
                    moved = True
        if not moved:
            break
    return np.clip(centres, 0.0, size - 1.0).astype(np.float32)


def make_count_dataset(
    n_samples: int, cfg, *, seed: int, mode: str = "A", balanced: bool = True
) -> dict:
    """Generate a balanced dot-count dataset.

    Returns a dict with ``images`` (N,S,S) float32, ``labels``, ``ns``, ``ink``,
    ``spread`` and ``radii``.
    """
    rng = np.random.default_rng(seed)
    counts = np.arange(cfg.n_min, cfg.n_max + 1)
    if balanced:
        ns = np.tile(counts, int(np.ceil(n_samples / len(counts))))[:n_samples]
        rng.shuffle(ns)
    else:
        ns = rng.integers(cfg.n_min, cfg.n_max + 1, n_samples)

    S = cfg_image_size(cfg)
    images = np.zeros((n_samples, S, S), dtype=np.float32)
    ink = np.zeros(n_samples, dtype=np.float32)
    spread = np.zeros(n_samples, dtype=np.float32)
    radius_mean = np.zeros(n_samples, dtype=np.float32)
    attempts = np.zeros(n_samples, dtype=np.int64)
    blob_ok = np.zeros(n_samples, dtype=bool)
    for i, n in enumerate(ns):
        stim = make_count_stimulus(int(n), cfg, rng, mode=mode)
        images[i] = stim.image
        ink[i] = stim.ink
        spread[i] = stim.spread
        radius_mean[i] = stim.radius_mean
        attempts[i] = stim.attempts
        blob_ok[i] = stim.blob_ok
    return {
        "images": images,
        "labels": (ns - cfg.n_min).astype(np.int64),
        "ns": ns.astype(np.int64),
        "ink": ink,
        "spread": spread,
        "radius_mean": radius_mean,
        "attempts": attempts,
        "blob_ok": blob_ok,
        "mode": mode,
    }


# --------------------------------------------------------------------------- #
def make_add_dataset(
    cfg,
    *,
    seed: int,
    reps_per_pair: int = 250,
    split: str = "train",
    a_min: int | None = None,
    a_max: int | None = None,
    holdout_pairs: tuple[tuple[int, int], ...] | None = None,
) -> dict:
    """Generate the two-epoch addition dataset.

    ``split="train"`` excludes the holdout pairs; ``split="test"`` contains only
    the holdout pairs (used for the compositional-generalisation test).  ``"all"``
    returns every ordered pair.
    """
    rng = np.random.default_rng(seed)
    a_min = cfg.a_min if a_min is None else a_min
    a_max = cfg.a_max if a_max is None else a_max
    holdout = set(cfg.holdout_pairs if holdout_pairs is None else holdout_pairs)

    pairs: list[tuple[int, int]] = []
    for a in range(a_min, a_max + 1):
        for b in range(a_min, a_max + 1):
            is_hold = (a, b) in holdout
            if split == "train" and is_hold:
                continue
            if split == "test" and not is_hold:
                continue
            pairs.extend([(a, b)] * reps_per_pair)

    S = cfg_image_size(cfg)
    n = len(pairs)
    img_a = np.zeros((n, S, S), dtype=np.float32)
    img_b = np.zeros((n, S, S), dtype=np.float32)
    av = np.zeros(n, dtype=np.int64)
    bv = np.zeros(n, dtype=np.int64)
    is_hold = np.zeros(n, dtype=bool)
    for i, (a, b) in enumerate(pairs):
        sa = make_count_stimulus(a, cfg, rng, mode="A")
        sb = make_count_stimulus(b, cfg, rng, mode="A")
        img_a[i], img_b[i] = sa.image, sb.image
        av[i], bv[i] = a, b
        is_hold[i] = (a, b) in holdout

    sum_min = 2 * a_min
    return {
        "image_a": img_a,
        "image_b": img_b,
        "a": av,
        "b": bv,
        "labels": (av + bv - sum_min).astype(np.int64),
        "sums": (av + bv).astype(np.int64),
        "is_holdout": is_hold,
        "n_classes": int(2 * a_max - sum_min + 1),
        "sum_min": int(sum_min),
        "holdout_pairs": sorted(holdout),
        "split": split,
    }


# --------------------------------------------------------------------------- #
def shortcut_report(dataset: dict, labels: np.ndarray | None = None) -> dict:
    """Quantify how much a non-numerical cue leaks the answer.

    Reports the accuracy of a nearest-centroid classifier that sees *only* the
    total ink, and (for the envelope-controlled set) the ink spread.  If those
    baselines exceed chance on a controlled condition, the control has failed.
    """
    y = dataset["labels"] if labels is None else labels
    ink = dataset["ink"]
    spread = dataset["spread"]

    def cent_clf(feat: np.ndarray, yv: np.ndarray) -> float:
        classes = np.unique(yv)
        means = np.array([feat[yv == c].mean() for c in classes])
        pred = classes[np.abs(feat[:, None] - means[None, :]).argmin(1)]
        return float((pred == yv).mean())

    classes = np.unique(y)
    chance = 1.0 / len(classes)
    if len(classes) < 2:
        return {"mode": dataset.get("mode"), "chance": 1.0}
    return {
        "mode": dataset.get("mode"),
        "chance": round(chance, 4),
        "ink_mean_by_class": {
            int(n): round(float(ink[y == i].mean()), 2)
            for i, n in enumerate(classes)
        },
        "ink_centroid_accuracy": round(cent_clf(ink, y), 4),
        "spread_centroid_accuracy": round(cent_clf(spread, y), 4),
        "ink_std_overall": round(float(ink.std()), 3),
    }
