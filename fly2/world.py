"""The world: what the brain is shown, and what it has to predict.

A "moment" is a short visual scene plus an odour.  Education is self-supervised: nothing
is labelled, and the only target is the next moment's odour, which the brain can only get
right by reading the current scene.  The odour is generated as a function of the scene's
numerosity, so the self-supervised task *requires* numerosity -- the same computation the
probes will later ask about, but arrived at without ever being asked.

Two stimulus families are used and mixed, and both are needed:

* natural layouts, where total ink carries the count -- the cue that a first attempt at this
  project accidentally taught, because training on it alone lets a brightness detector win;
* area-controlled layouts, where total ink is constant but the per-dot radius is a perfect
  size cue (unavoidable: fixing the area forces r ~ 1/sqrt(n)).

Mixing them puts each cue in half the data.  The probe stage still measures both ceilings,
so a result can be compared against what a cue-reader could reach instead of being asserted
to beat it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import spec


@dataclass
class Moment:
    images: np.ndarray          # (B, S, S) float32
    odour: np.ndarray           # (B, C) float32
    counts: np.ndarray          # (B,) int
    mode: np.ndarray            # (B,) "A" | "B"


def render_dots(size: int, centres, radii, *, supersample: int = 4) -> np.ndarray:
    """Anti-aliased discs, area-averaged exactly as in the previous stimulus generator."""
    ss = int(supersample)
    hi = size * ss
    yy, xx = np.mgrid[0:hi, 0:hi]
    canvas = np.zeros((hi, hi), dtype=np.float32)
    for (cx, cy), r in zip(centres, radii):
        rs = r * ss
        d2 = (xx + 0.5 - cx * ss) ** 2 + (yy + 0.5 - cy * ss) ** 2
        canvas = np.maximum(canvas, (d2 <= rs * rs).astype(np.float32))
    return canvas.reshape(size, ss, size, ss).mean(axis=(1, 3)).astype(np.float32)


def _layout(n: int, radii: np.ndarray, rng, size: int) -> np.ndarray:
    """Positions for ``n`` discs: grid first, ring fallback, both guaranteed separate."""
    mid = size / 2.0 - 0.5
    limit = size / 2.0 - 1.5
    r_max = float(radii.max())
    if n == 1:
        return np.array([[mid, mid]], dtype=np.float32)
    rho_min = r_max / np.sin(np.pi / n)
    rho_max = limit - r_max
    for _ in range(40):
        rho = rng.uniform(rho_min * 1.05, max(rho_min * 1.05, rho_max))
        ang = rng.random() * 2 * np.pi + np.arange(n) * 2 * np.pi / n
        pts = np.stack([mid + rho * np.cos(ang), mid + rho * np.sin(ang)], axis=1)
        d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() >= radii.max() * 2 + spec.WORLD["min_gap"]:
            return pts.astype(np.float32)
    rho = min(max(rho_min * 1.1, rho_min), rho_max)     # ring fallback
    ang = rng.random() * 2 * np.pi + np.arange(n) * 2 * np.pi / n
    return np.stack([mid + rho * np.cos(ang), mid + rho * np.sin(ang)],
                    axis=1).astype(np.float32)


def make_moment(batch: int, rng: np.random.Generator, *, mode: str | None = None,
                counts: np.ndarray | None = None) -> Moment:
    """One batch of scenes.  Counts are given by the caller when the world has a rule."""
    w = spec.WORLD
    S = int(w["image_size"])
    if counts is None:
        counts = rng.integers(w["n_min"], w["n_max"] + 1, batch)
    counts = np.asarray(counts)
    modes = np.array([mode or ("A" if i % 2 == 0 else "B") for i in range(batch)])
    images = np.zeros((batch, S, S), dtype=np.float32)
    for i, n in enumerate(counts):
        if modes[i] == "B":
            r = float(np.sqrt(w["area_target"] / (n * np.pi)))
            radii = np.clip(r * rng.normal(1.0, 0.04, n), w["radius_min"] * 0.5,
                            w["radius_max"] * 1.5).astype(np.float32)
        else:
            radii = rng.uniform(w["radius_min"], w["radius_max"], n).astype(np.float32)
        images[i] = render_dots(S, _layout(n, radii, rng, S), radii)
        if rng.random() < 0.5:
            images[i] = images[i][:, ::-1]
    odour = np.zeros((batch, w["n_odour_channels"]), dtype=np.float32)
    odour[np.arange(batch), (counts - w["n_min"]) % w["n_odour_channels"]] = 1.0
    return Moment(images=images, odour=odour, counts=counts, mode=modes)


def scene_sequence(batch: int, rng: np.random.Generator, *, steps: int,
                   change_every: int = 4) -> tuple[list[Moment], np.ndarray, np.ndarray]:
    """A stream of scenes whose counts advance by an unknown fixed step.

    The world has a rule: ``n(t+1) = n(t) + k`` modulo the range, with ``k`` drawn per
    stream and never revealed.  What the brain is asked to predict is the odour of the next
    count -- which it can only do by reading the current scene *and* having inferred ``k``
    from the scenes so far.  The odour it is shown belongs to the previous scene, so the
    odour channel is context, never the answer.

    A first version of this asked the brain to report the count of the scene currently on
    screen.  That needs no memory, no rule and no learning centre: a linear read of the
    visual stream saturates it, which the pilot measured (MSE 0.042 within a hundred
    updates).  Predicting a count that has not been seen yet is the version that requires
    the brain to build something.
    """
    n_scenes = max(2, steps // change_every)
    lo, hi = spec.WORLD["n_min"], spec.WORLD["n_max"]
    period = hi - lo + 1
    step = rng.integers(1, period, batch)
    first = rng.integers(lo, hi + 1, batch)
    counts = np.stack([((first - lo + step * t) % period) + lo for t in range(n_scenes + 1)])

    scenes: list[Moment] = []
    zero = np.zeros((batch, spec.WORLD["n_odour_channels"]), dtype=np.float32)
    for t in range(n_scenes):
        m = make_moment(batch, rng, counts=counts[t])
        prev = _onehot(counts[t - 1]) if t > 0 else zero
        scenes.append(Moment(images=m.images, odour=prev, counts=m.counts, mode=m.mode))
    return scenes, _onehot(counts[n_scenes]), step


def _onehot(counts: np.ndarray) -> np.ndarray:
    lo = spec.WORLD["n_min"]
    out = np.zeros((len(counts), spec.WORLD["n_odour_channels"]), dtype=np.float32)
    out[np.arange(len(counts)), (counts - lo) % spec.WORLD["n_odour_channels"]] = 1.0
    return out


def to_currents(brain, rng: np.random.Generator, *, batch: int, steps: int,
                change_every: int = 4, blind: str = "", show_odour: bool = False
                ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Per-step input currents and the target odour.

    ``blind`` optionally removes one sensory channel -- ``"odour"`` or ``"vision"`` -- which
    is how the pre-flight checks that the task is not solvable from the wrong one.
    """
    device = brain.device
    scenes, target, _ = scene_sequence(batch, rng, steps=steps,
                                      change_every=change_every)
    injections = []
    for m in scenes:
        current = torch.zeros(brain.n, batch, device=device)
        if blind != "vision":
            current = current + brain.inject_image(
                torch.as_tensor(m.images, device=device))
        if show_odour and blind != "odour":
            current = current + brain.inject_odor(
                torch.as_tensor(m.odour, device=device))
        injections += [current] * change_every
    return injections[:steps], torch.as_tensor(target, device=device)
