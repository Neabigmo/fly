"""Pre-flight gates as pure functions, so the checks themselves can be tested.

A gate that lives inline in a training script cannot be tested, and an untested check is
how a run gets started on a substrate that was never verified.  Every gate here is a
function of data that a test can synthesise -- including deliberately broken brains, so that
the gate is shown to *fail* when it should.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str


def check_drive(injections: list[torch.Tensor], retina: torch.Tensor,
                odor: torch.Tensor) -> Gate:
    """Both sensory channels must actually deliver current."""
    late = injections[min(len(injections) - 1, len(injections) // 4)]
    r = float(late[retina].abs().mean())
    o = float(late[odor].abs().mean())
    return Gate("drive variance", r > 0 and o > 0,
                f"retina |I| {r:.4f}, odour |I| {o:.4f}")


def check_dynamics(history: list[torch.Tensor], *, peak_max: float) -> Gate:
    """Activity must stay finite and bounded across the whole sequence."""
    peaks = [float(h.detach().abs().max()) for h in history]
    ok = bool(np.isfinite(peaks).all() and peaks[-1] < peak_max)
    return Gate("bounded state", ok,
                f"peak |h| {peaks[0]:.3g} -> {peaks[-1]:.3g} over {len(peaks)} steps "
                f"(limit {peak_max:g})")


def check_dead(history: list[torch.Tensor], *, max_fraction: float) -> Gate:
    """Most neurons must *vary* with the input.

    A constant bias keeps every neuron non-zero even when all weights are gone, so "output is
    exactly zero" measures nothing.  What matters is whether a neuron carries information:
    zero variance across the batch means it does not.
    """
    final = history[-1].detach()
    spread = final.std(dim=1)
    scale = final.abs().mean(dim=1).clamp_min(1e-9)
    dead = float(((spread / scale) < 1e-6).float().mean())
    return Gate("no dead tissue", dead < max_fraction,
                f"uninformative fraction {dead:.3f} (limit {max_fraction:g})")


def check_gradients(named: dict[str, torch.Tensor | None]) -> Gate:
    """Every trainable tensor that receives a gradient must receive a finite, non-zero one."""
    detail, ok = [], True
    for name, p in named.items():
        if p is None:
            detail.append(f"{name} absent")
            continue
        if p.grad is None:
            detail.append(f"{name} no grad")
            ok = False
            continue
        norm = float(p.grad.norm())
        detail.append(f"{name} {norm:.3g}")
        ok = ok and np.isfinite(norm) and norm > 0
    return Gate("gradient flow", ok, ", ".join(detail))


def check_learning(first: float, last: float, *, min_drop: float) -> Gate:
    drop = (first - last) / max(abs(first), 1e-12)
    return Gate("it learns", drop > min_drop, f"loss {first:.4f} -> {last:.4f} "
                                              f"({100 * drop:.1f}%)")


def check_blinding(loss_full: float, loss_blind: float, *, margin: float = 0.02) -> Gate:
    """Removing vision must hurt: otherwise the task is solved without the visual stream."""
    ratio = loss_blind / max(loss_full, 1e-12)
    return Gate("vision is required", ratio > 1.0 + margin,
                f"vision-blind loss {loss_blind:.4f} vs {loss_full:.4f} "
                f"(ratio {ratio:.3f}, need > {1 + margin:.2f})")


def _as_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def check_target_is_not_injected(scenes: list, target) -> Gate:
    """The answer must never appear in what the brain is shown.

    An earlier version of the world asked the brain to report the scene currently on
    screen, and a later one injected the very odour it was asked to predict.  Both are
    solvable by copying an input, both look like fast learning, and neither needs a brain.
    This gate exists so that failure mode cannot come back silently.
    """
    tgt = _as_numpy(target)
    hit = 0
    for m in scenes:
        o = np.asarray(m.odour)
        hit += int((np.abs(o - tgt).max(axis=1) < 1e-6).sum())
    return Gate("target not injected", hit == 0,
                f"{hit} (scene, trial) pairs inject the target odour")


def check_target_not_linearly_readable(inputs: list[torch.Tensor], target, *,
                                       tol: float = 0.25) -> Gate:
    """A linear read of the raw input stream must not solve the task.

    The brain is allowed to solve it; a ridge regression on the injected currents is not.
    If that regression succeeds, the task is a cue, not a computation.
    """
    x = torch.cat([i.t() for i in inputs], dim=1).detach().cpu().numpy()
    y = _as_numpy(target)
    n = len(x)
    if n < 8:
        return Gate("task needs computation", True, f"batch too small to test ({n})")
    g = np.random.default_rng(0)
    perm = g.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    xs = (x - mu) / sd
    w = np.linalg.solve(xs[tr].T @ xs[tr] + 1e-2 * np.eye(xs.shape[1]),
                        xs[tr].T @ y[tr])
    pred = np.argmax(xs[te] @ w, axis=1)
    acc = float((pred == np.argmax(y[te], axis=1)).mean())
    chance = 1.0 / y.shape[1]
    return Gate("task needs computation", acc < chance + tol,
                f"linear read of the inputs scores {acc:.3f} (chance {chance:.3f})")


def summarise(gates: list[Gate]) -> tuple[bool, list[str]]:
    bad = [g.name for g in gates if not g.ok]
    return (not bad), bad
