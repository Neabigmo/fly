"""Temporal decoding: where and when is each quantity represented?

For a trained addition network we record the recurrent state at every timestep
and fit an independent linear probe at each one to decode

* ``a``   -- the first count (must survive the delay for the task to be solvable),
* ``b``   -- the second count,
* ``a+b`` -- the sum, which is never presented but has to be produced.

The resulting curves show *when* the first count enters the representation, how
well it is held across the blank delay, and when the sum becomes readable.  This
is the analysis that distinguishes "it memorised a lookup table" from "it built
a persistent quantity representation and combined it".
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import ExperimentConfig
from ..retina.torch_encoder import TorchRetina
from ..train.trainer import encode


@torch.no_grad()
def collect_states(
    model,
    retina: TorchRetina,
    img_a: np.ndarray,
    img_b: np.ndarray,
    cfg: ExperimentConfig,
    *,
    n_samples: int = 1200,
    batch_size: int = 128,
    device="cpu",
) -> tuple[np.ndarray, dict]:
    """Return ``(T, n_readout, n_samples)`` states plus the labels."""
    n = min(n_samples, len(img_a))
    idx = np.random.default_rng(0).choice(len(img_a), size=n, replace=False)
    ia, ib = img_a[idx], img_b[idx]

    states = None
    for i in range(0, n, batch_size):
        ca = encode(ia[i : i + batch_size], retina, device)
        cb = encode(ib[i : i + batch_size], retina, device)
        blank = torch.zeros_like(ca)
        seq = (
            [ca] * cfg.time.steps_a
            + [blank] * cfg.time.steps_gap
            + [cb] * cfg.time.steps_b
        )
        _, history = model._run(seq, ca.shape[0])

        stacked = torch.stack(
            [h.index_select(0, model.readout_idx) for h in history]
        )  # (T, n_readout, B)
        block = stacked.cpu().numpy().astype(np.float32)
        if states is None:
            states = np.empty((block.shape[0], block.shape[1], n), dtype=np.float32)
        states[:, :, i : i + block.shape[2]] = block

    meta = {
        "a": (idx, cfg.stimulus.a_min),
        "steps_a": cfg.time.steps_a,
        "steps_gap": cfg.time.steps_gap,
        "steps_b": cfg.time.steps_b,
        "epoch_a": list(range(1, cfg.time.steps_a + 1)),
        "gap": list(
            range(cfg.time.steps_a + 1, cfg.time.steps_a + cfg.time.steps_gap + 1)
        ),
        "epoch_b": list(
            range(
                cfg.time.steps_a + cfg.time.steps_gap + 1,
                cfg.time.steps_a + cfg.time.steps_gap + cfg.time.steps_b + 1,
            )
        ),
    }
    return states, meta


def decode_over_time(
    states: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    train_frac: float = 0.7,
    seed: int = 0,
    max_iter: int = 400,
    C: float = 1.0,
) -> dict:
    """Linear-probe accuracy at every timestep for every target.

    ``states`` is ``(T, n_features, n_samples)``; ``targets`` maps a name to an
    integer label vector of length ``n_samples``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    T, n_feat, n = states.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    tr, te = perm[:n_train], perm[n_train:]

    out: dict[str, dict] = {"n_samples": int(n), "n_features": int(n_feat), "T": int(T)}
    for name, y in targets.items():
        y = np.asarray(y)
        curve, chance = [], 1.0 / len(np.unique(y))
        for t in range(T):
            X = states[t].T  # (n, n_feat)
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)
            clf.fit(scaler.transform(X[tr]), y[tr])
            curve.append(float((clf.predict(scaler.transform(X[te])) == y[te]).mean()))
        out[name] = {"curve": curve, "chance": chance, "final": curve[-1]}
    return out


def summarise_decoding(dec: dict, meta: dict) -> dict:
    """Add phase-wise summaries (first epoch / delay / second epoch)."""
    phases = {
        "epoch_a": meta["epoch_a"],
        "delay": meta["gap"],
        "epoch_b": meta["epoch_b"],
    }
    out = dict(dec)
    for name in ("a", "b", "sum"):
        if name not in dec:
            continue
        curve = dec[name]["curve"]
        out[f"{name}_phases"] = {
            phase: {
                "mean": float(np.mean([curve[t - 1] for t in steps if t - 1 < len(curve)])),
                "max": float(np.max([curve[t - 1] for t in steps if t - 1 < len(curve)])),
            }
            for phase, steps in phases.items()
        }
    if "a" in dec and "b" in dec:
        out["memory_retention"] = (
            out["a_phases"]["delay"]["mean"] / max(out["a_phases"]["epoch_a"]["mean"], 1e-9)
        )
    return out
