"""The frozen-fly linear probe must carry its feature standardisation.

A previous version fitted the probe on standardised features but returned only
the bare ``nn.Linear``, so every prediction was made on *raw* features.  The
symptom was unmistakable once looked for -- validation accuracy around 0.53 while
every test condition sat at exactly chance (0.2000) -- but it is easy to miss,
because a probe that always predicts one class still reports a plausible-looking
validation number that is computed on the correct (standardised) features.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flynum.config import ExperimentConfig
from flynum.logging_utils import get_logger
from flynum.train.trainer import StandardizedProbe, probe_predict, train_probe


class _Ctx:
    """Minimal RunContext stand-in (train_probe only needs .metric and .log)."""

    def __init__(self):
        self.log = get_logger("test_probe")
        self.records = []

    def metric(self, **kw):
        self.records.append(kw)


#: Fixed class centres so that train and test describe the *same* problem.  An
#: earlier version drew fresh centres per call, which made the synthetic task
#: unlearnable and produced a confusing "the probe does not learn" failure.
_CLASS_CENTRES = np.random.default_rng(20240915).normal(0, 1, (3, 8))


def _synthetic(n: int, d: int, seed: int = 0):
    """Features whose *scale* varies across dimensions, as in the real network.

    The spread is two to three orders of magnitude, which is what the connectome
    readout actually shows; a much wider spread would make the task purely about
    rescaling and would not reflect the real feature matrix.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    centres = _CLASS_CENTRES[:, :d]
    X = centres[y] + rng.normal(0, 1, (n, d))
    X = X * (10.0 ** (np.arange(d) / 3.0))[None, :]
    return X.astype(np.float32), y.astype(np.int64)


def test_probe_carries_its_scaler() -> None:
    X_tr, y_tr = _synthetic(600, 8, seed=1)
    X_te, y_te = _synthetic(300, 8, seed=2)

    cfg = ExperimentConfig()
    cfg.train.max_epochs = 40
    cfg.seeds["model_seed"] = 0
    probe, info = train_probe(cfg, X_tr, y_tr, X_te, y_te, _Ctx())

    assert isinstance(probe, StandardizedProbe), "probe must bundle its scaler"
    pred = probe_predict(probe, X_te)
    acc = float((pred == y_te).mean())
    assert acc > 0.6, f"probe accuracy {acc:.3f} suggests raw features were used"
    assert len(np.unique(pred)) > 1, "probe predicts a constant class"

    # the bundle must reproduce what was measured during fitting
    assert abs(acc - info["best_val_acc"]) < 0.25


def test_standardized_probe_matches_manual_application() -> None:
    X_tr, y_tr = _synthetic(400, 6, seed=3)
    X_te, _ = _synthetic(200, 6, seed=4)
    cfg = ExperimentConfig()
    cfg.train.max_epochs = 25
    probe, _ = train_probe(cfg, X_tr, y_tr, X_te, y_tr[: len(X_te)], _Ctx())

    # the bundle must equal "standardise, then apply the head"
    manual = (X_te - probe.feat_mu.numpy()) / probe.feat_sd.numpy()
    bundled = probe(torch.as_tensor(X_te)).detach().numpy()
    manual_out = probe.head(torch.as_tensor(manual)).detach().numpy()
    assert np.allclose(bundled, manual_out, atol=1e-5)
