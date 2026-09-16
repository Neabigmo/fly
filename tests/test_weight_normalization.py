"""Weight construction and the readout normalisation guard.

Two failure modes are covered here, both of which cost real debugging time:

1. **Row-sum heterogeneity.** Under global normalisation a neuron's drive is
   proportional to its in-degree.  That is biologically meaningful, but when the
   degree distribution is extreme the highest-degree neuron dominates: in the
   ``full`` circuit its row sum is 129x the mean.  ``normalization="row"`` must
   equalise total drive exactly.

2. **Standardising features that are effectively constant.** Dividing by a
   near-zero std amplifies numerical noise; the guard must remove only genuinely
   dead features and must not discard low-variance but usable ones.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flynum.models.connectome_rnn import ConnectomeRNN, make_base_weights
from flynum.models.sparse_lin import SparseConnectome


def _heterogeneous_graph(n: int = 60, seed: int = 0):
    """A graph with deliberately extreme in-degree heterogeneity."""
    rng = np.random.default_rng(seed)
    src, dst = [], []
    # neuron 0 receives a huge fan-in, everything else a few edges
    for _ in range(500):
        src.append(int(rng.integers(1, n)))
        dst.append(0)
    for _ in range(200):
        src.append(int(rng.integers(0, n)))
        dst.append(int(rng.integers(1, n)))
    pre = torch.as_tensor(src, dtype=torch.long)
    post = torch.as_tensor(dst, dtype=torch.long)
    w = torch.as_tensor(rng.integers(1, 30, len(src)), dtype=torch.float32)
    return pre, post, w


def test_global_normalisation_preserves_drive_differences() -> None:
    pre, post, w = _heterogeneous_graph()
    base, stats = make_base_weights(pre, post, w, 60, w_scale=1.0,
                                    normalization="global")
    row = torch.zeros(60).index_add_(0, post, base)
    assert abs(float(row.mean()) - 1.0) < 1e-4, "mean row sum must equal w_scale"
    # the high-degree neuron really is driven far harder
    assert float(row.max()) > 5 * float(row.mean())
    assert stats["row_sum_heterogeneity"] > 5


def test_row_normalisation_equalises_drive_exactly() -> None:
    pre, post, w = _heterogeneous_graph()
    base, _ = make_base_weights(pre, post, w, 60, w_scale=0.7, normalization="row")
    row = torch.zeros(60).index_add_(0, post, base)
    touched = row > 0
    assert torch.allclose(row[touched], torch.full_like(row[touched], 0.7), atol=1e-5), (
        "every neuron must receive the same total weight under row normalisation"
    )


def test_unknown_normalisation_is_rejected() -> None:
    pre, post, w = _heterogeneous_graph()
    with pytest.raises(ValueError, match="normalization"):
        make_base_weights(pre, post, w, 60, normalization="magic")


def test_readout_stats_guard_keeps_low_variance_features() -> None:
    """Only genuinely dead features may be neutralised."""
    n_feat = 200
    rng = np.random.default_rng(0)
    f = rng.normal(0, 1, (500, n_feat)).astype(np.float32)
    f[:, 0] = 0.0                      # dead
    f[:, 1] = 1e-9                     # numerical noise, effectively dead
    f[:, 2] *= 1e-3                    # low variance but perfectly usable

    conn = SparseConnectome(
        torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long), 1
    )
    model = ConnectomeRNN(
        conn,
        torch.ones(1),
        n_neurons=1,
        n_classes=2,
        n_columns=1,
        input_rows=torch.tensor([0], dtype=torch.long),
        input_cols=torch.tensor([0], dtype=torch.long),
        readout_idx=torch.arange(n_feat, dtype=torch.long),
        standardize=True,
    )
    info = model.fit_readout_stats(torch.as_tensor(f))
    assert info["n_constant_features"] >= 2, "dead features must be neutralised"
    # the low-variance-but-informative feature must survive with a usable scale
    assert float(model.feat_std[2]) > 0, "low-variance feature was discarded"
    z = (torch.as_tensor(f[:, 2]) - model.feat_mean[2]) / model.feat_std[2]
    assert torch.isfinite(z).all()
    assert float(z.std()) > 0.5, "surviving feature should end up roughly unit scale"
