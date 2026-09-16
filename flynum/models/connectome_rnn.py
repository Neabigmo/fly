"""Connectome-constrained recurrent network.

Dynamics (one continuous state per neuron, no spikes)::

    h(t+1) = (1 - alpha) * h(t) + alpha * phi( W h(t) + I(t) + b )

with ``W = (1 + delta) * w_scale * log1p(synapse_count) / c``.

The **topology is frozen**: zeros in ``W`` stay zero forever, because only the
per-edge multiplier ``delta`` (initialised to exactly 0) is optimised.  A
connection that the connectome does not contain can therefore never appear,
which is the biological constraint the study is built on.  A light penalty
``lambda * mean(delta^2)`` keeps the learned weights near the measured wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .sparse_lin import SparseConnectome


# --------------------------------------------------------------------------- #
def make_base_weights(
    pre: torch.Tensor,
    post: torch.Tensor,
    weight: torch.Tensor,
    n: int,
    *,
    w_scale: float = 1.0,
    normalization: str = "global",
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """``log1p(synapse_count)`` scaled into a usable range.

    ``normalization="global"`` (default)
        Divide by the **mean** post-synaptic row sum, so the average neuron
        receives total weight ``w_scale``.  This preserves the biologically
        meaningful fact that a neuron with many synapses is more strongly driven,
        and it is what the reported ``core`` results use.

    ``normalization="row"``
        Divide each neuron by **its own** row sum, so *every* neuron receives
        total weight ``w_scale``.  Necessary when the degree distribution is very
        heterogeneous: in the ``full`` circuit the highest-degree neuron has a row
        sum 129x the mean (20,694 vs 161), so under global normalisation it
        receives the equivalent of ``129 * w_scale`` and the recurrence explodes
        within the first training epoch regardless of learning rate.  ``core`` is
        far milder (max/mean = 59) and trains fine.

    The trade-off is real: row normalisation removes the drive-strength
    differences between neurons, so it is an approximation, not a strictly better
    choice.
    """
    if normalization not in ("global", "row"):
        raise ValueError(f"unknown normalization {normalization!r}")
    logw = torch.log1p(weight.to(device=device, dtype=torch.float32))
    post = post.to(device)
    row_sum = torch.zeros(n, device=device).index_add_(0, post, logw)

    if normalization == "row":
        per_neuron = row_sum.clamp_min(1e-9)
        base = logw / per_neuron[post] * float(w_scale)
        denom = float(row_sum.mean().clamp_min(1e-9))
    else:
        denom = float(row_sum.mean().clamp_min(1e-9))
        base = logw / denom * float(w_scale)

    stats = {
        "log1p_mean": float(logw.mean()),
        "log1p_std": float(logw.std()),
        "row_sum_mean_before": denom,
        "row_sum_max_before": float(row_sum.max()),
        "row_sum_heterogeneity": float(row_sum.max() / max(denom, 1e-9)),
        "normalization": normalization,
        "w_scale": float(w_scale),
        "base_mean": float(base.mean()),
        "base_max": float(base.max()),
    }
    return base.to(device), stats


# --------------------------------------------------------------------------- #
@dataclass
class ForwardDiagnostics:
    mean_activity: float
    frac_dead: float
    frac_saturated: float
    max_activity: float


class ConnectomeRNN(nn.Module):
    """Recurrent network over a frozen connectome with trainable edge gains."""

    def __init__(
        self,
        conn: SparseConnectome,
        base_weight: torch.Tensor,
        *,
        n_neurons: int,
        n_classes: int,
        n_columns: int,
        input_rows: torch.Tensor,
        input_cols: torch.Tensor,
        readout_idx: torch.Tensor,
        alpha: float = 0.2,
        nonlinearity: str = "relu",
        readout_window: int = 2,
        standardize: bool = True,
        learn_gains: bool = True,
        train_bias: bool = True,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.conn = conn
        self.n_neurons = int(n_neurons)
        self.n_columns = int(n_columns)
        self.alpha = float(alpha)
        self.readout_window = int(readout_window)
        self.learn_gains = bool(learn_gains)
        self.nonlinearity = nonlinearity
        self.standardize = bool(standardize)
        if nonlinearity not in ("relu", "tanh"):
            raise ValueError(f"unknown nonlinearity {nonlinearity!r}")

        self.register_buffer("base_weight", base_weight.to(device), persistent=False)
        self.register_buffer("input_rows", input_rows.to(device), persistent=False)
        self.register_buffer("input_cols", input_cols.to(device), persistent=False)
        self.register_buffer("readout_idx", readout_idx.to(device), persistent=False)
        n_readout = int(len(readout_idx))
        self.register_buffer(
            "feat_mean", torch.zeros(n_readout, device=device), persistent=False
        )
        self.register_buffer(
            "feat_std", torch.ones(n_readout, device=device), persistent=False
        )

        if learn_gains:
            self.delta = nn.Parameter(torch.zeros(conn.n_edges, device=device))
        else:
            self.register_parameter("delta", None)

        if train_bias:
            self.bias = nn.Parameter(torch.full((self.n_neurons,), 0.1, device=device))
        else:
            self.register_buffer(
                "bias_const", torch.full((self.n_neurons,), 0.1, device=device),
                persistent=False,
            )
            self.register_parameter("bias", None)

        # the readout pools the last ``readout_window`` steps by averaging, so
        # its input width is the number of readout neurons, not their product
        self.readout = nn.Linear(int(len(readout_idx)), int(n_classes)).to(device)

    # ------------------------------------------------------------------ #
    @property
    def edge_values(self) -> torch.Tensor:
        """Effective edge weights in edge-list order."""
        if self.delta is None:
            return self.base_weight
        return self.base_weight * (1.0 + self.delta)

    def gain_penalty(self, lam: float) -> torch.Tensor:
        if self.delta is None or lam <= 0:
            return torch.zeros((), device=self.base_weight.device)
        return lam * self.delta.pow(2).mean()

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    def _inject(self, column_values: torch.Tensor) -> torch.Tensor:
        """(B, n_cols) column activations -> (n_neurons, B) input current."""
        if column_values.shape[1] != self.n_columns:
            raise ValueError(
                f"expected {self.n_columns} column values, got {column_values.shape[1]}"
            )
        gathered = column_values.index_select(1, self.input_cols)   # (B, n_input)
        out = torch.zeros(
            self.n_neurons, column_values.shape[0],
            device=column_values.device, dtype=column_values.dtype,
        )
        out[self.input_rows] = gathered.t()
        return out

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) if self.nonlinearity == "relu" else torch.tanh(x)

    def _run(self, inputs: list[torch.Tensor], batch: int) -> tuple[torch.Tensor, list]:
        """Run the recurrence over a list of per-timestep column activations."""
        dev = self.base_weight.device
        h = torch.zeros(self.n_neurons, batch, device=dev)
        bias = self.bias if self.bias is not None else self.bias_const
        b = bias.unsqueeze(1)
        values = self.edge_values
        history = []
        for step in inputs:
            injection = self._inject(step)
            drive = self.conn.matmul(values, h) + b + injection
            h = (1.0 - self.alpha) * h + self.alpha * self._act(drive)
            history.append(h)
        return h, history

    # ------------------------------------------------------------------ #
    def pool(self, history: list[torch.Tensor]) -> torch.Tensor:
        """Mean-pool the last ``readout_window`` steps over the readout pool.

        Returns raw ``(B, n_readout)`` features, before any standardisation.
        """
        k = self.readout_window
        pooled = torch.stack(
            [h.index_select(0, self.readout_idx) for h in history[-k:]]
        )
        return pooled.mean(dim=0).t()

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Standardise (optionally) and apply the linear readout."""
        if self.standardize:
            z = (z - self.feat_mean) / self.feat_std
        return self.readout(z)

    @torch.no_grad()
    def fit_readout_stats(self, features: torch.Tensor) -> dict:
        """Freeze per-feature mean/std measured on the untrained network.

        A naive ``(z - mu) / sigma`` is dangerous here: many readout neurons
        carry no signal at all, so their ``sigma`` is essentially zero and
        dividing by it amplifies numerical noise into enormous activations
        (measured: the first epoch diverged with a cross-entropy of 37).  Any
        feature whose spread falls below a floor derived from the population
        median is therefore marked **constant**, set to exactly zero mean and
        unit scale, i.e. it contributes nothing rather than exploding.

        The floor is deliberately conservative (1e-3 of the median).  A more
        generous threshold silently discards low-variance but perfectly usable
        neurons, which measurably hurts a linear readout.
        """
        if not self.standardize or features.numel() == 0:
            return {"standardised": False}

        mean = features.mean(dim=0)
        std = features.std(dim=0)
        median_std = float(std.median())
        floor = max(median_std * 1e-6, 1e-12)
        constant = std < floor

        safe_std = torch.where(constant, torch.ones_like(std), std.clamp_min(floor))
        safe_mean = torch.where(constant, torch.zeros_like(mean), mean)
        self.feat_mean.copy_(safe_mean.to(self.feat_mean.device))
        self.feat_std.copy_(safe_std.to(self.feat_std.device))

        z = (features - safe_mean) / safe_std
        return {
            "standardised": True,
            "n_features": int(features.shape[1]),
            "n_constant_features": int(constant.sum()),
            "constant_fraction": round(float(constant.float().mean()), 4),
            "std_median": median_std,
            "std_floor": floor,
            "std_p1": float(std.quantile(0.01)),
            "std_p99": float(std.quantile(0.99)),
            "z_abs_p999": float(np.percentile(np.abs(z.numpy()), 99.9)),
            "z_abs_max": float(z.abs().max()),
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def raw_pooled_features(
        self, column_values: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """Pooled readout features with no gradient tracking."""
        was_training = self.training
        self.eval()
        _, history = self.forward_count(column_values, steps)
        out = self.pool(history)
        if was_training:
            self.train()
        return out

    def _readout(self, history: list[torch.Tensor]) -> torch.Tensor:
        return self.classify(self.pool(history))

    def forward_count(self, column_values: torch.Tensor, steps: int):
        """Constant stimulus for ``steps`` recurrent steps."""
        batch = column_values.shape[0]
        _, history = self._run([column_values] * steps, batch)
        return self._readout(history), history

    def forward_add(
        self,
        columns_a: torch.Tensor,
        columns_b: torch.Tensor,
        steps_a: int,
        steps_gap: int,
        steps_b: int,
    ):
        """Stimulus A, a blank delay, then stimulus B."""
        batch = columns_a.shape[0]
        blank = torch.zeros_like(columns_a)
        seq = (
            [columns_a] * steps_a + [blank] * steps_gap + [columns_b] * steps_b
        )
        _, history = self._run(seq, batch)
        return self._readout(history), history

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def diagnostics(self, history: list[torch.Tensor]) -> ForwardDiagnostics:
        h = history[-1]
        return ForwardDiagnostics(
            mean_activity=float(h.mean()),
            frac_dead=float((h.abs().max(dim=1).values < 1e-6).float().mean()),
            frac_saturated=float((h > 0.99).float().mean()) if self.nonlinearity == "relu" else float((h.abs() > 0.99).float().mean()),
            max_activity=float(h.max()),
        )
