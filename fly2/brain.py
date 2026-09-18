"""The whole-brain model: cell-type-aware dynamics over the real connectome.

    h(t+1) = (1 - alpha_i) h(t) + alpha_i * phi( sum_j W_ij h_j(t) + I_i(t) + b_i )

``alpha_i`` is per neuron and set by superclass: the optic lobe is fast and
feed-forward-ish, the central brain integrates over a longer window.  ``W`` is the measured
connectome with a trainable multiplier per edge, ``g = 1 + delta``, so topology is frozen
and only existing synapses change strength.  Inputs are injected at the lamina neurons
(vision, through the bilinear retinotopic map) and at the olfactory receptor neurons
(odour, through a fixed binding matrix).

Two read-outs are exposed, because the experiment needs both: the visual projection pool
(the perceptual read-out) and the MBON pool (the mushroom body's output).  Neither is a task
head during education -- education is self-supervised prediction -- they are the surfaces a
probe is fitted on afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import spec
from .sparse import EdgeWeights, base_weights


@dataclass
class BrainState:
    h: torch.Tensor
    history: list[torch.Tensor]


class WholeBrain(nn.Module):
    def __init__(self, data, *, device: torch.device | str = "cpu",
                 lesion_mb: bool = False, train_gains: bool = True,
                 seed: int = 0):
        super().__init__()
        self.n = data.n
        self.device = torch.device(device)
        self.lesion_mb = bool(lesion_mb)

        pre = torch.as_tensor(data.pre.astype("int64"))
        post = torch.as_tensor(data.post.astype("int64"))
        weight = torch.as_tensor(data.weight.astype("float32"))
        sign = torch.as_tensor(data.edge_sign.astype("float32"))
        self.edges = EdgeWeights(pre, post, self.n, device=self.device)

        # Values stay in the original edge order: SparseEdgeMatmul permutes them into CSR
        # order at the point of use, and keeping a pre-permuted tensor here would make the
        # reverse product index the wrong edges.
        base = base_weights(pre, post, weight, self.n, w_scale=spec.BRAIN["w_scale"],
                            sign=sign, device=self.device)
        if lesion_mb:
            # zero the MB's *outgoing* synapses: the neurons stay in the graph and keep
            # their other inputs, so the lesion removes one pathway rather than a node
            lesion = torch.as_tensor(
                _mb_lesion_mask(data), dtype=torch.bool).to(self.device)
            base = base.masked_fill(lesion, 0.0)
        self.register_buffer("base_weight", base, persistent=False)

        self.alpha = torch.as_tensor(data.alpha.astype("float32")).to(self.device)
        self.register_buffer("alpha_buf", self.alpha, persistent=False)
        self.bias = nn.Parameter(torch.full((self.n,), 0.1, device=self.device))
        self.delta = (torch.nn.Parameter(torch.zeros(self.edges.n_edges,
                                                     device=self.device))
                      if train_gains else None)

        self.retina = torch.as_tensor(data.retina, dtype=torch.long, device=self.device)
        self.retina_col = torch.as_tensor(data.retina_col, dtype=torch.long,
                                          device=self.device)
        self.odor = torch.as_tensor(data.odor, dtype=torch.long, device=self.device)
        self.readout_visual = torch.as_tensor(data.readout_visual, dtype=torch.long,
                                              device=self.device)
        self.readout_mb = torch.as_tensor(data.readout_mb, dtype=torch.long,
                                          device=self.device)
        self.kenyon = torch.as_tensor(data.kenyon, dtype=torch.long, device=self.device)
        self.register_buffer(
            "odour_binding", torch.as_tensor(data.odour_binding, dtype=torch.float32,
                                             device=self.device), persistent=False)
        self.register_buffer(
            "coverage", torch.as_tensor(data.coverage_mask, dtype=torch.float32,
                                        device=self.device), persistent=False)
        self.register_buffer(
            "col_u", torch.as_tensor(data.col_u, dtype=torch.float32,
                                     device=self.device), persistent=False)
        self.register_buffer(
            "col_v", torch.as_tensor(data.col_v, dtype=torch.float32,
                                     device=self.device), persistent=False)

    # ------------------------------------------------------------------ #
    @property
    def edge_values(self) -> torch.Tensor:
        return self.base_weight if self.delta is None else \
            self.base_weight * (1.0 + self.delta)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    def inject_image(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, S, S)`` images -> ``(N, B)`` current at the lamina neurons.

        The image is sampled bilinearly at each visual column's retinotopic coordinate, then
        every input neuron reads the column it belongs to.  Columns outside the image window
        are zeroed by the coverage mask, so a stimulus never leaks in through the border.
        """
        S = images.shape[-1]
        flat = images.reshape(images.shape[0], -1)
        u = self.col_u.clamp(0, S - 1)
        v = self.col_v.clamp(0, S - 1)
        u0 = u.floor().long()
        v0 = v.floor().long()
        u1 = (u0 + 1).clamp(0, S - 1)
        v1 = (v0 + 1).clamp(0, S - 1)
        du = (u - u0.float()).unsqueeze(0)
        dv = (v - v0.float()).unsqueeze(0)

        def gather(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            idx = (b * S + a).unsqueeze(0).expand(flat.shape[0], -1)
            return flat.gather(1, idx)

        cols = (gather(u0, v0) * (1 - du) * (1 - dv) + gather(u1, v0) * du * (1 - dv)
                + gather(u0, v1) * (1 - du) * dv + gather(u1, v1) * du * dv)
        cols = cols * self.coverage.unsqueeze(0)              # (B, n_columns)
        drive = torch.zeros(self.n, images.shape[0], device=images.device)
        # (n_columns, B) -> one row per input neuron, then scatter onto the lamina cells
        drive[self.retina] = cols.t()[self.retina_col]
        return drive

    def inject_odor(self, odour: torch.Tensor) -> torch.Tensor:
        """``(B, C)`` odour channels -> ``(N, B)`` current at the olfactory receptor neurons."""
        act = odour @ self.odour_binding.t()
        drive = torch.zeros(self.n, odour.shape[0], device=odour.device)
        drive[self.odor] = act.t()
        return drive

    # ------------------------------------------------------------------ #
    def run(self, injections: list[torch.Tensor]) -> BrainState:
        """Unroll the recurrence over a list of per-step input currents."""
        batch = injections[0].shape[1]
        h = torch.zeros(self.n, batch, device=self.device)
        values = self.edge_values
        bias = self.bias.unsqueeze(1)
        history: list[torch.Tensor] = []
        alpha = self.alpha.unsqueeze(1)
        for step in injections:
            drive = self.edges.matmul(values, h) + bias + step
            h = (1.0 - alpha) * h + alpha * torch.relu(drive)
            history.append(h)
        return BrainState(h=h, history=history)

    def pool(self, history: list[torch.Tensor], pool: str = "visual",
             window: int | None = None) -> torch.Tensor:
        idx = self.readout_visual if pool == "visual" else self.readout_mb
        k = window or spec.READOUT_WINDOW
        return torch.stack([h.index_select(0, idx) for h in history[-k:]]).mean(0).t()


def _mb_lesion_mask(data) -> "object":
    """Edges whose presynaptic cell is part of the mushroom body."""
    import numpy as np

    src = np.concatenate([data.kenyon, data.readout_mb, data.dopamine, data.octopamine])
    return np.isin(data.pre, src)
