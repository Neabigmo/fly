"""Sparse recurrence with an explicit, memory-bounded backward.

The connectome is a 188,778 x 188,778 matrix with 26 million non-zeros, so two things are
non-negotiable:

* the **weight gradient must never be dense** -- 188,778 squared floats is 142 GB;
* the **forward must be a real SpMM**, not an edge-wise gather: expanding ``value * h[pre]``
  over 26M edges for a batch of 16 materialises a 1.7 GB temporary *per step*, which is what
  made the first version of this module run at 3.7 s per update.

Ordering, which is where this module can go silently wrong
----------------------------------------------------------
CSR needs the edges sorted by post-synaptic cell, and the reverse product needs them sorted
by pre-synaptic cell.  The trainable tensor is therefore kept in the **original edge order**
and permuted at the point of use:

    forward    values[order]     -> CSR (post-sorted)
    W^T grad   values[order_r]   -> CSR (pre-sorted)

Keeping a permuted tensor as the parameter instead saves one gather and costs correctness:
``order_r`` then indexes the wrong edges, so the state gradient is computed with weights
attached to other synapses.  Nothing crashes and training still runs.  ``test_fly2.py``
checks both products against a dense reference for exactly this reason.
"""

from __future__ import annotations

import torch

#: Edges per chunk when accumulating the weight gradient: bounds the temporary at
#: ``chunk * batch`` floats instead of ``n_edges * batch``.
_CHUNK = 2_000_000


class SparseEdgeMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, order, crow, col, row, order_r, crow_r, col_r, n, h):
        w = torch.sparse_csr_tensor(crow, col, values[order], size=(n, n),
                                    check_invariants=False)
        ctx.save_for_backward(values, order, crow, col, row, order_r, crow_r, col_r, h)
        ctx.n = n
        return torch.sparse.mm(w, h)

    @staticmethod
    def backward(ctx, grad_out):
        (values, order, crow, col, row, order_r, crow_r, col_r, h) = ctx.saved_tensors
        n = ctx.n
        grad_out = grad_out.contiguous()

        wr = torch.sparse_csr_tensor(crow_r, col_r, values[order_r], size=(n, n),
                                     check_invariants=False)
        grad_h = torch.sparse.mm(wr, grad_out)

        grad_fwd = torch.empty_like(values)
        for start in range(0, col.numel(), _CHUNK):
            stop = min(start + _CHUNK, col.numel())
            g = grad_out.index_select(0, row[start:stop])
            a = h.index_select(0, col[start:stop])
            grad_fwd[start:stop] = (g * a).sum(dim=1)

        # the parameter lives in original edge order, so the CSR-ordered gradient is
        # scattered back rather than returned as-is
        grad_values = torch.empty_like(values)
        grad_values[order] = grad_fwd
        return grad_values, None, None, None, None, None, None, None, None, grad_h


class EdgeWeights:
    """The frozen sparsity pattern, stored as a forward and a reverse CSR."""

    def __init__(self, pre: torch.Tensor, post: torch.Tensor, n: int,
                 device: torch.device | str = "cpu"):
        self.n = int(n)
        pre64, post64 = pre.to(dtype=torch.int64), post.to(dtype=torch.int64)
        device = torch.device(device)

        self.order = torch.argsort(post64, stable=True).to(device)
        post_sorted = post64[self.order.cpu()].to(device)
        self.col = pre64[self.order.cpu()].to(device=device).contiguous()
        self.crow = torch.zeros(self.n + 1, dtype=torch.int64, device=device)
        self.crow[1:] = torch.bincount(post_sorted, minlength=self.n).cumsum(0)
        idx = torch.arange(self.col.numel(), device=device)
        self.row = torch.searchsorted(self.crow, idx, right=True) - 1

        self.order_r = torch.argsort(pre64, stable=True).to(device)
        pre_sorted = pre64[self.order_r.cpu()].to(device)
        self.col_r = post64[self.order_r.cpu()].to(device=device).contiguous()
        self.crow_r = torch.zeros(self.n + 1, dtype=torch.int64, device=device)
        self.crow_r[1:] = torch.bincount(pre_sorted, minlength=self.n).cumsum(0)

    @property
    def n_edges(self) -> int:
        return int(self.col.numel())

    def initial_values(self) -> torch.Tensor:
        return torch.zeros(self.n_edges)

    def matmul(self, values: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return SparseEdgeMatmul.apply(values, self.order, self.crow, self.col, self.row,
                                      self.order_r, self.crow_r, self.col_r, self.n, h)


def base_weights(pre: torch.Tensor, post: torch.Tensor, weight: torch.Tensor, n: int,
                 *, w_scale: float, sign: torch.Tensor | None = None,
                 device: torch.device | str = "cpu") -> torch.Tensor:
    """``log1p(count)`` scaled so that the average neuron receives ``w_scale`` in total.

    Row normalisation, not global: the whole brain's largest in-degree is 11,530 against a
    mean of 123, and a single global divisor lets the highest-degree cells drive the
    recurrence into divergence.  Signs are applied *after* normalisation, so ``|W|`` is
    unchanged and only the sign of the drive flips -- which is what lets excitatory and
    inhibitory input cancel instead of summing.

    The returned tensor is in the given (original) edge order, matching what
    :meth:`EdgeWeights.matmul` expects.
    """
    logw = torch.log1p(weight.to(device=device, dtype=torch.float32))
    post_d = post.to(device)
    row_sum = torch.zeros(n, dtype=torch.float32, device=device).index_add_(
        0, post_d, logw.abs())
    base = logw / row_sum[post_d].clamp_min(1e-9) * float(w_scale)
    if sign is not None:
        base = base * sign.to(device=device, dtype=torch.float32)
    return base
