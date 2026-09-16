"""Sparse linear operator with correct, memory-safe gradients.

Why this module exists
----------------------
``torch.sparse.mm(W, h)`` cannot be used directly when ``W.values()`` requires a
gradient: PyTorch's backward computes the dense outer product ``grad_out @ h^T``
and therefore allocates ``N^2`` floats (measured: **34.47 GiB at N = 96,194**).
It is also slow in COO format (measured 45.61 ms for a 12.5 M-nonzero product
that takes **1.90 ms** in CSR -- a 24x difference).

:class:`SparseLinFn` therefore implements the operator by hand:

forward
    ``out = W @ h`` using cuSPARSE CSR SpMM.

backward
    ``grad_h = Wᵀ @ grad_out`` (one CSR SpMM with the transposed pattern), and
    ``grad_w[e] = <grad_out[post_e], h[pre_e]>`` computed as an elementwise
    row-dot, which costs ``O(E * B)`` memory instead of ``O(N^2)``.

Numerical validation (dense reference, 3 random trials): forward error
5e-7, ``grad_w`` error 1.9e-6, ``grad_h`` error 1.4e-6.

Implementation gotcha (already burned once): when building the transposed CSR,
the new *row* index is the old *column* and the new *column* index is the old
*row*.  Swapping them silently computes ``W`` instead of ``Wᵀ``.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
def build_csr(pre: torch.Tensor, post: torch.Tensor, n: int):
    """Build CSR arrays (rows = postsynaptic neuron, cols = presynaptic).

    Returns ``(crow, col, perm)`` where ``perm[k]`` is the index, in the original
    edge list, of CSR entry ``k``.
    """
    perm = torch.argsort(post, stable=True)
    col = pre[perm].to(torch.int32)
    crow = torch.bincount(post, minlength=n).cumsum(0).to(torch.int32)
    crow = torch.cat([torch.zeros(1, dtype=torch.int32, device=pre.device), crow])
    return crow, col, perm


def build_transpose(crow: torch.Tensor, col: torch.Tensor, n: int):
    """Build the CSR pattern of ``Wᵀ``.

    Returns ``(trow, tcol, tsrt, row_of_nnz)`` where ``tsrt[j]`` maps the ``j``-th
    entry of the transposed pattern back to a CSR entry of ``W``.
    """
    row_of_nnz = torch.repeat_interleave(
        torch.arange(n, device=crow.device), torch.diff(crow).long()
    )
    new_row = col.long()          # transposed row  == original column
    new_col = row_of_nnz          # transposed col  == original row
    tsrt = torch.argsort(new_row, stable=True)
    tcol = new_col[tsrt].to(torch.int32)
    trow = torch.bincount(new_row, minlength=n).cumsum(0).to(torch.int32)
    trow = torch.cat([torch.zeros(1, dtype=torch.int32, device=crow.device), trow])
    return trow, tcol, tsrt, row_of_nnz


# --------------------------------------------------------------------------- #
class SparseLinFn(torch.autograd.Function):
    """``out = W @ h`` for a fixed sparsity pattern and trainable values."""

    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        h: torch.Tensor,
        crow: torch.Tensor,
        col: torch.Tensor,
        trow: torch.Tensor,
        tcol: torch.Tensor,
        tsrt: torch.Tensor,
        row_of_nnz: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(values, h, col, trow, tcol, tsrt, row_of_nnz)
        ctx.n = n
        w = torch.sparse_csr_tensor(crow, col, values, (n, n))
        return torch.sparse.mm(w, h)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        values, h, col, trow, tcol, tsrt, row_of_nnz = ctx.saved_tensors
        n = ctx.n

        wt = torch.sparse_csr_tensor(
            trow, tcol, values.index_select(0, tsrt), (n, n)
        )
        grad_h = torch.sparse.mm(wt, grad_out)

        # grad wrt each CSR value: row-dot of grad_out[post_e] and h[pre_e]
        grad_values = (
            grad_out.index_select(0, row_of_nnz)
            * h.index_select(0, col.long())
        ).sum(dim=1)

        return grad_values, grad_h, None, None, None, None, None, None, None


# --------------------------------------------------------------------------- #
class SparseConnectome:
    """Fixed sparsity pattern + trainable per-edge values.

    The pattern is built once and kept in CSR order; values live in the original
    edge order (so they line up with the edge list for analysis) and are
    permuted into CSR order on every forward.
    """

    def __init__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        n: int,
        device: torch.device | str = "cpu",
    ):
        self.n = int(n)
        self.n_edges = int(len(pre))
        self.device = torch.device(device)

        pre = pre.to(self.device)
        post = post.to(self.device)
        self.pre = pre
        self.post = post

        crow, col, perm = build_csr(pre, post, self.n)
        trow, tcol, tsrt, row_of_nnz = build_transpose(crow, col, self.n)

        self.crow = crow
        self.col = col
        self.perm = perm
        self.trow = trow
        self.tcol = tcol
        self.tsrt = tsrt
        self.row_of_nnz = row_of_nnz

    def to(self, device):
        dev = torch.device(device)
        for name in ("pre", "post", "crow", "col", "perm", "trow", "tcol", "tsrt", "row_of_nnz"):
            setattr(self, name, getattr(self, name).to(dev))
        self.device = dev
        return self

    def matmul(self, values_edge_order: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``W @ h`` where ``values_edge_order`` is indexed like the edge list."""
        values_csr = values_edge_order.index_select(0, self.perm)
        return SparseLinFn.apply(
            values_csr,
            h,
            self.crow,
            self.col,
            self.trow,
            self.tcol,
            self.tsrt,
            self.row_of_nnz,
            self.n,
        )

    def memory_bytes(self) -> int:
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in ("crow", "col", "perm", "trow", "tcol", "tsrt", "row_of_nnz")
        )
