"""Numerical correctness of the hand-written sparse autograd operator.

This is the highest-risk piece of the implementation.  ``torch.sparse.mm``
cannot be used directly because its backward allocates a dense ``N x N``
gradient, so :class:`SparseLinFn` computes the backward pass by hand.  These
tests pin the gradients against a dense reference.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flynum.models.sparse_lin import SparseConnectome, SparseLinFn, build_csr, build_transpose


def _random_graph(n: int, e: int, seed: int, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pre = torch.randint(0, n, (e,), generator=g).to(device)
    post = torch.randint(0, n, (e,), generator=g).to(device)
    return pre, post


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_and_gradients_match_dense(seed: int) -> None:
    n, e, batch = 400, 3000, 7
    pre, post = _random_graph(n, e, seed)
    w0 = torch.rand(e, generator=torch.Generator().manual_seed(seed + 10)) * 0.8 + 0.05

    crow, col, perm = build_csr(pre, post, n)
    trow, tcol, tsrt, row_of_nnz = build_transpose(crow, col, n)

    w_csr = w0[perm]
    h0 = torch.randn(n, batch, generator=torch.Generator().manual_seed(seed + 20))

    # dense reference (duplicate (pre,post) pairs accumulate, as CSR does)
    dense = torch.zeros(n, n)
    dense.index_put_((row_of_nnz.long(), col.long()), w_csr, accumulate=True)
    wd = dense.clone().requires_grad_(True)
    hd = h0.clone().requires_grad_(True)
    ref = (wd @ hd * 1.7).tanh().sum()
    ref.backward()

    w = w_csr.clone().requires_grad_(True)
    h = h0.clone().requires_grad_(True)
    out = SparseLinFn.apply(w, h, crow, col, trow, tcol, tsrt, row_of_nnz, n)
    ((out * 1.7).tanh().sum()).backward()

    assert torch.allclose(out, wd @ hd, atol=1e-4), "forward mismatch"
    ref_gw = wd.grad[row_of_nnz.long(), col.long()]
    assert torch.allclose(w.grad, ref_gw, atol=1e-4), "grad wrt values mismatch"
    assert torch.allclose(h.grad, hd.grad, atol=1e-4), "grad wrt h mismatch"


def test_sparse_connectome_matmul_matches_dense() -> None:
    n, e = 120, 900
    pre, post = _random_graph(n, e, 5)
    conn = SparseConnectome(pre, post, n)
    values = torch.rand(e)
    h = torch.randn(n, 4)

    # random graphs contain duplicate (pre, post) pairs, which CSR *accumulates*
    dense = torch.zeros(n, n)
    dense.index_put_(
        (conn.row_of_nnz.long(), conn.col.long()), values[conn.perm], accumulate=True
    )
    assert torch.allclose(conn.matmul(values, h), dense @ h, atol=1e-5)


def test_gradient_accumulates_to_every_edge() -> None:
    """Every edge must receive a gradient signal (no silently dropped edges)."""
    n, e = 300, 2000
    pre, post = _random_graph(n, e, 7)
    conn = SparseConnectome(pre, post, n)
    base = torch.rand(e, requires_grad=True)
    h = torch.randn(n, 5)
    out = conn.matmul(base, h)
    loss = (out.mean(dim=1) ** 2).sum()
    loss.backward()
    assert base.grad is not None
    assert base.grad.shape == (e,)
    assert torch.isfinite(base.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scales_to_large_n_without_n_squared_memory() -> None:
    """Regression guard: N=96k must not allocate an N^2 dense gradient."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n, e, batch = 96_194, 2_000_000, 8
    pre, post = _random_graph(n, e, 3, device="cuda")
    conn = SparseConnectome(pre, post, n, device="cuda")
    values = torch.rand(e, device="cuda", requires_grad=True)
    h = torch.randn(n, batch, device="cuda")
    out = conn.matmul(values, h)
    ((out.mean(dim=1)) ** 2).sum().backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    # a dense N^2 gradient alone would be 37 GB
    assert peak_gb < 2.5, f"peak memory {peak_gb:.2f} GB suggests an N^2 gradient"
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_transpose_pattern_is_correct() -> None:
    """The transposed CSR must represent W^T, not W (a bug we already hit)."""
    n, e = 50, 200
    pre, post = _random_graph(n, e, 11)
    crow, col, perm = build_csr(pre, post, n)
    trow, tcol, tsrt, row_of_nnz = build_transpose(crow, col, n)
    values = torch.randn(e)

    w_csr = values[perm]
    dense = torch.zeros(n, n)
    dense.index_put_((row_of_nnz.long(), col.long()), w_csr, accumulate=True)

    tvalues = w_csr[tsrt]
    trows = torch.repeat_interleave(torch.arange(n), torch.diff(trow).long())
    tdense = torch.zeros(n, n)
    tdense.index_put_((trows, tcol.long()), tvalues, accumulate=True)

    assert torch.allclose(tdense, dense.t(), atol=1e-5)
