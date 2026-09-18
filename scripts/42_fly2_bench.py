"""Where does a whole-brain update actually spend its time?

A 26-million-edge SpMM over a 32-step unroll is the dominant cost of fly2, and the first
implementation ran at 3.7 s per update -- three orders of magnitude off what the GPU can do.
This measures the pieces separately so the fix is chosen from data:

    forward only          one SpMM plus the state update
    forward + backward    the full training step
    chunk sweep           the edge-wise weight gradient at different chunk sizes
    batch sweep           how the cost scales with batch, which tells whether it is
                          bandwidth-bound (linear) or launch-bound (flat)

    python scripts/42_fly2_bench.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from fly2 import data as f2data  # noqa: E402
from fly2 import sparse  # noqa: E402


def timeit(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", default="4,8,16,32")
    args = ap.parse_args()
    log = get_logger("fly2.bench")
    dev = torch.device(args.device)

    d = f2data.load(paths.DATA_PROCESSED / "fly2")
    pre = torch.as_tensor(d.pre.astype("int64"))
    post = torch.as_tensor(d.post.astype("int64"))
    w = torch.as_tensor(d.weight.astype("float32"))
    sign = torch.as_tensor(d.edge_sign.astype("float32"))
    ew = sparse.EdgeWeights(pre, post, d.n, device=dev)
    base = sparse.base_weights(pre, post, w, d.n, w_scale=0.5, sign=sign,
                               device=dev)[ew.order]
    log.info("%d neurons, %d edges (%.1f M)", d.n, ew.n_edges, ew.n_edges / 1e6)

    for batch in [int(b) for b in args.batches.split(",")]:
        values = base.clone().requires_grad_(True)
        h = torch.rand(d.n, batch, device=dev)

        def fwd():
            return ew.matmul(values, h)

        t_fwd = timeit(fwd)
        y = fwd()
        t_bwd = timeit(lambda: y.sum().backward(retain_graph=True))

        def full():
            out = ew.matmul(values, h)
            out.sum().backward()

        t_full = timeit(full)
        log.info("batch %3d | forward %7.1f ms | backward %7.1f ms | step %7.1f ms "
                 "| 32 steps %5.2f s", batch, 1e3 * t_fwd, 1e3 * t_bwd, 1e3 * t_full,
                 32 * t_full)
        del values, h, y
        torch.cuda.empty_cache()

    # chunk size sweep for the edge-wise weight gradient
    batch = 16
    values = base.clone().requires_grad_(True)
    h = torch.rand(d.n, batch, device=dev)
    grad_out = torch.rand(d.n, batch, device=dev)
    log.info("--- weight-gradient chunk sweep (batch %d) ---", batch)
    for chunk in (500_000, 2_000_000, 8_000_000, 26_000_000):
        sparse._CHUNK = chunk

        def gv():
            out = torch.empty_like(values)
            for start in range(0, ew.n_edges, chunk):
                stop = min(start + chunk, ew.n_edges)
                g = grad_out.index_select(0, ew.row[start:stop])
                a = h.index_select(0, ew.col[start:stop])
                out[start:stop] = (g * a).sum(dim=1)

        log.info("  chunk %9d -> %7.1f ms", chunk, 1e3 * timeit(gv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
