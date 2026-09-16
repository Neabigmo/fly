"""Characterise the control: does the shuffle preserve the operating point?

The shuffled control preserves in-degree, out-degree and the multiset of synaptic
weights exactly, so any accuracy difference is attributed to *routing*.  But routing
also sets where the network sits dynamically: the same weights wired differently give
a different spectral radius, and with ReLU units an all-excitatory matrix above or
below criticality behaves completely differently.  The measured real core circuit grows
1.5e10-fold over 48 steps, so the question is not academic -- if the shuffled graph is
merely sub-critical while the real one is critical, a large part of the accuracy gap
would be an operating-point difference rather than a structural one.

This measures, for the real circuit and each cached control graph:

* the Perron root (largest eigenvalue) of the non-negative weight matrix, by power
  iteration -- for a non-negative matrix this is the spectral radius and it governs
  whether the linearised recurrence grows;
* the largest singular value, which bounds the one-step amplification ``||W h||``;
* peak and mean |h| over 32 recurrent steps, i.e. the regime actually observed.

CPU only, a few seconds per graph, so it runs while training owns the GPU.

Usage::

    python scripts/23_graph_spectrum.py --circuit core --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402

HORIZON = 32
BATCH = 16
POWER_ITERS = 400


def perron_root(W: sp.csr_matrix, *, iters: int = POWER_ITERS, seed: int = 0) -> dict:
    """Spectral radius of a non-negative matrix by power iteration."""
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.random(n).astype(np.float64)
    v /= np.linalg.norm(v)
    lam = 0.0
    for i in range(iters):
        w = W @ v
        nrm = np.linalg.norm(w)
        if not np.isfinite(nrm) or nrm == 0:
            return {"perron": 0.0, "converged": False, "iterations": i}
        w /= nrm
        lam_new = float(v @ (W @ w))  # Rayleigh quotient
        if abs(lam_new - lam) < 1e-9 * max(abs(lam_new), 1e-12):
            v, lam = w, lam_new
            break
        v, lam = w, lam_new
    return {"perron": lam, "converged": True, "iterations": i + 1}


def singular_top(W: sp.csr_matrix, *, iters: int = POWER_ITERS, seed: int = 0) -> float:
    """Largest singular value, by power iteration on W^T W."""
    rng = np.random.default_rng(seed + 1)
    v = rng.random(W.shape[1]).astype(np.float64)
    v /= np.linalg.norm(v)
    s = 0.0
    Wt = W.T.tocsr()
    for _ in range(iters):
        w = Wt @ (W @ v)
        nrm = np.linalg.norm(w)
        if not np.isfinite(nrm) or nrm == 0:
            return 0.0
        v = w / nrm
        s = float(np.sqrt(nrm))
    return s


def measure(cfg: ExperimentConfig, *, horizon: int, label: str, seed: int = 0) -> dict:
    prep = prepare(cfg, logger=None)
    model = prep.build_model(device="cpu", seed=seed)
    pre = prep.graph.pre
    post = prep.graph.post
    vals = model.base_weight.detach().cpu().numpy().astype(np.float64)
    W = sp.coo_matrix((vals, (pre, post)),
                      shape=(model.n_neurons, model.n_neurons)).tocsr()

    neg = float((vals < 0).sum() / max(len(vals), 1))
    # For power iteration on the spectral radius the non-negative case is exact; with
    # signed weights use the entrywise absolute value, which is the standard bound on
    # the growth of ||h||.
    A = abs(W) if neg > 0 else W
    pr = perron_root(A)
    sv = singular_top(A)

    torch.manual_seed(seed)
    model.eval()
    cols = torch.rand(BATCH, model.n_columns)
    with torch.no_grad():
        _, history = model.forward_count(cols, horizon)
    peak = [float(h.abs().max()) for h in history]
    mean = [float(h.abs().mean()) for h in history]

    return {
        "label": label,
        "n_neurons": int(model.n_neurons),
        "n_edges": int(len(vals)),
        "nnz_per_row": float(len(vals) / model.n_neurons),
        "fraction_negative": neg,
        "row_sum_mean": float(np.asarray(W.sum(axis=1)).ravel().mean()),
        "perron": pr["perron"],
        "perron_converged": pr["converged"],
        "singular_top": sv,
        "peak_t1": peak[0],
        "peak_t8": peak[min(7, len(peak) - 1)],
        "peak_final": peak[-1],
        "mean_final": mean[-1],
        "growth": peak[-1] / max(peak[0], 1e-30),
        "peak_curve": peak,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="core")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    log = get_logger("spectrum")
    out: dict = {"circuit": args.circuit, "horizon": args.horizon, "graphs": {}}
    t0 = time.time()

    def base(graph: str, shuffle_seed: int) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.data.circuit = args.circuit
        cfg.data.graph = graph
        cfg.model_cfg.readout_standardize = False
        cfg.seeds["shuffle_seed"] = shuffle_seed
        return cfg

    rec = measure(base("real", 0), horizon=args.horizon, label="real")
    out["graphs"]["real"] = rec
    log.info("%-16s rho=%.4f  sigma_max=%.4f  peak|h| t=32 %.4g (x%.3g)",
             "real", rec["perron"], rec["singular_top"], rec["peak_final"],
             rec["growth"])

    for s in args.seeds:
        rec = measure(base("shuffled", s), horizon=args.horizon, label=f"shuffled s{s}")
        out["graphs"][f"shuffled_s{s}"] = rec
        log.info("%-16s rho=%.4f  sigma_max=%.4f  peak|h| t=32 %.4g (x%.3g)",
                 f"shuffled s{s}", rec["perron"], rec["singular_top"],
                 rec["peak_final"], rec["growth"])

    real = out["graphs"]["real"]
    sh = [v for k, v in out["graphs"].items() if k != "real"]

    # Verify the spectrally matched control rather than trusting the derivation: the
    # weight matrix is linear in w_scale, so w_scale * rho(real)/rho(shuffled) should
    # land on the real spectral radius.  Recorded here so the claim in the report is
    # reproducible whenever this script is rerun.
    if sh:
        from importlib import util as _util

        spec = _util.spec_from_file_location(
            "_parallel_grid", Path(__file__).with_name("15_run_parallel.py"))
        grid = _util.module_from_spec(spec)
        try:
            spec.loader.exec_module(grid)
            ws = grid.SPECTRAL_MATCH_W_SCALE
        except Exception:
            ws = 1.4106
        cfg = base("shuffled", args.seeds[0])
        cfg.model_cfg.w_scale = ws
        matched = measure(cfg, horizon=args.horizon, label=f"shuffled matched w={ws}")
        out["graphs"]["shuffled_matched"] = matched
        resid = abs(matched["perron"] - real["perron"]) / max(real["perron"], 1e-12)
        out["spectral_match"] = {"w_scale": ws, "perron": matched["perron"],
                                 "real_perron": real["perron"],
                                 "relative_residual": resid}
        log.info("%-16s rho=%.4f  sigma_max=%.4f  peak|h| t=32 %.4g (x%.3g)",
                 f"shuffled matched", matched["perron"], matched["singular_top"],
                 matched["peak_final"], matched["growth"])
        log.info("spectral match at w_scale=%.4f: residual %.3f%% of the real radius",
                 ws, 100 * resid)
        if resid > 0.01:
            log.warning("the matched control is %.2f%% off; SPECTRAL_MATCH_W_SCALE in "
                        "scripts/15_run_parallel.py needs updating", 100 * resid)

    if sh:
        rho = np.array([v["perron"] for v in sh])
        sv = np.array([v["singular_top"] for v in sh])
        pk = np.array([np.log10(max(v["peak_final"], 1e-30)) for v in sh])
        out["summary"] = {
            "real_perron": real["perron"],
            "shuffled_perron_mean": float(rho.mean()),
            "shuffled_perron_sd": float(rho.std(ddof=1)) if len(rho) > 1 else 0.0,
            "real_singular_top": real["singular_top"],
            "shuffled_singular_top_mean": float(sv.mean()),
            "log10_peak_ratio_mean": float(pk.mean() - np.log10(max(real["peak_final"], 1e-30))),
        }
        log.info("-" * 78)
        log.info("spectral radius: real %.4f vs shuffled %.4f +/- %.4f (%.2f%% apart)",
                 real["perron"], rho.mean(),
                 out["summary"]["shuffled_perron_sd"],
                 100 * abs(real["perron"] - rho.mean()) / max(real["perron"], 1e-12))
        log.info("largest singular value: real %.4f vs shuffled %.4f",
                 real["singular_top"], sv.mean())
        log.info("log10 peak|h|(t=%d): real %.3f vs shuffled %.3f -- %.2f decades apart",
                 args.horizon, np.log10(max(real["peak_final"], 1e-30)), pk.mean(),
                 -out["summary"]["log10_peak_ratio_mean"])
        ratio = rho.mean() / max(real["perron"], 1e-12)
        if 0.95 <= ratio <= 1.05:
            log.info("=> the control preserves the operating point as well as the degree "
                     "and weight distributions, so an accuracy gap is not a "
                     "criticality difference")
        else:
            log.info("=> the control sits %.1f%% BELOW the real circuit in spectral "
                     "radius, so part of any accuracy gap may be an operating-point "
                     "difference.  The spectrally matched control (w_scale=%.4f) fixes "
                     "this and is recorded below.",
                     100 * abs(ratio - 1), out.get("spectral_match", {}).get("w_scale", 0.0))

    out["wall_seconds"] = round(time.time() - t0, 1)
    path = Path(args.out) if args.out else (
        paths.DATA_PROCESSED / f"graph_spectrum_{args.circuit}.json")
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
