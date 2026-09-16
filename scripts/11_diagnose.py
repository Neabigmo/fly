"""Diagnose why the frozen/taught network is or is not learning.

The first pilot reached only ~0.70 validation accuracy on the natural condition,
well below the Level-1 bar of 0.90.  Before more compute is spent, this script
isolates which factor is responsible:

* the **scale and tail** of the readout features (a heavy tail makes a linear
  readout ill-conditioned),
* the number of recurrent **steps** (the propagation audit showed the visual
  projection neurons were still rising steeply at t=8, i.e. the computation was
  being truncated),
* the **weight scale**,
* the **nonlinearity** (ReLU is unbounded; tanh bounds the state).

For each setting it reports activity percentiles and the accuracy of a
standardised linear probe on the frozen representation, which is an upper bound
on what the readout can extract from that state.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.logging_utils import RunContext  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402
from flynum.retina.torch_encoder import TorchRetina  # noqa: E402
from flynum.stimuli.dots import make_count_dataset  # noqa: E402
from flynum.train.trainer import encode  # noqa: E402


@torch.no_grad()
def pooled_features(model, retina, images, steps, device, batch=128):
    """Mean-pooled readout features, plus the raw state statistics."""
    feats, stats = [], []
    for i in range(0, len(images), batch):
        cols = encode(images[i : i + batch], retina, device)
        _, history = model.forward_count(cols, steps)
        k = model.readout_window
        pooled = torch.stack(
            [h.index_select(0, model.readout_idx) for h in history[-k:]]
        ).mean(0)
        feats.append(pooled.t().cpu().numpy())
        h = history[-1]
        stats.append(
            [
                float(h.mean()),
                float(torch.quantile(h.flatten()[:200_000], 0.5)),
                float(torch.quantile(h.flatten()[:200_000], 0.99)),
                float(h.max()),
            ]
        )
    f = np.concatenate(feats).astype(np.float32)
    s = np.array(stats).mean(0)
    return f, s


def probe_accuracy(f_tr, y_tr, f_te, y_te, *, standardise: bool, C: float = 1.0):
    if standardise:
        sc = StandardScaler().fit(f_tr)
        f_tr, f_te = sc.transform(f_tr), sc.transform(f_te)
    clf = LogisticRegression(C=C, max_iter=400, n_jobs=-1)
    clf.fit(f_tr, y_tr)
    return float((clf.predict(f_te) == y_te).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--circuit", default="core")
    ap.add_argument("--steps", nargs="*", type=int, default=[8, 12, 16])
    ap.add_argument("--w-scale", nargs="*", type=float, default=[0.6, 1.0])
    ap.add_argument("--act", nargs="*", default=["relu", "tanh"])
    args = ap.parse_args()

    ctx = RunContext("diagnose")
    log = ctx.log
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ExperimentConfig()
    cfg.data.circuit = args.circuit
    prep = prepare(cfg, logger=log)
    retina = TorchRetina(prep.encoder, device=device)

    ds = make_count_dataset(
        args.n_train + args.n_test, cfg.stimulus, seed=777, mode="A", balanced=True
    )
    images = ds["images"]
    labels = ds["labels"]
    f_tr_i, f_te_i = slice(0, args.n_train), slice(args.n_train, None)

    log.info("-" * 78)
    log.info(
        "%-6s %6s %6s | %7s %7s %7s %7s | %7s %7s %7s",
        "act", "wscale", "steps", "h_mean", "h_p50", "h_p99", "h_max",
        "f_p99", "probe", "probeZ",
    )
    rows = []
    for act, ws, steps in product(args.act, args.w_scale, args.steps):
        cfg.model_cfg.nonlinearity = act
        cfg.model_cfg.w_scale = ws
        model = prep.build_model(device=device)
        model.eval()
        f_tr, s_tr = pooled_features(model, retina, images[f_tr_i], steps, device)
        f_te, _ = pooled_features(model, retina, images[f_te_i], steps, device)
        acc_raw = probe_accuracy(
            f_tr, labels[f_tr_i], f_te, labels[f_te_i], standardise=False
        )
        acc_z = probe_accuracy(
            f_tr, labels[f_tr_i], f_te, labels[f_te_i], standardise=True
        )
        row = {
            "act": act, "w_scale": ws, "steps": steps,
            "h_mean": float(s_tr[0]),
            "h_p50": float(s_tr[1]),
            "h_p99": float(s_tr[2]),
            "h_max": float(s_tr[3]),
            "f_p99": float(np.percentile(np.abs(f_tr), 99)),
            "probe_raw": acc_raw,
            "probe_standardised": acc_z,
            "chance": 0.2,
        }
        rows.append(row)
        log.info(
            "%-6s %6.2f %6d | %7.3f %7.3f %7.3f %7.3f | %7.2f %7.4f %7.4f",
            act, ws, steps, row["h_mean"], row["h_p50"], row["h_p99"], row["h_max"],
            row["f_p99"], acc_raw, acc_z,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = paths.DATA_PROCESSED / "diagnosis.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: r["probe_standardised"])
    log.info("-" * 78)
    log.info(
        "best frozen-probe setting: act=%s w_scale=%.2f steps=%d -> %.4f "
        "(standardised) / %.4f (raw)",
        best["act"], best["w_scale"], best["steps"],
        best["probe_standardised"], best["probe_raw"],
    )
    ctx.finish("ok", best=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
