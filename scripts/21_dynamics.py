"""Peak activity over time, with and without signed synapses.

The Fly-v2 claim is not "signed synapses are more biological" -- it is that they
make h(t) **bounded**, which is what any task with a delay needs.  This writes the
evidence: peak and mean |h| over 48 steps for the unsigned and signed core circuit
at several weight scales, plus the transmitter coverage those signs came from.

Runs on the CPU (a few seconds) so it can be produced while a training job owns
the GPU.  Output goes to ``data/processed/dynamics_signed.json`` and is consumed by
``plot_signed_dynamics``.

Usage::

    python scripts/21_dynamics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never compete with a training job

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402

HORIZON = 48
BATCH = 16


def measure(cfg: ExperimentConfig, steps: int, seed: int = 0) -> dict:
    """Run a constant stimulus for ``steps`` steps and record |h| statistics."""
    prep = prepare(cfg, logger=None)
    model = prep.build_model(device="cpu", seed=seed)
    model.eval()
    torch.manual_seed(seed)
    cols = torch.rand(BATCH, model.n_columns)
    peak, mean, track = [], [], []
    with torch.no_grad():
        _, history = model.forward_count(cols, steps)
    for t, h in enumerate(history, start=1):
        peak.append(float(h.abs().max()))
        mean.append(float(h.abs().mean()))
        track.append(t)
    return {
        "steps": track,
        "peak_abs": peak,
        "mean_abs": mean,
        "peak_t8": peak[min(7, len(peak) - 1)],
        "peak_final": peak[-1],
        "growth": peak[-1] / max(peak[0], 1e-30),
        "finite": bool(np.isfinite(peak).all()),
        "signed": bool(cfg.model_cfg.signed_synapses),
        "w_scale": cfg.model_cfg.w_scale,
        "sign_report": prep.sign_report,
        "n_edges": int(prep.graph.pre.size),
        "n_neurons": int(prep.subgraph.n_neurons),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="core")
    ap.add_argument("--steps", type=int, default=HORIZON, dest="steps")
    ap.add_argument("--scales", nargs="*", type=float, default=[1.0, 0.5, 0.2])
    args = ap.parse_args()

    log = get_logger("dynamics")
    out: dict = {"circuit": args.circuit, "horizon": args.steps, "records": {}}
    t0 = time.time()

    for signed in (False, True):
        for ws in args.scales:
            if not signed and ws != 1.0:
                continue  # the unsigned curve is only needed at the study's scale
            cfg = ExperimentConfig()
            cfg.data.circuit = args.circuit
            cfg.model = "M1"
            cfg.model_cfg.w_scale = ws
            cfg.model_cfg.signed_synapses = signed
            cfg.model_cfg.readout_standardize = False
            label = ("signed" if signed else "unsigned") + f" w={ws:g}"
            rec = measure(cfg, args.steps)
            out["records"][label] = rec
            if signed and rec["sign_report"]:
                out["sign_report"] = rec["sign_report"]
            log.info(
                "%-16s peak|h| t=1 %.3g -> t=8 %.3g -> t=%d %.3g (x%.3g) finite=%s",
                label, rec["peak_abs"][0], rec["peak_t8"], args.steps,
                rec["peak_final"], rec["growth"], rec["finite"],
            )

    out["wall_seconds"] = round(time.time() - t0, 1)
    path = paths.DATA_PROCESSED / f"dynamics_signed_{args.circuit}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", path)

    # The headline comparison, stated the way the report needs it.
    u = out["records"].get("unsigned w=1")
    s = out["records"].get("signed w=0.5")
    if u and s:
        log.info(
            "boundedness at t=%d: unsigned peaks at %.3g (x%.3g) while signed at "
            "w_scale=0.5 peaks at %.3g (x%.3g) -- %.3g times lower",
            args.steps, u["peak_final"], u["growth"], s["peak_final"], s["growth"],
            u["peak_final"] / max(s["peak_final"], 1e-30),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
