"""Stage 0g -- calibrate the weight scale and audit signal propagation.

Two questions must be answered before any training run is worth starting:

1. **Is the network alive?**  With ``W = w_scale * log1p(n_ij) / c`` the scale
   ``w_scale`` sets how much recurrent drive a neuron receives.  Too small and
   activity dies out before reaching the readout; too large and every unit
   saturates.  We sweep ``w_scale`` and pick the largest value that keeps the
   mean activity inside a healthy band.

2. **Does the image reach the readout population in ``T`` steps?**  The optic
   lobe is a deep feed-forward chain; if the visual projection neurons are still
   silent at step 8 then no amount of training will help and the dynamics or the
   step budget must change.  We therefore measure, per superclass, how activity
   builds up over time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.data.manifest import Manifest  # noqa: E402
from flynum.logging_utils import RunContext  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402
from flynum.retina.torch_encoder import TorchRetina  # noqa: E402
from flynum.stimuli.dots import make_count_dataset  # noqa: E402

#: Sweep range.  The low end matters for ``full``: its degree heterogeneity is far
#: larger than ``core``'s (E/N 125 vs 35), so the same mean row sum produces much
#: bigger excursions and the stable scale is an order of magnitude smaller.
W_SCALE_GRID = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
#: healthy band for the mean activity at the final step
ACTIVITY_LO, ACTIVITY_HI = 0.01, 0.60
#: a unit is "dead" if it never leaves zero across the batch
DEAD_FRACTION_MAX = 0.60

#: Long-horizon stability check.  A weight scale is only accepted if, run for
#: this many steps, the activity neither explodes nor keeps growing.
STABILITY_HORIZON = 32
STABILITY_MAX = 100.0
STABILITY_GROWTH_MAX = 2.0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--circuit", default="core", choices=["core", "full"])
    ap.add_argument("--batch-size", type=int, default=64,
                    help="the 'full' circuit needs a small batch to fit in 8 GB")
    ap.add_argument(
        "--criterion", default="activity", choices=["activity", "stability"],
        help="'activity' (default) picks the largest value whose activity at the "
             "task horizon is in the healthy band -- this is what the counting "
             "grid was run with. 'stability' additionally demands boundedness "
             "over a long horizon, which is stricter and picks a much smaller "
             "value. The full stability sweep is always recorded either way, so "
             "the marginal stability of the recurrence is never hidden.",
    )
    args = ap.parse_args()

    ctx = RunContext(f"stage0_calibration_{args.circuit}")
    log = ctx.log
    log.info("=" * 78)
    log.info("CALIBRATION + SIGNAL PROPAGATION AUDIT  (circuit=%s)", args.circuit)
    log.info("=" * 78)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # w_scale is circuit-specific: the base weights are normalised so the *mean*
    # post-synaptic row sum equals w_scale, but the spread of row sums grows with
    # the circuit's degree heterogeneity (E/N is 35 for core and 125 for full).
    # Reusing the core value on 'full' diverges immediately (measured loss 109).
    cfg = ExperimentConfig()
    cfg.data.circuit = args.circuit
    prep = prepare(cfg, logger=log)
    retina = TorchRetina(prep.encoder, device=device)

    ds = make_count_dataset(args.batch_size, cfg.stimulus, seed=11, mode="A")
    images = torch.as_tensor(ds["images"], dtype=torch.float32, device=device)
    cols = retina.encode(images)

    # ---- 1. w_scale sweep --------------------------------------------- #
    # The sweep records BOTH the activity at the task horizon (t = steps) and a
    # long-horizon stability check.  Activity at t=steps alone is not a stability
    # criterion: the connectome recurrence with unsigned weights is only
    # marginally stable, so a value can look healthy at t=8 while growing 1.6x per
    # step.  Training then diverges (measured: the 'full' circuit with the value
    # chosen on the t=8 criterion alone blew up at epoch 0 with loss 109).
    sweep = []
    log.info("-" * 78)
    log.info(
        "%-9s %9s %9s %8s %7s %9s %9s %8s %s",
        "w_scale", "act(t=1)", "act(t=T)", "dead%", "sat%", "act(t=32)",
        "max(t=32)", "growth", "stable",
    )
    for ws in W_SCALE_GRID:
        cfg.model_cfg.w_scale = ws
        model = prep.build_model(device=device)
        model.eval()
        with torch.no_grad():
            _, history = model.forward_count(cols, cfg.time.steps)
            acts = torch.stack([h.mean() for h in history]).cpu().numpy()
            final = history[-1]
            dead = float((final.abs().max(dim=1).values < 1e-6).float().mean())
            sat = float((final > 0.99).float().mean())
            vpn_act = float(final.index_select(0, model.readout_idx).mean())

            _, long_hist = model.forward_count(cols, STABILITY_HORIZON)
            long_acts = torch.stack([h.mean() for h in long_hist]).cpu().numpy()
            third = STABILITY_HORIZON // 3
            growth = float(long_acts[-1] / max(long_acts[third], 1e-12))
            long_max = float(long_hist[-1].max())
        stable = bool(long_max < STABILITY_MAX and growth < STABILITY_GROWTH_MAX)
        rec = {
            "w_scale": ws,
            "activity_t1": float(acts[0]),
            "activity_final": float(acts[-1]),
            "activity_curve": acts.tolist(),
            "dead_fraction": dead,
            "saturated_fraction": sat,
            "vpn_activity": vpn_act,
            "max_activity": float(final.max()),
            "activity_long": float(long_acts[-1]),
            "max_long": long_max,
            "growth_long": growth,
            "stable": stable,
        }
        sweep.append(rec)
        log.info(
            "%-9.2f %9.4f %9.4f %7.1f%% %6.1f%% %9.4f %9.2f %8.2f %s",
            ws, rec["activity_t1"], rec["activity_final"], 100 * dead, 100 * sat,
            rec["activity_long"], long_max, growth, "yes" if stable else "NO",
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    healthy = [
        r for r in sweep
        if ACTIVITY_LO <= r["activity_final"] <= ACTIVITY_HI
        and r["dead_fraction"] <= DEAD_FRACTION_MAX
        and (r["stable"] or args.criterion == "activity")
    ]
    if healthy:
        chosen = max(healthy, key=lambda r: r["w_scale"])
        reason = (
            "largest w_scale inside the healthy activity band"
            + (f" and bounded after {STABILITY_HORIZON} steps"
               if args.criterion == "stability" else
               f" (activity criterion at the task horizon t={cfg.time.steps}; "
               f"note this value is only marginally stable)")
        )
    else:
        # fall back to the value closest to the middle of the band
        target = 0.5 * (ACTIVITY_LO + ACTIVITY_HI)
        chosen = min(sweep, key=lambda r: abs(r["activity_final"] - target))
        reason = "no healthy value found; closest to the band centre"
    log.info("-" * 78)
    log.info(
        "chosen w_scale = %.2f  (activity %.4f, dead %.1f%%, VPN %.4f, "
        "growth over %d steps %.2f, max %.2f)  [%s]",
        chosen["w_scale"], chosen["activity_final"],
        100 * chosen["dead_fraction"], chosen["vpn_activity"],
        STABILITY_HORIZON, chosen["growth_long"], chosen["max_long"], reason,
    )

    # ---- 2. signal propagation audit ---------------------------------- #
    log.info("-" * 78)
    log.info("signal propagation by superclass (mean activity per step)")
    cfg.model_cfg.w_scale = chosen["w_scale"]
    model = prep.build_model(device=device)
    model.eval()
    with torch.no_grad():
        _, history = model.forward_count(cols, cfg.time.steps)
        stack = torch.stack(history).cpu().numpy()      # (T, N, B)

    sub_ann = prep.ann.index_of(prep.subgraph.body_ids)
    superclass = prep.ann.superclass[sub_ann]
    input_mask = np.zeros(prep.subgraph.n_neurons, dtype=bool)
    input_mask[prep.input_rows] = True
    groups = ["ol_intrinsic", "visual_projection", "visual_centrifugal"]
    table = {}
    for g in groups:
        m = superclass == g
        if m.sum() == 0:
            continue
        curve = stack[:, m, :].mean(axis=(1, 2))
        table[g] = curve.tolist()
        log.info("  %-20s n=%6d | %s", g, int(m.sum()),
                 " ".join(f"{v:.3f}" for v in curve))
    curve = stack[:, input_mask, :].mean(axis=(1, 2))
    table["input_cells"] = curve.tolist()
    log.info("  %-20s n=%6d | %s", "input_cells", int(input_mask.sum()),
             " ".join(f"{v:.3f}" for v in curve))

    vpn_curve = np.asarray(table.get("visual_projection", [0.0]))
    reaches = bool(vpn_curve[-1] > max(0.02, 5 * vpn_curve[0]))
    log.info(
        "  VPN activity at t=1 %.4f -> t=T %.4f  | signal reaches readout: %s",
        float(vpn_curve[0]), float(vpn_curve[-1]), reaches,
    )

    out = {
        "device": device,
        "circuit": args.circuit,
        "criterion": args.criterion,
        "sweep": sweep,
        "chosen_w_scale": chosen["w_scale"],
        "chosen_reason": reason,
        "activity_band": [ACTIVITY_LO, ACTIVITY_HI],
        "stability_horizon": STABILITY_HORIZON,
        "stable_values": [r["w_scale"] for r in sweep if r["stable"]],
        "marginally_stable": not chosen["stable"],
        "propagation": table,
        "signal_reaches_readout": reaches,
        "steps": cfg.time.steps,
        "alpha": cfg.model_cfg.alpha,
        "nonlinearity": cfg.model_cfg.nonlinearity,
        "n_input_neurons": prep.info["n_input_neurons"],
        "n_readout_neurons": prep.info["n_readout_neurons"],
    }
    path = paths.DATA_PROCESSED / f"calibration_{args.circuit}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    man = Manifest()
    man.add("calibration", path, chosen_w_scale=chosen["w_scale"], reaches=reaches)
    man.save()
    log.info("wrote %s", path)

    ctx.finish("ok", w_scale=chosen["w_scale"], reaches_readout=reaches)
    log.info("CALIBRATION: DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
