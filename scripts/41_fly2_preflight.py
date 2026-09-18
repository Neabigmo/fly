"""fly2 stage 0: the gate that has to pass before any long run.

Order matters.  The cheap correctness gates run first, on CPU, and abort the script before a
GPU is touched -- because every failure in the first fly2 attempt (a missing cache field, an
unregistered buffer, a wrong axis, weights in the wrong order, a world whose answer was on
screen) was findable without a GPU, and each one cost a training run to discover.

    phase A  correctness, no GPU:  the target is not injected, the task is not a linear read
                                   of the inputs, both sensory channels deliver current
    phase B  dynamics, one forward pass:  bounded state, no uninformative tissue,
                                   finite non-zero gradients everywhere
    phase C  a short pilot:  the loss falls, vision is required, and the measured seconds
                                   per update set the real budget
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from fly2 import data as f2data  # noqa: E402
from fly2 import gates, spec, world  # noqa: E402
from fly2.brain import WholeBrain  # noqa: E402


def readout_head(brain: WholeBrain, width: int, device) -> torch.nn.Linear:
    return torch.nn.Linear(len(brain.readout_visual), width).to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=spec.WORLD["batch"])
    ap.add_argument("--steps", type=int, default=spec.WORLD["steps"])
    ap.add_argument("--updates", type=int, default=spec.GATES["pilot_updates"])
    ap.add_argument("--blind-updates", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lesion-mb", action="store_true")
    ap.add_argument("--skip-pilot", action="store_true")
    args = ap.parse_args()

    log = get_logger("fly2.preflight")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    d = f2data.load(paths.DATA_PROCESSED / "fly2")
    log.info("=" * 88)
    log.info("FLY2 STAGE 0 | %d neurons / %d edges | device %s | %d steps x batch %d",
             d.n, d.n_edges, device, args.steps, args.batch)
    for k, v in d.summary().items():
        log.info("  %-24s %s", k, v)

    # ---------------------------------------------------------------- phase A
    log.info("-" * 88)
    log.info("PHASE A  correctness (no GPU work beyond building the module)")
    probe = WholeBrain(d, device="cpu", lesion_mb=args.lesion_mb)
    rng = np.random.default_rng(args.seed)
    scenes, target, _ = world.scene_sequence(args.batch, rng, steps=args.steps)
    injections_cpu, target_cpu = world.to_currents(
        probe, np.random.default_rng(args.seed), batch=args.batch, steps=args.steps)
    a_gates = [
        gates.check_target_is_not_injected(scenes, target),
        gates.check_target_not_linearly_readable(injections_cpu, target_cpu),
        gates.check_drive(injections_cpu, torch.as_tensor(d.retina),
                          torch.as_tensor(d.odor)),
    ]
    for g in a_gates:
        log.info("  %-24s %-4s %s", g.name, "PASS" if g.ok else "FAIL", g.detail)
    ok, bad = gates.summarise(a_gates)
    if not ok:
        log.error("STAGE 0 BLOCKED in phase A on %s -- no GPU run started", bad)
        return 1

    # ---------------------------------------------------------------- phase B
    log.info("-" * 88)
    log.info("PHASE B  dynamics (one forward pass)")
    brain = WholeBrain(d, device=device, lesion_mb=args.lesion_mb).to(device)
    log.info("  trainable parameters: %d", brain.n_trainable())
    injections, target = world.to_currents(brain, rng, batch=args.batch,
                                           steps=args.steps)
    state = brain.run(injections)
    head = readout_head(brain, target.shape[1], device)
    loss = torch.nn.functional.mse_loss(head(brain.pool(state.history)), target)
    loss.backward()
    b_gates = [
        gates.check_dynamics(state.history, peak_max=spec.GATES["h_peak_max"]),
        gates.check_dead(state.history, max_fraction=spec.GATES["dead_fraction_max"]),
        gates.check_gradients({"delta": brain.delta, "bias": brain.bias,
                               "readout": head.weight}),
    ]
    for g in b_gates:
        log.info("  %-24s %-4s %s", g.name, "PASS" if g.ok else "FAIL", g.detail)
    ok, bad = gates.summarise(b_gates)
    if not ok:
        log.error("STAGE 0 BLOCKED in phase B on %s -- no long run started", bad)
        return 1

    # ---------------------------------------------------------------- phase C
    if args.skip_pilot:
        log.info("phase C skipped by request")
        return 0
    log.info("-" * 88)
    log.info("PHASE C  pilot (%d updates) and speed", args.updates)

    def pilot(updates: int, blind: str) -> tuple[float, float, float]:
        torch.manual_seed(args.seed)
        b = WholeBrain(d, device=device, lesion_mb=args.lesion_mb).to(device)
        h = readout_head(b, 24, device)
        params = [{"params": [b.bias], "lr": spec.LEARN["lr_gain"]}]
        if b.delta is not None:
            params.append({"params": [b.delta], "lr": spec.LEARN["lr_gain"]})
        params.append({"params": list(h.parameters()), "lr": spec.LEARN["lr_readout"]})
        opt = torch.optim.AdamW(params)
        r = np.random.default_rng(args.seed)
        first, hist, t0 = None, [], time.time()
        for _ in range(updates):
            inj, tgt = world.to_currents(b, r, batch=args.batch, steps=args.steps,
                                         blind=blind)
            p = h(b.pool(b.run(inj).history))
            loss = torch.nn.functional.mse_loss(p, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [q for q in b.parameters() if q.requires_grad] + list(h.parameters()),
                spec.LEARN["grad_clip"])
            opt.step()
            hist.append(float(loss.detach()))
            first = hist[0]
        return first, float(np.mean(hist[-10:])), time.time() - t0

    first, last, dt = pilot(args.updates, "")
    speed = dt / max(args.updates, 1)
    log.info("  pilot loss %.4f -> %.4f in %.0f s (%.3f s/update; %.1f h per 50k updates)",
             first, last, dt, speed, speed * 50_000 / 3600)
    _, blind_loss, _ = pilot(args.blind_updates, "vision")
    c_gates = [
        gates.check_learning(first, last, min_drop=spec.GATES["pilot_loss_drop"]),
        gates.check_blinding(last, blind_loss),
    ]
    for g in c_gates:
        log.info("  %-24s %-4s %s", g.name, "PASS" if g.ok else "FAIL", g.detail)
    ok, bad = gates.summarise(c_gates)
    log.info("=" * 88)
    log.info("STAGE 0 %s", "PASSED" if ok else f"BLOCKED on {bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
