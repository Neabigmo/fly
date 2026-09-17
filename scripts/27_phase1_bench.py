"""Measure Phase I throughput, then project each cell's wall clock.

Run this on whichever machine is about to train.  It does not touch the stimulus pools or
the training data at all: the cost of an update is set by the sequence length, the batch
size and the circuit, and those are what it reproduces with random column activations.  A
timing run that needed the real pools would take longer to set up than the thing it is
measuring.

    python scripts/27_phase1_bench.py                 # every task, current device
    python scripts/27_phase1_bench.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.devices import pick_device  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from flynum.phase1 import spec  # noqa: E402
from flynum.phase1.data import split_of  # noqa: E402
from flynum.phase1.run import (Cell, evaluate_split, internal_metrics,  # noqa: E402
                               make_config, phase_columns, sequence)
from flynum.phase1.tasks import TASKS, SequencePlan  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402
from flynum.retina.torch_encoder import TorchRetina  # noqa: E402
from flynum.train.trainer import make_optimizer  # noqa: E402
import torch.nn as nn  # noqa: E402

#: Where each cell is meant to run, so the projection is about the actual plan.
PLAN = {
    "A (line A)": [("C0", "count", 1), ("A1-S", "add", 1), ("A2-S", "addsub", 1),
                   ("A3-S", "two_step", 1)],
    "B (line B)": [("B1-S", "add", 1), ("B2-S", "cyc7", 1)],
}


def bench_task(model, plan: SequencePlan, cfg, *, device, batch: int, iters: int,
               warmup: int = 3) -> float:
    """Seconds per optimiser update for one task's sequence length."""
    opt = make_optimizer(model, cfg)
    lossf = nn.CrossEntropyLoss()
    cols = [torch.rand(batch, model.n_columns, device=device)
            for _ in range(len(plan.task.content))]
    y = torch.randint(0, plan.task.n_classes, (batch,), device=device)
    for i in range(warmup + iters):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        logits = model.classify(model.pool(model._run(sequence(cols, plan), batch)[1]))[0]
        loss = lossf(logits, y) + model.gain_penalty(cfg.model_cfg.lambda_gain)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def bench_probe(model, retina, plan: SequencePlan, cfg, *, device, eval_items: int,
                probe_steps: int) -> float:
    """Seconds for one full probe step: seven splits plus the internal measurements."""
    n_items = max(len(plan.task.items), 1)
    rows = max(1, int(np.ceil(eval_items / n_items)))
    images = np.random.rand(rows * n_items, len(plan.task.content),
                            cfg.stimulus.image_size, cfg.stimulus.image_size).astype(np.float32)
    pool = {"images": images, "item_index": np.repeat(np.arange(n_items), rows),
            "items": np.array(plan.task.items, dtype=np.int64)}
    split = split_of(pool, plan.task, n=eval_items, seed=0)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    evaluate_split(model, retina, split, plan, batch_size=256, device=device)
    internal_metrics(model, retina, split, plan, probe_steps=probe_steps, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="")
    ap.add_argument("--batch", type=int, default=spec.OPTIM["batch_size"])
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--eval-items", type=int, default=1500)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    log = get_logger("phase1.bench")
    device = torch.device(args.device or pick_device())
    log.info("=" * 84)
    log.info("PHASE I THROUGHPUT | device=%s | batch=%d", device, args.batch)
    if device.type == "cuda":
        log.info("  %s | %.1f GB", torch.cuda.get_device_name(device),
                 torch.cuda.get_device_properties(device).total_memory / 1e9)

    cell = Cell(run="bench", task="add", brain="scratch", budget=1, probes=(0,))
    cfg = make_config(cell)
    t0 = time.time()
    prep = prepare(cfg, logger=None, n_classes=13, label_offset=2)
    retina = TorchRetina(prep.encoder, device=device)
    log.info("  prepared core in %.1f s: %d neurons, %d edges, readout pool %d",
             time.time() - t0, prep.subgraph.n_neurons, prep.subgraph.n_edges,
             len(prep.readout_idx))
    torch.manual_seed(0)
    model = prep.build_model(device=device, n_heads=1)

    report: dict = {"device": str(device), "batch": args.batch, "tasks": {}}
    for name, task in TASKS.items():
        plan = SequencePlan(task)
        spu = bench_task(model, plan, cfg, device=device, batch=args.batch,
                         iters=args.iters)
        ptime = bench_probe(model, retina, plan, cfg, device=device,
                            eval_items=args.eval_items, probe_steps=args.probe_steps)
        report["tasks"][name] = {
            "steps": plan.n_steps(), "seconds_per_update": spu,
            "updates_per_second": 1 / spu, "seconds_per_probe": ptime,
        }
        log.info("  %-9s %2d steps | %.4f s/update (%6.2f upd/s) | %.1f s per probe",
                 name, plan.n_steps(), spu, 1 / spu, ptime)

    log.info("-" * 84)
    log.info("  %-8s %-9s %9s %7s %10s %10s %9s", "cell", "task", "budget",
             "probes", "train h", "probe h", "total h")
    for group, cells in PLAN.items():
        total = 0.0
        for run, task, count in cells:
            plan = SequencePlan(TASKS[task])
            spec_cell = [t for t in (spec.FOUNDATION + spec.LINE_A + spec.LINE_B)
                         if t.run == run][0]
            n_probes = len(spec_cell.probes)
            train_h = spec_cell.budget * report["tasks"][task]["seconds_per_update"] / 3600
            probe_h = n_probes * report["tasks"][task]["seconds_per_probe"] / 3600
            total += (train_h + probe_h) * count
            log.info("  %-8s %-9s %9d %7d %10.2f %10.2f %9.2f", run, task,
                     spec_cell.budget, n_probes, train_h, probe_h, train_h + probe_h)
        report[group] = total
        log.info("  %s planned subtotal: %.1f h", group, total)

    out = Path(args.out) if args.out else paths.DATA_PROCESSED / "phase1_bench.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
