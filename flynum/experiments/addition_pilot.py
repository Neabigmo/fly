"""Can a brain that has learned to count be taught addition more easily?

A deliberately narrow 2x2 pilot, one seed, four runs::

                    taught all 16 pairs        taught 15, holding out 2+3
  from scratch      A1 Scratch-Full            B1 Scratch-Holdout
  count-pretrained  A2 Pretrained-Full         B2 Pretrained-Holdout

Nothing else varies.  Same seed, same dataset, same batch-order seeding, same
optimiser, same learning rates, same **number of optimiser updates**, same Fly-v2
dynamics, same readout initialisation.  The equal-update requirement is not a detail:
the project's own sample-efficiency result turned out to be a compute artefact, so two
arms that see different numbers of gradient steps are not comparable no matter how
identical everything else looks.

Why 2+3 specifically: the answer 5 stays reachable through 1+4, 3+2 and 4+1, which are
all taught.  Success on 2+3 is therefore composition, not "has it seen class 5".

The teaching is **explicit**: the sum objective is active from the first update, with no
operand-only warm-up lesson.  The earlier curriculum showed that count-pretraining
sharpens the operand representation (a: 0.39 -> 0.71) without producing the sum on its
own, so this pilot stops hoping the operation appears by itself and teaches it, while
still recording the operand heads to see where the bottleneck sits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..logging_utils import RunContext
from ..pipeline import prepare
from ..retina.torch_encoder import TorchRetina
from ..train.data import build_add_data
from ..train.trainer import encode, make_optimizer
from ..devices import pick_device

DEVICE = pick_device()

#: Update counts at which every metric is recorded.  Fixed in advance so the two
#: teaching modes and the two brain initialisations are read at identical compute.
PROBE_STEPS: tuple[int, ...] = (0, 500, 1000, 2000, 5000, 10000)

#: The single ordered pair held out of teaching in the holdout condition.
HOLDOUT_PAIR: tuple[int, int] = (2, 3)


@dataclass
class PilotConfig:
    """One cell of the 2x2."""

    label: str = ""
    #: run id of a counting run whose recurrent parameters seed this brain; "" = scratch
    pretrain_from: str = ""
    #: True  -> teach all 16 ordered pairs (the held-out pair is put back into training)
    #: False -> teach the 15 that remain when HOLDOUT_PAIR is withheld
    teach_holdout: bool = False
    budget_steps: int = 10_000
    probe_steps: tuple[int, ...] = field(default_factory=lambda: PROBE_STEPS)
    #: items used per evaluation, subsampled deterministically
    eval_items: int = 2000

    @property
    def brain(self) -> str:
        return "pretrained" if self.pretrain_from else "scratch"

    @property
    def teaching(self) -> str:
        return "full" if self.teach_holdout else "holdout"


def _subset(d: dict, n: int) -> dict:
    """Deterministic, class-balanced-ish prefix of a split (it is already shuffled)."""
    if n >= len(d["a"]):
        return d
    idx = np.linspace(0, len(d["a"]) - 1, n).astype(int)
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (len(d["a"]),) else v)
            for k, v in d.items()}


def _concat(*splits: dict) -> dict:
    """Concatenate the per-item arrays of several splits, keeping metadata as-is.

    ``make_add_dataset`` also returns scalars and an 0-d ``holdout_pairs`` object
    array; a blanket ``np.concatenate`` over every ndarray dies on those with
    "all the input arrays must have same number of dimensions".
    """
    n0 = len(splits[0]["a"])
    out: dict = {}
    for k, v in splits[0].items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n0:
            out[k] = np.concatenate([s[k] for s in splits])
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_pilot(model, retina: TorchRetina, data: dict, cfg: ExperimentConfig,
                   *, eval_items: int, device=DEVICE) -> dict:
    """Per-head accuracy on a split, plus P(5 | 2+3) and the operand probes.

    ``acc_sum`` is the sum head's accuracy; ``p_y5`` is the probability mass the sum
    head puts on class 5 (label 3, since labels are ``a + b - 2``), which is the
    quantity Figure B plots for the held-out pair.
    """
    model.eval()
    a_steps, gap, b_steps = cfg.time.steps_a, cfg.time.steps_gap, cfg.time.steps_b
    d = _subset(data, eval_items)
    n = len(d["a"])
    ca_parts, cb_parts, sa, sb, ss, p5 = [], [], [], [], [], []
    for i in range(0, n, 128):
        ca = encode(d["image_a"][i:i + 128], retina, device)
        cb = encode(d["image_b"][i:i + 128], retina, device)
        blank = torch.zeros_like(ca)
        seq = [ca] * a_steps + [blank] * gap + [cb] * b_steps
        _, history = model._run(seq, ca.shape[0])
        logits = model.classify(model.pool(history))
        sa.append(logits[0].argmax(1).cpu().numpy())
        sb.append(logits[1].argmax(1).cpu().numpy())
        ss.append(logits[2].argmax(1).cpu().numpy())
        p5.append(torch.softmax(logits[2], dim=1)[:, 3].cpu().numpy())
    pa, pb, ps, py5 = (np.concatenate(sa), np.concatenate(sb),
                       np.concatenate(ss), np.concatenate(p5))
    return {
        "n": int(n),
        "acc_a": float((pa == d["a"] - 1).mean()),
        "acc_b": float((pb == d["b"] - 1).mean()),
        "acc_sum": float((ps == d["labels"]).mean()),
        "p_y5": float(py5.mean()),
        # how often the sum head literally answers 5 on this split
        "pred_y5_rate": float((ps == 3).mean()),
    }


def run_addition_pilot(cfg: ExperimentConfig, pc: PilotConfig, *, logger=None) -> dict:
    from ..logging_utils import get_logger

    log = logger or get_logger("pilot")
    run_id = (f"9_pilot_{pc.brain}_{pc.teaching}"
              f"_s{cfg.seeds['model_seed']}")
    ctx = RunContext(run_id, {**cfg.to_dict(), "pilot": pc.__dict__})
    log = ctx.log
    log.info("=" * 78)
    log.info("PILOT %s | brain=%s teaching=%s | budget=%d updates | holdout=%s",
             pc.label or run_id, pc.brain, pc.teaching, pc.budget_steps, HOLDOUT_PAIR)

    seed = cfg.seeds["model_seed"]
    prep = prepare(cfg, logger=log)
    # One dataset construction for all four cells: the held-out pair is always the
    # split boundary, so every cell shares a byte-identical (2+3) probe set.  A cell
    # that teaches everything simply puts the probe set back into its training data.
    data = build_add_data(cfg, logger=log, reps_per_pair=cfg.stimulus.add_reps_per_pair)
    seen, probe = data["train"], data["test"]
    train_ds = _concat(seen, probe) if pc.teach_holdout else seen
    log.info("  taught pairs: %d items%s | held-out probe: %d items",
             len(train_ds["a"]), " (incl. 2+3)" if pc.teach_holdout else " (2+3 withheld)",
             len(probe["a"]))

    retina = TorchRetina(prep.encoder, device=DEVICE)
    # identical readout initialisation in every cell: same seed, same construction order
    torch.manual_seed(seed)
    n_op = cfg.stimulus.a_max - cfg.stimulus.a_min + 1
    model = prep.build_model(device=DEVICE, n_heads=3,
                             head_classes=(n_op, n_op, prep.n_classes))
    transferred = None
    if pc.pretrain_from:
        ckpt = _find_checkpoint(pc.pretrain_from)
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint for {pc.pretrain_from!r}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        transferred = model.load_recurrent_from(state)
        log.info("  warm start: %d recurrent tensors from %s; %d readout tensors left "
                 "at the same fresh initialisation as the scratch cells",
                 len(transferred["transferred"]), pc.pretrain_from,
                 len(transferred["skipped"]))
    else:
        log.info("  scratch: gains g=1 everywhere, readout freshly initialised")

    opt = make_optimizer(model, cfg)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)          # same seeding in all four cells
    y_a, y_b, y_s = train_ds["a"] - 1, train_ds["b"] - 1, train_ds["labels"]

    history: list[dict] = []
    t0 = time.time()
    opt_steps = 0

    def record(tag: str) -> None:
        seen_ev = evaluate_pilot(model, retina, seen, cfg, eval_items=pc.eval_items)
        probe_ev = evaluate_pilot(model, retina, probe, cfg, eval_items=pc.eval_items)
        full_ev = (evaluate_pilot(model, retina, _concat(seen, probe), cfg,
                                  eval_items=pc.eval_items)
                   if pc.teach_holdout else None)
        rec = {
            "updates": opt_steps,
            "acc_seen": seen_ev["acc_sum"],
            "acc_a_seen": seen_ev["acc_a"],
            "acc_b_seen": seen_ev["acc_b"],
            "acc_2p3": probe_ev["acc_sum"],
            "p_y5_given_2p3": probe_ev["p_y5"],
            "acc_a_2p3": probe_ev["acc_a"],
            "acc_b_2p3": probe_ev["acc_b"],
            "elapsed": round(time.time() - t0, 1),
            "tag": tag,
        }
        if full_ev:
            rec["acc_all16"] = full_ev["acc_sum"]
        ctx.metric(**rec)
        history.append(rec)
        log.info("  [%6d updates] seen %.4f | 2+3 acc %.4f  P(5|2+3) %.4f | a %.3f b %.3f%s",
                 opt_steps, rec["acc_seen"], rec["acc_2p3"], rec["p_y5_given_2p3"],
                 rec["acc_a_2p3"], rec["acc_b_2p3"],
                 f" | all16 {rec['acc_all16']:.4f}" if full_ev else "")

    record("initial")
    probes = [p for p in sorted(set(pc.probe_steps)) if p > 0]
    stop = False
    epoch = 0
    while not stop:
        model.train()
        perm = rng.permutation(len(y_a))
        for i in range(0, len(perm), cfg.train.batch_size):
            if opt_steps >= pc.budget_steps:
                stop = True
                break
            idx = perm[i:i + cfg.train.batch_size]
            ca = encode(train_ds["image_a"][idx], retina, DEVICE)
            cb = encode(train_ds["image_b"][idx], retina, DEVICE)
            blank = torch.zeros_like(ca)
            seq = ([ca] * cfg.time.steps_a + [blank] * cfg.time.steps_gap
                   + [cb] * cfg.time.steps_b)
            _, hist = model._run(seq, ca.shape[0])
            logits = model.classify(model.pool(hist))
            loss = (ce(logits[0], torch.as_tensor(y_a[idx], dtype=torch.long, device=DEVICE))
                    + ce(logits[1], torch.as_tensor(y_b[idx], dtype=torch.long, device=DEVICE))
                    + ce(logits[2], torch.as_tensor(y_s[idx], dtype=torch.long, device=DEVICE)))
            loss = loss + model.gain_penalty(cfg.model_cfg.lambda_gain)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
            opt.step()
            opt_steps += 1
            if probes and opt_steps >= probes[0]:
                # evaluate exactly ON the probe step, then drop it
                record(f"updates={probes[0]}")
                probes.pop(0)
        epoch += 1
        if epoch > 4000:
            break
    if opt_steps != pc.budget_steps:
        log.warning("stopped at %d updates, not the %d budget", opt_steps, pc.budget_steps)

    summary = {
        "run_id": ctx.run_id,
        "task": "add_pilot",
        "console": pc.label,
        "brain": pc.brain,
        "teaching": pc.teaching,
        "pretrain_from": pc.pretrain_from,
        "teach_holdout": pc.teach_holdout,
        "holdout_pair": list(HOLDOUT_PAIR),
        "model_seed": seed,
        "budget_steps": pc.budget_steps,
        "optimizer_steps": opt_steps,
        "n_taught_items": int(len(train_ds["a"])),
        "n_probe_items": int(len(probe["a"])),
        "transferred": transferred,
        "history": history,
        "final": history[-1] if history else {},
        "wall_seconds": round(ctx.elapsed(), 1),
    }
    (ctx.dir / "full_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    ctx.finish("ok", stage="9", task="add_pilot", model="M1",
               circuit=cfg.data.circuit, graph=cfg.data.graph, model_seed=seed,
               wall_seconds=round(ctx.elapsed(), 1))
    log.info("PILOT DONE %s | final seen %.4f | 2+3 acc %.4f P(5|2+3) %.4f | %.1f min",
             ctx.run_id, summary["final"].get("acc_seen", float("nan")),
             summary["final"].get("acc_2p3", float("nan")),
             summary["final"].get("p_y5_given_2p3", float("nan")),
             ctx.elapsed() / 60)
    return summary


def _find_checkpoint(run_id: str):
    from .. import paths

    for cand in (paths.run_dir(run_id) / "ckpt" / "best.pt",
                 paths.run_dir(run_id) / "ckpt" / "last.pt"):
        if cand.exists():
            return cand
    return None
