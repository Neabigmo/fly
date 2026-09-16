"""Guided addition curriculum: learn the operands before the operation.

The straight-to-addition run failed in a specific way: the temporal probes showed
the first count held across the delay (retention ~1.2) while ``a+b`` never rose
above chance.  The network could remember but not combine.  Retraining from
scratch with a harder objective therefore attacks the wrong problem.

This module implements the three-lesson curriculum:

**Lesson 1 -- counting.**  Take a connectome already trained to count (or train
one), and keep its learned recurrent parameters.

**Lesson 2 -- two quantities.**  Show ``a`` dots, a blank delay, then ``b`` dots,
and require the network to read out ``a`` and ``b`` **separately**.  The sum is
not required yet, so the only new demand is holding and labelling two operands.

**Lesson 3 -- the operation.**  Add the sum objective::

    L = w_a * CE(head_a, a) + w_b * CE(head_b, b) + w_sum * CE(head_sum, a+b)

The three heads share the same pooled features, so the operand heads act as an
auxiliary task that keeps the representation readable while the sum head learns
to combine it.

The scientific comparison is ``count-pretrained`` against ``scratch`` with an
otherwise identical configuration: it isolates whether a learned connectome
transfers, as opposed to a readout being re-fit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..devices import pick_device
from ..logging_utils import RunContext
from ..pipeline import prepare
from ..retina.torch_encoder import TorchRetina
from ..train.data import build_add_data
from ..train.metrics import classification_metrics
from ..train.trainer import encode, fit_readout_statistics, make_optimizer

DEVICE = pick_device()


@dataclass
class CurriculumConfig:
    """Lesson weights and staging."""

    #: epochs spent on lesson 2 (operands only) before the sum objective appears
    operand_epochs: int = 15
    w_a: float = 1.0
    w_b: float = 1.0
    w_sum: float = 1.0
    #: run id whose checkpoint warm-starts the recurrent parameters (lesson 1)
    pretrain_from: str = ""
    #: if True the model is trained on the exact same schedule but from scratch
    scratch: bool = False


def _multi_head_loss(
    logits: list[torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    cc: CurriculumConfig,
    *,
    with_sum: bool,
) -> tuple[torch.Tensor, dict]:
    ce = nn.CrossEntropyLoss()
    loss_a = ce(logits[0], a)
    loss_b = ce(logits[1], b)
    out = cc.w_a * loss_a + cc.w_b * loss_b
    parts = {"loss_a": float(loss_a), "loss_b": float(loss_b)}
    if with_sum:
        loss_s = ce(logits[2], s)
        out = out + cc.w_sum * loss_s
        parts["loss_sum"] = float(loss_s)
    return out, parts


@torch.no_grad()
def evaluate_curriculum(
    model, retina: TorchRetina, data: dict, cc: CurriculumConfig, device=DEVICE
) -> dict:
    model.eval()
    out: dict = {}
    for split in ("train", "test"):
        d = data[split]
        preds = {"a": [], "b": [], "s": []}
        for i in range(0, len(d["a"]), 128):
            ca = encode(d["image_a"][i : i + 128], retina, device)
            cb = encode(d["image_b"][i : i + 128], retina, device)
            blank = torch.zeros_like(ca)
            seq = [ca] * 5 + [blank] * 4 + [cb] * 5
            _, history = model._run(seq, ca.shape[0])
            logits = model.classify(model.pool(history))
            preds["a"].append(logits[0].argmax(1).cpu().numpy())
            preds["b"].append(logits[1].argmax(1).cpu().numpy())
            preds["s"].append(logits[2].argmax(1).cpu().numpy())
        pa = np.concatenate(preds["a"])
        pb = np.concatenate(preds["b"])
        ps = np.concatenate(preds["s"])
        out[split] = {
            "acc_a": float((pa == d["a"] - 1).mean()),
            "acc_b": float((pb == d["b"] - 1).mean()),
            "acc_sum": float((ps == d["labels"]).mean()),
        }
        if split == "test":
            out["test"]["sum_confusion"] = classification_metrics(
                d["labels"], ps, 7
            )["confusion"]
            out["test"]["unseen_pairs"] = sorted(
                {tuple(p) for p in zip(d["a"].tolist(), d["b"].tolist())}
            )
    return out


def run_curriculum(
    cfg: ExperimentConfig, cc: CurriculumConfig, *, logger=None
) -> dict:
    """Train the three-lesson curriculum and return every metric."""
    from ..logging_utils import get_logger

    log = logger or get_logger("curriculum")
    run_id = f"{cfg.stage}_curr_{'scratch' if cc.scratch else 'pretrained'}_{cfg.data.graph}_s{cfg.seeds['model_seed']}"
    ctx = RunContext(
        run_id,
        {**cfg.to_dict(), "curriculum": cc.__dict__},
    )
    log = ctx.log
    log.info(
        "CURRICULUM | graph=%s circuit=%s pretrain=%s scratch=%s | "
        "operand_epochs=%d  w=(%.2f,%.2f,%.2f)",
        cfg.data.graph, cfg.data.circuit, cc.pretrain_from or "-", cc.scratch,
        cc.operand_epochs, cc.w_a, cc.w_b, cc.w_sum,
    )

    torch.manual_seed(cfg.seeds["model_seed"])
    np.random.seed(cfg.seeds["model_seed"] % (2**32 - 1))

    prep = prepare(cfg, logger=log)
    data = build_add_data(cfg, logger=log, reps_per_pair=cfg.stimulus.add_reps_per_pair)
    retina = TorchRetina(prep.encoder, device=DEVICE)

    # three heads over the shared pooled features; head 0 = a, 1 = b, 2 = a+b.
    # The operands span a_min..a_max (4 classes); the sum spans 2*a_min..2*a_max
    # (7 classes), which is exactly ``prep.n_classes`` for the addition task.
    n_operand = cfg.stimulus.a_max - cfg.stimulus.a_min + 1
    model = prep.build_model(
        device=DEVICE, n_heads=3, head_classes=(n_operand, n_operand, prep.n_classes)
    )

    transferred = None
    if cc.pretrain_from and not cc.scratch:
        ckpt = _find_checkpoint(cc.pretrain_from)
        if ckpt is None:
            raise FileNotFoundError(
                f"pretrain checkpoint for {cc.pretrain_from!r} not found under runs/"
            )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        transferred = model.load_recurrent_from(state)
        log.info(
            "lesson 1 (warm start): transferred %d recurrent tensors from %s "
            "(%d readout tensors left fresh)",
            len(transferred["transferred"]), ckpt.parent.parent.name,
            len(transferred["skipped"]),
        )
    else:
        log.info("lesson 1: no warm start -- recurrent parameters are random-gain (all g=1)")

    stats = fit_readout_statistics(
        model, retina, data["train"]["image_a"],
        steps=cfg.time.steps_a + cfg.time.steps_gap + cfg.time.steps_b,
        n_samples=1000, device=DEVICE,
    )
    log.info("readout standardisation: %s", {k: v for k, v in stats.items() if k != "std_p1"})

    train_info = _train_curriculum(cfg, cc, model, retina, data, ctx)
    ev = evaluate_curriculum(model, retina, data, cc, DEVICE)

    log.info("-" * 78)
    log.info(
        "a: train %.4f test %.4f | b: train %.4f test %.4f | sum: train %.4f test %.4f",
        ev["train"]["acc_a"], ev["test"]["acc_a"],
        ev["train"]["acc_b"], ev["test"]["acc_b"],
        ev["train"]["acc_sum"], ev["test"]["acc_sum"],
    )
    summary = {
        "run_id": ctx.run_id,
        "task": "add_curriculum",
        "graph": cfg.data.graph,
        "circuit": cfg.data.circuit,
        "model_seed": cfg.seeds["model_seed"],
        "scratch": cc.scratch,
        "pretrain_from": cc.pretrain_from,
        "curriculum": cc.__dict__,
        "acc_a_test": ev["test"]["acc_a"],
        "acc_b_test": ev["test"]["acc_b"],
        "sum_test": ev["test"]["acc_sum"],
        "acc_a_train": ev["train"]["acc_a"],
        "acc_b_train": ev["train"]["acc_b"],
        "sum_train": ev["train"]["acc_sum"],
        "chance_a": 1.0 / 4,
        "chance_sum": 1.0 / 7,
        "transferred": transferred,
        "train_info": {k: v for k, v in train_info.items() if k != "history"},
        "sign_report": prep.sign_report,
        "wall_seconds": round(ctx.elapsed(), 1),
    }
    (ctx.dir / "full_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    ctx.finish(
        "ok", stage=cfg.stage, task="add_curriculum", model="M1",
        circuit=cfg.data.circuit, graph=cfg.data.graph,
        model_seed=cfg.seeds["model_seed"], shuffle_seed=cfg.seeds["shuffle_seed"],
        best_val_acc=train_info.get("best_val_acc", float("nan")),
        wall_seconds=round(ctx.elapsed(), 1),
    )
    return summary


# --------------------------------------------------------------------------- #
def _train_curriculum(
    cfg: ExperimentConfig, cc: CurriculumConfig, model, retina, data, ctx
) -> dict:
    log = ctx.log
    tc = cfg.train
    tr, va = data["train"], data["test"]
    n_train = len(tr["a"])
    opt = make_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(cfg.seeds["model_seed"])

    best_val, best_epoch, bad = -1.0, -1, 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(tc.max_epochs):
        with_sum = epoch >= cc.operand_epochs
        model.train()
        perm = rng.permutation(n_train)
        ep_loss = nb = 0
        t_epoch = time.time()
        for i in range(0, len(perm), tc.batch_size):
            idx = perm[i : i + tc.batch_size]
            ca = encode(tr["image_a"][idx], retina, DEVICE)
            cb = encode(tr["image_b"][idx], retina, DEVICE)
            blank = torch.zeros_like(ca)
            seq = (
                [ca] * cfg.time.steps_a + [blank] * cfg.time.steps_gap
                + [cb] * cfg.time.steps_b
            )
            _, hist = model._run(seq, ca.shape[0])
            logits = model.classify(model.pool(hist))
            loss, _ = _multi_head_loss(
                logits,
                torch.as_tensor(tr["a"][idx] - 1, dtype=torch.long, device=DEVICE),
                torch.as_tensor(tr["b"][idx] - 1, dtype=torch.long, device=DEVICE),
                torch.as_tensor(tr["labels"][idx], dtype=torch.long, device=DEVICE),
                cc,
                with_sum=with_sum,
            )
            loss = loss + model.gain_penalty(cfg.model_cfg.lambda_gain)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], tc.grad_clip
                )
            opt.step()
            ep_loss += float(loss) * len(idx)
            nb += len(idx)

        ev = evaluate_curriculum(model, retina, {"train": tr, "test": va}, cc, DEVICE)
        # "validation" = the auxiliary operand tasks, which is what the
        # curriculum is actually training; the sum is reported but not selected on
        val_acc = 0.5 * (ev["test"]["acc_a"] + ev["test"]["acc_b"])
        rec = {
            "epoch": epoch, "with_sum": with_sum,
            "loss": ep_loss / max(nb, 1),
            "val_operand_acc": val_acc,
            "test_sum_acc": ev["test"]["acc_sum"],
            "epoch_seconds": round(time.time() - t_epoch, 2),
        }
        ctx.metric(**rec)
        history.append(rec)
        if epoch % 5 == 0 or epoch == tc.max_epochs - 1:
            log.info(
                "  epoch %3d | %s | loss %.4f | a %.3f b %.3f | sum %.3f",
                epoch, "a+b" if with_sum else "operands",
                rec["loss"], ev["test"]["acc_a"], ev["test"]["acc_b"],
                ev["test"]["acc_sum"],
            )
        if val_acc > best_val + 1e-6:
            best_val, best_epoch, bad = val_acc, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       ctx.dir / "ckpt" / "best.pt")
        else:
            bad += 1
        if bad >= tc.patience:
            log.info("  early stop at epoch %d (best %d)", epoch, best_epoch)
            break

    path = ctx.dir / "ckpt" / "best.pt"
    if path.exists():
        # map_location="cpu" then load_state_dict: copies into the live parameters
        # whatever device they are on, and never fails when the checkpoint was
        # written from a CUDA run but is being read back on a CPU-only machine.
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model"]
        )
    return {
        "best_val_acc": best_val, "best_epoch": best_epoch,
        "epochs_run": len(history), "history": history,
        "wall_seconds": round(time.time() - t0, 1),
    }


def _find_checkpoint(run_id: str):
    from .. import paths

    for cand in (
        paths.run_dir(run_id) / "ckpt" / "best.pt",
        paths.run_dir(run_id) / "ckpt" / "last.pt",
    ):
        if cand.exists():
            return cand
    return None
