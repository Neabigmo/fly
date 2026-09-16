"""Training loops.

Two protocols, both required by the study design:

**M1 -- taught fly.**  The connectome topology is frozen but every existing
synapse carries a trainable gain, and the linear readout is trained jointly.
Backpropagation runs through time across the unrolled recurrence.

**M0 -- frozen fly.**  The network weights are not touched at all; only a linear
readout is fitted.  Because the recurrent state is then independent of the
parameters, the pooled features are computed **once** and the probe is fitted on
the cache.  This is both the standard linear-probe protocol and roughly 50x
cheaper, since no BPTT is needed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..logging_utils import RunContext
from ..models.connectome_rnn import ConnectomeRNN
from ..retina.torch_encoder import TorchRetina
from .metrics import bootstrap_ci, classification_metrics

#: A mean cross-entropy above this means the readout has blown up; the run is
#: abandoned instead of silently consuming hours of GPU time.
DIVERGENCE_LOSS = 25.0


# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    confusion: list
    per_class_recall: list
    correct: np.ndarray
    pred: np.ndarray
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "confusion": self.confusion,
            "per_class_recall": self.per_class_recall,
            **self.extra,
        }


def encode(images: np.ndarray, retina_map: TorchRetina, device) -> torch.Tensor:
    t = torch.as_tensor(np.ascontiguousarray(images), dtype=torch.float32, device=device)
    return retina_map.encode(t)


# --------------------------------------------------------------------------- #
def predict_count(
    model: ConnectomeRNN,
    retina_map: TorchRetina,
    images: np.ndarray,
    *,
    steps: int,
    batch_size: int = 256,
    device="cpu",
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            cols = encode(images[i : i + batch_size], retina_map, device)
            logits, _ = model.forward_count(cols, steps)
            out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def predict_add(
    model: ConnectomeRNN,
    retina_map: TorchRetina,
    img_a: np.ndarray,
    img_b: np.ndarray,
    cfg: ExperimentConfig,
    *,
    batch_size: int = 256,
    device="cpu",
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(img_a), batch_size):
            ca = encode(img_a[i : i + batch_size], retina_map, device)
            cb = encode(img_b[i : i + batch_size], retina_map, device)
            logits, _ = model.forward_add(
                ca, cb, cfg.time.steps_a, cfg.time.steps_gap, cfg.time.steps_b
            )
            out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def evaluate(
    y_true: np.ndarray, pred: np.ndarray, n_classes: int, **extra
) -> EvalResult:
    m = classification_metrics(y_true, pred, n_classes)
    correct = (np.asarray(y_true).astype(int) == pred.astype(int)).astype(np.float64)
    ci = bootstrap_ci(correct)
    return EvalResult(
        accuracy=m["accuracy"],
        macro_f1=m["macro_f1"],
        confusion=m["confusion"],
        per_class_recall=m["per_class_recall"],
        correct=correct,
        pred=pred,
        extra={"acc_ci_low": ci[0], "acc_ci_high": ci[1], "chance": m["chance"], **extra},
    )


# --------------------------------------------------------------------------- #
def make_optimizer(model: ConnectomeRNN, cfg: ExperimentConfig):
    tc = cfg.train
    groups = []
    if model.delta is not None:
        groups.append({"params": [model.delta], "lr": tc.lr_gain, "weight_decay": 0.0})
    if model.bias is not None:
        groups.append({"params": [model.bias], "lr": tc.lr_gain, "weight_decay": 0.0})
    groups.append(
        {
            "params": list(model.readout.parameters()),
            "lr": tc.lr_readout,
            "weight_decay": tc.wd_readout,
        }
    )
    return torch.optim.AdamW(groups)


def _make_scheduler(opt, cfg: ExperimentConfig, epochs: int):
    if not cfg.train.cosine:
        return None
    if cfg.train.max_steps > 0:
        # the schedule is applied per optimiser step instead (see _step_lr)
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))


def _step_lr(base_lrs: list[float], step: int, total: int) -> list[float]:
    """Cosine learning rates as a function of *optimiser step*.

    With a fixed step budget this replaces the epoch-based scheduler, which would
    otherwise give a different amount of annealing depending on how many batches
    an epoch happens to contain.
    """
    if total <= 0:
        return list(base_lrs)
    frac = min(step / total, 1.0)
    scale = 0.5 * (1.0 + np.cos(np.pi * frac))
    return [lr * scale for lr in base_lrs]


# --------------------------------------------------------------------------- #
def train_taught(
    cfg: ExperimentConfig,
    model: ConnectomeRNN,
    retina_map: TorchRetina,
    train_xy,
    val_xy,
    ctx: RunContext,
    *,
    device="cpu",
    logger=None,
) -> dict:
    """Full BPTT training of the per-edge gains together with the readout."""
    logger = logger or ctx.log
    tc = cfg.train
    img_tr, y_tr = train_xy
    img_va, y_va = val_xy
    n_classes = int(max(y_tr.max(), y_va.max())) + 1
    opt = make_optimizer(model, cfg)
    sched = _make_scheduler(opt, cfg, tc.max_epochs)
    ce = nn.CrossEntropyLoss()

    rng = np.random.default_rng(cfg.seeds["model_seed"])
    best_val = -1.0
    best_epoch = -1
    bad = 0
    history: list[dict] = []
    ckpt_dir = ctx.dir / "ckpt"
    t0 = time.time()
    opt_steps = 0
    base_lrs = [g["lr"] for g in opt.param_groups]
    base_lrs_scaled = list(base_lrs)

    epochs_iterated = 0
    for epoch in range(tc.max_epochs):
        epochs_iterated = epoch + 1
        model.train()
        perm = rng.permutation(len(img_tr))
        ep_loss = 0.0
        ep_correct = 0
        nb = 0
        te = time.time()
        budget_hit = False
        for i in range(0, len(perm), tc.batch_size):
            if tc.max_steps and opt_steps >= tc.max_steps:
                budget_hit = True
                break
            if tc.max_steps:
                for group, lr in zip(opt.param_groups, base_lrs_scaled):
                    group["lr"] = lr
            idx = perm[i : i + tc.batch_size]
            cols = encode(img_tr[idx], retina_map, device)
            y = torch.as_tensor(y_tr[idx], dtype=torch.long, device=device)
            logits, _ = model.forward_count(cols, cfg.time.steps)
            loss = ce(logits, y) + model.gain_penalty(cfg.model_cfg.lambda_gain)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], tc.grad_clip
                )
            opt.step()
            opt_steps += 1
            if tc.max_steps:
                base_lrs_scaled = _step_lr(base_lrs, opt_steps, tc.max_steps)
            ep_loss += float(loss) * len(idx)
            ep_correct += int((logits.argmax(1) == y).sum())
            nb += len(idx)

        train_acc = ep_correct / max(nb, 1)
        mean_loss = ep_loss / max(nb, 1)

        # Under a fixed step budget the epoch is only a bookkeeping unit: a run
        # with 4x fewer samples needs 4x more epochs to spend the same number of
        # updates.  Evaluate on a cadence that keeps the *number of validation
        # points* comparable across N, and always on the last epoch.
        every = tc.eval_every if (tc.max_steps and tc.eval_every > 0) else 1
        if not ((epoch % every == 0) or budget_hit or epoch == tc.max_epochs - 1):
            if not np.isfinite(mean_loss) or mean_loss > DIVERGENCE_LOSS:
                logger.error(
                    "  DIVERGED at epoch %d (train_loss=%s) - aborting this run",
                    epoch, mean_loss,
                )
                ctx.metric(epoch=epoch, train_loss=mean_loss, diverged=True)
                history.append({"epoch": epoch, "train_loss": mean_loss, "diverged": True})
                break
            continue

        val_pred = predict_count(
            model, retina_map, img_va, steps=cfg.time.steps, device=device
        )
        val_acc = float((val_pred == y_va).mean())
        rec = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr_readout": opt.param_groups[-1]["lr"],
            "epoch_seconds": round(time.time() - te, 2),
            "elapsed": round(time.time() - t0, 1),
        }
        if not np.isfinite(mean_loss) or mean_loss > DIVERGENCE_LOSS or not np.isfinite(val_acc):
            logger.error(
                "  DIVERGED at epoch %d (train_loss=%s, val_acc=%s) - aborting this run "
                "rather than burning GPU time; check --lr-gain/--lr-readout",
                epoch, mean_loss, val_acc,
            )
            ctx.metric(**rec, diverged=True)
            history.append(rec)
            break
        if device != "cpu" and torch.cuda.is_available():
            rec["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if model.delta is not None:
            with torch.no_grad():
                rec["delta_abs_mean"] = float(model.delta.abs().mean())
                rec["delta_abs_max"] = float(model.delta.abs().max())
        ctx.metric(**rec)
        history.append(rec)
        logger.info(
            "  epoch %3d | loss %.4f | train %.4f | val %.4f | %.1fs",
            epoch, rec["train_loss"], train_acc, val_acc, rec["epoch_seconds"],
        )

        if val_acc > best_val + 1e-6:
            best_val, best_epoch, bad = val_acc, epoch, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                ckpt_dir / "best.pt",
            )
        else:
            bad += 1
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
            ckpt_dir / "last.pt",
        )
        if sched is not None:
            sched.step()
        if tc.max_steps and opt_steps >= tc.max_steps:
            logger.info("  reached the fixed budget of %d optimiser steps", tc.max_steps)
            break
        if not tc.max_steps and bad >= tc.patience:
            logger.info("  early stop at epoch %d (best %d, val %.4f)", epoch, best_epoch, best_val)
            break

    if (ckpt_dir / "best.pt").exists():
        state = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    return {
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "epochs_iterated": epochs_iterated,
        "optimizer_steps": opt_steps,
        "history": history,
        "wall_seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_features(
    model: ConnectomeRNN,
    retina_map: TorchRetina,
    images: np.ndarray,
    *,
    steps: int,
    batch_size: int = 256,
    device="cpu",
) -> np.ndarray:
    """Raw pooled readout features for a frozen network.

    Raw means *before* the model's own standardisation: the linear probe fits
    its own scaler on the training split, which keeps M0 and M1 comparable while
    making sure the probe never sees test statistics.
    """
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        cols = encode(images[i : i + batch_size], retina_map, device)
        _, history = model.forward_count(cols, steps)
        out.append(model.pool(history).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def cache_features_add(
    model: ConnectomeRNN,
    retina_map: TorchRetina,
    img_a: np.ndarray,
    img_b: np.ndarray,
    cfg: ExperimentConfig,
    *,
    batch_size: int = 256,
    device="cpu",
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(img_a), batch_size):
        ca = encode(img_a[i : i + batch_size], retina_map, device)
        cb = encode(img_b[i : i + batch_size], retina_map, device)
        _, history = model.forward_add(
            ca, cb, cfg.time.steps_a, cfg.time.steps_gap, cfg.time.steps_b
        )
        out.append(model.pool(history).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def fit_readout_statistics(
    model: ConnectomeRNN,
    retina_map: TorchRetina,
    images: np.ndarray,
    *,
    steps: int,
    n_samples: int = 2000,
    batch_size: int = 128,
    device="cpu",
) -> dict:
    """Freeze the readout standardisation from the untrained network."""
    if not model.standardize or len(images) == 0:
        return {"standardised": False}
    idx = np.linspace(0, len(images) - 1, min(n_samples, len(images))).astype(int)
    feats = []
    for i in range(0, len(idx), batch_size):
        cols = encode(images[idx[i : i + batch_size]], retina_map, device)
        feats.append(model.raw_pooled_features(cols, steps).cpu())
    f = torch.cat(feats, dim=0)
    info = model.fit_readout_stats(f)
    info["n_stats_samples"] = int(f.shape[0])
    if info.get("standardised") and info["z_abs_max"] > 100:
        raise RuntimeError(
            "readout standardisation produced extreme values "
            f"(max |z| = {info['z_abs_max']:.1f}) - the feature statistics are "
            "unreliable; check the network dynamics before training"
        )
    return info


class StandardizedProbe(nn.Module):
    """Linear probe together with the feature standardisation it was fitted with.

    The scaler has to travel with the head.  Returning a bare ``nn.Linear`` and
    then feeding it *raw* features at prediction time silently produces constant
    predictions -- validation accuracy (computed on standardised features) looked
    like 0.53 while every test set sat at exactly chance.
    """

    def __init__(self, head: nn.Module, mu: np.ndarray, sd: np.ndarray):
        super().__init__()
        self.head = head
        self.register_buffer("feat_mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("feat_sd", torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head((x - self.feat_mu) / self.feat_sd)


def train_probe(
    cfg: ExperimentConfig,
    features_tr: np.ndarray,
    y_tr: np.ndarray,
    features_va: np.ndarray,
    y_va: np.ndarray,
    ctx: RunContext,
    *,
    device="cpu",
    logger=None,
) -> tuple[nn.Module, dict]:
    """Fit the linear readout on cached frozen-network features.

    Features are standardised with **training-split** statistics only, so the
    probe never sees test information.  The returned module carries those
    statistics, and the same constant-feature guard used by the model applies
    here: a feature whose spread is essentially zero would otherwise be divided
    by ~0 and inject pure noise at enormous amplitude.
    """
    logger = logger or ctx.log
    tc = cfg.train
    n_classes = int(max(y_tr.max(), y_va.max())) + 1
    mu = features_tr.mean(axis=0, keepdims=True)
    sd = features_tr.std(axis=0, keepdims=True)
    # Only genuinely dead features may be discarded.  A relative floor cannot
    # tell "low variance but informative" from "numerical noise" when the spread
    # of feature scales is wide, so it is kept extremely conservative (1e-6 of
    # the median): the guard exists to stop a divide-by-~0 blow-up, not to do
    # feature selection.
    floor = max(float(np.median(sd)) * 1e-6, 1e-12)
    constant = sd < floor
    sd = np.where(constant, 1.0, np.maximum(sd, floor))
    mu = np.where(constant, 0.0, mu)
    features_tr = (features_tr - mu) / sd
    features_va = (features_va - mu) / sd
    logger.info(
        "  probe input: %d features, standardised (%d constant, mean std=%.4g)",
        features_tr.shape[1], int(constant.sum()), float(sd.mean()),
    )
    head = nn.Linear(features_tr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=tc.lr_readout, weight_decay=tc.wd_readout)
    sched = _make_scheduler(opt, cfg, tc.max_epochs)
    ce = nn.CrossEntropyLoss()

    rng = np.random.default_rng(cfg.seeds["model_seed"])
    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for epoch in range(tc.max_epochs):
        head.train()
        perm = rng.permutation(len(y_tr))
        for i in range(0, len(perm), tc.batch_size):
            idx = perm[i : i + tc.batch_size]
            x = torch.as_tensor(features_tr[idx], device=device)
            y = torch.as_tensor(y_tr[idx], dtype=torch.long, device=device)
            loss = ce(head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            xv = torch.as_tensor(features_va, device=device)
            val_acc = float(
                (head(xv).argmax(1).cpu().numpy() == y_va).mean()
            )
        ctx.metric(epoch=epoch, val_acc=val_acc, probe=True)
        if val_acc > best_val + 1e-6:
            best_val, bad = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if sched is not None:
            sched.step()
        if bad >= tc.patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    logger.info("  probe: best val %.4f in %.1fs", best_val, time.time() - t0)
    probe = StandardizedProbe(head, mu.squeeze(0), sd.squeeze(0)).to(device)
    probe.eval()
    return probe, {
        "best_val_acc": best_val,
        "epochs_run": epoch + 1,
        "protocol": "frozen linear probe",
        "n_features": int(features_tr.shape[1]),
        "n_constant_features": int(constant.sum()),
        "wall_seconds": round(time.time() - t0, 1),
    }


def probe_predict(head: nn.Module, features: np.ndarray, device="cpu") -> np.ndarray:
    head.eval()
    with torch.no_grad():
        x = torch.as_tensor(features, device=device)
        return head(x).argmax(1).cpu().numpy()


# --------------------------------------------------------------------------- #
def save_training_summary(ctx: RunContext, name: str, payload: dict) -> Path:
    path = ctx.dir / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
