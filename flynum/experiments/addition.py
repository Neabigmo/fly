"""Two-epoch addition: ``a`` dots, a delay, then ``b`` dots -> ``a + b``.

The delay is what makes this more than two counting trials glued together: the
network has to *hold* the first count while the second stimulus is presented.
Because the readout only sees the state after the second epoch, information
about ``a`` must survive in the recurrent dynamics.

Compositional generalisation is tested by removing specific ordered pairs (by
default ``2+3`` and ``3+2``) from training entirely.  Their sum is still
reachable through other pairs, so a correct answer on a held-out pair means the
network learned to combine counts rather than to memorise a lookup table.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..config import ExperimentConfig
from ..devices import pick_device
from ..logging_utils import RunContext
from ..pipeline import prepare
from ..retina.torch_encoder import TorchRetina
from ..train.data import addition_matrix, build_add_data
from ..train.metrics import bootstrap_ci, classification_metrics, unseen_pair_report
from ..train.trainer import (
    cache_features_add,
    fit_readout_statistics,
    predict_add,
    probe_predict,
    train_probe,
)
from ..train.trainer import _make_scheduler, make_optimizer  # noqa: F401 (re-export)

DEVICE = pick_device()


def run_addition_experiment(
    cfg: ExperimentConfig, *, force: bool = False, logger=None
) -> dict:
    """Execute one addition run (M0 probe or M1 BPTT)."""
    run_id = cfg.run_id()
    ctx = RunContext(run_id, cfg.to_dict())
    logger = ctx.log
    logger.info("=" * 78)
    logger.info(
        "ADDITION RUN | model=%s graph=%s circuit=%s | a,b in [%d,%d] | holdout %s",
        cfg.model, cfg.data.graph, cfg.data.circuit,
        cfg.stimulus.a_min, cfg.stimulus.a_max, cfg.stimulus.holdout_pairs,
    )
    logger.info("=" * 78)

    torch.manual_seed(cfg.seeds["model_seed"])
    np.random.seed(cfg.seeds["model_seed"] % (2**32 - 1))

    prep = prepare(cfg, logger=logger, force=force)
    data = build_add_data(
        cfg, logger=logger, reps_per_pair=cfg.stimulus.add_reps_per_pair
    )
    tr, te = data["train"], data["test"]
    n_classes = prep.n_classes
    logger.info(
        "addition data: %d train items (%d pairs), %d holdout items, %d classes",
        len(tr["a"]),
        len({(int(x), int(y)) for x, y in zip(tr["a"], tr["b"])}),
        len(te["a"]),
        n_classes,
    )

    retina = TorchRetina(prep.encoder, device=DEVICE)
    model = prep.build_model(device=DEVICE)
    n_params = model.n_trainable()
    logger.info(
        "model: %d trainable parameters | steps A=%d gap=%d B=%d | readout=%s",
        n_params, cfg.time.steps_a, cfg.time.steps_gap, cfg.time.steps_b, prep.readout_name,
    )

    t0 = time.time()
    add_stats = fit_readout_statistics(
        model, retina, tr["image_a"],
        steps=cfg.time.steps_a + cfg.time.steps_gap + cfg.time.steps_b,
        n_samples=cfg.model_cfg.readout_stats_samples,
        device=DEVICE,
    )
    if add_stats.get("standardised"):
        logger.info("readout standardisation fitted (%s)", add_stats)
    if cfg.model == "M0":
        f_tr = cache_features_add(
            model, retina, tr["image_a"], tr["image_b"], cfg, device=DEVICE
        )
        f_te = cache_features_add(
            model, retina, te["image_a"], te["image_b"], cfg, device=DEVICE
        )
        # validate on a slice of the training pairs
        n_val = min(len(f_tr) // 10, 2000)
        head, train_info = train_probe(
            cfg, f_tr[n_val:], tr["labels"][n_val:], f_tr[:n_val], tr["labels"][:n_val],
            ctx, device=DEVICE, logger=logger,
        )
        pred_test = probe_predict(head, f_te, device=DEVICE)
        pred_train = probe_predict(head, f_tr, device=DEVICE)
    elif cfg.model == "M1":
        train_info = _train_taught_add(cfg, model, retina, tr, ctx, logger)
        pred_test = predict_add(
            model, retina, te["image_a"], te["image_b"], cfg, device=DEVICE
        )
        pred_train = predict_add(
            model, retina, tr["image_a"], tr["image_b"], cfg, device=DEVICE
        )
    else:
        raise ValueError(f"unknown model {cfg.model!r} (M0 | M1)")

    # ---- metrics ------------------------------------------------------- #
    metrics_train = classification_metrics(tr["labels"], pred_train, n_classes)
    metrics_test = classification_metrics(te["labels"], pred_test, n_classes)
    unseen = unseen_pair_report(
        te["labels"], pred_test, np.ones(len(te["labels"]), dtype=bool), n_classes
    )
    # held-out vs the equivalent seen pairs (same sums)
    seen_sums = np.isin(tr["sums"], te["sums"])
    seen_slice_pred = pred_train[seen_sums]
    seen_slice_y = tr["labels"][seen_sums]
    seen_acc = float((seen_slice_pred == seen_slice_y).mean()) if seen_sums.any() else float("nan")

    matrix_test = addition_matrix(te["a"], te["b"], te["labels"], pred_test,
                                  cfg.stimulus.a_min, cfg.stimulus.a_max)
    matrix_train = addition_matrix(tr["a"], tr["b"], tr["labels"], pred_train,
                                   cfg.stimulus.a_min, cfg.stimulus.a_max)

    correct_unseen = (pred_test == te["labels"]).astype(np.float64)
    ci = bootstrap_ci(correct_unseen)

    logger.info("-" * 78)
    logger.info(
        "train acc %.4f | held-out-pair acc %.4f (n=%d, chance %.3f) | "
        "seen-pairs-on-train %.4f",
        metrics_train["accuracy"], metrics_test["accuracy"], len(te["labels"]),
        metrics_test["chance"], seen_acc,
    )
    logger.info("addition matrix (accuracy per a+b):")
    for a in range(cfg.stimulus.a_min, cfg.stimulus.a_max + 1):
        row = " ".join(f"{matrix_test[f'{a}+{b}']['accuracy']:.2f}"
                       for b in range(cfg.stimulus.a_min, cfg.stimulus.a_max + 1))
        logger.info("  a=%d | %s", a, row)

    np.savez_compressed(
        ctx.dir / "predictions.npz",
        test_pred=pred_test.astype(np.int16),
        test_label=te["labels"].astype(np.int16),
        test_a=te["a"].astype(np.int16),
        test_b=te["b"].astype(np.int16),
        train_pred=pred_train.astype(np.int16),
        train_label=tr["labels"].astype(np.int16),
        train_a=tr["a"].astype(np.int16),
        train_b=tr["b"].astype(np.int16),
    )

    summary = {
        **prep.info,
        "run_id": run_id,
        "stage": cfg.stage,
        "task": "add",
        "model": cfg.model,
        "n_params": n_params,
        "holdout_pairs": [list(p) for p in cfg.stimulus.holdout_pairs],
        "addition_train_accuracy": metrics_train["accuracy"],
        "addition_test_accuracy": metrics_test["accuracy"],
        "unseen_pair_accuracy": metrics_test["accuracy"],
        "unseen_ci": list(ci),
        "seen_pair_accuracy_on_train": seen_acc,
        "unseen": unseen,
        "matrix_test": matrix_test,
        "matrix_train": matrix_train,
        "confusion_test": metrics_test["confusion"],
        "train_info": {k: v for k, v in train_info.items() if k != "history"},
        "wall_seconds_total": round(ctx.elapsed(), 1),
        "train_seconds": round(time.time() - t0, 1),
    }
    (ctx.dir / "full_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    ctx.finish(
        "ok",
        stage=cfg.stage,
        task="add",
        model=cfg.model,
        circuit=cfg.data.circuit,
        graph=cfg.data.graph,
        model_seed=cfg.seeds["model_seed"],
        shuffle_seed=cfg.seeds["shuffle_seed"],
        best_val_acc=train_info.get("best_val_acc", float("nan")),
        wall_seconds=round(ctx.elapsed(), 1),
    )
    logger.info(
        "SUMMARY | held-out-pair acc %.4f | %.1f min", metrics_test["accuracy"],
        ctx.elapsed() / 60,
    )
    return summary


# --------------------------------------------------------------------------- #
def _train_taught_add(cfg, model, retina, tr, ctx: RunContext, logger) -> dict:
    """BPTT for the addition task (mirrors :func:`train_taught`)."""
    import torch.nn as nn

    from ..train.trainer import encode

    tc = cfg.train
    n_val = min(len(tr["labels"]) // 10, 2000)
    img_a_va, img_b_va = tr["image_a"][:n_val], tr["image_b"][:n_val]
    y_va = tr["labels"][:n_val]
    img_a_tr, img_b_tr, y_tr = tr["image_a"][n_val:], tr["image_b"][n_val:], tr["labels"][n_val:]

    opt = make_optimizer(model, cfg)
    sched = _make_scheduler(opt, cfg, tc.max_epochs)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(cfg.seeds["model_seed"])

    best_val, best_epoch, bad = -1.0, -1, 0
    history = []
    t0 = time.time()
    for epoch in range(tc.max_epochs):
        model.train()
        perm = rng.permutation(len(y_tr))
        ep_loss = 0.0
        nb = 0
        te = time.time()
        for i in range(0, len(perm), tc.batch_size):
            idx = perm[i : i + tc.batch_size]
            ca = encode(img_a_tr[idx], retina, DEVICE)
            cb = encode(img_b_tr[idx], retina, DEVICE)
            y = torch.as_tensor(y_tr[idx], dtype=torch.long, device=DEVICE)
            logits, _ = model.forward_add(
                ca, cb, cfg.time.steps_a, cfg.time.steps_gap, cfg.time.steps_b
            )
            loss = ce(logits, y) + model.gain_penalty(cfg.model_cfg.lambda_gain)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], tc.grad_clip
                )
            opt.step()
            ep_loss += float(loss) * len(idx)
            nb += len(idx)

        val_pred = predict_add(model, retina, img_a_va, img_b_va, cfg, device=DEVICE)
        val_acc = float((val_pred == y_va).mean())
        rec = {
            "epoch": epoch,
            "train_loss": ep_loss / max(nb, 1),
            "val_acc": val_acc,
            "epoch_seconds": round(time.time() - te, 2),
        }
        ctx.metric(**rec)
        history.append(rec)
        logger.info("  epoch %3d | loss %.4f | val %.4f | %.1fs",
                    epoch, rec["train_loss"], val_acc, rec["epoch_seconds"])
        if val_acc > best_val + 1e-6:
            best_val, best_epoch, bad = val_acc, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       ctx.dir / "ckpt" / "best.pt")
        else:
            bad += 1
        if sched is not None:
            sched.step()
        if bad >= tc.patience:
            logger.info("  early stop at epoch %d (best %d)", epoch, best_epoch)
            break

    best_path = ctx.dir / "ckpt" / "best.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=False)["model"]
        )
    return {
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "wall_seconds": round(time.time() - t0, 1),
        "protocol": "BPTT on per-edge gains + readout",
    }
