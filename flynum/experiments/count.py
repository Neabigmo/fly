"""Run one dot-counting experiment end to end.

Protocol
--------
* **M1 (taught fly)** -- connectome topology frozen, per-edge gains and the
  linear readout trained jointly with BPTT.
* **M0 (frozen fly)** -- network untouched; a linear probe is fitted on pooled
  features cached in a single forward pass.  This is the diagnostic that asks
  whether the *unmodified* connectome already carries a linearly readable
  numerosity code.

Every run evaluates on four test conditions (see :mod:`flynum.stimuli.dots`) and
writes per-sample predictions so that the real-vs-shuffled comparison can be
made *paired*, item by item.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import paths
from ..config import ExperimentConfig
from ..devices import pick_device
from ..logging_utils import RunContext
from ..pipeline import prepare
from ..retina.torch_encoder import TorchRetina
from ..train.data import build_count_data
from ..train.metrics import classification_metrics, unseen_pair_report
from ..train.trainer import (
    cache_features,
    fit_readout_statistics,
    predict_count,
    probe_predict,
    train_probe,
    train_taught,
)

DEVICE = pick_device()


def _evaluate_all_modes(
    predict_fn, data, n_classes: int, ctx: RunContext, logger
) -> tuple[dict, dict]:
    """Evaluate on conditions A-D and stash per-sample predictions."""
    metrics: dict = {}
    predictions: dict = {}
    for mode in ("A", "B", "C", "D"):
        ds = data.tests[mode]
        pred = predict_fn(ds["images"])
        y = ds["labels"]
        m = classification_metrics(y, pred, n_classes)
        m["correct"] = (pred == y).astype(np.uint8)
        metrics[mode] = m
        predictions[f"{mode}_pred"] = pred.astype(np.int16)
        predictions[f"{mode}_label"] = y.astype(np.int16)
        logger.info(
            "  test %s: acc %.4f (chance %.2f) | macro-F1 %.4f | per-class %s",
            mode,
            m["accuracy"],
            m["chance"],
            m["macro_f1"],
            " ".join(f"{v:.2f}" for v in m["per_class_recall"]),
        )
        ctx.metric(split=f"test_{mode}", accuracy=m["accuracy"], macro_f1=m["macro_f1"])
    return metrics, predictions


def run_count_experiment(
    cfg: ExperimentConfig, *, force: bool = False, logger=None
) -> dict:
    """Execute one counting run and return its summary dict."""
    run_id = cfg.run_id()
    ctx = RunContext(run_id, cfg.to_dict())
    run_id = ctx.run_id          # may carry an _rN suffix if this config was rerun
    logger = ctx.log
    logger.info("=" * 78)
    logger.info(
        "COUNT RUN | model=%s graph=%s circuit=%s N=%d model_seed=%d shuffle_seed=%d",
        cfg.model, cfg.data.graph, cfg.data.circuit, cfg.train.n_train,
        cfg.seeds["model_seed"], cfg.seeds["shuffle_seed"],
    )
    logger.info("=" * 78)

    torch.manual_seed(cfg.seeds["model_seed"])
    np.random.seed(cfg.seeds["model_seed"] % (2**32 - 1))

    prep = prepare(cfg, logger=logger, force=force)
    data = build_count_data(cfg, logger=logger)
    n_classes = prep.n_classes

    retina = TorchRetina(prep.encoder, device=DEVICE)
    model = prep.build_model(device=DEVICE)
    n_params = model.n_trainable()
    logger.info(
        "model: %d trainable parameters | readout=%s (%d neurons) | alpha=%.2f steps=%d",
        n_params, prep.readout_name, len(prep.readout_idx),
        cfg.model_cfg.alpha, cfg.time.steps,
    )

    img_tr, y_tr = data.subset(cfg.train.n_train, seed=cfg.seeds["split_seed"])
    logger.info("training set: %d images (class counts %s)",
                len(y_tr), np.bincount(y_tr, minlength=n_classes).tolist())

    # Freeze the readout standardisation from the untrained network, before any
    # gradient step, so M0 and M1 see identically scaled features.
    stats = fit_readout_statistics(
        model, retina, img_tr,
        steps=cfg.time.steps,
        n_samples=cfg.model_cfg.readout_stats_samples,
        device=DEVICE,
    )
    if stats.get("standardised"):
        logger.info(
            "readout standardisation: %d features, %d constant (%.2f%%), "
            "std median %.4g, |z| p99.9 = %.1f, max %.1f",
            stats["n_features"], stats["n_constant_features"],
            100 * stats["constant_fraction"], stats["std_median"],
            stats["z_abs_p999"], stats["z_abs_max"],
        )
    elif stats.get("standardised") is False:
        if stats.get("refused"):
            logger.warning("readout standardisation REFUSED: %s", stats["refused"])
        else:
            logger.info("readout standardisation disabled")

    t0 = time.time()
    if cfg.model == "M0":
        logger.info("M0 frozen-fly protocol: caching features (single forward pass)")
        feats_tr = cache_features(model, retina, img_tr, steps=cfg.time.steps, device=DEVICE)
        feats_va = cache_features(
            model, retina, data.val_images, steps=cfg.time.steps, device=DEVICE
        )
        logger.info("  feature matrix %s (%.1f MB)", feats_tr.shape, feats_tr.nbytes / 1e6)
        head, probe_info = train_probe(
            cfg, feats_tr, y_tr, feats_va, data.val_labels, ctx, device=DEVICE, logger=logger
        )

        def predict_fn(images, _model=model, _head=head):
            f = cache_features(_model, retina, images, steps=cfg.time.steps, device=DEVICE)
            return probe_predict(_head, f, device=DEVICE)

        train_info = dict(probe_info)
    elif cfg.model == "M1":
        train_info = train_taught(
            cfg, model, retina, (img_tr, y_tr), (data.val_images, data.val_labels),
            ctx, device=DEVICE, logger=logger,
        )
        train_info["protocol"] = "BPTT on per-edge gains + readout"

        def predict_fn(images, _model=model):
            return predict_count(
                _model, retina, images, steps=cfg.time.steps, device=DEVICE
            )
    else:
        raise ValueError(f"unknown model {cfg.model!r} (M0 | M1)")

    logger.info("-" * 78)
    metrics, predictions = _evaluate_all_modes(predict_fn, data, n_classes, ctx, logger)

    np.savez_compressed(ctx.dir / "predictions.npz", **predictions)

    basis = classification_metrics(
        data.tests["A"]["labels"], predictions["A_pred"], n_classes
    )
    best_val = train_info.get("best_val_acc", float("nan"))
    summary = {
        **prep.info,
        "run_id": run_id,
        "stage": cfg.stage,
        "task": cfg.task,
        "model": cfg.model,
        "n_train": int(cfg.train.n_train),
        "model_seed": cfg.seeds["model_seed"],
        "shuffle_seed": cfg.seeds["shuffle_seed"],
        "n_params": n_params,
        "best_val_acc": best_val,
        "test_acc_a": metrics["A"]["accuracy"],
        "test_acc_b": metrics["B"]["accuracy"],
        "test_acc_c": metrics["C"]["accuracy"],
        "test_acc_d": metrics["D"]["accuracy"],
        "test_macro_f1_a": basis["macro_f1"],
        "confusion_a": metrics["A"]["confusion"],
        "per_class_recall_a": metrics["A"]["per_class_recall"],
        "per_mode": {
            m: {k: v for k, v in metrics[m].items() if k != "correct"} for m in metrics
        },
        "train_info": {k: v for k, v in train_info.items() if k != "history"},
        "wall_seconds_total": round(ctx.elapsed(), 1),
        "train_seconds": round(time.time() - t0, 1),
        "stimulus_qc": data.qc,
    }
    ctx.finish(
        "ok",
        stage=cfg.stage,
        task=cfg.task,
        model=cfg.model,
        circuit=cfg.data.circuit,
        graph=cfg.data.graph,
        n_train=cfg.train.n_train,
        model_seed=cfg.seeds["model_seed"],
        shuffle_seed=cfg.seeds["shuffle_seed"],
        best_val_acc=best_val,
        test_acc_a=metrics["A"]["accuracy"],
        test_acc_b=metrics["B"]["accuracy"],
        test_acc_c=metrics["C"]["accuracy"],
        n_params=n_params,
        epochs_run=train_info.get("epochs_run", ""),
        optimizer_steps=train_info.get("optimizer_steps", ""),
        wall_seconds=round(ctx.elapsed(), 1),
    )
    (ctx.dir / "full_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "SUMMARY | test A %.4f | B %.4f | C %.4f | D %.4f | best val %.4f | %.1f min",
        metrics["A"]["accuracy"], metrics["B"]["accuracy"], metrics["C"]["accuracy"],
        metrics["D"]["accuracy"], best_val, ctx.elapsed() / 60,
    )
    return summary


# --------------------------------------------------------------------------- #
def run_count_grid(
    configs: list[ExperimentConfig], *, force: bool = False, logger=None
) -> list[dict]:
    """Run several counting experiments sequentially, tolerating failures."""
    from ..logging_utils import get_logger

    log = logger or get_logger("count_grid")
    out = []
    for i, cfg in enumerate(configs, 1):
        log.info("[%d/%d] %s", i, len(configs), cfg.run_id())
        try:
            out.append(run_count_experiment(cfg, force=force, logger=log))
        except Exception as exc:  # keep the grid going, record the failure
            log.exception("run %s FAILED: %s", cfg.run_id(), exc)
            out.append({"run_id": cfg.run_id(), "status": "error", "error": str(exc)})
    okruns = [r for r in out if r.get("status") != "error"]
    if okruns:
        log.info(
            "grid done: %d/%d ok | mean test A %.4f",
            len(okruns), len(out), float(np.mean([r["test_acc_a"] for r in okruns])),
        )
    return out
