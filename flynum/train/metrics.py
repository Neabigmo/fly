"""Evaluation metrics.

The five primary metrics of the study, plus the uncertainty machinery needed to
say something honest about the real-vs-shuffled gap:

1. count accuracy on the natural condition (A),
2. area-controlled accuracy (B),
3. addition accuracy,
4. unseen-pair accuracy,
5. the real-minus-shuffled gap, with bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    return cm


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict:
    """Accuracy, macro-F1, per-class recall and the confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0

    per_class_recall = []
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_recall.append(float(recall))
        f1s.append(float(f1))

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class_recall": per_class_recall,
        "confusion": cm.tolist(),
        "n": int(total),
        "chance": 1.0 / n_classes,
    }


def bootstrap_ci(
    correct: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean of 0/1 correctness values."""
    correct = np.asarray(correct, dtype=np.float64)
    if len(correct) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    means = correct[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_gap(
    correct_real: np.ndarray,
    correct_shuf: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap the real-minus-shuffled accuracy gap.

    Both arrays must be indexed identically (same test items), which is what the
    paired evaluation protocol guarantees.
    """
    a = np.asarray(correct_real, dtype=np.float64)
    b = np.asarray(correct_shuf, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"paired gap needs equal lengths, got {len(a)} and {len(b)}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "acc_real": float(a.mean()),
        "acc_shuffled": float(b.mean()),
        "gap": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_gap_le_zero": float((diffs <= 0).mean()),
        "n": int(len(a)),
    }


def unseen_pair_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holdout: np.ndarray,
    n_classes: int,
) -> dict:
    """Accuracy on the compositional-generalisation (held-out) pairs."""
    is_holdout = np.asarray(is_holdout, dtype=bool)
    out = {"n_unseen": int(is_holdout.sum()), "n_seen": int((~is_holdout).sum())}
    for name, sel in (("seen", ~is_holdout), ("unseen", is_holdout)):
        if sel.sum():
            m = classification_metrics(y_true[sel], y_pred[sel], n_classes)
            out[f"accuracy_{name}"] = m["accuracy"]
            out[f"macro_f1_{name}"] = m["macro_f1"]
        else:
            out[f"accuracy_{name}"] = float("nan")
    out["generalisation_drop"] = out.get("accuracy_seen", float("nan")) - out.get(
        "accuracy_unseen", float("nan")
    )
    return out
