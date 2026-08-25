from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, brier_score_loss, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def binary_metrics(y, p) -> dict[str, float | None]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, np.c_[1 - p, p], labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["roc_auc"] = None
    return out


def probability_objective(y, p) -> float:
    m = binary_metrics(y, p)
    return float(m["log_loss"] + 0.25 * m["brier"] + 0.10 * m["ece"])
