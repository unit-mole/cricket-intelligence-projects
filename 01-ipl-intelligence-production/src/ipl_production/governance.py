from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from cricket_intel.metrics import binary_metrics

EXPECTED = {
    "accuracy": 0.5422535211267606,
    "balanced_accuracy": 0.55,
    "f1": 0.5255474452554745,
    "log_loss": 0.685847206895017,
    "brier": 0.24636225510466153,
    "ece": 0.0878613566915955,
    "roc_auc": 0.5713414634146342,
}


def frozen_metrics() -> dict[str, float]:
    return dict(EXPECTED)


def recompute_strict_metrics(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    required = {"target", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strict predictions missing columns: {sorted(missing)}")
    return binary_metrics(df["target"].astype(int).to_numpy(), df["probability"].astype(float).to_numpy())


def metrics_match(expected: dict[str, float], actual: dict[str, float], atol: float = 5e-10) -> tuple[bool, dict[str, float]]:
    diffs = {}
    ok = True
    for key, exp in expected.items():
        act = float(actual[key])
        diffs[key] = act - float(exp)
        if not np.isclose(act, float(exp), atol=atol, rtol=0.0):
            ok = False
    return ok, diffs


def load_experiment_table(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "reports/FROZEN_V1_V2_V3_V4_COMPARISON.csv")


def load_decisions(root: Path) -> dict:
    return json.loads((root / "reports/EXPERIMENT_DECISIONS.json").read_text(encoding="utf-8"))
