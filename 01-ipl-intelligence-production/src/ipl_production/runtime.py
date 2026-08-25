from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cricket_intel.features import matchup_from_snapshot
from cricket_intel.simulation import active_ipl_teams, simulate_ipl


@dataclass(frozen=True)
class ProductionPaths:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def data(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def pretoss_bundle(self) -> Path:
        return self.artifacts / "pretoss_model_bundle.joblib"

    @property
    def posttoss_bundle(self) -> Path:
        return self.artifacts / "posttoss_model_bundle.joblib"

    @property
    def snapshot(self) -> Path:
        return self.data / "latest_team_snapshot_v2.json"

    @property
    def ready_flag(self) -> Path:
        return self.artifacts / "PRODUCTION_READY.flag"


REQUIRED_RUNTIME_FILES = (
    "artifacts/pretoss_model_bundle.joblib",
    "artifacts/posttoss_model_bundle.joblib",
    "data/processed/latest_team_snapshot_v2.json",
    "reports/strict_test_predictions_pretoss.csv",
    "reports/model_comparison_pretoss.csv",
    "reports/FROZEN_CHAMPION_METRICS.json",
    "reports/FROZEN_V1_V2_V3_V4_COMPARISON.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_runtime_files(root: Path) -> list[str]:
    return [rel for rel in REQUIRED_RUNTIME_FILES if not (root / rel).exists()]


def ensure_ready(root: Path, require_flag: bool = True) -> None:
    missing = missing_runtime_files(root)
    if missing:
        raise RuntimeError(
            "Production assets are missing: " + ", ".join(missing) + ". "
            "Run 01_IMPORT_V2_CHAMPION.bat and 02_VERIFY_PRODUCTION_PACKAGE.bat."
        )
    if require_flag and not (root / "artifacts/PRODUCTION_READY.flag").exists():
        raise RuntimeError(
            "Production assets exist but have not passed the verification gate. "
            "Run 02_VERIFY_PRODUCTION_PACKAGE.bat before launching."
        )


def load_runtime(root: Path, require_flag: bool = True) -> dict[str, Any]:
    ensure_ready(root, require_flag=require_flag)
    paths = ProductionPaths(root)
    snapshot = json.loads(paths.snapshot.read_text(encoding="utf-8"))
    pre = joblib.load(paths.pretoss_bundle)
    post = joblib.load(paths.posttoss_bundle)
    return {
        "snapshot": snapshot,
        "pretoss_bundle": pre,
        "posttoss_bundle": post,
        "config": json.loads((root / "configs/production.json").read_text(encoding="utf-8")),
    }


def parse_lineup(text: str | None) -> list[str]:
    if not text:
        return []
    normalized = str(text).replace("\n", ",").replace(";", ",")
    out: list[str] = []
    seen: set[str] = set()
    for raw in normalized.split(","):
        name = raw.strip()
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def normalized_venue(value: str | None) -> str:
    value = str(value or "")
    return "" if value in {"Neutral / unknown", "Neutral", "Unknown"} else value


def predict_match(
    snapshot: dict,
    pretoss_bundle,
    posttoss_bundle,
    mode: str,
    team_a: str,
    team_b: str,
    venue: str = "",
    toss_side: str = "Team A",
    toss_decision: str = "field",
    lineup_a: list[str] | None = None,
    lineup_b: list[str] | None = None,
) -> dict[str, Any]:
    if team_a == team_b:
        raise ValueError("Choose two different teams")
    mode_key = "pretoss" if str(mode).lower().startswith("pre") else "posttoss"
    bundle = pretoss_bundle if mode_key == "pretoss" else posttoss_bundle
    toss_winner = ""
    if mode_key == "posttoss":
        toss_winner = team_a if toss_side == "Team A" else team_b
    X = matchup_from_snapshot(
        snapshot,
        team_a,
        team_b,
        venue=normalized_venue(venue),
        mode=mode_key,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        lineup1=lineup_a or None,
        lineup2=lineup_b or None,
    )
    p = float(bundle.predict_proba(X)[:, 1][0])
    return {
        "team_a": team_a,
        "team_b": team_b,
        "p_team_a": p,
        "p_team_b": 1.0 - p,
        "mode": mode_key,
        "feature_row": X,
    }


def local_sensitivity(bundle, X: pd.DataFrame, team_a: str, limit: int = 12) -> pd.DataFrame:
    base = float(bundle.predict_proba(X)[:, 1][0])
    rows = []
    for col in X.columns:
        value = float(X.iloc[0][col])
        if abs(value) < 1e-12:
            continue
        neutral = X.copy()
        neutral.loc[:, col] = 0.0
        p0 = float(bundle.predict_proba(neutral)[:, 1][0])
        delta = base - p0
        rows.append(
            {
                "feature": col,
                "feature_value": value,
                "probability_delta": delta,
                "direction": f"Helps {team_a}" if delta > 0 else f"Hurts {team_a}",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "feature_value", "probability_delta", "direction"])
    return (
        pd.DataFrame(rows)
        .assign(_abs=lambda d: d.probability_delta.abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        .head(int(limit))
        .reset_index(drop=True)
    )


def team_intelligence(snapshot: dict, team: str) -> pd.DataFrame:
    s = snapshot["teams"][team]
    matches = float(s.get("matches", 0) or 0)
    metrics = [
        ("Elo", s.get("elo")),
        ("Fast Elo", s.get("fast_elo")),
        ("Slow Elo", s.get("slow_elo")),
        ("Matches", s.get("matches")),
        ("Win rate", (float(s.get("wins", 0) or 0) / matches) if matches else 0.0),
        ("Last-5 form", s.get("form5")),
        ("Last-10 form", s.get("form10")),
        ("EWMA form", s.get("form_ewm")),
        ("Recent runs / innings", s.get("runs10")),
        ("Recent run rate", s.get("run_rate10")),
        ("Powerplay batting RR", s.get("pp_bat_rr")),
        ("Middle-over batting RR", s.get("middle_bat_rr")),
        ("Death-over batting RR", s.get("death_bat_rr")),
        ("Powerplay bowling RR conceded", s.get("pp_bowl_rr")),
        ("Middle bowling RR conceded", s.get("middle_bowl_rr")),
        ("Death bowling RR conceded", s.get("death_bowl_rr")),
        ("Chasing win rate", s.get("chase_form")),
        ("Defending win rate", s.get("defend_form")),
        ("Squad proxy batting runs", s.get("squad_batting_runs")),
        ("Squad proxy bowling wickets", s.get("squad_bowling_wickets")),
        ("Last match date", s.get("last_date")),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value"])


def player_intelligence(snapshot: dict, team: str, limit: int = 30) -> pd.DataFrame:
    rows = []
    for player, p in (snapshot.get("players", {}) or {}).items():
        if p.get("last_team") != team:
            continue
        rows.append(
            {
                "player": player,
                "matches": p.get("matches", 0),
                "recent_runs_per_match": p.get("runs_per_match", 0),
                "recent_strike_rate": p.get("strike_rate", 0),
                "recent_wickets_per_match": p.get("wickets_per_match", 0),
                "recent_economy": p.get("economy", 0),
                "last_date": p.get("last_date"),
            }
        )
    if not rows:
        return pd.DataFrame({"Info": ["No player history available in the production snapshot."]})
    return (
        pd.DataFrame(rows)
        .sort_values(["last_date", "matches"], ascending=[False, False])
        .head(int(limit))
        .reset_index(drop=True)
    )


def simulate_championship(pretoss_bundle, snapshot: dict, n: int = 25000, seed: int = 42) -> pd.DataFrame:
    return simulate_ipl(pretoss_bundle, snapshot, n=int(n), seed=int(seed))


def symmetry_error(bundle, X: pd.DataFrame) -> float:
    p = float(bundle.predict_proba(X)[:, 1][0])
    q = float(bundle.predict_proba(-X)[:, 1][0])
    return abs((p + q) - 1.0)


def active_teams(snapshot: dict) -> list[str]:
    return active_ipl_teams(snapshot)
