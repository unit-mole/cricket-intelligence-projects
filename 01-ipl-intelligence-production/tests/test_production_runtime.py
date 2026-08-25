from pathlib import Path
import json

import numpy as np
import pandas as pd

from ipl_production.runtime import ProductionPaths, missing_runtime_files, normalized_venue, parse_lineup, sha256_file
from ipl_production.governance import frozen_metrics, load_decisions, load_experiment_table, metrics_match

ROOT = Path(__file__).resolve().parents[1]


def test_parse_lineup_handles_commas_semicolons_and_newlines():
    assert parse_lineup("A, B; C\nD") == ["A", "B", "C", "D"]


def test_parse_lineup_deduplicates_preserving_order():
    assert parse_lineup("A,B,A,C,B") == ["A", "B", "C"]


def test_parse_lineup_blank():
    assert parse_lineup("") == []
    assert parse_lineup(None) == []


def test_normalized_neutral_venue():
    assert normalized_venue("Neutral / unknown") == ""
    assert normalized_venue("Neutral") == ""
    assert normalized_venue("Unknown") == ""


def test_normalized_real_venue_is_preserved():
    assert normalized_venue("Wankhede Stadium") == "Wankhede Stadium"


def test_production_paths_contract(tmp_path: Path):
    p = ProductionPaths(tmp_path)
    assert p.pretoss_bundle == tmp_path / "artifacts/pretoss_model_bundle.joblib"
    assert p.snapshot == tmp_path / "data/processed/latest_team_snapshot_v2.json"


def test_missing_runtime_files_detects_empty_project(tmp_path: Path):
    missing = missing_runtime_files(tmp_path)
    assert "artifacts/pretoss_model_bundle.joblib" in missing
    assert "reports/FROZEN_CHAMPION_METRICS.json" in missing


def test_sha256_file_is_stable(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("ipl-v2", encoding="utf-8")
    assert sha256_file(p) == sha256_file(p)
    first = sha256_file(p)
    p.write_text("ipl-v2-changed", encoding="utf-8")
    assert sha256_file(p) != first


def test_frozen_metrics_contract():
    m = frozen_metrics()
    assert set(m) == {"accuracy", "balanced_accuracy", "f1", "log_loss", "brier", "ece", "roc_auc"}
    assert abs(m["accuracy"] - 0.5422535211267606) < 1e-12
    assert abs(m["roc_auc"] - 0.5713414634146342) < 1e-12


def test_metrics_match_accepts_small_rounding():
    m = frozen_metrics()
    actual = {k: v + 1e-8 for k, v in m.items()}
    ok, _ = metrics_match(m, actual, atol=1e-7)
    assert ok


def test_metrics_match_rejects_real_change():
    m = frozen_metrics()
    actual = dict(m)
    actual["accuracy"] += 0.01
    ok, diffs = metrics_match(m, actual, atol=1e-7)
    assert not ok
    assert diffs["accuracy"] > 0


def test_experiment_table_has_four_versions_and_v2_champion():
    df = load_experiment_table(ROOT)
    assert len(df) == 4
    assert df.loc[df.system.eq("V2_pretoss"), "status"].iloc[0] == "CHAMPION"
    assert df.loc[df.system.eq("V2_pretoss"), "roc_auc"].iloc[0] == df.roc_auc.max()


def test_v2_has_best_probability_objective_in_frozen_table():
    df = load_experiment_table(ROOT)
    best = df.sort_values("probability_objective").iloc[0]
    assert best.system == "V2_pretoss"


def test_governance_decision_freezes_v2_and_no_v5():
    d = load_decisions(ROOT)
    assert d["V2"]["status"] == "champion"
    assert "no V5" in d["final_decision"]


def test_frozen_report_json_matches_runtime_constants():
    report = json.loads((ROOT / "reports/FROZEN_CHAMPION_METRICS.json").read_text(encoding="utf-8"))
    assert report["champion"] == "V2"
    assert report["strict_test_rows"] == 142
    for k, v in frozen_metrics().items():
        assert abs(report["metrics"][k] - v) < 1e-12
