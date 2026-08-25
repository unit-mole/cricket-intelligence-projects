from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipl_production.governance import frozen_metrics, metrics_match, recompute_strict_metrics
from ipl_production.runtime import ProductionPaths, active_teams, missing_runtime_files, sha256_file, symmetry_error
from cricket_intel.features import matchup_from_snapshot


def fail(msg: str) -> None:
    (ROOT / "artifacts/PRODUCTION_READY.flag").unlink(missing_ok=True)
    raise SystemExit(msg)


def main() -> None:
    missing = missing_runtime_files(ROOT)
    if missing:
        fail("Missing production assets:\n  - " + "\n  - ".join(missing))

    manifest_path = ROOT / "artifacts/PRODUCTION_ASSET_MANIFEST.json"
    if not manifest_path.exists():
        fail("Missing PRODUCTION_ASSET_MANIFEST.json. Run 01_IMPORT_V2_CHAMPION.bat first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checksum_failures = []
    for item in manifest.get("files", []):
        p = ROOT / item["destination"]
        if not p.exists() or sha256_file(p) != item["sha256"]:
            checksum_failures.append(item["destination"])
    if checksum_failures:
        fail("Checksum verification failed:\n  - " + "\n  - ".join(checksum_failures))

    validation = json.loads((ROOT / "data/metadata/source_v2_data_validation.json").read_text(encoding="utf-8"))
    if validation.get("mode") != "cricsheet_current_v2" or validation.get("status") != "PASS":
        fail("Source V2 data validation is not the accepted cricsheet_current_v2 PASS run.")

    strict_path = ROOT / "reports/strict_test_predictions_pretoss.csv"
    strict = pd.read_csv(strict_path)
    if len(strict) != 142:
        fail(f"Expected 142 strict V2 rows (2025-2026); found {len(strict)}")
    if "year" in strict.columns and sorted(strict["year"].astype(int).unique().tolist()) != [2025, 2026]:
        fail("Strict prediction years are not exactly [2025, 2026].")
    if "match_id" in strict.columns and strict["match_id"].astype(str).duplicated().any():
        fail("Duplicate match_id detected in strict predictions.")
    actual = recompute_strict_metrics(strict_path)
    ok, diffs = metrics_match(frozen_metrics(), actual, atol=6e-7)
    if not ok:
        fail("Frozen champion metrics do not reproduce:\n" + json.dumps({"actual": actual, "diff": diffs}, indent=2))

    paths = ProductionPaths(ROOT)
    snapshot = json.loads(paths.snapshot.read_text(encoding="utf-8"))
    pre = joblib.load(paths.pretoss_bundle)
    post = joblib.load(paths.posttoss_bundle)
    teams = active_teams(snapshot)
    if len(teams) != 10:
        fail(f"Expected 10 active IPL teams in production snapshot; found {len(teams)}")

    pair_checks = []
    for i in range(min(5, len(teams) - 1)):
        a, b = teams[i], teams[i + 1]
        Xpre = matchup_from_snapshot(snapshot, a, b, venue="", mode="pretoss")
        p = float(pre.predict_proba(Xpre)[:, 1][0])
        err = symmetry_error(pre, Xpre)
        if not (0.0 < p < 1.0) or err > 1e-8:
            fail(f"Pre-toss runtime sanity check failed for {a} vs {b}: p={p}, symmetry_error={err}")
        Xpost = matchup_from_snapshot(snapshot, a, b, venue="", mode="posttoss", toss_winner=a, toss_decision="field")
        pp = float(post.predict_proba(Xpost)[:, 1][0])
        err_post = symmetry_error(post, Xpost)
        if not (0.0 < pp < 1.0) or err_post > 1e-8:
            fail(f"Post-toss runtime sanity check failed for {a} vs {b}")
        pair_checks.append({"team_a": a, "team_b": b, "pretoss_probability": p, "posttoss_probability": pp, "pretoss_symmetry_error": err, "posttoss_symmetry_error": err_post})

    metadata = json.loads((ROOT / "artifacts/model_metadata_pretoss.json").read_text(encoding="utf-8"))
    if str(metadata.get("project_version")) != "2.0.0":
        fail("Imported pre-toss bundle metadata is not V2.0.0.")
    if metadata.get("strict_test_years") != [2025, 2026]:
        fail("Imported V2 metadata strict test years do not match [2025, 2026].")

    report = {
        "status": "PASS",
        "production_champion": "V2",
        "strict_metrics_recomputed": actual,
        "strict_rows": len(strict),
        "strict_years": [2025, 2026],
        "active_teams": teams,
        "runtime_pair_checks": pair_checks,
        "checksum_files_verified": len(manifest.get("files", [])),
        "source_data_mode": validation.get("mode"),
        "source_data_date_max": validation.get("date_max"),
        "governance": "Frozen V2 champion; no retraining occurred in production packaging.",
    }
    (ROOT / "reports/PRODUCTION_VERIFICATION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "artifacts/PRODUCTION_READY.flag").write_text("V2_PRODUCTION_READY\n", encoding="utf-8")

    print("=" * 76)
    print("IPL PRODUCTION VERIFICATION: PASS")
    print("=" * 76)
    print("Champion            : V2")
    print("Strict window       : 2025-2026")
    print(f"Strict rows         : {len(strict)}")
    print(f"Accuracy            : {actual['accuracy']:.4f}")
    print(f"ROC-AUC             : {actual['roc_auc']:.4f}")
    print(f"Log Loss            : {actual['log_loss']:.4f}")
    print(f"Brier               : {actual['brier']:.4f}")
    print(f"Runtime pair checks : {len(pair_checks)}")
    print("Probability symmetry: PASS")
    print("Asset checksums      : PASS")
    print("Production flag      : artifacts\\PRODUCTION_READY.flag")


if __name__ == "__main__":
    main()
