from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipl_production.governance import frozen_metrics, metrics_match, recompute_strict_metrics
from ipl_production.runtime import sha256_file


REQUIRED = {
    "artifacts/pretoss_model_bundle.joblib": "artifacts/pretoss_model_bundle.joblib",
    "artifacts/posttoss_model_bundle.joblib": "artifacts/posttoss_model_bundle.joblib",
    "artifacts/model_metadata_pretoss.json": "artifacts/model_metadata_pretoss.json",
    "artifacts/model_metadata_posttoss.json": "artifacts/model_metadata_posttoss.json",
    "data/processed/latest_team_snapshot_v2.json": "data/processed/latest_team_snapshot_v2.json",
    "data/processed/feature_schema_v2.json": "data/processed/feature_schema_v2.json",
    "data/metadata/data_validation.json": "data/metadata/source_v2_data_validation.json",
    "reports/strict_test_predictions_pretoss.csv": "reports/strict_test_predictions_pretoss.csv",
    "reports/strict_test_predictions_posttoss.csv": "reports/strict_test_predictions_posttoss.csv",
    "reports/strict_test_by_year_pretoss.csv": "reports/strict_test_by_year_pretoss.csv",
    "reports/strict_test_by_year_posttoss.csv": "reports/strict_test_by_year_posttoss.csv",
    "reports/model_comparison_pretoss.csv": "reports/model_comparison_pretoss.csv",
    "reports/model_comparison_posttoss.csv": "reports/model_comparison_posttoss.csv",
}

OPTIONAL = {
    "reports/expanding_window_backtest_pretoss.csv": "reports/expanding_window_backtest_pretoss.csv",
    "reports/expanding_window_backtest_posttoss.csv": "reports/expanding_window_backtest_posttoss.csv",
    "reports/permutation_importance_pretoss.csv": "reports/permutation_importance_pretoss.csv",
    "reports/permutation_importance_posttoss.csv": "reports/permutation_importance_posttoss.csv",
    "reports/championship_simulation_v2.csv": "reports/championship_simulation_v2.csv",
    "reports/github_metrics.json": "reports/source_v2_github_metrics.json",
    "artifacts/ensemble_weights_pretoss.json": "artifacts/ensemble_weights_pretoss.json",
    "artifacts/ensemble_weights_posttoss.json": "artifacts/ensemble_weights_posttoss.json",
}


def resolve_source(arg: str | None) -> Path:
    candidates = []
    if arg:
        candidates.append(Path(arg).expanduser())
    candidates.extend(
        [
            ROOT.parent / "IPL_Intelligence_Lab_V2",
            ROOT.parent / "IPL_Intelligence_Lab_V2_Complete_Local_Project" / "IPL_Intelligence_Lab_V2",
        ]
    )
    for c in candidates:
        c = c.resolve()
        if (c / "artifacts/pretoss_model_bundle.joblib").exists():
            return c
    lines = "\n".join(f"  - {c}" for c in candidates)
    raise SystemExit(
        "Could not locate a trained V2 sibling project. Checked:\n" + lines +
        "\nKeep IPL_Intelligence_Production next to IPL_Intelligence_Lab_V2, or pass --source."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the already-selected V2 champion into the production repo")
    parser.add_argument("--source", help="Optional explicit path to trained IPL_Intelligence_Lab_V2")
    args = parser.parse_args()
    src = resolve_source(args.source)

    missing = [rel for rel in REQUIRED if not (src / rel).exists()]
    if missing:
        raise SystemExit("The V2 project is not complete. Missing required files:\n  - " + "\n  - ".join(missing))

    validation = json.loads((src / "data/metadata/data_validation.json").read_text(encoding="utf-8"))
    if validation.get("mode") != "cricsheet_current_v2" or validation.get("status") != "PASS":
        raise SystemExit(
            "Refusing to freeze V2: source data_validation.json must report "
            "mode=cricsheet_current_v2 and status=PASS."
        )

    actual = recompute_strict_metrics(src / "reports/strict_test_predictions_pretoss.csv")
    ok, diffs = metrics_match(frozen_metrics(), actual, atol=6e-7)
    if not ok:
        raise SystemExit(
            "Refusing to freeze V2: recomputed strict metrics do not match the accepted champion benchmark.\n"
            + json.dumps({"expected": frozen_metrics(), "actual": actual, "diff": diffs}, indent=2)
        )

    copied = []
    for src_rel, dst_rel in REQUIRED.items():
        sp, dp = src / src_rel, ROOT / dst_rel
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)
        copied.append((src_rel, dst_rel, True))
    for src_rel, dst_rel in OPTIONAL.items():
        sp = src / src_rel
        if not sp.exists():
            continue
        dp = ROOT / dst_rel
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)
        copied.append((src_rel, dst_rel, False))

    manifest = {
        "production_project": "IPL_Intelligence_Production",
        "champion": "V2",
        "source_folder": str(src),
        "source_data_mode": validation.get("mode"),
        "source_data_status": validation.get("status"),
        "source_date_max": validation.get("date_max"),
        "strict_metrics_recomputed": actual,
        "files": [],
    }
    for src_rel, dst_rel, required in copied:
        dp = ROOT / dst_rel
        manifest["files"].append(
            {
                "source": src_rel,
                "destination": dst_rel,
                "required": required,
                "bytes": dp.stat().st_size,
                "sha256": sha256_file(dp),
            }
        )
    (ROOT / "artifacts/PRODUCTION_ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Any prior ready flag becomes invalid after re-import until verification runs again.
    (ROOT / "artifacts/PRODUCTION_READY.flag").unlink(missing_ok=True)

    print("=" * 76)
    print("V2 CHAMPION IMPORT COMPLETE")
    print("=" * 76)
    print(f"Source              : {src}")
    print(f"Data mode           : {validation.get('mode')}")
    print(f"Data status         : {validation.get('status')}")
    print(f"Files imported      : {len(copied)}")
    print("Strict metrics      : VERIFIED against frozen V2 benchmark")
    print("Production ready?   : NOT YET - run 02_VERIFY_PRODUCTION_PACKAGE.bat")


if __name__ == "__main__":
    main()
