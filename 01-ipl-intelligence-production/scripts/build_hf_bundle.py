from __future__ import annotations
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ipl_production.runtime import ensure_ready


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ensure_ready(ROOT, require_flag=True)
    out = ROOT / "dist/huggingface_space"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copy(ROOT / "app.py", out / "app.py")
    copy(ROOT / "deployment/requirements_hf.txt", out / "requirements.txt")
    copy(ROOT / "deployment/README_HF_TEMPLATE.md", out / "README.md")
    copy(ROOT / "configs/production.json", out / "configs/production.json")

    for pkg in ["cricket_intel", "ipl_production"]:
        shutil.copytree(ROOT / "src" / pkg, out / pkg)

    for rel in [
        "artifacts/pretoss_model_bundle.joblib",
        "artifacts/posttoss_model_bundle.joblib",
        "artifacts/model_metadata_pretoss.json",
        "artifacts/model_metadata_posttoss.json",
        "artifacts/PRODUCTION_ASSET_MANIFEST.json",
        "artifacts/PRODUCTION_READY.flag",
        "data/processed/latest_team_snapshot_v2.json",
        "reports/FROZEN_CHAMPION_METRICS.json",
        "reports/FROZEN_V1_V2_V3_V4_COMPARISON.csv",
        "reports/EXPERIMENT_DECISIONS.json",
        "reports/strict_test_predictions_pretoss.csv",
        "reports/model_comparison_pretoss.csv",
        "reports/strict_test_by_year_pretoss.csv",
    ]:
        copy(ROOT / rel, out / rel)

    for rel in [
        "reports/expanding_window_backtest_pretoss.csv",
        "reports/permutation_importance_pretoss.csv",
    ]:
        p = ROOT / rel
        if p.exists():
            copy(p, out / rel)

    zip_path = ROOT / "dist/IPL_Intelligence_HuggingFace_Space.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out))
    print("Hugging Face Space bundle created:")
    print(out)
    print(zip_path)

if __name__ == "__main__":
    main()
