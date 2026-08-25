from __future__ import annotations
from pathlib import Path
import json
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ipl_production.runtime import ensure_ready, sha256_file

EXCLUDES = {".venv", ".git", "__pycache__", ".pytest_cache", "dist"}


def main():
    ensure_ready(ROOT, require_flag=True)
    out = ROOT / "dist/github_release"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    large = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in EXCLUDES for part in p.relative_to(ROOT).parts):
            continue
        size = p.stat().st_size
        if size >= 50 * 1024 * 1024:
            large.append({"file": str(p.relative_to(ROOT)).replace("\\", "/"), "mb": round(size / 1024 / 1024, 2), "sha256": sha256_file(p)})

    audit = {
        "status": "PASS",
        "champion": "V2",
        "large_files": large,
        "git_lfs_required_for_joblib": any(x["file"].endswith(".joblib") for x in large),
        "note": "Repository includes .gitattributes for *.joblib Git LFS. Run git lfs install before the first push if model files are committed.",
    }
    (out / "GITHUB_RELEASE_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "GITHUB_PUSH_CHECKLIST.md").write_text(
        "# GitHub push checklist\n\n"
        "1. Confirm `artifacts/PRODUCTION_READY.flag` exists.\n"
        "2. Review README metrics and experiment history.\n"
        "3. Install Git LFS: `git lfs install`. The repo tracks `*.joblib` through LFS.\n"
        "4. Initialize Git only in `IPL_Intelligence_Production` — do not push `.venv/` or `dist/`.\n"
        "5. Run `python -m pytest -q` before committing.\n"
        "6. Commit the frozen V2 model assets only after their local verification passed.\n"
        "7. Create the Hugging Face Space separately from `06_BUILD_HF_DEPLOYMENT_BUNDLE.bat`.\n\n"
        "## Portfolio framing\n\n"
        "Keep the V1→V4 comparison table visible. The key story is controlled experimentation: V2 won; V3 and V4 were rejected instead of hidden.\n",
        encoding="utf-8",
    )

    zip_path = out / "IPL_Intelligence_Production_GitHub_Snapshot.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if any(part in EXCLUDES for part in rel.parts):
                continue
            zf.write(p, arcname=str(Path("IPL_Intelligence_Production") / rel))
    print("GitHub release audit: PASS")
    print("Large files:", len(large))
    print("Snapshot:", zip_path)
    if large:
        print("Git LFS note: *.joblib is already configured in .gitattributes")

if __name__ == "__main__":
    main()
