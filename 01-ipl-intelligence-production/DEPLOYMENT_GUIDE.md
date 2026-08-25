# Deployment Guide

Do not deploy until the local production verification and test suite pass.

## Phase 1 — local production freeze

Run:

```text
00_SETUP_ENVIRONMENT.bat
CHECK_SYSTEM.bat
01_IMPORT_V2_CHAMPION.bat
02_VERIFY_PRODUCTION_PACKAGE.bat
03_RUN_TESTS.bat
04_LAUNCH_LOCAL_APP.bat
```

Confirm the Gradio application loads, a few team matchups return probabilities, the experiment-history table shows V2 as champion, and the tournament simulator completes successfully.

## Phase 2 — GitHub

Run:

```text
05_PREPARE_GITHUB_RELEASE.bat
```

Then review `dist/github_release/GITHUB_RELEASE_AUDIT.json` and the checklist. The repository contains `.gitattributes` configured to store `*.joblib` with Git LFS.

Recommended first-push flow from the production folder:

```text
git init
git lfs install
git add .
git commit -m "Production release: frozen IPL V2 champion"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Before pushing, verify `.venv/` and `dist/` are excluded.

## Phase 3 — Hugging Face Space

Run:

```text
06_BUILD_HF_DEPLOYMENT_BUNDLE.bat
```

The script creates a self-contained Space folder and ZIP containing only the frozen production assets needed for inference.

Create a new **Gradio** Space, then upload the contents of:

```text
dist/huggingface_space/
```

The generated Space README already contains the required YAML metadata and the accepted model metrics.

## Phase 4 — post-deployment QA

After the Space builds:

1. Compare one pre-toss prediction locally and on the Space using the same teams/venue.
2. Verify probabilities match within floating-point tolerance.
3. Run a 5,000-simulation tournament scenario.
4. Open the Experiment History tab and confirm V2 is labeled champion and V3/V4 rejected.
5. Confirm the methodology/limitations tab is visible.

Only after these checks should the public Space link be added to GitHub and the portfolio website.
