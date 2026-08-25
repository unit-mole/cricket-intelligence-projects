# Deployment Guide

1. Complete local import, verification, tests, and app QA.
2. Run `05_PREPARE_GITHUB_RELEASE.bat`.
3. Run `07_INSTALL_IN_CRICKET_MONOREPO.bat`.
4. Review `git status`, then commit/push the umbrella repository.
5. Run `06_BUILD_HF_DEPLOYMENT_BUNDLE.bat`.
6. Create public Gradio Space `icc-world-cup-intelligence` and push the contents of `dist/huggingface_space`.
