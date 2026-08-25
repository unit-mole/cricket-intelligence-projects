# Build Validation Report

This report describes package-level validation performed before delivery. It is **not** a substitute for the user's local V2 artifact verification because the accepted trained V2 bundles live in the user's existing `IPL_Intelligence_Lab_V2` folder.

## Delivery checks

- Production repository created independently of V1/V2/V3/V4 experiment folders.
- Frozen V1→V4 comparison included with V2 labeled champion.
- Frozen V2 strict metrics included from the accepted local experiment.
- Production import script refuses any source other than `cricsheet_current_v2` with `PASS` validation.
- Import script recomputes the accepted V2 strict metrics before copying artifacts.
- Production verification checks file checksums, 142 strict rows, 2025–2026 years, V2 metadata, runtime predictions and probability symmetry.
- Local app remains locked until `PRODUCTION_READY.flag` exists.
- GitHub release script includes Git LFS guidance for model bundles.
- Hugging Face builder creates a self-contained Gradio Space only after verification.
- No V5 training code is included.
- No fabricated production model metrics or fake trained joblib assets are included in the delivery ZIP.
- Python source compilation: **PASS**.
- Packaged unit/integrity test suite: **25 passed**.

The final local verification steps are intentionally performed on the user's machine after importing the already-trained V2 champion.
