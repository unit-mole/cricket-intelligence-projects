# Build Validation Report

- Project: ICC World Cup Intelligence Production
- Production model: V1 actual local champion imported at runtime
- Tournament engine: V2 corrected 2027 structure
- Public monorepo folder: `02-world-cup-intelligence`
- Hugging Face Space target: `icc-world-cup-intelligence`
- Python source compile check: PASS
- Packaged structural/unit tests: **25 passed**
- Production model assets intentionally not fabricated in this ZIP; `01_IMPORT_FINAL_COMPONENTS.bat` imports the user's completed local V1/V2 run and revalidates the exact same-window governance result.
- Explainability implementation: manual permutation log-loss importance against the frozen V1 ensemble; no estimator `fit` wrapper is required.
- 2027 simulator contract: 3 + 30 + 21 + 2 + 1 = 57 matches.
