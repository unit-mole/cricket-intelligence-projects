# ICC World Cup Intelligence Engine

A production-oriented ODI World Cup forecasting and tournament-intelligence system that separates **model selection** from **software selection**.

The production architecture combines:

- **V1 forecasting model** — retained as the production champion after the exact V1/V2 acceptance gate;
- **V2 2027 tournament engine** — retained because it corrected the official 57-match tournament structure and expanded engineering validation.

## Final model-governance decision

| System | Accuracy | ROC-AUC | Log Loss | Brier | ECE | Decision |
|---|---:|---:|---:|---:|---:|---|
| **V1 actual model** | **59.02%** | 0.6270 | 0.6680 | 0.2377 | **0.0543** | **MODEL CHAMPION** |
| V2 challenger | 60.93% | **0.6421** | **0.6679** | **0.2367** | 0.0655 | Rejected by pre-registered probability objective |

V2 improved several individual metrics, but its probability objective was `0.733609` versus `0.732888` for V1, where lower is better. The project did **not** change the acceptance rule after seeing the strict test results.

## Production architecture

```text
Accepted V1 model
       +
Corrected V2 2027 simulator
       +
Expanded V2 tests
       +
Fixed production explainability
       |
       v
ICC World Cup Intelligence Engine
```

## 2027 tournament structure

The simulator implements the project’s validated 57-match contract:

- Super Series: 3 matches
- Round 2: 30 matches (two groups of six)
- Super 7: 21 matches
- Semifinals: 2 matches (1st vs 4th, 2nd vs 3rd)
- Final: 1 match
- Total: 57 matches

Scenario participant seeding and scenario group allocation are clearly labelled and must not be presented as an official qualifier list or official group draw unless authoritative inputs are supplied.

## Application modules

- Executive Overview
- Match Predictor
- 2027 Tournament Simulator
- Team Intelligence
- Model Evaluation
- Experiment History
- Historical Backtesting
- Explainability
- Methodology & Limitations

## Production explainability fix

V2’s generic `permutation_importance` call could not treat the custom strategy bundle as a fitted sklearn estimator. Production packaging therefore computes permutation importance manually against the frozen V1 ensemble: each feature is shuffled repeatedly and importance is measured as the increase in strict-window log loss. The frozen V1 model itself is not retrained or altered.

## Local workflow

```bat
00_SETUP_ENVIRONMENT.bat
CHECK_SYSTEM.bat
01_IMPORT_FINAL_COMPONENTS.bat
02_VERIFY_PRODUCTION_PACKAGE.bat
03_RUN_TESTS.bat
04_LAUNCH_LOCAL_APP.bat
05_PREPARE_GITHUB_RELEASE.bat
06_BUILD_HF_DEPLOYMENT_BUNDLE.bat
07_INSTALL_IN_CRICKET_MONOREPO.bat
```

## GitHub destination

`cricket-intelligence-projects/02-world-cup-intelligence/`

## Hugging Face destination

Recommended Space: `anmol-unitmole/icc-world-cup-intelligence`

## Responsible use

This is a probabilistic sports-analytics and portfolio application. It is not a guarantee of match outcomes and is not betting advice.
