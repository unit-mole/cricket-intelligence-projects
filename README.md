# Cricket Intelligence Projects

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EC4E20.svg)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-2E8B57.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-FFCC00.svg)](https://catboost.ai/)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive%20ML%20Apps-orange.svg)](https://www.gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-2%20Live%20Spaces-ffd21e.svg)](https://huggingface.co/anmol-unitmole)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Project--Specific%20CI-2088ff.svg)](https://github.com/unit-mole/cricket-intelligence-projects/actions)
[![Sports Analytics](https://img.shields.io/badge/Domain-Cricket%20%26%20Sports%20Analytics-2ea44f.svg)](https://github.com/unit-mole/cricket-intelligence-projects)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A structured portfolio of **two completed end-to-end cricket machine-learning projects** covering IPL match forecasting, ODI World Cup forecasting, calibrated probability estimation, chronological validation, model governance, historical backtesting, explainability, and probabilistic tournament simulation.

Each project is developed beyond notebook-only experimentation and includes reproducible source code, point-in-time feature engineering, controlled model comparison, strict out-of-time evaluation, automated testing, GitHub Actions validation, production packaging, and a publicly accessible Hugging Face application.

**Portfolio status:** 2 completed and deployed cricket intelligence projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)  
**Deployment portfolio:** 2 Hugging Face Spaces  
**Primary focus:** Sports Analytics · Probabilistic Machine Learning · Model Governance · Tournament Simulation · ML Deployment

---

## Portfolio Objective

This repository demonstrates how machine-learning systems can be built for cricket forecasting while preserving the principles that matter in real predictive systems:

- leakage-safe feature construction;
- chronological validation;
- out-of-time testing;
- calibrated probabilities;
- model comparison under common evaluation windows;
- explicit acceptance and rejection gates;
- historical backtesting;
- model-derived explainability;
- tournament simulation;
- production model freezing;
- reproducible packaging;
- automated testing and CI;
- transparent limitations;
- public deployment.

The portfolio is intentionally designed to show more than raw predictive accuracy.

A central theme across both projects is:

> A newer or more complex model is promoted only when the evidence supports it.

The portfolio is intended to demonstrate skills relevant to:

- Data Science;
- Machine Learning;
- Applied Artificial Intelligence;
- Sports Analytics;
- Predictive Modeling;
- Probabilistic Forecasting;
- Model Evaluation;
- Model Governance;
- Analytics Engineering;
- ML application development;
- Quality-oriented analytical thinking.

---

## Completed Projects

| No. | Project | Cricket Problem | Production Decision | Primary Deployment | Status |
|---:|---|---|---|---|---|
| 1 | [IPL Intelligence Production](01-ipl-intelligence-production/) | IPL pre-match forecasting + tournament simulation | **V2 selected after V1-V4 comparison** | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/ipl-intelligence-production) |
| 2 | [ICC World Cup Intelligence Engine](02-world-cup-intelligence/) | ODI match forecasting + corrected 2027 World Cup simulation | **V1 model retained; V2 simulator adopted** | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/icc-world-cup-intelligence) |

---

## Portfolio at a Glance

| Coverage Area | Demonstrated Through |
|---|---|
| IPL forecasting | Project 01 |
| ODI forecasting | Project 02 |
| Elo rating systems | Projects 01 and 02 |
| Recent-form modeling | Projects 01 and 02 |
| Opponent-adjusted form | Projects 01 and 02 |
| Venue / match context | Projects 01 and 02 |
| Player / squad experiments | Project 01 |
| Model-family comparison | Projects 01 and 02 |
| Probability calibration | Projects 01 and 02 |
| Chronological validation | Projects 01 and 02 |
| Strict out-of-time testing | Projects 01 and 02 |
| Model governance | Projects 01 and 02 |
| Acceptance / rejection gates | Projects 01 and 02 |
| Historical backtesting | Projects 01 and 02 |
| Tournament simulation | Projects 01 and 02 |
| 2027 World Cup format modeling | Project 02 |
| Explainability | Projects 01 and 02 |
| Interactive applications | Projects 01 and 02 |
| Production model packaging | Projects 01 and 02 |
| Git LFS model storage | Projects 01 and 02 |
| GitHub Actions | Projects 01 and 02 |
| Hugging Face deployment | Projects 01 and 02 |

---

## What the Portfolio Covers

The two projects are intentionally related but technically different.

### IPL Match Intelligence

The IPL project focuses on franchise cricket, where team identity alone can become stale because of:

- auctions;
- player transfers;
- roster turnover;
- season transitions;
- changing tactical roles;
- changing venues and matchups.

The project therefore explored:

- team-history features;
- Elo ratings;
- recent form;
- phase-specific batting and bowling;
- player-strength and squad-strength ideas;
- matchup context;
- venue context;
- pre-toss and post-toss forecasting;
- calibrated ensemble selection.

Four controlled model versions were evaluated before production selection.

### ODI World Cup Intelligence

The World Cup project focuses on international ODI forecasting, where team strength evolves across:

- bilateral series;
- major tournaments;
- squad selection cycles;
- player retirements;
- host conditions;
- travel and venue changes.

The project therefore explores:

- ODI Elo systems;
- recent form;
- exponentially weighted form;
- season form;
- opponent-adjusted form;
- strength of schedule;
- head-to-head context;
- tournament-stage simulation;
- same-window V1 vs V2 acceptance logic.

The final production architecture intentionally combines the **V1 match model** with the **V2 corrected 2027 tournament engine**.

---

## Project Summaries

### 01 — IPL Intelligence Production

[![Open Project 01](https://img.shields.io/badge/Open-Project%2001-2ea44f.svg)](01-ipl-intelligence-production/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/ipl-intelligence-production)

This project builds a production-oriented IPL match forecasting and championship-simulation platform from a controlled four-version modeling investigation.

The final production model is **V2**, selected after V1, V3, and V4 were compared under strict temporal evaluation and rejected when they failed to generalize as well.

**Key capabilities:**

- current IPL / Cricsheet data pipeline;
- point-in-time feature generation;
- ball-by-ball derived features;
- lineup-aware historical data;
- pre-toss and post-toss forecasting;
- Elo and recent-form features;
- multiple classical ML and boosting models;
- calibration;
- strict 2025-2026 out-of-time evaluation;
- V1-V4 model governance;
- historical backtesting;
- tournament simulation;
- explainability;
- Gradio application;
- Hugging Face deployment;
- GitHub Actions validation.

### Final IPL production metrics

| Metric | V2 Result |
|---|---:|
| Accuracy | **54.23%** |
| Balanced Accuracy | **55.00%** |
| F1 Score | **52.55%** |
| ROC-AUC | **0.5713** |
| Log Loss | **0.68585** |
| Brier Score | **0.24636** |
| ECE | **0.08786** |

### IPL model-development journey

| Version | Accuracy | ROC-AUC | Log Loss | Brier | Decision |
|---|---:|---:|---:|---:|---|
| V1 architecture replay | 52.82% | 0.5614 | 0.6896 | 0.2482 | Superseded |
| **V2** | **54.23%** | **0.5713** | **0.6858** | **0.2464** | **CHAMPION** |
| V3 | 47.89% | 0.5317 | 0.6976 | 0.2522 | Rejected |
| V4 | 44.37% | 0.4195 | 0.6947 | 0.2508 | Rejected |

**Final project finding:** V2 remained the strongest tested architecture. V3 and V4 were retained as failed challengers rather than hidden, demonstrating explicit production-model governance.

---

### 02 — ICC World Cup Intelligence Engine

[![Open Project 02](https://img.shields.io/badge/Open-Project%2002-2ea44f.svg)](02-world-cup-intelligence/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/icc-world-cup-intelligence)

This project builds an ODI World Cup forecasting and tournament-intelligence system using current structured ODI data, chronological validation, same-window champion/challenger testing, and a corrected 2027 World Cup simulator.

The production architecture deliberately separates model selection from software selection:

```text
V1 match-probability model
          +
V2 corrected 2027 tournament simulator
          ↓
ICC World Cup Intelligence Production
```

**Key capabilities:**

- current ODI data pipeline;
- chronological feature generation;
- multi-speed Elo features;
- recent and exponentially weighted form;
- opponent-adjusted form;
- strength-of-schedule features;
- best-single vs ensemble comparisons;
- strict 2023-2026 evaluation;
- exact V1-vs-V2 same-window comparison;
- pre-registered acceptance gate;
- corrected 57-match 2027 simulator;
- historical backtesting;
- explainability;
- Gradio application;
- Hugging Face deployment;
- GitHub Actions validation.

### World Cup exact same-window comparison

| Version | Accuracy | ROC-AUC | Log Loss | Brier | ECE | Decision |
|---|---:|---:|---:|---:|---:|---|
| **V1** | **59.02%** | 0.6270 | 0.66803 | 0.23771 | **0.05430** | **MODEL CHAMPION** |
| V2 | **60.93%** | **0.6421** | **0.66788** | **0.23670** | 0.06553 | Rejected by pre-registered gate |

The comparison uses the exact same **366 strict matches**.

### World Cup acceptance result

```text
V1 probability objective = 0.732888
V2 probability objective = 0.733609

Lower is better.

DECISION:
V2_REJECTED_USE_V1_MODEL_WITH_V2_SIMULATOR
```

**Final project finding:** V2 improved several headline metrics, but it failed the pre-registered probability-objective gate. V1 therefore remained the match-model champion while V2's corrected 2027 simulator was retained.

---

## 2027 World Cup Tournament Simulation

The World Cup production project implements the corrected 57-match structure:

```text
Super Series
3 teams
3 matches
1 qualifier
      |
      v
Round 2
12 teams
2 groups of 6
30 matches
      |
      v
7 teams advance
      |
      v
Super 7
21 matches
      |
      v
Top 4
      |
      v
Semifinals
1st vs 4th
2nd vs 3rd
      |
      v
Final
```

| Stage | Matches |
|---|---:|
| Super Series | 3 |
| Round 2 | 30 |
| Super 7 | 21 |
| Semifinals | 2 |
| Final | 1 |
| **Total** | **57** |

The simulator reports separate:

- Super Series probability;
- Round 2 probability;
- Super 7 probability;
- semifinal probability;
- final probability;
- championship probability.

---

## Machine-Learning Architecture Coverage

| Modeling Area | Demonstrated Through |
|---|---|
| Logistic Regression | Projects 01 and 02 |
| Random Forest | Projects 01 and 02 |
| Extra Trees | Projects 01 and 02 |
| HistGradientBoosting | Projects 01 and 02 |
| Gradient Boosting | Projects 01 and 02 |
| XGBoost | Projects 01 and 02 |
| LightGBM | Projects 01 and 02 |
| CatBoost | Projects 01 and 02 |
| Weighted / mean ensembles | Projects 01 and 02 |
| Calibration | Projects 01 and 02 |
| Stacked model experiments | Project 02 |
| Elo systems | Projects 01 and 02 |
| Player-first model experiments | Project 01 |
| Feature-family ablation | Projects 01 and 02 |
| Acceptance-gate model governance | Projects 01 and 02 |

---

## Evaluation Coverage

The projects use metrics aligned with probabilistic sports forecasting rather than relying on one headline number.

| Evaluation Area | Metrics / Methods |
|---|---|
| Classification | Accuracy, balanced accuracy, F1 |
| Ranking / discrimination | ROC-AUC |
| Probability quality | Log loss, Brier score |
| Calibration | ECE |
| Historical stability | Expanding-window backtesting |
| Version comparison | Exact same-window evaluation |
| Feature investigation | Feature-family ablation |
| Explainability | Permutation / perturbation sensitivity |
| Model governance | Explicit accept / reject gates |
| Tournament validity | Stage counts, probability mass, advancement logic |

### Why multiple metrics matter

- Accuracy alone can hide poor probability estimates.
- ROC-AUC measures ranking quality but not calibration.
- Log loss penalizes confident wrong predictions.
- Brier score measures squared probability error.
- ECE measures calibration mismatch.
- Season-level performance can reveal regime shifts hidden by aggregate metrics.
- Model-selection rules should be fixed before reviewing the final holdout.
- A newer model should not be promoted solely because one metric improves.

---

## What the Repository Demonstrates

### End-to-End Sports ML Delivery

The repository demonstrates the complete path from a historical notebook project to a public machine-learning application:

- legacy audit;
- data-source redesign;
- current-data ingestion;
- data validation;
- point-in-time feature engineering;
- model baselines;
- multiple model families;
- temporal hyperparameter tuning;
- validation and calibration;
- strict out-of-time testing;
- model-version comparison;
- model acceptance / rejection;
- production-model freezing;
- tournament simulation;
- explainability;
- historical backtesting;
- automated tests;
- CI validation;
- Git LFS artifact management;
- Gradio application development;
- Hugging Face deployment;
- portfolio documentation.

### Model Selection Based on Evidence

The projects do not assume that the newest model is automatically the production model.

Examples:

- IPL V2 beat V1 and remained stronger than V3 and V4.
- IPL V3 was formally rejected.
- IPL V4 was formally rejected.
- No IPL V5 was created without genuinely new data.
- World Cup V2 improved accuracy and ROC-AUC.
- World Cup V2 still failed the pre-registered probability objective.
- World Cup V1 therefore remained the production model.
- World Cup V2's superior tournament simulator was retained independently.

This distinction demonstrates that:

> model selection, feature selection, and software-component selection are separate engineering decisions.

### Reliable and Reusable Engineering

The repository includes:

- modular Python source;
- reusable feature pipelines;
- deterministic seeds;
- saved configurations;
- frozen production metrics;
- production asset manifests;
- checksum validation;
- Git LFS tracking for `.joblib` artifacts;
- automated tests;
- GitHub Actions workflows;
- deployment-specific bundles;
- model cards;
- experiment-history documents;
- deployment guides;
- responsible-use documentation.

---

## Repository Convention

The repository is organized as a two-project monorepo:

```text
cricket-intelligence-projects/
|
|-- .github/
|   `-- workflows/
|       |-- ipl-tests.yml
|       `-- world-cup-tests.yml
|
|-- 01-ipl-intelligence-production/
|   |-- artifacts/
|   |-- configs/
|   |-- data/
|   |-- deployment/
|   |-- images/
|   |-- reports/
|   |-- scripts/
|   |-- src/
|   |-- tests/
|   |-- app.py
|   |-- README.md
|   |-- MODEL_CARD.md
|   |-- EXPERIMENT_HISTORY.md
|   |-- DEPLOYMENT_GUIDE.md
|   `-- requirements.txt
|
|-- 02-world-cup-intelligence/
|   |-- artifacts/
|   |-- configs/
|   |-- data/
|   |-- deployment/
|   |-- images/
|   |-- reports/
|   |-- scripts/
|   |-- src/
|   |-- tests/
|   |-- app.py
|   |-- README.md
|   |-- MODEL_CARD.md
|   |-- EXPERIMENT_HISTORY.md
|   |-- DEPLOYMENT_GUIDE.md
|   `-- requirements.txt
|
|-- .gitattributes
|-- .gitignore
|-- LICENSE
`-- README.md
```

The historical development folders used locally for IPL V1-V4 and World Cup V1-V2 are intentionally kept outside the public production monorepo.

The GitHub repository contains the final production packages plus the experiment evidence needed to explain the model-selection decisions.

---

## Continuous Integration

The repository uses project-specific GitHub Actions workflows rather than one oversized pipeline.

### IPL workflow

```text
.github/workflows/ipl-tests.yml
```

Validates the IPL production project.

### World Cup workflow

```text
.github/workflows/world-cup-tests.yml
```

Validates the World Cup production project.

Depending on the project, CI checks:

- required folder structure;
- Python syntax;
- JSON / CSV artifacts;
- production flags;
- model-bundle availability;
- test suites;
- tournament-format invariants;
- probability bounds;
- model-governance artifacts;
- README resources;
- oversized-file handling.

[![Open GitHub Actions](https://img.shields.io/badge/Open-GitHub%20Actions-2088ff?style=for-the-badge)](https://github.com/unit-mole/cricket-intelligence-projects/actions)

---

## Deployment Portfolio

Both projects are deployed independently so each application can be reviewed as a complete case study.

| Project | Platform | Live Application |
|---|---|---|
| IPL Intelligence Production | Hugging Face Spaces | [Open IPL App](https://huggingface.co/spaces/anmol-unitmole/ipl-intelligence-production) |
| ICC World Cup Intelligence Engine | Hugging Face Spaces | [Open World Cup App](https://huggingface.co/spaces/anmol-unitmole/icc-world-cup-intelligence) |

### IPL Space

```text
anmol-unitmole/ipl-intelligence-production
```

### World Cup Space

```text
anmol-unitmole/icc-world-cup-intelligence
```

Both applications use Gradio and load the frozen production model artifacts rather than retraining on startup.

---

## Run a Project Locally

Each project contains its own detailed setup instructions.

### 1. Clone the repository

```bash
git clone https://github.com/unit-mole/cricket-intelligence-projects.git
cd cricket-intelligence-projects
```

### 2. Enter a project

IPL:

```bash
cd 01-ipl-intelligence-production
```

or World Cup:

```bash
cd 02-world-cup-intelligence
```

### 3. Create a virtual environment

**Windows**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run tests

```bash
python -m pytest -q
```

### 6. Launch the application

```bash
python app.py
```

Then open the Gradio URL shown in the terminal.

Always follow the selected project's own `README.md` for project-specific details.

---

## Responsible Use

The repository is intended for education, experimentation, sports analytics, technical demonstration, and portfolio presentation.

General limitations include:

- cricket outcomes are inherently uncertain;
- public predictions are probabilistic forecasts, not guarantees;
- models may not know last-minute injuries or squad changes;
- toss, pitch, and weather information may be unavailable;
- historical team behavior can become stale;
- franchise and international teams change over time;
- binary win probabilities do not separately model tie / no-result probability;
- historical metrics should not be generalized beyond evaluated conditions;
- model-derived feature sensitivity is not causal explanation;
- the applications are not betting or financial-advice systems;
- predictions should not be used as the sole basis for high-stakes decisions.

Negative results and rejected models are intentionally retained where they add technical value.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| IPL analytics | Project 01 |
| ODI analytics | Project 02 |
| Sports forecasting | Projects 01 and 02 |
| Probabilistic classification | Projects 01 and 02 |
| Elo modeling | Projects 01 and 02 |
| Recent-form modeling | Projects 01 and 02 |
| Opponent-adjusted form | Projects 01 and 02 |
| Player / squad modeling | Project 01 |
| Venue context | Projects 01 and 02 |
| Feature-family ablation | Projects 01 and 02 |
| Calibration | Projects 01 and 02 |
| Model ensembles | Projects 01 and 02 |
| Strict temporal validation | Projects 01 and 02 |
| Model governance | Projects 01 and 02 |
| Historical backtesting | Projects 01 and 02 |
| Monte Carlo simulation | Projects 01 and 02 |
| 2027 World Cup simulator | Project 02 |
| Explainability | Projects 01 and 02 |
| Production packaging | Projects 01 and 02 |
| Git LFS | Projects 01 and 02 |
| Gradio deployment | Projects 01 and 02 |
| Hugging Face Spaces | Projects 01 and 02 |
| GitHub Actions | Projects 01 and 02 |

---

## Core Skills Demonstrated

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `Sports Analytics` · `Cricket Analytics` · `IPL Forecasting` · `ODI Forecasting` · `Probabilistic Machine Learning` · `Classification` · `Elo Ratings` · `Time-Aware Feature Engineering` · `Recent Form` · `Opponent-Adjusted Form` · `Historical Backtesting` · `Chronological Validation` · `Out-of-Time Testing` · `Probability Calibration` · `ROC-AUC` · `Log Loss` · `Brier Score` · `ECE` · `Feature Ablation` · `Model Governance` · `Acceptance Gates` · `Ensemble Modeling` · `Monte Carlo Simulation` · `Tournament Simulation` · `Explainability` · `Plotly` · `Gradio` · `Hugging Face Spaces` · `Git LFS` · `pytest` · `GitHub Actions` · `CI/CD` · `Production ML Packaging` · `Responsible Model Communication`

---

## Portfolio Positioning

**One-line description:** Two end-to-end cricket machine-learning projects covering IPL and ODI World Cup forecasting, leakage-safe temporal modeling, calibrated probabilities, explicit model governance, historical backtesting, tournament simulation, explainability, CI, and live Hugging Face deployment.

**Pinned repository description:** Professional cricket ML portfolio featuring IPL and ICC World Cup intelligence systems with chronological validation, Elo and form modeling, calibrated probabilities, controlled model-version experiments, historical backtesting, Monte Carlo tournament simulation, explainability, GitHub Actions, and two live Hugging Face applications.

This portfolio connects naturally to a Quality Data Scientist background because the same analytical principles apply to:

- point-in-time data integrity;
- leakage prevention;
- controlled experiment comparison;
- model release gates;
- monitoring across time;
- transparent failure analysis;
- reproducible validation;
- evidence-based production decisions.

---

## License

This repository is distributed under the [MIT License](LICENSE).

Cricket data, third-party libraries, model dependencies, and external data sources remain subject to their original licenses and usage conditions.

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Sports Analytics | Predictive Modeling | Probabilistic Forecasting | Analytics Engineering | Quality Analytics
