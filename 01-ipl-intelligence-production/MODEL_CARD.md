# Model Card — IPL Intelligence Production V2 Champion

## Model summary

The production model is the frozen **IPL Intelligence Lab V2 pre-toss calibrated ensemble**, selected after controlled V1→V4 experimentation. The production repository does not perform additional hyperparameter search or retraining after that decision.

## Intended use

- IPL pre-match win-probability analysis.
- Optional post-toss scenario analysis.
- Tournament/championship simulation.
- Educational demonstration of leakage-safe temporal ML, calibration and model governance.

It is not intended for guaranteed outcome prediction or betting decisions.

## Evaluation

Strict out-of-time window: **2025–2026**, **142 matches**.

| Metric | Result |
|---|---:|
| Accuracy | 0.5422535 |
| Balanced Accuracy | 0.5500000 |
| F1 | 0.5255474 |
| ROC-AUC | 0.5713415 |
| Log Loss | 0.6858472 |
| Brier | 0.2463623 |
| ECE | 0.0878614 |

## Selection history

- V1 architecture replay: baseline.
- V2: accepted production champion.
- V3: rejected after feature-selection/stacking challenger underperformed.
- V4: rejected after player-first challenger underperformed. Paired bootstrap P(V4 better than V2)=0.259.

## Data and features

The V2 pipeline uses Cricsheet IPL match/ball-by-ball information to build point-in-time features including multi-speed Elo, recent form, opponent-adjusted form, phase batting/bowling summaries, scoring/wicket-pressure indicators, venue context, and player/squad proxy features. Historical exact lineups are used where available; live inference can accept an announced XI.

## Leakage controls

Historical match features are computed before that match is applied to the evolving state. Current-match outcomes and score information are therefore unavailable to the prediction being evaluated. Model tuning, ensemble selection, calibration and strict testing are separated chronologically.

## Calibration

The production bundle contains the calibrated ensemble selected by V2. Probability symmetry is explicitly enforced by comparing the matchup and its reversed orientation.

## Limitations

- The strict window is relatively small.
- Predictive discrimination is modest rather than dominant.
- Player availability/injury and other unverified future information are not fabricated.
- Tournament simulation without an official future schedule is a neutral-venue scenario.
- Distribution shift can occur between IPL seasons due to roster and tactical changes.

## Production governance

The repository imports the accepted V2 artifacts and verifies them against frozen per-match metrics/checksums before launch. No V5 is recommended without genuinely new authoritative information.
