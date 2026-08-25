# Experiment History — Why V2 Won

The final production decision was made through controlled model governance rather than by keeping the most complex model.

## Same-window comparison

| Version | Hypothesis | Accuracy | ROC-AUC | Log Loss | Brier | Outcome |
|---|---|---:|---:|---:|---:|---|
| V1 replay | Leakage-safe baseline architecture | 0.5282 | 0.5614 | 0.68957 | 0.24819 | Superseded |
| **V2** | Richer temporal/phase/player features + calibrated selected ensemble | **0.5423** | **0.5713** | **0.68585** | **0.24636** | **Accepted** |
| V3 | Feature-family ablation + aggressive pruning + stacking | 0.4789 | 0.5317 | 0.69762 | 0.25218 | Rejected |
| V4 | Player-first / roster intelligence challenger | 0.4437 | 0.4195 | 0.69470 | 0.25078 | Rejected |

## V1 — baseline

The architecture replay established a stricter benchmark than the old notebook, using chronological information and removing same-match score leakage. It was useful as an experimental starting point but did not beat V2.

## V2 — accepted champion

V2 expanded the point-in-time state with stronger Elo variants, form, phase performance, player/squad proxies and calibrated model selection. It delivered the best same-window probability objective and the strongest accuracy/ROC-AUC among the controlled versions.

## V3 — rejected

V3 tested the hypothesis that aggressive feature-family pruning plus a stacked strategy would generalize better. Diagnostics selected only 26 features, but strict 2025–2026 performance fell to 47.89% accuracy and 0.5317 ROC-AUC. The quality gate rejected it.

## V4 — rejected

V4 tested a more player-first/roster-oriented approach intended to address recent-season drift. It did not improve the strict benchmark. The final paired bootstrap reported **P(V4 better than V2)=0.259**, and the pre-registered gate recommended V2.

## Final conclusion

The production model is **not the newest model**. It is the model that survived the strongest evidence available. This is a deliberate portfolio example of model selection, negative-result reporting and test-set governance.
