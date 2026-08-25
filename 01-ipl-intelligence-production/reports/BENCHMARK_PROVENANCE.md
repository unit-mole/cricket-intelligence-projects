# Frozen benchmark provenance

The production repository intentionally **does not retrain after final model selection**. The headline metrics are the accepted results produced by the local V1→V4 experimental program on the same strict 2025–2026 comparison window wherever possible.

- **V1** is labeled `V1_architecture_replay` because it was replayed with strict chronology on the shared comparison matches; it is not presented as the old notebook's original score.
- **V2** is the accepted production champion and comes from the actual sibling V2 strict per-match predictions.
- **V3** and **V4** were formal challengers and were rejected by their pre-registered quality gates.
- V4's final paired bootstrap reported **P(V4 better than V2) = 0.259**, providing no evidence to replace V2.

The import/verification scripts recompute the V2 strict metrics from `strict_test_predictions_pretoss.csv` and reject the production freeze if they do not match the frozen benchmark.
