from __future__ import annotations

from pathlib import Path
import json

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ipl_production.runtime import (
    active_teams,
    load_runtime,
    local_sensitivity,
    parse_lineup,
    player_intelligence,
    predict_match as runtime_predict,
    simulate_championship,
    team_intelligence,
)
from ipl_production.governance import load_decisions, load_experiment_table

ROOT = Path(__file__).resolve().parent
RT = load_runtime(ROOT, require_flag=True)
SNAPSHOT = RT["snapshot"]
PRE_BUNDLE = RT["pretoss_bundle"]
POST_BUNDLE = RT["posttoss_bundle"]
CONFIG = RT["config"]
TEAMS = active_teams(SNAPSHOT)
VENUES = ["Neutral / unknown"] + list(SNAPSHOT.get("venues", []))
FROZEN = json.loads((ROOT / "reports/FROZEN_CHAMPION_METRICS.json").read_text(encoding="utf-8"))
EXPERIMENTS = load_experiment_table(ROOT)
DECISIONS = load_decisions(ROOT)


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _probability_chart(team_a: str, team_b: str, p: float):
    df = pd.DataFrame({"Team": [team_a, team_b], "Win Probability": [p, 1.0 - p]})
    fig = px.bar(df, x="Team", y="Win Probability", text=df["Win Probability"].map(_pct), range_y=[0, 1])
    fig.update_layout(title="Calibrated match win probability", yaxis_tickformat=".0%", margin=dict(t=55, l=40, r=20, b=30))
    fig.update_traces(textposition="outside")
    return fig


def predict_ui(mode, team_a, team_b, venue, toss_side, toss_decision, lineup_a_text, lineup_b_text):
    if team_a == team_b:
        return "### Choose two different teams.", None, pd.DataFrame()
    result = runtime_predict(
        SNAPSHOT, PRE_BUNDLE, POST_BUNDLE, mode, team_a, team_b, venue,
        toss_side=toss_side, toss_decision=toss_decision,
        lineup_a=parse_lineup(lineup_a_text), lineup_b=parse_lineup(lineup_b_text),
    )
    p = result["p_team_a"]
    edge = abs(p - 0.5)
    if edge < 0.04:
        signal = "Very close matchup — the model sees little separation."
    elif edge < 0.09:
        signal = "Competitive matchup with a modest model edge."
    else:
        signal = "The model identifies a clearer edge, but IPL outcomes remain highly stochastic."
    winner = team_a if p >= 0.5 else team_b
    summary = (
        f"## {team_a} {_pct(p)} · {team_b} {_pct(1-p)}\n"
        f"**Model lean:** {winner}  \n"
        f"{signal}\n\n"
        "*Probabilities come from the frozen, calibrated V2 champion. They are analytical forecasts, not certainty or betting advice.*"
    )
    bundle = PRE_BUNDLE if result["mode"] == "pretoss" else POST_BUNDLE
    sensitivity = local_sensitivity(bundle, result["feature_row"], team_a)
    return summary, _probability_chart(team_a, team_b, p), sensitivity


def simulate_ui(n):
    out = simulate_championship(PRE_BUNDLE, SNAPSHOT, n=int(n), seed=42)
    fig = px.bar(
        out,
        x="team",
        y="championship_probability",
        text=out["championship_probability"].map(_pct),
        title="Neutral-schedule Monte Carlo championship distribution",
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Championship probability")
    fig.update_traces(textposition="outside")
    return out, fig


def version_plot():
    df = EXPERIMENTS.copy()
    df["version"] = df["system"].str.extract(r"(V\d)", expand=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["version"], y=df["accuracy"], mode="lines+markers+text", name="Accuracy", text=df["accuracy"].map(lambda x: f"{x:.3f}"), textposition="top center"))
    fig.add_trace(go.Scatter(x=df["version"], y=df["roc_auc"], mode="lines+markers+text", name="ROC-AUC", text=df["roc_auc"].map(lambda x: f"{x:.3f}"), textposition="bottom center"))
    fig.update_layout(title="Controlled model journey on the same strict comparison window", yaxis=dict(range=[0.35, 0.65]), xaxis_title="Model version", yaxis_title="Score")
    return fig


def frozen_metric_table():
    m = FROZEN["metrics"]
    rows = [
        ("Accuracy", m["accuracy"]),
        ("Balanced Accuracy", m["balanced_accuracy"]),
        ("F1", m["f1"]),
        ("ROC-AUC", m["roc_auc"]),
        ("Log Loss", m["log_loss"]),
        ("Brier Score", m["brier"]),
        ("ECE", m["ece"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Strict 2025-2026 result"])


def load_optional_csv(name: str, empty_message: str) -> pd.DataFrame:
    p = ROOT / "reports" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame({"Info": [empty_message]})


CSS = """
.gradio-container {max-width: 1500px !important;}
#hero {padding: 22px 24px; border: 1px solid rgba(127,127,127,.28); border-radius: 18px;}
#champion {padding: 14px 18px; border: 1px solid rgba(127,127,127,.22); border-radius: 14px;}
"""

TITLE = CONFIG["app"]["title"]
SUBTITLE = CONFIG["app"]["subtitle"]

with gr.Blocks(title=TITLE, css=CSS) as demo:
    gr.Markdown(
        f"# 🏏 {TITLE}\n### {SUBTITLE}\n"
        "**Production champion: V2.** Selected only after V1, V3 and V4 were evaluated and rejected on controlled out-of-time evidence.",
        elem_id="hero",
    )

    with gr.Tab("Executive Overview"):
        gr.Markdown(
            "### Production decision\n"
            "V2 is frozen as the final IPL model. The project deliberately preserves failed challengers because model governance is part of the result — not an afterthought."
        )
        with gr.Row():
            gr.Dataframe(value=frozen_metric_table(), interactive=False, label="Frozen V2 strict metrics")
            gr.Dataframe(value=EXPERIMENTS[["system", "status", "accuracy", "roc_auc", "log_loss", "brier"]], interactive=False, label="V1 → V4 same-window comparison")
        gr.Plot(value=version_plot(), label="Model journey")
        gr.Markdown(
            "**Evaluation protocol:** temporal tuning on earlier seasons, ensemble selection on 2022–2023, calibration on 2024, then strict 2025–2026 evaluation. "
            "Production bundles were refit only after those strict metrics were recorded."
        )

    with gr.Tab("Match Predictor"):
        gr.Markdown("Generate a calibrated probability from information available before the match. Pre-toss is the default production forecasting mode.")
        with gr.Row():
            mode = gr.Radio(["Pre-Toss", "Post-Toss"], value="Pre-Toss", label="Prediction mode")
            team_a = gr.Dropdown(TEAMS, value=TEAMS[0], label="Team A")
            team_b = gr.Dropdown(TEAMS, value=TEAMS[1], label="Team B")
            venue = gr.Dropdown(VENUES, value=VENUES[0], label="Venue")
        with gr.Row():
            toss_side = gr.Radio(["Team A", "Team B"], value="Team A", label="Toss winner (post-toss only)")
            toss_decision = gr.Radio(["bat", "field"], value="field", label="Toss decision")
        with gr.Row():
            lineup_a = gr.Textbox(lines=2, label="Optional announced Team A XI", placeholder="Comma-separated names; blank uses the snapshot's recent squad proxy")
            lineup_b = gr.Textbox(lines=2, label="Optional announced Team B XI", placeholder="Comma-separated names; blank uses the snapshot's recent squad proxy")
        pred_btn = gr.Button("Generate prediction", variant="primary")
        pred_text = gr.Markdown()
        pred_plot = gr.Plot()
        gr.Markdown("#### Model-derived explanation\nFeature neutralization sensitivity shows how the final probability changes when one feature difference is set to neutral. It is not an LLM-generated reason.")
        pred_reason = gr.Dataframe(interactive=False)
        pred_btn.click(predict_ui, [mode, team_a, team_b, venue, toss_side, toss_decision, lineup_a, lineup_b], [pred_text, pred_plot, pred_reason])

    with gr.Tab("Tournament Simulator"):
        gr.Markdown(
            "The simulator uses the **pre-toss V2 production bundle**. It intentionally does not fabricate an unannounced future IPL schedule or venues; "
            "without an official schedule it runs a neutral-venue double round robin followed by IPL playoffs."
        )
        simulations = gr.Dropdown([5000, 10000, 25000, 50000, 100000], value=CONFIG["app"]["default_simulations"], label="Monte Carlo simulations")
        sim_btn = gr.Button("Run championship simulation", variant="primary")
        sim_table = gr.Dataframe(interactive=False)
        sim_plot = gr.Plot()
        sim_btn.click(simulate_ui, simulations, [sim_table, sim_plot])

    with gr.Tab("Team & Player Intelligence"):
        exp_team = gr.Dropdown(TEAMS, value=TEAMS[0], label="Team")
        with gr.Row():
            team_stats = gr.Dataframe(value=team_intelligence(SNAPSHOT, TEAMS[0]), interactive=False, label="Team state")
            player_stats = gr.Dataframe(value=player_intelligence(SNAPSHOT, TEAMS[0]), interactive=False, label="Recent player intelligence")
        exp_team.change(lambda t: team_intelligence(SNAPSHOT, t), exp_team, team_stats)
        exp_team.change(lambda t: player_intelligence(SNAPSHOT, t), exp_team, player_stats)

    with gr.Tab("Model Evaluation"):
        gr.Markdown("### Frozen production benchmark\nThese are the accepted V2 strict results. The production package verifies them from per-match predictions before the app is unlocked.")
        gr.Dataframe(value=frozen_metric_table(), interactive=False)
        gr.Markdown("### Candidate model comparison")
        gr.Dataframe(value=load_optional_csv("model_comparison_pretoss.csv", "Import V2 champion assets first."), interactive=False)
        gr.Markdown("### Strict performance by year")
        gr.Dataframe(value=load_optional_csv("strict_test_by_year_pretoss.csv", "Import V2 champion assets first."), interactive=False)

    with gr.Tab("Experiment History"):
        gr.Markdown(
            "### Why four versions were built\n"
            "The final model was not chosen from a single run. Each version tested a different hypothesis and had to earn promotion on out-of-time evidence."
        )
        gr.Dataframe(value=EXPERIMENTS, interactive=False)
        gr.Plot(value=version_plot())
        gr.Markdown(
            "**V1 — baseline:** established the controlled architecture replay.  \n"
            "**V2 — accepted:** strongest same-window probability quality and discrimination.  \n"
            "**V3 — rejected:** feature-family pruning + stacking failed to generalize.  \n"
            "**V4 — rejected:** player-first challenger did not solve recent-season drift; paired bootstrap P(V4 better than V2)=0.259.  \n\n"
            "**Final governance decision:** freeze V2. No V5 without genuinely new authoritative information."
        )

    with gr.Tab("Historical Backtesting"):
        gr.Markdown("Annual expanding-window results expose how performance changes through IPL history rather than hiding everything behind one aggregate score.")
        gr.Dataframe(value=load_optional_csv("expanding_window_backtest_pretoss.csv", "Backtest CSV was not present in the sibling V2 run; the production app still remains valid."), interactive=False)

    with gr.Tab("Explainability"):
        gr.Markdown(
            "### Global and local model interpretation\n"
            "The Match Predictor provides local feature-neutralization sensitivity. If the V2 permutation-importance report was generated locally, it is displayed below as a global production sensitivity view."
        )
        gr.Dataframe(value=load_optional_csv("permutation_importance_pretoss.csv", "Run the optional V2 explainability report or use local prediction sensitivity."), interactive=False)

    with gr.Tab("Methodology & Limitations"):
        gr.Markdown(
            """
### Production methodology

**Cricsheet IPL JSON → point-in-time state → leakage-safe V2 features → temporal hyperparameter tuning → weak-model rejection → constrained ensemble → calibration → strict 2025–2026 test → production refit → frozen production import.**

### Scientific guardrails

- No current-match runs or wickets are used before that historical match is predicted.
- Hyperparameters are tuned only on earlier temporal folds.
- Weak candidate models can receive zero ensemble weight.
- Calibration is chosen before the strict test window.
- The strict 2025–2026 rows remain separate from the later production refit.
- Probability symmetry is enforced: reversing the matchup should complement the probability.
- The production package verifies model files by checksum and recomputes the frozen strict metrics before unlocking the application.
- V3 and V4 remain documented as rejected experiments; their weaker scores were not hidden.

### Limitations

- The accepted strict window contains **142 matches**, so small metric differences should not be over-interpreted.
- V2's ROC-AUC of about **0.571** indicates modest, not dominant, pre-match signal.
- Cricket is highly stochastic; the model produces probabilities rather than deterministic outcomes.
- Neutral-schedule tournament simulation is a scenario when an official future schedule is not supplied.
- This project is an analytics/research system, **not betting advice**.
"""
        )

if __name__ == "__main__":
    demo.launch()
