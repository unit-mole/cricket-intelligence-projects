from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr
try:
    import spaces
except Exception:
    class _Spaces:
        def GPU(self,duration=1):
            return lambda fn: fn
    spaces=_Spaces()
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from world_cup_production.runtime import ProductionRuntime,ensure_ready
from world_cup_production.simulator_2027 import simulate_world_cup_2027,default_scenario_participants,official_match_counts
TITLE='ICC World Cup Intelligence Engine'
CSS="""
.gradio-container {max-width: 1180px !important; margin: auto !important;}
.hero {border:1px solid #333; border-radius:16px; padding:18px; margin-bottom:10px;}
"""
ensure_ready(ROOT,True); rt=ProductionRuntime(ROOT)
champ=json.loads((ROOT/'reports/FROZEN_CHAMPION_METRICS.json').read_text(encoding='utf-8')); comp=pd.read_csv(ROOT/'reports/FROZEN_V1_V2_COMPARISON.csv'); pi=pd.read_csv(ROOT/'reports/production_permutation_importance.csv'); back=pd.read_csv(ROOT/'reports/v1_expanding_window_backtest.csv')
scenario=default_scenario_participants(rt.snapshot)

@spaces.GPU(duration=2)
def predict_ui(a,b):
    if a==b: return 'Choose two different teams.',None
    p=rt.probability(a,b); q=1-p
    txt=f'{a}: {p*100:.1f}% | {b}: {q*100:.1f}%\n\nModel lean: {a if p>=.5 else b}'
    df=pd.DataFrame({'Team':[a,b],'Win Probability':[p,q]}); fig=px.bar(df,x='Team',y='Win Probability',text=df['Win Probability'].map(lambda x:f'{x:.1%}'),range_y=[0,1],title='Calibrated ODI win probability')
    return txt,fig

@spaces.GPU(duration=45)
def simulate_ui(n):
    n=int(n); out=simulate_world_cup_2027(rt.probability,rt.snapshot,scenario,n=n,seed=42)
    show=out.copy()
    fig=px.bar(show.head(14),x='team',y='championship_probability',title='2027 scenario championship probability')
    note=f"Scenario-seeded participant list, not an official qualifier list. Match structure: {out.attrs['match_counts']}. Group mode: {out.attrs['group_assignment_mode']}."
    return show,note,fig

m=champ['same_window_metrics']; metrics_df=pd.DataFrame({'Metric':['Accuracy','Balanced Accuracy','F1','ROC-AUC','Log Loss','Brier','ECE'], 'V1 production result':[m['accuracy'],m['balanced_accuracy'],m['f1'],m['roc_auc'],m['log_loss'],m['brier'],m['ece']]})
journey=comp.copy(); journey_fig=px.bar(journey,x='system',y=['accuracy','roc_auc'],barmode='group',title='V1 vs V2 exact same-window comparison')
pi_fig=px.bar(pi.head(12),x='importance_mean',y='feature',orientation='h',error_x='importance_std',title='Manual permutation importance: increase in log loss')
back_fig=px.line(back,x='year',y=['accuracy','roc_auc'],markers=True,title='V1 expanding-window historical backtest') if {'year','accuracy','roc_auc'}.issubset(back.columns) else None

with gr.Blocks(title=TITLE) as demo:
    gr.HTML('<div class="hero"><h1>ICC World Cup Intelligence Engine</h1><h3>Leakage-Safe ODI Forecasting, Model Governance and 2027 Tournament Simulation</h3><b>Production architecture: V1 model champion + V2 corrected 2027 simulator.</b></div>')
    with gr.Tabs():
        with gr.Tab('Executive Overview'):
            gr.Markdown('## Production decision\nV2 improved several individual metrics but failed the pre-registered probability-objective gate. The V1 model remains champion; the corrected V2 tournament engine is retained.')
            with gr.Row(): gr.Dataframe(metrics_df,interactive=False); gr.Dataframe(journey,interactive=False)
            gr.Plot(journey_fig)
        with gr.Tab('Match Predictor'):
            gr.Markdown('Generate a two-team ODI win probability using the frozen V1 production model.')
            with gr.Row(): a=gr.Dropdown(rt.teams,value=rt.teams[0],label='Team A'); b=gr.Dropdown(rt.teams,value=rt.teams[1],label='Team B')
            btn=gr.Button('Generate prediction',variant='primary'); out=gr.Markdown(); plot=gr.Plot(); btn.click(predict_ui,[a,b],[out,plot])
        with gr.Tab('2027 Tournament Simulator'):
            gr.Markdown('## Official-format structure\n3 Super Series matches -> 30 Round-2 matches -> 21 Super-7 matches -> 2 semifinals -> 1 final = **57 matches**.\n\nParticipant seeding is a scenario unless an official qualifier list/group draw is supplied.')
            n=gr.Dropdown([500,1000,2500,5000,10000],value=1000,label='Monte Carlo simulations'); sb=gr.Button('Run simulation',variant='primary'); st=gr.Dataframe(); sn=gr.Markdown(); sf=gr.Plot(); sb.click(simulate_ui,n,[st,sn,sf])
        with gr.Tab('Team Intelligence'):
            td=pd.DataFrame([{'team':t,**{k:v for k,v in rt.snapshot['teams'][t].items() if k in ['elo','matches','wins','form5','form10','form20','last_date']}} for t in rt.teams]); gr.Dataframe(td,interactive=False)
        with gr.Tab('Model Evaluation'):
            gr.Dataframe(journey,interactive=False); gr.Markdown('**Acceptance rule:** V2 had to improve the pre-registered probability objective. It did not, so it was rejected even though several individual V2 metrics were higher.')
        with gr.Tab('Experiment History'):
            gr.Markdown('## V1\nStrong current-data baseline and accepted production model.\n\n## V2\nTargeted challenger with stronger feature diagnostics and corrected 2027 simulator. Model rejected by the pre-registered probability-objective gate; simulator retained.')
            gr.Plot(journey_fig)
        with gr.Tab('Historical Backtesting'):
            if back_fig is not None: gr.Plot(back_fig)
            gr.Dataframe(back,interactive=False)
        with gr.Tab('Explainability'):
            gr.Markdown('Manual permutation importance is computed directly against the frozen V1 ensemble by measuring the increase in strict-window log loss after feature shuffling. This fixes the V2 StrategyBundle permutation wrapper issue without altering the production model.')
            gr.Plot(pi_fig); gr.Dataframe(pi.head(20),interactive=False)
        with gr.Tab('Methodology & Limitations'):
            gr.Markdown("""## Methodology\n- point-in-time ODI features\n- chronological validation and strict future testing\n- exact V1/V2 same-window model governance\n- frozen V1 model; no retraining during production packaging\n- corrected 2027 57-match tournament structure\n\n## Limitations\n- match forecasts are probabilistic, not guarantees\n- scenario participants and group allocation are not presented as an official qualifier list or official group draw\n- injuries, weather, final squads and future contextual data require authoritative point-in-time sources\n- binary match probabilities do not separately model tie/no-result outcomes\n- this is an analytics/portfolio application, not betting advice""")
if __name__=='__main__': demo.launch(css=CSS)
