from pathlib import Path
import json,sys,joblib,pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from world_cup_production.runtime import ProductionRuntime,sha256_file
from world_cup_production.simulator_2027 import official_match_counts,simulate_world_cup_2027,default_scenario_participants

def fail(msg):
    (ROOT/'artifacts/PRODUCTION_READY.flag').unlink(missing_ok=True); raise SystemExit(msg)
def main():
    req=['artifacts/model_bundle.joblib','artifacts/PRODUCTION_ASSET_MANIFEST.json','data/processed/latest_team_snapshot.json','reports/FROZEN_V1_V2_COMPARISON.csv','reports/FROZEN_CHAMPION_METRICS.json','reports/production_permutation_importance.csv','reports/V2_ACCEPTANCE_REPORT.json']
    missing=[x for x in req if not (ROOT/x).exists()]
    if missing: fail('Missing production assets:\n  - '+'\n  - '.join(missing))
    man=json.loads((ROOT/'artifacts/PRODUCTION_ASSET_MANIFEST.json').read_text(encoding='utf-8'))
    bad=[]
    for x in man['files']:
        p=ROOT/x['path']
        if not p.exists() or sha256_file(p)!=x['sha256']: bad.append(x['path'])
    if bad: fail('Checksum mismatch:\n  - '+'\n  - '.join(bad))
    dv=json.loads((ROOT/'data/metadata/source_v1_data_validation.json').read_text(encoding='utf-8'))
    if dv.get('mode')!='cricsheet_current' or dv.get('status')!='PASS': fail('V1 source data validation is not current-data PASS.')
    dec=json.loads((ROOT/'reports/V2_ACCEPTANCE_REPORT.json').read_text(encoding='utf-8'))
    if dec.get('decision')!='V2_REJECTED_USE_V1_MODEL_WITH_V2_SIMULATOR': fail('V2 governance decision changed unexpectedly.')
    rt=ProductionRuntime(ROOT); teams=rt.teams
    if len(teams)<14: fail('Production snapshot contains fewer than 14 teams.')
    checks=[]
    for i in range(5):
        a,b=teams[i],teams[i+1]; p=rt.probability(a,b)
        if not 0<p<1: fail(f'Invalid probability {a} vs {b}: {p}')
        checks.append({'a':a,'b':b,'p':p})
    participants=default_scenario_participants(rt.snapshot); sim=simulate_world_cup_2027(rt.probability,rt.snapshot,participants,n=300,seed=42)
    if sim.attrs['match_counts']!=official_match_counts(): fail('2027 simulator match-count contract failed.')
    if len(sim)!=14: fail('Simulator did not return 14 participants.')
    if abs(float(sim.championship_probability.sum())-1)>1e-9: fail('Championship probability mass does not sum to 1.')
    if not sim.filter(like='_probability').apply(lambda s:s.between(0,1).all()).all(): fail('Simulation probabilities outside [0,1].')
    pi=pd.read_csv(ROOT/'reports/production_permutation_importance.csv')
    if len(pi)==0 or pi.importance_mean.isna().any(): fail('Explainability report is empty/invalid.')
    report={'status':'PASS','champion_model':'V1','tournament_engine':'V2_2027','data_mode':'cricsheet_current','runtime_pair_checks':checks,'scenario_participants':participants,'official_match_counts':official_match_counts(),'explainability_features':len(pi),'governance_decision':dec.get('decision')}
    (ROOT/'reports/PRODUCTION_VERIFICATION_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (ROOT/'artifacts/PRODUCTION_READY.flag').write_text('V1_MODEL_PLUS_V2_2027_SIMULATOR_READY\n',encoding='utf-8')
    print('='*78); print('ICC WORLD CUP PRODUCTION VERIFICATION: PASS'); print('='*78); print('Champion model       : V1'); print('Tournament engine    : V2 corrected 2027'); print('Data mode            : cricsheet_current'); print('Runtime pair checks  : PASS'); print('2027 match counts    :',official_match_counts()); print('Probability mass     : PASS'); print('Explainability       : PASS'); print('Production flag      : artifacts\\PRODUCTION_READY.flag')
if __name__=='__main__': main()
