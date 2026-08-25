from __future__ import annotations
from pathlib import Path
import argparse,json,shutil,sys,joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from world_cup_production.governance import binary_metrics,probability_objective
from world_cup_production.runtime import sha256_file

def resolve(name,arg=None):
    c=[]
    if arg: c.append(Path(arg))
    c.append(ROOT.parent/name)
    for p in c:
        p=p.expanduser().resolve()
        if p.exists(): return p
    raise SystemExit('Could not locate sibling '+name+'. Keep production next to V1/V2 or pass an explicit source path.')

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)

def manual_permutation(bundle,X,y,features,repeats=5,seed=42):
    from sklearn.metrics import log_loss
    rng=np.random.default_rng(seed); base=log_loss(y,np.clip(bundle.predict_proba(X)[:,1],1e-6,1-1e-6),labels=[0,1]); rows=[]
    for f in features:
        vals=[]
        for _ in range(repeats):
            Xp=X.copy(); Xp[f]=rng.permutation(Xp[f].to_numpy()); pp=np.clip(bundle.predict_proba(Xp)[:,1],1e-6,1-1e-6); vals.append(log_loss(y,pp,labels=[0,1])-base)
        rows.append({'feature':f,'importance_mean':float(np.mean(vals)),'importance_std':float(np.std(vals))})
    return pd.DataFrame(rows).sort_values('importance_mean',ascending=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v1'); ap.add_argument('--v2'); a=ap.parse_args()
    v1=resolve('ICC_World_Cup_Intelligence_Engine',a.v1); v2=resolve('ICC_World_Cup_Intelligence_Engine_V2',a.v2)
    required_v1=['artifacts/model_bundle.joblib','artifacts/model_metadata.json','data/processed/features.csv','data/processed/feature_schema.json','data/processed/latest_team_snapshot.json','data/metadata/data_validation.json','reports/model_comparison.csv','reports/expanding_window_backtest.csv']
    required_v2=['reports/V1_V2_EXACT_COMPARISON.csv','reports/FINAL_MODEL_ACCEPTANCE_REPORT.json','reports/strict_test_predictions_v2.csv','reports/strict_test_by_year.csv','artifacts/model_metadata_v2.json','data/metadata/world_cup_2027_format.json','data/metadata/world_cup_2027_venues.csv']
    missing=[f'V1:{x}' for x in required_v1 if not (v1/x).exists()]+[f'V2:{x}' for x in required_v2 if not (v2/x).exists()]
    if missing: raise SystemExit('Missing required completed-run assets:\n  - '+'\n  - '.join(missing))
    dv=json.loads((v1/'data/metadata/data_validation.json').read_text(encoding='utf-8'))
    if dv.get('mode')!='cricsheet_current' or dv.get('status')!='PASS': raise SystemExit('V1 must be the completed current-data run: mode=cricsheet_current, status=PASS')
    decision=json.loads((v2/'reports/FINAL_MODEL_ACCEPTANCE_REPORT.json').read_text(encoding='utf-8'))
    if decision.get('decision')!='V2_REJECTED_USE_V1_MODEL_WITH_V2_SIMULATOR': raise SystemExit('Expected the completed V2 rejection decision; refusing to package a different model-selection outcome.')
    v2pred=pd.read_csv(v2/'reports/strict_test_predictions_v2.csv')
    vf=pd.read_csv(v1/'data/processed/features.csv')
    common=vf.merge(v2pred[['match_id']].astype({'match_id':str}),assign=None) if False else None
    vf['match_id']=vf['match_id'].astype(str); ids=v2pred['match_id'].astype(str); common=vf[vf['match_id'].isin(set(ids))].copy()
    common=common.drop_duplicates('match_id')
    order={m:i for i,m in enumerate(ids.tolist())}; common['_order']=common['match_id'].map(order); common=common.sort_values('_order')
    if len(common)!=len(v2pred): raise SystemExit(f'Exact common-window comparison failed: V1 rows={len(common)}, V2 rows={len(v2pred)}')
    cols=json.loads((v1/'data/processed/feature_schema.json').read_text(encoding='utf-8'))['feature_columns']
    bundle=joblib.load(v1/'artifacts/model_bundle.joblib'); p1=bundle.predict_proba(common[cols])[:,1]
    m1=binary_metrics(common['target'].to_numpy(),p1); m1['probability_objective']=probability_objective(m1)
    p2=v2pred['probability_team1'].to_numpy(); y2=v2pred['target'].to_numpy()
    if not np.array_equal(common['target'].to_numpy(dtype=int), y2.astype(int)):
        raise SystemExit('Target mismatch between the V1 exact-common rows and V2 strict predictions.')
    m2=binary_metrics(y2,p2); m2['probability_objective']=probability_objective(m2)
    for label,actual,expected in [('V1',m1,decision.get('v1_metrics') or {}),('V2',m2,decision.get('v2_metrics') or {})]:
        for k in ['accuracy','balanced_accuracy','f1','log_loss','brier','ece','roc_auc']:
            if k in expected and abs(float(actual[k])-float(expected[k]))>2e-6:
                raise SystemExit(f'{label} governance-metric mismatch for {k}: recomputed={actual[k]}, acceptance_report={expected[k]}')
    comp=pd.DataFrame([{'system':'V1_MODEL_CHAMPION','status':'CHAMPION','comparison_type':'exact same 366-match window','common_rows':len(common),**m1},{'system':'V2_MODEL_CHALLENGER','status':'REJECTED_MODEL','comparison_type':'exact same 366-match window','common_rows':len(common),**m2}])
    (ROOT/'reports').mkdir(exist_ok=True); comp.to_csv(ROOT/'reports/FROZEN_V1_V2_COMPARISON.csv',index=False)
    (ROOT/'reports/FROZEN_CHAMPION_METRICS.json').write_text(json.dumps({'champion_model':'V1','same_window_rows':len(common),'same_window_metrics':m1,'v2_metrics':m2,'decision':'V1 model + V2 official-2027 simulator','source_window':{'start':str(v2pred['date'].min()) if 'date' in v2pred else None,'end':str(v2pred['date'].max()) if 'date' in v2pred else None}},indent=2),encoding='utf-8')
    mappings={
      v1/'artifacts/model_bundle.joblib':ROOT/'artifacts/model_bundle.joblib', v1/'artifacts/model_metadata.json':ROOT/'artifacts/model_metadata_v1.json', v1/'data/processed/latest_team_snapshot.json':ROOT/'data/processed/latest_team_snapshot.json', v1/'data/processed/feature_schema.json':ROOT/'data/processed/feature_schema.json', v1/'data/metadata/data_validation.json':ROOT/'data/metadata/source_v1_data_validation.json', v1/'reports/model_comparison.csv':ROOT/'reports/v1_model_comparison.csv', v1/'reports/expanding_window_backtest.csv':ROOT/'reports/v1_expanding_window_backtest.csv', v2/'reports/strict_test_predictions_v2.csv':ROOT/'reports/v2_strict_test_predictions.csv', v2/'reports/strict_test_by_year.csv':ROOT/'reports/v2_strict_test_by_year.csv', v2/'reports/FINAL_MODEL_ACCEPTANCE_REPORT.json':ROOT/'reports/V2_ACCEPTANCE_REPORT.json', v2/'artifacts/model_metadata_v2.json':ROOT/'artifacts/model_metadata_v2.json', v2/'data/metadata/world_cup_2027_format.json':ROOT/'data/metadata/world_cup_2027_format.json', v2/'data/metadata/world_cup_2027_venues.csv':ROOT/'data/metadata/world_cup_2027_venues.csv'
    }
    for s,d in mappings.items(): copy(s,d)
    pi=manual_permutation(bundle,common[cols].copy(),common['target'].to_numpy(),cols,repeats=5); pi.to_csv(ROOT/'reports/production_permutation_importance.csv',index=False)
    manifest={'production_project':'ICC_World_Cup_Intelligence_Production','champion_model':'V1','accepted_simulator':'V2_2027','source_v1':str(v1),'source_v2':str(v2),'v1_data_mode':dv.get('mode'),'v1_status':dv.get('status'),'frozen_v1_metrics':m1,'v2_challenger_metrics':m2,'files':[]}
    for p in [ROOT/'artifacts/model_bundle.joblib',ROOT/'artifacts/model_metadata_v1.json',ROOT/'data/processed/latest_team_snapshot.json',ROOT/'data/processed/feature_schema.json',ROOT/'reports/FROZEN_V1_V2_COMPARISON.csv',ROOT/'reports/FROZEN_CHAMPION_METRICS.json',ROOT/'reports/production_permutation_importance.csv']:
        manifest['files'].append({'path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    (ROOT/'artifacts/PRODUCTION_ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); (ROOT/'artifacts/PRODUCTION_READY.flag').unlink(missing_ok=True)
    print('='*78); print('WORLD CUP PRODUCTION COMPONENT IMPORT COMPLETE'); print('='*78); print('Champion model       : V1 actual local model bundle'); print('Tournament engine    : V2 corrected 2027 structure'); print('Exact common rows    :',len(common)); print(f"V1 accuracy / AUC    : {m1['accuracy']:.4f} / {m1['roc_auc']:.4f}"); print(f"V2 accuracy / AUC    : {m2['accuracy']:.4f} / {m2['roc_auc']:.4f}"); print('V2 model decision    : REJECTED by pre-registered objective'); print('Explainability       : rebuilt with manual permutation log-loss importance'); print('Production ready?    : NOT YET - run 02_VERIFY_PRODUCTION_PACKAGE.bat')
if __name__=='__main__': main()
