
from __future__ import annotations
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression as Platt
from .features import FEATURE_COLUMNS
from .metrics import binary_metrics
from .utils import seed_everything, dump_json

class ProbabilityEnsemble:
    def __init__(self, models, weights, calibrator=None): self.models=models; self.weights=weights; self.calibrator=calibrator
    def raw_probability(self,X):
        ps=[]
        for name,m in self.models.items(): ps.append(m.predict_proba(X)[:,1])
        P=np.vstack(ps).T; w=np.array([self.weights[n] for n in self.models]); w=w/w.sum(); return P@w
    def predict_proba(self,X):
        p=self.raw_probability(X)
        if self.calibrator is not None: p=self.calibrator.predict_proba(p.reshape(-1,1))[:,1]
        return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)


def _optional_models(seed=42, use_gpu=True):
    out={}
    try:
        from xgboost import XGBClassifier
        kw={"n_estimators":500,"max_depth":4,"learning_rate":0.035,"subsample":0.85,"colsample_bytree":0.9,"eval_metric":"logloss","random_state":seed,"n_jobs":-1}
        if use_gpu: kw.update({"device":"cuda","tree_method":"hist"})
        out["xgboost"]=XGBClassifier(**kw)
    except Exception: pass
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"]=LGBMClassifier(n_estimators=500,num_leaves=31,learning_rate=0.03,subsample=0.85,colsample_bytree=0.9,random_state=seed,verbosity=-1)
    except Exception: pass
    try:
        from catboost import CatBoostClassifier
        out["catboost"]=CatBoostClassifier(iterations=600,depth=6,learning_rate=0.035,loss_function="Logloss",verbose=False,random_seed=seed,
            task_type="GPU" if use_gpu else "CPU",allow_writing_files=False)
    except Exception: pass
    return out


def model_candidates(seed=42,use_gpu=True):
    base={
      "logistic":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2000,C=0.5,random_state=seed))]),
      "random_forest":RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_leaf=3,class_weight="balanced_subsample",n_jobs=-1,random_state=seed),
      "extra_trees":ExtraTreesClassifier(n_estimators=500,max_depth=12,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=seed),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=400,learning_rate=0.04,max_leaf_nodes=23,l2_regularization=1.0,random_state=seed),
      "gradient_boosting":GradientBoostingClassifier(n_estimators=350,learning_rate=0.03,max_depth=3,random_state=seed),
    }
    base.update(_optional_models(seed,use_gpu)); return base


def chronological_split(df, train_frac=.70, val_frac=.15):
    df=df.sort_values("date").reset_index(drop=True); n=len(df); a=int(n*train_frac); b=int(n*(train_frac+val_frac))
    return df.iloc[:a],df.iloc[a:b],df.iloc[b:]

def augment_symmetry(X,y):
    X2=pd.concat([X,-X],ignore_index=True); y2=pd.concat([pd.Series(y).reset_index(drop=True),1-pd.Series(y).reset_index(drop=True)],ignore_index=True); return X2,y2

def train_all(feature_csv: Path, artifacts: Path, reports: Path, seed=42, use_gpu=True):
    seed_everything(seed); df=pd.read_csv(feature_csv,parse_dates=["date"]).dropna(subset=FEATURE_COLUMNS+["target"])
    tr,va,te=chronological_split(df)
    Xtr,ytr=augment_symmetry(tr[FEATURE_COLUMNS],tr.target); Xv,yv=augment_symmetry(va[FEATURE_COLUMNS],va.target); Xt,yt=augment_symmetry(te[FEATURE_COLUMNS],te.target)
    models={}; rows=[]; probs={}
    for name,m in model_candidates(seed,use_gpu).items():
        try:
            m.fit(Xtr,ytr); pv=m.predict_proba(Xv)[:,1]; pt=m.predict_proba(Xt)[:,1]; met=binary_metrics(yt,pt); vm=binary_metrics(yv,pv)
            row={"model":name,**{f"val_{k}":v for k,v in vm.items()},**{f"test_{k}":v for k,v in met.items()}}; rows.append(row); models[name]=m; probs[name]=(pv,pt)
        except Exception as e:
            rows.append({"model":name,"error":str(e)})
    good=[r for r in rows if "val_log_loss" in r]
    if not good: raise RuntimeError("No model trained successfully")
    inv={r["model"]:1/max(r["val_log_loss"],1e-6) for r in good}; s=sum(inv.values()); weights={k:v/s for k,v in inv.items()}
    pv=sum(weights[k]*probs[k][0] for k in weights); pt_raw=sum(weights[k]*probs[k][1] for k in weights)
    cal=Platt(max_iter=1000).fit(pv.reshape(-1,1),yv); pt=cal.predict_proba(pt_raw.reshape(-1,1))[:,1]
    ensmet=binary_metrics(yt,pt); val_cal=cal.predict_proba(pv.reshape(-1,1))[:,1]; valmet=binary_metrics(yv,val_cal)
    rows.append({"model":"calibrated_ensemble",**{f"val_{k}":v for k,v in valmet.items()},**{f"test_{k}":v for k,v in ensmet.items()}})
    ensemble=ProbabilityEnsemble(models,weights,cal)
    artifacts.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    joblib.dump(ensemble,artifacts/"model_bundle.joblib"); dump_json(weights,artifacts/"ensemble_weights.json")
    pd.DataFrame(rows).to_csv(reports/"model_comparison.csv",index=False)
    data_mode="UNKNOWN"
    dv=feature_csv.parent.parent/"metadata/data_validation.json"
    if dv.exists():
        try: data_mode=json.loads(dv.read_text()).get("mode","UNKNOWN")
        except Exception: pass
    meta={"seed":seed,"data_mode":data_mode,"rows":len(df),"train_rows":len(tr),"validation_rows":len(va),"test_rows":len(te),
          "train_end":str(tr.date.max().date()),"validation_start":str(va.date.min().date()),"validation_end":str(va.date.max().date()),"test_start":str(te.date.min().date()),"test_end":str(te.date.max().date()),
          "features":FEATURE_COLUMNS,"ensemble_test_metrics":ensmet,"ensemble_validation_metrics":valmet,"component_weights":weights}
    dump_json(meta,artifacts/"model_metadata.json"); return pd.DataFrame(rows),meta

def load_bundle(path: Path): return joblib.load(path)
