from __future__ import annotations
from pathlib import Path
import hashlib,json,joblib,pandas as pd,numpy as np
from cricket_intel.features import matchup_from_snapshot

class ProductionRuntime:
    def __init__(self,root:Path):
        self.root=Path(root)
        self.bundle=joblib.load(self.root/'artifacts/model_bundle.joblib')
        self.snapshot=json.loads((self.root/'data/processed/latest_team_snapshot.json').read_text(encoding='utf-8'))
    @property
    def teams(self):
        return sorted(self.snapshot.get('teams',{}),key=lambda t:float(self.snapshot['teams'][t].get('elo',1500)),reverse=True)
    def probability(self,a,b):
        if a==b: return 0.5
        X=matchup_from_snapshot(self.snapshot,a,b)
        return float(np.clip(self.bundle.predict_proba(X)[:,1][0],1e-6,1-1e-6))

def sha256_file(path:Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def ensure_ready(root:Path,require_flag=True):
    req=['artifacts/model_bundle.joblib','data/processed/latest_team_snapshot.json','reports/FROZEN_V1_V2_COMPARISON.csv','reports/FROZEN_CHAMPION_METRICS.json']
    missing=[x for x in req if not (root/x).exists()]
    if require_flag and not (root/'artifacts/PRODUCTION_READY.flag').exists(): missing.append('artifacts/PRODUCTION_READY.flag')
    if missing: raise RuntimeError('Production package is not ready. Missing: '+', '.join(missing))
