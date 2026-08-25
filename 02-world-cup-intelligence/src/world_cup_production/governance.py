from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,log_loss,brier_score_loss,roc_auc_score

def ece_score(y,p,bins=10):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); edges=np.linspace(0,1,bins+1); total=0.0
    for i in range(bins):
        m=(p>=edges[i]) & (p<(edges[i+1]) if i<bins-1 else p<=edges[i+1])
        if m.any(): total += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(total)

def binary_metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6); pred=(p>=.5).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'f1':float(f1_score(y,pred,zero_division=0)),'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),'ece':ece_score(y,p),'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float('nan')}

def probability_objective(m):
    # Same objective family used by the V2 acceptance gate; lower is better.
    return float(m['log_loss'] + 0.25*m['brier'] + 0.10*m['ece'])
