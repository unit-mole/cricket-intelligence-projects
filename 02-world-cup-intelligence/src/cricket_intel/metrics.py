
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, brier_score_loss, roc_auc_score

def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); total=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
        if mask.any(): total += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(total)

def binary_metrics(y,p):
    pred=(np.asarray(p)>=0.5).astype(int)
    out={"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"f1":float(f1_score(y,pred)),
         "log_loss":float(log_loss(y,np.c_[1-np.asarray(p),np.asarray(p)],labels=[0,1])),"brier":float(brier_score_loss(y,p)),"ece":ece(y,p)}
    try: out["roc_auc"]=float(roc_auc_score(y,p))
    except Exception: out["roc_auc"]=None
    return out
