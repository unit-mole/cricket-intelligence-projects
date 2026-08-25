import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from cricket_intel.models import ProbabilityEnsembleV2, IdentityCalibrator, augment_symmetry, optimize_ensemble_weights


def test_symmetry_augmentation_balances_orientation():
    X = pd.DataFrame({"a":[1.0,2.0],"b":[3.0,4.0]})
    y = pd.Series([1,0])
    Xa, ya = augment_symmetry(X,y)
    assert len(Xa)==4
    assert np.allclose(Xa.iloc[2:].to_numpy(), -X.to_numpy())
    assert ya.tolist()==[1,0,0,1]


def test_bundle_probability_is_exactly_symmetric():
    X = pd.DataFrame({"f1":[-2,-1,1,2],"f2":[-.5,-.2,.2,.5]})
    y = np.array([0,0,1,1])
    model=LogisticRegression().fit(X,y)
    bundle=ProbabilityEnsembleV2({"m":model},{"m":1.0},IdentityCalibrator(),["f1","f2"],"pretoss")
    q=pd.DataFrame({"f1":[1.3],"f2":[.4]})
    p=float(bundle.predict_proba(q)[0,1])
    r=float(bundle.predict_proba(-q)[0,1])
    assert abs(p-(1-r))<1e-12


def test_weight_optimizer_can_reject_bad_model():
    y=np.array([0,0,0,1,1,1,0,1,0,1])
    P=pd.DataFrame({
        "good":[.1,.2,.25,.8,.75,.9,.2,.8,.3,.7],
        "bad":[.9,.8,.75,.2,.25,.1,.8,.2,.7,.3],
        "flat":[.5]*10,
    })
    w=optimize_ensemble_weights(y,P,["good","bad","flat"])
    assert w.get("good",0)>0.8
    assert w.get("bad",0)<0.05
