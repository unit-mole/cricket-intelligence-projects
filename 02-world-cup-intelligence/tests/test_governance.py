import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from world_cup_production.governance import binary_metrics,probability_objective

def test_metrics_bounds():
    m=binary_metrics([0,1,0,1],[.2,.8,.3,.7]); assert 0<=m['accuracy']<=1 and 0<=m['brier']<=1 and 0<=m['ece']<=1
def test_perfect_better_than_coin():
    a=binary_metrics([0,1,0,1],[.01,.99,.01,.99]); b=binary_metrics([0,1,0,1],[.5,.5,.5,.5]); assert probability_objective(a)<probability_objective(b)
def test_objective_lower_is_better(): assert probability_objective({'log_loss':.6,'brier':.2,'ece':.05}) < probability_objective({'log_loss':.7,'brier':.25,'ece':.1})
def test_ece_zeroish_for_balanced_coin(): assert binary_metrics([0,1]*50,[.5]*100)['ece']<1e-12
