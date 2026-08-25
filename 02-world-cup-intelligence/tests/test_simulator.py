import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from world_cup_production.simulator_2027 import *

def teams(): return [f'T{i}' for i in range(14)]
def cache(ts):
    c={}
    for i,a in enumerate(ts):
        for b in ts[i+1:]: c[(a,b)]=.55; c[(b,a)]=.45
    return c

def test_participant_count(): validate_participants(teams())
def test_duplicate_rejected():
    import pytest
    with pytest.raises(ValueError): validate_participants(teams()[:-1]+['T0'])
def test_wrong_count_rejected():
    import pytest
    with pytest.raises(ValueError): validate_participants(teams()[:-1])
def test_official_counts(): assert official_match_counts()=={'super_series':3,'round2':30,'super7':21,'semifinals':2,'final':1,'total':57}
def test_one_trace_counts():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert tr.match_counts==official_match_counts()
def test_super_series_three_teams():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert len(tr.super_series_teams)==3
def test_one_super_series_survivor():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert tr.super_series_winner in tr.super_series_teams
def test_round2_twelve():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert len(tr.round2_teams)==12
def test_groups_six_each():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert len(tr.group_a)==len(tr.group_b)==6
def test_super7_seven():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert len(tr.super7_teams)==7
def test_semifinalists_four():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert len(tr.semifinalists)==4
def test_semifinal_pairing_1v4_2v3():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert tr.semifinal_pairings==[(tr.semifinalists[0],tr.semifinalists[3]),(tr.semifinalists[1],tr.semifinalists[2])]
def test_two_finalists():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert len(tr.finalists)==2
def test_one_champion():
    t=teams(); tr=simulate_one_2027(cache(t),{x:1500-i for i,x in enumerate(t)},t,np.random.default_rng(1)); assert tr.champion in tr.finalists
